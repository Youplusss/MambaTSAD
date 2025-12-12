# mambatsad/engine/trainer_adv.py
# -*- coding: utf-8 -*-
"""
实验分支：共享 encoder + 对抗训练版本 Trainer。

本文件配合 mambatsad/models/hybrid_shared_adv.py 使用，用于探索
类似 STAMP / DADA 的「预测 + 重构 + 对抗」训练策略。

与主分支 TSADTrainer 的主要区别：
--------------------------------
1. 只支持混合模型（hybrid_shared_adv），即同时包含重构头与预测头；
2. 可选两阶段训练：
   - warmup 阶段（前 adv_warmup_epochs 轮）：
       只优化重构损失 + 预测损失，相当于普通的多任务学习；
   - adversarial 阶段：
       每个 batch 先更新「生成器」（预测头 + 共享 encoder），
       再更新「判别器」（重构头 + 共享 encoder），
       目标为：
           · 生成器：让预测结果既接近真实未来，又能被重构头
                     认为是“正常”（重构误差小）；
           · 判别器：要求对真实序列重构误差小，对预测序列
                     重构误差至少大于一定 margin，
                     强制放大正常 / 异常之间的重构差异。

3. 评估阶段完全复用 TSADTrainer 的思想：
   - 计算重构分数 / 预测分数；
   - 使用 search_best_f1_threshold_with_auto_flip 自动选择
     分数方向与最优阈值；
   - 使用 fuse_scores_by_zscore 做分数融合；
   - 记录并可视化混合分支的得分曲线。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from mambatsad.utils.metrics import search_best_f1_threshold_with_auto_flip
from mambatsad.utils.score_fusion import fuse_scores_by_zscore
from mambatsad.utils.visualization import plot_scores_with_labels


class AdvHybridTrainer:
    """共享 encoder + 对抗训练版混合模型 Trainer。"""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        labels_list: Sequence[np.ndarray],
        log_dir: str,
        win_size: int,
        pred_len: int,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        use_adv_training: bool = True,
        adv_warmup_epochs: int = 5,
        lambda_rec: float = 1.0,
        lambda_pred: float = 1.0,
        lambda_adv1: float = 0.5,
        lambda_adv2: float = 0.5,
        use_point_adjust: bool = True,
    ) -> None:
        """
        参数与 main_adv.py 中保持一致。
        """
        self.device = device
        self.model = model.to(device)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels_list = list(labels_list)
        self.log_dir = log_dir

        self.win_size = win_size
        self.pred_len = pred_len
        if win_size <= pred_len:
            raise ValueError(
                f"win_size={win_size} 必须大于 pred_len={pred_len} 才能进行预测。"
            )
        self.context_len = win_size - pred_len

        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and (device.type == "cuda")
        self.use_adv_training = use_adv_training
        self.adv_warmup_epochs = max(int(adv_warmup_epochs), 0)

        self.lambda_rec = float(lambda_rec)
        self.lambda_pred = float(lambda_pred)
        self.lambda_adv1 = float(lambda_adv1)
        self.lambda_adv2 = float(lambda_adv2)
        if self.lambda_rec < 0 or self.lambda_pred < 0:
            raise ValueError("lambda_rec / lambda_pred 必须为非负数。")

        # 对抗 margin：希望「预测序列的重构误差」
        # 至少比「真实序列的重构误差」大 margin
        self.adv_margin = 0.01

        self.use_point_adjust = use_point_adjust

        # 统一使用 AdamW；对抗阶段会拆成 main/gen/disc 三个优化器
        self.optimizer_main = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 生成器：预测头 + 共享 encoder
        gen_params = list(self.model.shared_encoder.parameters()) + list(
            self.model.forecast_head.parameters()
        )
        self.optimizer_gen = torch.optim.AdamW(
            gen_params, lr=lr, weight_decay=weight_decay
        )

        # 判别器：重构头 + 共享 encoder
        disc_params = list(self.model.shared_encoder.parameters()) + list(
            self.model.recon_head.parameters()
        )
        self.optimizer_disc = torch.optim.AdamW(
            disc_params, lr=lr, weight_decay=weight_decay
        )

        self.best_f1: float = -1.0
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = -1

    # ==================================================================
    # 工具函数
    # ==================================================================

    def _safe_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """与主 Trainer 中一致的安全 MSE。"""
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        pred = torch.clamp(pred, -1e4, 1e4)
        target = torch.clamp(target, -1e4, 1e4)
        diff = pred - target
        return (diff ** 2).mean()

    def _init_score_buffers(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        recon_scores_list: List[np.ndarray] = []
        forecast_scores_list: List[np.ndarray] = []
        for labels in self.labels_list:
            T = len(labels)
            recon_scores_list.append(np.zeros(T, dtype=np.float32))
            forecast_scores_list.append(np.zeros(T, dtype=np.float32))
        return recon_scores_list, forecast_scores_list

    # ==================================================================
    # 训练主入口
    # ==================================================================

    def train(self, epochs: int, logger) -> None:
        no_improve = 0

        for epoch in range(1, epochs + 1):
            if (not self.use_adv_training) or (epoch <= self.adv_warmup_epochs):
                logger.info(
                    f"========== Epoch {epoch}/{epochs} [warmup / non-adv] =========="
                )
                train_stats = self._train_epoch_warmup(epoch, logger)
            else:
                logger.info(
                    f"========== Epoch {epoch}/{epochs} [adversarial] =========="
                )
                train_stats = self._train_epoch_adversarial(epoch, logger)

            metrics_recon, metrics_forecast, metrics_hybrid = self.evaluate(
                epoch, logger
            )
            curr_metrics = metrics_hybrid if metrics_hybrid else metrics_recon
            curr_f1 = float(curr_metrics.get("f1", 0.0))

            if curr_f1 > self.best_f1 + 1e-6:
                self.best_f1 = curr_f1
                self.best_metrics = curr_metrics
                self.best_epoch = epoch
                no_improve = 0
                best_path = os.path.join(self.log_dir, "best_model_adv.pt")
                torch.save(self.model.state_dict(), best_path)
                logger.info(
                    f"[ADV] 发现更优模型 (F1={curr_f1:.4f})，已保存至 {best_path}"
                )
            else:
                no_improve += 1
                logger.info(
                    f"[ADV] 当前 F1={curr_f1:.4f}，已连续 {no_improve} 个 epoch 未提升。"
                )

        logger.info(
            f"[ADV] 训练结束，最佳 F1={self.best_f1:.4f} (epoch={self.best_epoch})，"
            f"最佳指标={self.best_metrics}"
        )

    # ==================================================================
    # warmup：普通混合模型训练
    # ==================================================================

    def _train_epoch_warmup(self, epoch: int, logger) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_rec_sum = 0.0
        loss_fore_sum = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-Warmup-{epoch}"):
            x = batch["window"].to(self.device)  # [B, L, D]
            y_true = x[:, -self.pred_len :, :]  # [B, T_pred, D]

            self.optimizer_main.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                out = self.model(x)
                x_hat = out["recon"]
                y_pred = out["pred"]

                loss_recon = self._safe_mse_loss(x_hat, x)
                loss_fore = self._safe_mse_loss(y_pred, y_true)
                loss = self.lambda_rec * loss_recon + self.lambda_pred * loss_fore

            if not torch.isfinite(loss):
                logger.warning("Warmup 阶段遇到 NaN/Inf loss，跳过该 batch。")
                continue

            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer_main.step()

            total_loss += float(loss.detach().cpu())
            loss_rec_sum += float(loss_recon.detach().cpu())
            loss_fore_sum += float(loss_fore.detach().cpu())
            n_batches += 1

        avg_total = total_loss / max(n_batches, 1)
        avg_rec = loss_rec_sum / max(n_batches, 1)
        avg_fore = loss_fore_sum / max(n_batches, 1)
        logger.info(
            f"[Warmup-Epoch {epoch}] "
            f"loss_total={avg_total:.6f}, loss_rec={avg_rec:.6f}, loss_fore={avg_fore:.6f}"
        )

        return {
            "loss": avg_total,
            "loss_recon": avg_rec,
            "loss_forecast": avg_fore,
        }

    # ==================================================================
    # adversarial：生成器 + 判别器交替更新
    # ==================================================================

    def _set_requires_grad(self, module: nn.Module, flag: bool) -> None:
        for p in module.parameters():
            p.requires_grad = flag

    def _train_epoch_adversarial(self, epoch: int, logger) -> Dict[str, float]:
        self.model.train()
        total_g = 0.0
        total_d = 0.0
        rec_pos_sum = 0.0
        rec_neg_sum = 0.0
        pred_sum = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-ADV-{epoch}"):
            x = batch["window"].to(self.device)  # [B, L, D]
            y_true = x[:, -self.pred_len :, :]  # [B, T_pred, D]
            x_context = x[:, : self.context_len, :]  # [B, L_c, D]

            # ==========================================================
            # (1) 更新生成器：shared_encoder + forecast_head
            # ==========================================================
            self._set_requires_grad(self.model.recon_head, False)
            self._set_requires_grad(self.model.forecast_head, True)
            self._set_requires_grad(self.model.shared_encoder, True)

            self.optimizer_gen.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                # 生成预测序列
                h_context = self.model.encode(x)[:, : self.context_len, :]
                y_pred = self.model.decode_forecast(h_context, x_context)

                # 预测损失：逼近真实未来
                loss_pred = self._safe_mse_loss(y_pred, y_true)

                # 构造“伪样本”：用预测序列拼接上下文
                x_neg = torch.cat([x_context, y_pred], dim=1)  # [B, L, D]
                h_neg = self.model.encode(x_neg)
                x_hat_neg = self.model.decode_recon(h_neg)

                # 希望生成器产生「看起来正常」的序列 —— 即被重构头重构得很好
                loss_rec_neg = self._safe_mse_loss(x_hat_neg, x)

                loss_g = self.lambda_pred * loss_pred + self.lambda_adv1 * loss_rec_neg

            if torch.isfinite(loss_g):
                loss_g.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer_gen.step()
            else:
                logger.warning("生成器更新遇到 NaN/Inf loss，跳过该 batch。")
                continue

            # ==========================================================
            # (2) 更新判别器：shared_encoder + recon_head
            # ==========================================================
            self._set_requires_grad(self.model.recon_head, True)
            self._set_requires_grad(self.model.forecast_head, False)
            self._set_requires_grad(self.model.shared_encoder, True)

            self.optimizer_disc.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                # 正样本：真实序列的重构误差应尽量小
                h_pos = self.model.encode(x)
                x_hat_pos = self.model.decode_recon(h_pos)
                loss_rec_pos = self._safe_mse_loss(x_hat_pos, x)

                # 负样本：使用「当前生成器」产生的预测序列
                with torch.no_grad():
                    h_context_det = self.model.encode(x)[:, : self.context_len, :]
                    y_pred_det = self.model.decode_forecast(h_context_det, x_context)
                    x_neg_det = torch.cat([x_context, y_pred_det], dim=1)

                # 重新编码负样本以更新 shared_encoder + recon_head
                h_neg2 = self.model.encode(x_neg_det)
                x_hat_neg2 = self.model.decode_recon(h_neg2)
                loss_rec_neg2 = self._safe_mse_loss(x_hat_neg2, x)

                # margin-based 对抗目标：
                #   希望 loss_rec_neg2 >= loss_rec_pos + margin
                adv_term = torch.relu(
                    self.adv_margin + loss_rec_pos - loss_rec_neg2
                )

                loss_d = self.lambda_rec * loss_rec_pos + self.lambda_adv2 * adv_term

            if torch.isfinite(loss_d):
                loss_d.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer_disc.step()
            else:
                logger.warning("判别器更新遇到 NaN/Inf loss，跳过该 batch。")
                continue

            # 记录若干统计量（注意这里使用 pos/neg2 对应判别器的视角）
            total_g += float(loss_g.detach().cpu())
            total_d += float(loss_d.detach().cpu())
            rec_pos_sum += float(loss_rec_pos.detach().cpu())
            rec_neg_sum += float(loss_rec_neg2.detach().cpu())
            pred_sum += float(loss_pred.detach().cpu())
            n_batches += 1

        if n_batches == 0:
            logger.warning("对抗训练整个 epoch 没有有效 batch（可能全部 NaN/Inf）。")
            return {
                "loss": 0.0,
                "loss_recon": 0.0,
                "loss_forecast": 0.0,
                "loss_gen": 0.0,
                "loss_disc": 0.0,
            }

        avg_g = total_g / n_batches
        avg_d = total_d / n_batches
        avg_rec_pos = rec_pos_sum / n_batches
        avg_rec_neg = rec_neg_sum / n_batches
        avg_pred = pred_sum / n_batches

        logger.info(
            f"[ADV-Epoch {epoch}] "
            f"loss_gen={avg_g:.6f}, loss_disc={avg_d:.6f}, "
            f"rec_pos={avg_rec_pos:.6f}, rec_neg={avg_rec_neg:.6f}, "
            f"loss_pred={avg_pred:.6f}"
        )

        return {
            "loss": avg_g + avg_d,
            "loss_recon": avg_rec_pos,
            "loss_forecast": avg_pred,
            "loss_gen": avg_g,
            "loss_disc": avg_d,
        }

    # ==================================================================
    # 评估逻辑（与主 Trainer 基本一致，简化为 hybrid 一种情况）
    # ==================================================================

    def evaluate(
        self, epoch: int, logger
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        self.model.eval()

        recon_scores_list, forecast_scores_list = self._init_score_buffers()
        labels_list = self.labels_list

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Eval-ADV-Epoch{epoch}"):
                x = batch["window"].to(self.device)  # [B, L, D]
                seq_idx = batch["seq_idx"].cpu().numpy().astype(int)
                start = batch["start"].cpu().numpy().astype(int)
                B = x.shape[0]

                out = self.model(x)
                x_hat = out["recon"]
                y_pred = out["pred"]
                y_true = x[:, -self.pred_len :, :]

                rec_err = ((x_hat - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
                fore_err = ((y_pred - y_true) ** 2).mean(dim=(1, 2)).cpu().numpy()

                for i in range(B):
                    idx = seq_idx[i]
                    s = start[i]
                    e = s + self.win_size

                    # 重构分数：整个窗口均赋值 rec_err[i]
                    recon_scores_list[idx][s:e] = np.maximum(
                        recon_scores_list[idx][s:e], rec_err[i]
                    )

                    # 预测分数：只作用于未来 T_pred 段
                    s_f = s + self.context_len
                    e_f = s + self.win_size
                    forecast_scores_list[idx][s_f:e_f] = np.maximum(
                        forecast_scores_list[idx][s_f:e_f], fore_err[i]
                    )

        labels_all = np.concatenate(labels_list, axis=0)
        recon_scores = np.concatenate(recon_scores_list, axis=0)
        forecast_scores = np.concatenate(forecast_scores_list, axis=0)

        metrics_recon: Dict[str, float] = {}
        metrics_forecast: Dict[str, float] = {}
        metrics_hybrid: Dict[str, float] = {}

        # -------- 重构分支 --------
        thr_r, metrics_r = search_best_f1_threshold_with_auto_flip(
            recon_scores,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )
        metrics_recon = metrics_r
        logger.info(
            "[ADV-Eval-recon] "
            f"F1={metrics_r['f1']:.4f}, P={metrics_r['precision']:.4f}, "
            f"R={metrics_r['recall']:.4f}, AUC={metrics_r['auc']:.4f}, "
            f"thr={metrics_r['threshold']:.6f}, "
            f"need_flip={metrics_r.get('need_flip', False)}, "
            f"dir={metrics_r.get('direction', 'greater')}, "
            f"point_adjust={self.use_point_adjust}"
        )

        # -------- 预测分支 --------
        thr_f, metrics_f = search_best_f1_threshold_with_auto_flip(
            forecast_scores,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )
        metrics_forecast = metrics_f
        logger.info(
            "[ADV-Eval-forecast] "
            f"F1={metrics_f['f1']:.4f}, P={metrics_f['precision']:.4f}, "
            f"R={metrics_f['recall']:.4f}, AUC={metrics_f['auc']:.4f}, "
            f"thr={metrics_f['threshold']:.6f}, "
            f"need_flip={metrics_f.get('need_flip', False)}, "
            f"dir={metrics_f.get('direction', 'greater')}, "
            f"point_adjust={self.use_point_adjust}"
        )

        # -------- 混合分支：z-score + 线性加权 --------
        fused_scores = fuse_scores_by_zscore(
            recon_scores,
            forecast_scores,
            w_recon=self.lambda_rec,
            w_forecast=self.lambda_pred,
        )
        thr_h, metrics_h = search_best_f1_threshold_with_auto_flip(
            fused_scores,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )
        metrics_hybrid = metrics_h
        logger.info(
            "[ADV-Eval-hybrid] "
            f"F1={metrics_h['f1']:.4f}, P={metrics_h['precision']:.4f}, "
            f"R={metrics_h['recall']:.4f}, AUC={metrics_h['auc']:.4f}, "
            f"thr={metrics_h['threshold']:.6f}, "
            f"need_flip={metrics_h.get('need_flip', False)}, "
            f"dir={metrics_h.get('direction', 'greater')}, "
            f"point_adjust={self.use_point_adjust}; "
            f"recon_F1={metrics_recon.get('f1', 0.0):.4f}, "
            f"forecast_F1={metrics_forecast.get('f1', 0.0):.4f}"
        )

        # 可视化第一条实体
        first_len = len(labels_list[0])
        scores_vis = fused_scores[:first_len]
        if metrics_h.get("need_flip", False):
            scores_vis = -scores_vis
        thr_vis = metrics_h["threshold"]
        vis_path = os.path.join(self.log_dir, f"scores_epoch{epoch}_adv_hybrid.png")
        plot_scores_with_labels(
            scores=scores_vis,
            labels=labels_list[0],
            threshold=thr_vis,
            save_path=vis_path,
        )
        logger.info(f"[ADV] 已保存可视化图：{vis_path}")

        return metrics_recon, metrics_forecast, metrics_hybrid
