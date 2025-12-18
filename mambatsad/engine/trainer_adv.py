# mambatsad/engine/trainer_adv.py
# -*- coding: utf-8 -*-
"""
TSADAdvTrainer：在原 hybrid 模型基础上的「重构+预测+输入对抗扰动」训练器。

设计目标
--------
1. 复用已有的重构分支 / 预测分支实现：
   - 通过 mambatsad.models.hybrid.build_hybrid_model 构建模型；
   - 不再使用之前单独写的 HybridSharedAdvModel；
2. 训练流程与 TSADTrainer 保持尽量一致：
   - 同样基于滑动窗口多实体数据集（SMD/MSL/SWAT/WADI/SMAP）；
   - 同样在测试集上使用 search_best_f1_threshold_with_auto_flip +
     fuse_scores_by_zscore 做阈值搜索与分数融合；
3. 对抗训练采用「输入 FGSM 扰动」的形式：
   - 先在当前模型参数下，仅对输入窗口 x 求梯度，生成 x_adv；
   - 再用 x（干净）和 x_adv（对抗）共同反向，强化模型对微小扰动的鲁棒性。
4. 暂不包含伪标签逻辑，简化为一个稳健的 baseline 版本。

注意事项
--------
- 由于没有直接复用 TSADTrainer 的内部实现，本文件是“平行”的一个 Trainer，
  不会影响原有 main.py + TSADTrainer 的行为；
- 你可以用 main.py 跑“普通版”，用 main_adv.py 跑“对抗训练版”，二者互不干扰。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from mambatsad.models.hybrid import build_hybrid_model
from mambatsad.utils.metrics import search_best_f1_threshold_with_auto_flip
from mambatsad.utils.score_fusion import fuse_scores_by_zscore
from mambatsad.utils.visualization import plot_scores_with_labels


class TSADAdvTrainer:
    """
    对抗训练版的混合模型 Trainer（仅 hybrid 分支）。

    参数基本与 main_adv.py 中保持一致。
    """

    def __init__(
        self,
        device: torch.device,
        input_dim: int,
        win_size: int,
        pred_len: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        labels_list: Sequence[np.ndarray],
        entity_ids: Sequence[str],
        logger,
        writer,
        log_dir: str,
        lr: float = 1e-4,
        weight_decay: float = 5e-4,
        max_grad_norm: float = 1.0,
        patience: int = 8,
        use_point_adjust: bool = True,
        use_amp: bool = True,
        lambda_recon: float = 1.0,
        lambda_forecast: float = 1.0,
        use_adv_training: bool = True,
        adv_epsilon: float = 0.05,
        adv_beta: float = 0.5,
        adv_warmup_epochs: int = 5,
    ) -> None:
        self.device = device
        self.input_dim = int(input_dim)
        self.win_size = int(win_size)
        self.pred_len = int(pred_len)
        if self.win_size <= self.pred_len:
            raise ValueError(
                f"win_size={self.win_size} 必须大于 pred_len={self.pred_len}，"
                "才能进行预测式异常检测。"
            )
        self.context_len = self.win_size - self.pred_len

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels_list: List[np.ndarray] = [
            np.asarray(lab, dtype=int) for lab in labels_list
        ]
        self.entity_ids = list(entity_ids)
        self.logger = logger
        self.writer = writer
        self.log_dir = log_dir

        self.max_grad_norm = float(max_grad_norm)
        self.patience = int(patience)
        self.use_point_adjust = bool(use_point_adjust)

        self.lambda_recon = float(lambda_recon)
        self.lambda_forecast = float(lambda_forecast)
        if self.lambda_recon < 0 or self.lambda_forecast < 0:
            raise ValueError("lambda_recon / lambda_forecast 必须为非负数。")

        # 对抗训练相关
        self.use_adv_training = bool(use_adv_training)
        self.adv_epsilon = float(adv_epsilon)
        self.adv_beta = float(adv_beta)
        self.adv_warmup_epochs = max(int(adv_warmup_epochs), 0)

        # -------- 构建 hybrid 模型（复用原重构+预测分支结构） --------
        self.model: nn.Module = build_hybrid_model(
            input_dim=self.input_dim,
            win_size=self.win_size,
            pred_len=self.pred_len,
        ).to(self.device)
        self.logger.info(f"[ADV] 模型结构：\n{self.model}")

        # 优化器 + AMP
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        if use_amp and self.device.type == "cuda":
            self.scaler: GradScaler | None = GradScaler(enabled=True)
        else:
            self.scaler = None

        # 训练过程指标
        self.best_f1: float = -1.0
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = -1

    # ==================================================================
    # 工具函数
    # ==================================================================
    @staticmethod
    def _safe_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        在 float32 精度下安全地计算 MSE 损失，尽量规避 NaN/Inf。

        做法：
        1. nan_to_num 把 NaN/Inf 修成有限值；
        2. clamp 到较保守的数值范围；
        3. 再做均方。
        """
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        pred = torch.clamp(pred, -1e4, 1e4)
        target = torch.clamp(target, -1e4, 1e4)
        diff = pred - target
        return (diff * diff).mean()

    def _init_score_buffers(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        # 每个实体分配一条重构/预测分数序列，长度与标签相同
        recon_scores_list: List[np.ndarray] = []
        forecast_scores_list: List[np.ndarray] = []
        for labels in self.labels_list:
            T = len(labels)
            recon_scores_list.append(np.zeros(T, dtype=np.float32))
            forecast_scores_list.append(np.zeros(T, dtype=np.float32))
        return recon_scores_list, forecast_scores_list

    # ==================================================================
    # 生成 FGSM 对抗样本
    # ==================================================================
    def _generate_adversarial_examples(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        基于当前模型参数，对输入窗口 x 生成 FGSM 对抗样本。

        这里只对输入求一次梯度，不更新模型参数：
        1) x_adv_src = x.clone().requires_grad_(True)
        2) 正常前向计算重构+预测损失；
        3) 对 x_adv_src 反向，取梯度的符号作为扰动方向；
        4) x_adv = x + epsilon * sign(grad)；若梯度异常则返回 None。

        为了稳定性，这里 *不* 使用 AMP，而是强制在 float32 下计算。
        """
        self.model.eval()  # 生成扰动时使用 eval 模式，避免 Dropout 噪声
        x_adv_src = x.detach().clone().requires_grad_(True)

        # 只对 x_adv_src 相关的图计算梯度
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            out = self.model(x_adv_src)
            x_hat = out["recon"]
            y_pred = out["pred"]

            loss_rec = self._safe_mse_loss(x_hat, x_adv_src)
            loss_fore = self._safe_mse_loss(y_pred, y_true)
            loss = self.lambda_recon * loss_rec + self.lambda_forecast * loss_fore

        if not torch.isfinite(loss):
            self.logger.warning("生成对抗样本时 loss 非有限，跳过对抗扰动。")
            return None

        loss.backward()
        grad = x_adv_src.grad
        if grad is None or (not torch.isfinite(grad).all()):
            self.logger.warning(
                "生成对抗样本时梯度为 None 或包含 NaN/Inf，跳过对抗扰动。"
            )
            return None

        grad = torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)

        # 按样本做 L∞ 归一化，避免梯度爆炸
        grad_flat = grad.view(grad.size(0), -1)
        grad_norm = grad_flat.abs().max(dim=1, keepdim=True).values.view(-1, 1, 1)
        grad_norm = torch.clamp(grad_norm, min=1e-6)
        grad_normalized = grad / grad_norm

        x_adv = x + self.adv_epsilon * grad_normalized.sign()
        return x_adv.detach()

    # ==================================================================
    # 训练主循环
    # ==================================================================
    def train(self, num_epochs: int) -> None:
        no_improve = 0

        for epoch in range(1, num_epochs + 1):
            use_adv = (
                self.use_adv_training and (epoch > self.adv_warmup_epochs)
            )
            phase = "ADV" if use_adv else "WARMUP/NO-ADV"
            self.logger.info(
                f"========== [ADV] Epoch {epoch}/{num_epochs} ({phase}) =========="
            )

            train_stats = self._train_one_epoch(epoch, use_adv=use_adv)
            self.logger.info(
                f"[ADV-Train] Epoch {epoch}: "
                f"loss={train_stats['loss']:.6f}, "
                f"loss_clean={train_stats['loss_clean']:.6f}, "
                f"loss_adv={train_stats['loss_adv']:.6f}"
            )

            # 评估
            metrics_recon, metrics_forecast, metrics_hybrid = self.evaluate(epoch)

            curr_metrics = metrics_hybrid if metrics_hybrid else metrics_recon
            curr_f1 = float(curr_metrics.get("f1", 0.0))

            if curr_f1 > self.best_f1 + 1e-6:
                self.best_f1 = curr_f1
                self.best_metrics = curr_metrics
                self.best_epoch = epoch
                no_improve = 0

                best_path = os.path.join(self.log_dir, "best_model_adv.pt")
                torch.save(self.model.state_dict(), best_path)
                self.logger.info(
                    f"[ADV] 发现更优模型 (F1={curr_f1:.4f})，已保存至 {best_path}"
                )
            else:
                no_improve += 1
                self.logger.info(
                    f"[ADV] 当前 F1={curr_f1:.4f}，已连续 {no_improve} 个 epoch 未提升。"
                )

            if no_improve >= self.patience:
                self.logger.info(
                    f"[ADV] 早停触发：连续 {self.patience} 轮未提升，停止训练。"
                )
                break

        self.logger.info(
            f"[ADV] 训练结束，最佳 F1={self.best_f1:.4f} (epoch={self.best_epoch})，"
            f"最佳指标={self.best_metrics}"
        )

    # ==================================================================
    # 单个 epoch 的训练过程
    # ==================================================================
    def _train_one_epoch(self, epoch: int, use_adv: bool) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_clean = 0.0
        total_adv = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-ADV-{epoch}"):
            # batch["window"] 可能是 numpy，也可能已经是 tensor
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device).float()  # [B, L, D]
            y_true = x[:, -self.pred_len :, :]  # [B, T_pred, D]

            x_adv = None
            if use_adv:
                x_adv = self._generate_adversarial_examples(x, y_true)

            # ---------------- 真正的参数更新（干净 + 对抗） ----------------
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler is not None):
                out_clean = self.model(x)
                x_hat_clean = out_clean["recon"]
                y_pred_clean = out_clean["pred"]

                loss_rec_clean = self._safe_mse_loss(x_hat_clean, x)
                loss_fore_clean = self._safe_mse_loss(y_pred_clean, y_true)
                loss_clean = (
                    self.lambda_recon * loss_rec_clean
                    + self.lambda_forecast * loss_fore_clean
                )

                if use_adv and (x_adv is not None):
                    out_adv = self.model(x_adv)
                    x_hat_adv = out_adv["recon"]
                    y_pred_adv = out_adv["pred"]
                    y_true_adv = x_adv[:, -self.pred_len :, :]

                    loss_rec_adv = self._safe_mse_loss(x_hat_adv, x_adv)
                    loss_fore_adv = self._safe_mse_loss(y_pred_adv, y_true_adv)
                    loss_adv = (
                        self.lambda_recon * loss_rec_adv
                        + self.lambda_forecast * loss_fore_adv
                    )
                else:
                    loss_adv = torch.tensor(0.0, device=self.device)

                loss = loss_clean + self.adv_beta * loss_adv

            if not torch.isfinite(loss):
                self.logger.warning(
                    f"[ADV-Train] Epoch {epoch} 遇到 NaN/Inf loss，跳过该 batch。"
                )
                continue

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_clean += float(loss_clean.detach().cpu())
            total_adv += float(loss_adv.detach().cpu())
            n_batches += 1

        if n_batches == 0:
            self.logger.warning(
                f"[ADV-Train] Epoch {epoch} 没有任何有效 batch（可能全部被 NaN/Inf 跳过）。"
            )
            return {"loss": 0.0, "loss_clean": 0.0, "loss_adv": 0.0}

        avg_loss = total_loss / n_batches
        avg_clean = total_clean / n_batches
        avg_adv = total_adv / n_batches

        if self.writer is not None:
            self.writer.add_scalar("adv_train/loss_total", avg_loss, epoch)
            self.writer.add_scalar("adv_train/loss_clean", avg_clean, epoch)
            self.writer.add_scalar("adv_train/loss_adv", avg_adv, epoch)

        return {"loss": avg_loss, "loss_clean": avg_clean, "loss_adv": avg_adv}

    # ==================================================================
    # 评估：与原 TSADTrainer 的思路保持一致
    # ==================================================================
    def evaluate(
        self,
        epoch: int,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        self.model.eval()
        recon_scores_list, forecast_scores_list = self._init_score_buffers()
        labels_list = self.labels_list

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Eval-ADV-{epoch}"):
                x = batch["window"]
                if isinstance(x, np.ndarray):
                    x = torch.from_numpy(x)
                x = x.to(self.device).float()  # [B, L, D]

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
                    idx = int(seq_idx[i])
                    s = int(start[i])
                    e = s + self.win_size

                    # 重构分数：窗口内每个时间点都赋同一 score（取最大）
                    recon_scores_list[idx][s:e] = np.maximum(
                        recon_scores_list[idx][s:e], rec_err[i]
                    )

                    # 预测分数：只在未来 pred_len 段打分
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
        self.logger.info(
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
        self.logger.info(
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
            w_recon=self.lambda_recon,
            w_forecast=self.lambda_forecast,
        )
        thr_h, metrics_h = search_best_f1_threshold_with_auto_flip(
            fused_scores,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )
        metrics_hybrid = metrics_h
        self.logger.info(
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

        # TensorBoard 记录
        if self.writer is not None:
            self.writer.add_scalar("adv_eval/recon_F1", metrics_recon["f1"], epoch)
            self.writer.add_scalar("adv_eval/forecast_F1", metrics_forecast["f1"], epoch)
            self.writer.add_scalar("adv_eval/hybrid_F1", metrics_hybrid["f1"], epoch)

        # 可视化第一条实体的分数曲线
        first_len = len(labels_list[0])
        scores_vis = fused_scores[:first_len]
        if metrics_h.get("need_flip", False):
            scores_vis = -scores_vis
        thr_vis = metrics_h["threshold"]

        vis_path = os.path.join(
            self.log_dir,
            f"scores_epoch{epoch}_adv_hybrid.png",
        )
        plot_scores_with_labels(
            scores=scores_vis,
            labels=labels_list[0],
            threshold=thr_vis,
            save_path=vis_path,
        )
        self.logger.info(f"[ADV] 已保存可视化图：{vis_path}")

        return metrics_recon, metrics_forecast, metrics_hybrid