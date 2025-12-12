# mambatsad/engine/trainer.py
# -*- coding: utf-8 -*-
"""
统一的训练 / 评估逻辑（非对抗版）。

TSADTrainer 支持三种训练模式：
- branch="recon"   : 只训练重构分支；
- branch="forecast": 只训练预测分支；
- branch="hybrid"  : 同时训练重构 + 预测分支，并在评估时融合两种分数。

核心设计要点
-----------
1. 与原项目保持接口兼容：
   - main.py 仍然只需要构建 TSADTrainer 并调用 train()；
   - 模型工厂函数沿用 build_recon_model / build_forecast_model / build_hybrid_model。

2. 数值稳定性：
   - _safe_mse_loss 中对 pred/target 做 nan_to_num + clamp；
   - 训练时做梯度裁剪，避免梯度爆炸；
   - 若出现 NaN/Inf loss，则跳过该 batch。

3. 阈值搜索与“分数方向自动翻转”：
   - 使用 search_best_f1_threshold_with_auto_flip 而不是最原始的 search_best_f1_threshold；
   - auto_flip 会同时在 scores 与 -scores 上搜索 F1/AUC，
     自动判断「越大越异常」还是「越小越异常」更合理；
   - metrics 中会带上 need_flip / direction 字段，方便后续可视化。

4. 分数组合：
   - 混合模型中，先对两路分数做 z-score，再使用
     mambatsad.utils.score_fusion.fuse_scores_by_zscore 做线性加权融合；
   - 默认权重为 λ_rec : λ_forecast，与你在 loss 中的加权保持一致。
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

from mambatsad.models.recon import build_recon_model
from mambatsad.models.forecast import build_forecast_model
from mambatsad.models.hybrid import build_hybrid_model
from mambatsad.utils.metrics import (
    search_best_f1_threshold_with_auto_flip,
)
from mambatsad.utils.score_fusion import fuse_scores_by_zscore
from mambatsad.utils.visualization import plot_scores_with_labels


class TSADTrainer:
    """时间序列异常检测统一 Trainer。"""

    def __init__(
        self,
        branch: str,
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
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        patience: int = 8,
        use_point_adjust: bool = True,
        use_amp: bool = True,
        lambda_recon: float = 1.0,
        lambda_forecast: float = 1.0,
    ) -> None:
        """
        参数
        ----
        branch
            "recon" / "forecast" / "hybrid"
        device
            torch.device("cuda" or "cpu")
        input_dim
            输入特征维度 D
        win_size
            滑动窗口长度 L
        pred_len
            预测步数 T_pred（仅 forecast/hybrid 使用）
        train_loader / test_loader
            训练 / 测试 DataLoader
        labels_list
            每个实体的完整测试标签序列列表，按时间展开
        entity_ids
            实体 ID 列表（如 "machine-1-1"）
        logger
            日志记录器
        writer
            TensorBoard SummaryWriter
        log_dir
            日志及模型权重保存目录
        lr
            学习率
        weight_decay
            权重衰减
        max_grad_norm
            梯度裁剪阈值
        patience
            早停耐心轮数（基于 F1）
        use_point_adjust
            评估阶段是否使用 point-adjust 技巧
        use_amp
            是否启用混合精度训练
        lambda_recon / lambda_forecast
            混合模型中两路 loss 的加权系数
        """
        self.branch = branch.lower()
        assert self.branch in {"recon", "forecast", "hybrid"}

        self.device = device
        self.input_dim = input_dim
        self.win_size = win_size
        self.pred_len = pred_len
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels_list = list(labels_list)
        self.entity_ids = list(entity_ids)
        self.logger = logger
        self.writer = writer
        self.log_dir = log_dir

        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.use_point_adjust = use_point_adjust
        self.use_amp = use_amp and (device.type == "cuda")

        self.lambda_recon = float(lambda_recon)
        self.lambda_forecast = float(lambda_forecast)
        if self.lambda_recon < 0 or self.lambda_forecast < 0:
            raise ValueError("lambda_recon / lambda_forecast 必须为非负数。")

        # ------------------ 预测相关的上下文长度 ------------------
        if self.branch in {"forecast", "hybrid"}:
            if pred_len <= 0:
                raise ValueError("pred_len 必须为正整数。")
            if win_size <= pred_len:
                raise ValueError(
                    f"win_size={win_size} 必须大于 pred_len={pred_len}，"
                    "才能进行预测式异常检测。"
                )
            self.context_len = win_size - pred_len
        else:
            self.context_len = None

        # ------------------ 构建模型 ------------------
        if self.branch == "recon":
            model = build_recon_model(input_dim=input_dim)
            logger.info("训练模式：仅重构分支 (recon)。")
        elif self.branch == "forecast":
            assert self.context_len is not None
            model = build_forecast_model(
                input_dim=input_dim,
                seq_len=self.context_len,
                pred_len=pred_len,
            )
            logger.info(
                "训练模式：仅预测分支 (forecast)，"
                f"context_len={self.context_len}, pred_len={pred_len}。"
            )
        else:  # hybrid
            model = build_hybrid_model(
                input_dim=input_dim,
                win_size=win_size,
                pred_len=pred_len,
            )
            logger.info(
                "训练模式：混合模型 (hybrid)，"
                f"context_len={win_size - pred_len}, pred_len={pred_len}。"
            )

        self.model: nn.Module = model.to(device)
        logger.info(f"模型结构：\n{self.model}")

        # ------------------ 优化器 ------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # 记录最佳指标
        self.best_f1: float = -1.0
        self.best_metrics: Dict[str, float] = {}
        self.best_epoch: int = -1

    # ==================================================================
    # 内部工具函数
    # ==================================================================

    def _safe_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        在 float32 精度下安全地计算 MSE 损失，避免 AMP / 数值溢出问题。

        处理步骤：
        1. 对 pred/target 做 nan_to_num，防止 NaN/Inf；
        2. clamp 到 [-1e4, 1e4]，限制数值范围；
        3. 使用 float32 计算均方误差。
        """
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e6, neginf=-1e6)
        pred = torch.clamp(pred, -1e4, 1e4)
        target = torch.clamp(target, -1e4, 1e4)
        diff = pred - target
        loss = (diff ** 2).mean()
        return loss

    # ==================================================================
    # 训练主循环
    # ==================================================================

    def train(self, num_epochs: int) -> None:
        """主训练入口。"""

        no_improve_epochs = 0

        for epoch in range(1, num_epochs + 1):
            self.logger.info(
                f"========== Epoch {epoch}/{num_epochs} ({self.branch}) =========="
            )

            if self.branch == "recon":
                train_stats = self._train_epoch_recon(epoch)
            elif self.branch == "forecast":
                train_stats = self._train_epoch_forecast(epoch)
            else:
                train_stats = self._train_epoch_hybrid(epoch)

            # ====================== 评估 ======================
            metrics_recon, metrics_forecast, metrics_hybrid = self.evaluate(epoch)

            # 根据不同模式选择“主指标”
            if self.branch == "recon":
                curr_metrics = metrics_recon
            elif self.branch == "forecast":
                curr_metrics = metrics_forecast
            else:
                curr_metrics = metrics_hybrid

            curr_f1 = float(curr_metrics.get("f1", 0.0))

            # TensorBoard 记录
            self.writer.add_scalar("train/loss", train_stats["loss"], epoch)
            if "loss_recon" in train_stats:
                self.writer.add_scalar("train/loss_recon", train_stats["loss_recon"], epoch)
            if "loss_forecast" in train_stats:
                self.writer.add_scalar(
                    "train/loss_forecast", train_stats["loss_forecast"], epoch
                )

            self.writer.add_scalar("eval/f1", curr_f1, epoch)

            # ------------------ 早停 / 保存最优模型 ------------------
            if curr_f1 > self.best_f1 + 1e-6:
                self.best_f1 = curr_f1
                self.best_metrics = curr_metrics
                self.best_epoch = epoch
                no_improve_epochs = 0

                best_path = os.path.join(self.log_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                self.logger.info(f"发现更优模型 (F1={curr_f1:.4f})，已保存至 {best_path}")
            else:
                no_improve_epochs += 1
                self.logger.info(
                    f"当前 F1={curr_f1:.4f}，已连续 {no_improve_epochs} 个 epoch 未提升。"
                )
                if no_improve_epochs >= self.patience:
                    self.logger.info(
                        f"早停触发：patience={self.patience}，最佳 F1={self.best_f1:.4f} "
                        f"(epoch={self.best_epoch})，对应指标={self.best_metrics}"
                    )
                    break

        self.logger.info(
            f"训练结束，最佳 F1={self.best_f1:.4f} (epoch={self.best_epoch})，"
            f"最佳指标={self.best_metrics}"
        )

    # ==================================================================
    # 各分支训练一个 epoch
    # ==================================================================

    def _train_epoch_recon(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-RECON-{epoch}"):
            x = batch["window"].to(self.device)  # [B, L, D]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                x_hat = self.model(x)  # [B, L, D]
                loss = self._safe_mse_loss(x_hat, x)

            if not torch.isfinite(loss):
                self.logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch。")
                continue

            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.logger.info(f"[Epoch {epoch}] 重构分支训练损失: {avg_loss:.6f}")
        return {"loss": avg_loss, "loss_recon": avg_loss}

    def _train_epoch_forecast(self, epoch: int) -> Dict[str, float]:
        assert self.context_len is not None

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-FORECAST-{epoch}"):
            x = batch["window"].to(self.device)  # [B, L, D]
            x_ctx = x[:, : self.context_len, :]
            y_true = x[:, -self.pred_len :, :]  # [B, pred_len, D]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                y_pred = self.model(x_ctx)  # [B, pred_len, D]
                loss = self._safe_mse_loss(y_pred, y_true)

            if not torch.isfinite(loss):
                self.logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch。")
                continue

            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        self.logger.info(f"[Epoch {epoch}] 预测分支训练损失: {avg_loss:.6f}")
        return {"loss": avg_loss, "loss_forecast": avg_loss}

    def _train_epoch_hybrid(self, epoch: int) -> Dict[str, float]:
        assert self.context_len is not None

        self.model.train()
        total_loss = 0.0
        loss_rec_sum = 0.0
        loss_fore_sum = 0.0
        n_batches = 0

        for batch in tqdm(self.train_loader, desc=f"Train-HYBRID-{epoch}"):
            x = batch["window"].to(self.device)  # [B, L, D]
            y_true = x[:, -self.pred_len :, :]  # [B, pred_len, D]

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.use_amp):
                out = self.model(x)
                x_hat = out["recon"]
                y_pred = out["pred"]

                loss_recon = self._safe_mse_loss(x_hat, x)
                loss_fore = self._safe_mse_loss(y_pred, y_true)
                loss = (
                    self.lambda_recon * loss_recon
                    + self.lambda_forecast * loss_fore
                )

            if not torch.isfinite(loss):
                self.logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch。")
                continue

            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            total_loss += float(loss.detach().cpu())
            loss_rec_sum += float(loss_recon.detach().cpu())
            loss_fore_sum += float(loss_fore.detach().cpu())
            n_batches += 1

        avg_total = total_loss / max(n_batches, 1)
        avg_rec = loss_rec_sum / max(n_batches, 1)
        avg_fore = loss_fore_sum / max(n_batches, 1)

        self.logger.info(
            f"[Epoch {epoch}] 混合模型训练损失: "
            f"total={avg_total:.6f}, recon={avg_rec:.6f}, forecast={avg_fore:.6f}"
        )
        return {
            "loss": avg_total,
            "loss_recon": avg_rec,
            "loss_forecast": avg_fore,
        }

    # ==================================================================
    # 评估逻辑
    # ==================================================================

    def _init_score_buffers(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """为每个实体构建按时间的得分数组初值（全 0）。"""
        recon_scores_list: List[np.ndarray] = []
        forecast_scores_list: List[np.ndarray] = []

        for labels in self.labels_list:
            T = len(labels)
            recon_scores_list.append(np.zeros(T, dtype=np.float32))
            forecast_scores_list.append(np.zeros(T, dtype=np.float32))

        return recon_scores_list, forecast_scores_list

    def evaluate(
        self, epoch: int
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        在测试集上计算重构 / 预测 / 混合得分，并根据 F1 最优阈值报告指标。

        返回
        ----
        metrics_recon, metrics_forecast, metrics_hybrid
        对于非 hybrid 分支，其中的一部分可能是空字典。
        """
        self.model.eval()

        use_recon = self.branch in {"recon", "hybrid"}
        use_forecast = self.branch in {"forecast", "hybrid"}

        recon_scores_list, forecast_scores_list = self._init_score_buffers()
        labels_list = self.labels_list

        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc=f"Eval-{self.branch}-Epoch{epoch}"):
                x = batch["window"].to(self.device)  # [B, L, D]
                seq_idx = batch["seq_idx"].cpu().numpy().astype(int)
                start = batch["start"].cpu().numpy().astype(int)
                B = x.shape[0]

                if self.branch == "recon":
                    x_hat = self.model(x)
                    rec_err = ((x_hat - x) ** 2).mean(dim=(1, 2)).cpu().numpy()
                    fore_err = None
                elif self.branch == "forecast":
                    assert self.context_len is not None
                    x_ctx = x[:, : self.context_len, :]
                    y_true = x[:, -self.pred_len :, :]
                    y_pred = self.model(x_ctx)
                    fore_err = ((y_pred - y_true) ** 2).mean(dim=(1, 2)).cpu().numpy()
                    rec_err = None
                else:  # hybrid
                    assert self.context_len is not None
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

                    if use_recon and rec_err is not None:
                        # 全窗口都赋予同一标量得分，使用 max 聚合
                        recon_scores_list[idx][s:e] = np.maximum(
                            recon_scores_list[idx][s:e], rec_err[i]
                        )

                    if use_forecast and fore_err is not None:
                        # 预测误差只作用于未来时间段
                        s_f = s + (self.context_len or 0)
                        e_f = s + self.win_size
                        forecast_scores_list[idx][s_f:e_f] = np.maximum(
                            forecast_scores_list[idx][s_f:e_f], fore_err[i]
                        )

        # 展平成全局 1D 数组
        labels_all = np.concatenate(labels_list, axis=0)
        recon_scores = (
            np.concatenate(recon_scores_list, axis=0) if use_recon else None
        )
        forecast_scores = (
            np.concatenate(forecast_scores_list, axis=0) if use_forecast else None
        )

        metrics_recon: Dict[str, float] = {}
        metrics_forecast: Dict[str, float] = {}
        metrics_hybrid: Dict[str, float] = {}

        # ------------------ 重构分支 ------------------
        if use_recon and recon_scores is not None:
            thr_rec, metrics_rec = search_best_f1_threshold_with_auto_flip(
                recon_scores,
                labels_all,
                use_point_adjust=self.use_point_adjust,
            )
            metrics_recon = metrics_rec
            self.logger.info(
                "[Eval-recon] "
                f"F1={metrics_rec['f1']:.4f}, P={metrics_rec['precision']:.4f}, "
                f"R={metrics_rec['recall']:.4f}, AUC={metrics_rec['auc']:.4f}, "
                f"thr={metrics_rec['threshold']:.6f}, "
                f"need_flip={metrics_rec.get('need_flip', False)}, "
                f"dir={metrics_rec.get('direction', 'greater')}, "
                f"point_adjust={self.use_point_adjust}"
            )
        else:
            thr_rec = None

        # ------------------ 预测分支 ------------------
        if use_forecast and forecast_scores is not None:
            thr_fore, metrics_for = search_best_f1_threshold_with_auto_flip(
                forecast_scores,
                labels_all,
                use_point_adjust=self.use_point_adjust,
            )
            metrics_forecast = metrics_for
            self.logger.info(
                "[Eval-forecast] "
                f"F1={metrics_for['f1']:.4f}, P={metrics_for['precision']:.4f}, "
                f"R={metrics_for['recall']:.4f}, AUC={metrics_for['auc']:.4f}, "
                f"thr={metrics_for['threshold']:.6f}, "
                f"need_flip={metrics_for.get('need_flip', False)}, "
                f"dir={metrics_for.get('direction', 'greater')}, "
                f"point_adjust={self.use_point_adjust}"
            )
        else:
            thr_fore = None

        # ------------------ 混合分支（仅 hybrid 使用） ------------------
        if self.branch == "hybrid" and recon_scores is not None and forecast_scores is not None:
            # 先做归一化 + 线性融合（权重比例跟 loss 相同）
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
                "[Eval-hybrid] "
                f"F1={metrics_h['f1']:.4f}, P={metrics_h['precision']:.4f}, "
                f"R={metrics_h['recall']:.4f}, AUC={metrics_h['auc']:.4f}, "
                f"thr={metrics_h['threshold']:.6f}, "
                f"need_flip={metrics_h.get('need_flip', False)}, "
                f"dir={metrics_h.get('direction', 'greater')}, "
                f"point_adjust={self.use_point_adjust}; "
                f"recon_F1={metrics_recon.get('f1', 0.0):.4f}, "
                f"forecast_F1={metrics_forecast.get('f1', 0.0):.4f}"
            )

            # --- 可视化第一条实体的得分 ---
            first_len = len(labels_list[0])
            scores_vis = fused_scores[:first_len]
            if metrics_h.get("need_flip", False):
                scores_vis = -scores_vis
            thr_vis = metrics_h["threshold"]
            vis_path = os.path.join(
                self.log_dir, f"scores_epoch{epoch}_hybrid.png"
            )
            plot_scores_with_labels(
                scores=scores_vis,
                labels=labels_list[0],
                threshold=thr_vis,
                save_path=vis_path,
            )
            self.logger.info(f"已保存可视化图：{vis_path}")

        return metrics_recon, metrics_forecast, metrics_hybrid
