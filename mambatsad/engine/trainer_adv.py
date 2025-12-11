# mambatsad/engine/trainer_adv.py
# -*- coding: utf-8 -*-
"""
实验版 Trainer：用于 Shared-Encoder + 对抗训练 + 伪标签混合模型。

与主分支的 trainer 区分开，避免互相影响。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from mambatsad.models.hybrid_shared_adv import (
    MambaTSADHybridSharedAdv,
)
from mambatsad.utils.metrics import search_best_f1_threshold
from mambatsad.utils.visualization import plot_scores_with_labels


class AdvHybridTrainer:
    """
    实验版混合模型 Trainer：

    支持：
    - 共享 encoder 的混合模型（MambaTSADHybridSharedAdv）；
    - STAMP 风格的两阶段对抗训练（use_adv_training）；
    - 伪标签过滤训练集（在 main_adv.py 中完成）。
    """

    def __init__(
        self,
        model: MambaTSADHybridSharedAdv,
        device: torch.device,
        train_loader: DataLoader,
        test_loader: DataLoader,
        labels_list: List[np.ndarray],
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
        self.model = model.to(device)
        self.device = device

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.labels_list = [np.asarray(x, dtype=int) for x in labels_list]

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if pred_len <= 0:
            raise ValueError("pred_len 必须为正整数。")
        if win_size <= pred_len:
            raise ValueError(
                f"win_size({win_size}) 必须大于 pred_len({pred_len})，"
                "否则无法同时做重构与预测。"
            )

        self.win_size = win_size
        self.pred_len = pred_len
        self.context_len = win_size - pred_len

        # 第一阶段：优化整个模型
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        # 第二阶段：只优化 shared_encoder + recon_head
        self.optimizer_recon = torch.optim.Adam(
            list(self.model.shared_encoder.parameters())
            + list(self.model.recon_head.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.max_grad_norm = max_grad_norm
        self.scaler = GradScaler(enabled=use_amp)

        self.use_adv_training = use_adv_training
        self.adv_warmup_epochs = adv_warmup_epochs

        self.lambda_rec = lambda_rec
        self.lambda_pred = lambda_pred
        self.lambda_adv1 = lambda_adv1
        self.lambda_adv2 = lambda_adv2

        self.use_point_adjust = use_point_adjust

        self.best_f1 = -1.0
        self.best_metrics: Dict[str, float] = {}

    # ----------------- 公共工具函数 -----------------
    @staticmethod
    def _safe_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """防守式 MSE：在极端情况下避免 nan/inf。"""
        diff = pred - target
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e6, neginf=-1e6)
        loss = (diff**2).mean()
        return loss

    # ----------------- 训练循环 -----------------
    def train(self, epochs: int, logger) -> None:
        """主训练入口。"""
        for epoch in range(1, epochs + 1):
            logger.info(
                "========== Epoch %d/%d (adv-hybrid) ==========",
                epoch,
                epochs,
            )

            if not self.use_adv_training:
                self._train_epoch_plain(epoch, logger)
            else:
                if epoch <= self.adv_warmup_epochs:
                    self._train_epoch_stage1(epoch, logger)
                else:
                    self._train_epoch_stage2(epoch, logger)

            # epoch 结束后做一次评估
            metrics_recon, metrics_fore, metrics_hybrid = self.evaluate(
                epoch, logger
            )

            # 以混合 F1 作为早停 / 最佳选择指标
            f1 = metrics_hybrid["f1"]
            if f1 > self.best_f1 + 1e-6:
                self.best_f1 = f1
                self.best_metrics = metrics_hybrid
                best_path = os.path.join(
                    self.log_dir,
                    "best_model_adv_hybrid.pt",
                )
                torch.save(self.model.state_dict(), best_path)
                logger.info("发现更优模型，已保存至 %s", best_path)

        logger.info(
            "训练结束，最佳混合 F1=%.4f, 最佳指标=%s",
            self.best_f1,
            self.best_metrics,
        )

    def _train_epoch_plain(self, epoch: int, logger) -> None:
        """不使用对抗训练时的普通多任务联合训练。"""
        self.model.train()

        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        pred_loss_sum = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x = batch["window"].to(self.device)  # [B, L, D]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                out = self.model(x)
                if not isinstance(out, dict):
                    raise ValueError("AdvHybrid 模型 forward 必须返回 dict。")

                x_hat = out["recon"]  # [B, L, D]
                y_pred = out["pred"]  # [B, pred_len, D]

                y_true = x[:, -self.pred_len :, :]  # [B, pred_len, D]

                rec_loss = self._safe_mse_loss(x_hat, x)
                pred_loss = self._safe_mse_loss(y_pred, y_true)

                loss = self.lambda_rec * rec_loss + self.lambda_pred * pred_loss

            if not torch.isfinite(loss):
                logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch")
                continue

            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += float(loss.detach().cpu())
            rec_loss_sum += float(rec_loss.detach().cpu())
            pred_loss_sum += float(pred_loss.detach().cpu())
            n_batches += 1

        if n_batches > 0:
            logger.info(
                "[Epoch %d] 训练损失(plain): "
                "total=%.6f, recon=%.6f, forecast=%.6f",
                epoch,
                total_loss_sum / n_batches,
                rec_loss_sum / n_batches,
                pred_loss_sum / n_batches,
            )

    def _train_epoch_stage1(self, epoch: int, logger) -> None:
        """
        对抗训练第 1 阶段（参考 STAMP 的 Loss1）：

        L1 = λ_rec * L_rec_pos + λ_pred * L_pred + λ_adv1 * L_rec_neg
        """
        self.model.train()

        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        pred_loss_sum = 0.0
        adv_loss_sum = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x = batch["window"].to(self.device)  # [B, L, D]

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                out = self.model(x)
                if not isinstance(out, dict):
                    raise ValueError("AdvHybrid 模型 forward 必须返回 dict。")

                x_hat = out["recon"]  # [B, L, D]
                y_pred = out["pred"]  # [B, pred_len, D]
                y_true = x[:, -self.pred_len :, :]  # [B, pred_len, D]

                # 正样本重构 + 预测
                rec_loss_pos = self._safe_mse_loss(x_hat, x)
                pred_loss = self._safe_mse_loss(y_pred, y_true)

                # 构造负样本：用预测结果替换真实未来
                x_neg = torch.cat(
                    [x[:, : self.context_len, :], y_pred],
                    dim=1,
                )  # [B, L, D]
                h_neg = self.model.encode(x_neg)
                x_hat_neg = self.model.decode_recon(h_neg)
                rec_loss_neg = self._safe_mse_loss(x_hat_neg, x)

                loss = (
                    self.lambda_rec * rec_loss_pos
                    + self.lambda_pred * pred_loss
                    + self.lambda_adv1 * rec_loss_neg
                )

            if not torch.isfinite(loss):
                logger.warning(
                    "遇到非有限 loss (NaN/Inf)，跳过该 batch (stage1)"
                )
                continue

            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss_sum += float(loss.detach().cpu())
            rec_loss_sum += float(rec_loss_pos.detach().cpu())
            pred_loss_sum += float(pred_loss.detach().cpu())
            adv_loss_sum += float(rec_loss_neg.detach().cpu())
            n_batches += 1

        if n_batches > 0:
            logger.info(
                "[Epoch %d] 阶段1对抗训练损失: "
                "total=%.6f, recon_pos=%.6f, pred=%.6f, recon_neg=%.6f",
                epoch,
                total_loss_sum / n_batches,
                rec_loss_sum / n_batches,
                pred_loss_sum / n_batches,
                adv_loss_sum / n_batches,
            )

    def _train_epoch_stage2(self, epoch: int, logger) -> None:
        """
        对抗训练第 2 阶段（参考 STAMP 的 Loss2）：

        只更新 shared_encoder + recon_head，预测头冻结：
        L2 = L_rec_pos - λ_adv2 * L_rec_neg
        """
        self.model.train()

        # 冻结预测头
        for p in self.model.forecast_head.parameters():
            p.requires_grad = False

        total_loss_sum = 0.0
        rec_loss_sum = 0.0
        adv_loss_sum = 0.0
        n_batches = 0

        for batch in self.train_loader:
            x = batch["window"].to(self.device)  # [B, L, D]

            self.optimizer_recon.zero_grad(set_to_none=True)

            with autocast(enabled=self.scaler.is_enabled()):
                # 正样本重构
                h_pos = self.model.encode(x)
                x_hat_pos = self.model.decode_recon(h_pos)
                rec_loss_pos = self._safe_mse_loss(x_hat_pos, x)

                # 使用冻结预测头生成负样本
                with torch.no_grad():
                    h = self.model.encode(x)
                    h_ctx = h[:, : self.context_len, :]
                    x_ctx = x[:, : self.context_len, :]
                    y_pred = self.model.decode_forecast(h_ctx, x_ctx)
                    x_neg = torch.cat(
                        [x[:, : self.context_len, :], y_pred],
                        dim=1,
                    )
                    h_neg = self.model.encode(x_neg)
                    x_hat_neg = self.model.decode_recon(h_neg)
                    rec_loss_neg = self._safe_mse_loss(x_hat_neg, x)

                loss = rec_loss_pos - self.lambda_adv2 * rec_loss_neg

            if not torch.isfinite(loss):
                logger.warning(
                    "遇到非有限 loss (NaN/Inf)，跳过该 batch (stage2)"
                )
                continue

            self.scaler.scale(loss).backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer_recon)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.shared_encoder.parameters())
                    + list(self.model.recon_head.parameters()),
                    self.max_grad_norm,
                )
            self.scaler.step(self.optimizer_recon)
            self.scaler.update()

            total_loss_sum += float(loss.detach().cpu())
            rec_loss_sum += float(rec_loss_pos.detach().cpu())
            adv_loss_sum += float(rec_loss_neg.detach().cpu())
            n_batches += 1

        if n_batches > 0:
            logger.info(
                "[Epoch %d] 阶段2对抗训练损失: "
                "total=%.6f, recon_pos=%.6f, recon_neg=%.6f",
                epoch,
                total_loss_sum / n_batches,
                rec_loss_sum / n_batches,
                adv_loss_sum / n_batches,
            )

        # 解冻预测头（防止后续 epoch 切回其他模式）
        for p in self.model.forecast_head.parameters():
            p.requires_grad = True

    # ----------------- 评估 / 得分融合 -----------------
    def evaluate(
        self,
        epoch: int,
        logger,
    ) -> Tuple[Dict, Dict, Dict]:
        """
        在测试集上计算重构 / 预测 / 混合得分，并根据 F1 最优阈值报告指标。
        """
        self.model.eval()

        labels_list = self.labels_list

        # 为每个实体构建按时间的得分数组
        recon_scores_list: List[np.ndarray] = []
        forecast_scores_list: List[np.ndarray] = []

        with torch.no_grad():
            # 初始化 scores 为零数组，后续用 "max-overlapping" 聚合
            for labels in labels_list:
                T = len(labels)
                recon_scores_list.append(np.zeros(T, dtype=np.float32))
                forecast_scores_list.append(np.zeros(T, dtype=np.float32))

            for batch in self.test_loader:
                x = batch["window"].to(self.device)  # [B, L, D]
                seq_idx = (
                    batch["seq_idx"].cpu().numpy().astype(int)
                )  # [B]
                start = batch["start"].cpu().numpy().astype(int)  # [B]
                B = x.shape[0]

                out = self.model(x)
                if not isinstance(out, dict):
                    raise ValueError("AdvHybrid 模型 forward 必须返回 dict。")

                x_hat = out["recon"]
                y_pred = out["pred"]
                y_true = x[:, -self.pred_len :, :]

                # 重构误差（标量）: MSE over 全窗口
                rec_err = (
                    (x_hat - x) ** 2
                ).mean(dim=(1, 2)).cpu().numpy()  # [B]

                # 预测误差（标量）: 只对未来部分
                fore_err = (
                    (y_pred - y_true) ** 2
                ).mean(dim=(1, 2)).cpu().numpy()  # [B]

                # 将每个窗口的标量得分填回对应实体的时间轴上，采用 max 聚合
                for i in range(B):
                    idx = seq_idx[i]
                    s = start[i]
                    e = s + self.win_size

                    e = min(e, len(recon_scores_list[idx]))
                    if e <= s:
                        continue

                    recon_scores_list[idx][s:e] = np.maximum(
                        recon_scores_list[idx][s:e],
                        rec_err[i],
                    )

                    # 预测误差只作用于未来时间段
                    s_f = s + self.context_len
                    e_f = s + self.win_size
                    e_f = min(e_f, len(forecast_scores_list[idx]))
                    if e_f <= s_f:
                        continue

                    forecast_scores_list[idx][s_f:e_f] = np.maximum(
                        forecast_scores_list[idx][s_f:e_f],
                        fore_err[i],
                    )

        # 展平为全局 1D 数组
        recon_scores = np.concatenate(recon_scores_list, axis=0)
        forecast_scores = np.concatenate(forecast_scores_list, axis=0)
        labels_all = np.concatenate(labels_list, axis=0)

        # 为了更稳健的融合，对每个分支做一次 z-score 归一化
        def _zscore(x: np.ndarray) -> np.ndarray:
            m = np.mean(x)
            s = np.std(x)
            if s < 1e-8:
                return np.zeros_like(x)
            return (x - m) / s

        recon_z = _zscore(recon_scores)
        forecast_z = _zscore(forecast_scores)

        # 先分别搜索各自的最佳阈值
        _, metrics_recon = search_best_f1_threshold(
            recon_z,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )
        _, metrics_fore = search_best_f1_threshold(
            forecast_z,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )

        logger.info(
            "[Eval-recon(adv-hybrid)] "
            "F1=%.4f, P=%.4f, R=%.4f, AUC=%.4f, thr=%.6f, point_adjust=%s",
            metrics_recon["f1"],
            metrics_recon["precision"],
            metrics_recon["recall"],
            metrics_recon["auc"],
            metrics_recon["threshold"],
            self.use_point_adjust,
        )
        logger.info(
            "[Eval-forecast(adv-hybrid)] "
            "F1=%.4f, P=%.4f, R=%.4f, AUC=%.4f, thr=%.6f, point_adjust=%s",
            metrics_fore["f1"],
            metrics_fore["precision"],
            metrics_fore["recall"],
            metrics_fore["auc"],
            metrics_fore["threshold"],
            self.use_point_adjust,
        )

        # 再做融合：简单起见，使用加权和（权重可以调参）
        alpha = 0.5
        beta = 0.5
        hybrid_scores = alpha * recon_z + beta * forecast_z

        thr_h, metrics_hybrid = search_best_f1_threshold(
            hybrid_scores,
            labels_all,
            use_point_adjust=self.use_point_adjust,
        )

        logger.info(
            "[Eval-hybrid(adv)] "
            "F1=%.4f, P=%.4f, R=%.4f, AUC=%.4f, thr=%.6f, point_adjust=%s; "
            "recon_F1=%.4f, forecast_F1=%.4f",
            metrics_hybrid["f1"],
            metrics_hybrid["precision"],
            metrics_hybrid["recall"],
            metrics_hybrid["auc"],
            metrics_hybrid["threshold"],
            self.use_point_adjust,
            metrics_recon["f1"],
            metrics_fore["f1"],
        )

        # 画一张 hybrid 得分的可视化图（使用第一条实体的得分）
        first_len = len(labels_list[0])
        first_scores = hybrid_scores[:first_len]
        first_labels = labels_list[0]

        vis_path = os.path.join(
            self.log_dir,
            f"scores_epoch{epoch}_adv_hybrid.png",
        )
        plot_scores_with_labels(
            scores=first_scores,
            labels=first_labels,
            threshold=metrics_hybrid["threshold"],
            save_path=vis_path,
        )
        logger.info("已保存可视化图：%s", vis_path)

        return metrics_recon, metrics_fore, metrics_hybrid
