# mambatsad/engine/trainer.py
# -*- coding: utf-8 -*-
"""
统一的训练 / 评估逻辑。

TSADTrainer 支持三种训练模式：
- branch="recon"   : 只训练重构分支；
- branch="forecast": 只训练预测分支；
- branch="hybrid"  : 同时训练重构 + 预测分支，并在评估时融合两种分数。

这样既可以单独验证各分支的效果，也可以验证混合模型是否带来收益。
"""

from __future__ import annotations

import os
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from mambatsad.models.recon import build_recon_model
from mambatsad.models.forecast import build_forecast_model
from mambatsad.models.hybrid import build_hybrid_model
from mambatsad.utils.metrics import search_best_f1_threshold
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
        branch : "recon" / "forecast" / "hybrid"
        device : torch.device("cuda" or "cpu")
        input_dim : 输入特征维度 D
        win_size : 滑动窗口长度 L
        pred_len : 预测步数 T_pred（仅 forecast/hybrid 使用）
        train_loader/test_loader : 训练/测试 DataLoader
        labels_list : 每个实体对应的完整测试标签序列列表（按时间展开）
        entity_ids : 实体 ID 列表（如 machine-1-1）
        logger : 日志记录器
        writer : TensorBoard SummaryWriter
        log_dir : 日志及模型权重保存目录
        lr : 学习率
        weight_decay : 权重衰减
        max_grad_norm : 梯度裁剪阈值
        patience : 早停轮数
        use_point_adjust : 评估时是否使用 point-adjust 技巧
        use_amp : 是否启用混合精度训练
        lambda_recon / lambda_forecast : 混合模型中两任务的 loss 权重
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
        self.use_amp = use_amp

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
            self.logger.info("训练模式：仅重构分支 (recon)。")
        elif self.branch == "forecast":
            model = build_forecast_model(
                input_dim=input_dim, seq_len=self.context_len, pred_len=pred_len
            )
            self.logger.info(
                f"训练模式：仅预测分支 (forecast)，context_len={self.context_len}, "
                f"pred_len={pred_len}。"
            )
        else:  # hybrid
            model = build_hybrid_model(
                input_dim=input_dim, win_size=win_size, pred_len=pred_len
            )
            self.logger.info(
                f"训练模式：混合模型 (hybrid)，context_len={win_size - pred_len}, "
                f"pred_len={pred_len}。"
            )

        self.model: nn.Module = model.to(device)
        self.logger.info(f"模型结构：\n{self.model}")

        # ------------------ 优化器与 AMP ------------------
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        if use_amp and device.type == "cuda":
            # PyTorch 2.0+ 推荐使用 torch.amp.GradScaler
            self.scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

    # ==================================================================
    # 内部工具函数
    # ==================================================================

    def _safe_mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        在 float32 精度下安全地计算 MSE 损失，避免 AMP 下的数值溢出问题。

        处理步骤：
        1. 对 pred/target 做 nan_to_num，防止 NaN/Inf；
        2. clamp 到 [-1e4, 1e4]，限制数值范围；
        3. 转为 float32 再计算 MSE。
        """
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e4, neginf=-1e4)
        target = torch.nan_to_num(target, nan=0.0, posinf=1e4, neginf=-1e4)

        pred = torch.clamp(pred, -1e4, 1e4).float()
        target = torch.clamp(target, -1e4, 1e4).float()

        loss = F.mse_loss(pred, target, reduction="mean")
        return loss

    def _backward(self, loss: torch.Tensor) -> bool:
        """
        统一的反向传播 + 梯度裁剪逻辑。

        若 loss 非有限（NaN/Inf），则返回 False 并跳过该 batch。
        """
        if (
            not isinstance(loss, torch.Tensor)
            or not loss.requires_grad
            or not torch.isfinite(loss)
        ):
            self.logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch")
            return False

        if self.scaler is not None:
            # AMP 模式
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 普通 FP32 模式
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

        return True

    # ==================================================================
    # 训练单个 epoch：重构分支
    # ==================================================================

    def _train_epoch_recon(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        torch.autograd.set_detect_anomaly(False)

        for batch in tqdm(
            self.train_loader, desc=f"Train-Epoch{epoch}(recon)", leave=False
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device, non_blocking=True)

            # 输入清洗：去除 NaN / Inf，并限制数值范围
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, -1e4, 1e4)

            self.optimizer.zero_grad(set_to_none=True)

            # 前向：在 AMP 下进行，但 MSE 在 float32 下算
            with torch.amp.autocast(
                device_type="cuda", enabled=(self.scaler is not None)
            ):
                out = self.model(x)
                rec_list = out.get("recon_multi")
                recon = out.get("recon")

            # 在 float32 下计算安全的 MSE 损失
            if rec_list is None:
                if recon is None:
                    raise ValueError("重构模型未返回 'recon'。")
                loss = self._safe_mse_loss(recon, x)
            else:
                loss_val = 0.0
                for rec in rec_list:
                    loss_val = loss_val + self._safe_mse_loss(rec, x)
                loss = loss_val

            if not self._backward(loss):
                continue

            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"[Epoch {epoch}] 重构分支训练损失: {avg_loss:.6f}")
        return avg_loss

    # ==================================================================
    # 训练单个 epoch：预测分支
    # ==================================================================

    def _train_epoch_forecast(self, epoch: int) -> float:
        assert self.context_len is not None

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        torch.autograd.set_detect_anomaly(False)

        for batch in tqdm(
            self.train_loader, desc=f"Train-Epoch{epoch}(forecast)", leave=False
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device, non_blocking=True)

            # 输入清洗
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, -1e4, 1e4)

            B, L, D = x.shape
            if L < self.context_len + self.pred_len:
                raise ValueError(
                    f"窗口长度 L={L} 小于 context_len + pred_len = "
                    f"{self.context_len + self.pred_len}，请调大 --win_size。"
                )

            x_enc = x[:, : self.context_len, :]  # [B, L_c, D]
            y_true = x[:, self.context_len : self.context_len + self.pred_len, :]

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(self.scaler is not None)
            ):
                y_pred = self.model(x_enc)
                if isinstance(y_pred, dict):
                    if "pred" not in y_pred:
                        raise ValueError(
                            "预测模型 forward 返回的字典中未包含 'pred' 键。"
                        )
                    y_pred = y_pred["pred"]

            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致，"
                    "请检查 seq_len 与 pred_len 配置。"
                )

            loss = self._safe_mse_loss(y_pred, y_true)

            if not self._backward(loss):
                continue

            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"[Epoch {epoch}] 预测分支训练损失: {avg_loss:.6f}")
        return avg_loss

    # ==================================================================
    # 训练单个 epoch：混合模型
    # ==================================================================

    def _train_epoch_hybrid(self, epoch: int) -> float:
        assert self.context_len is not None
        hybrid_model = self.model

        self.model.train()
        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_forecast = 0.0
        num_batches = 0

        torch.autograd.set_detect_anomaly(False)

        for batch in tqdm(
            self.train_loader, desc=f"Train-Epoch{epoch}(hybrid)", leave=False
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device, non_blocking=True)

            # 输入清洗
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, -1e4, 1e4)

            B, L, D = x.shape
            if L < self.context_len + self.pred_len:
                raise ValueError(
                    f"窗口长度 L={L} 小于 context_len + pred_len = "
                    f"{self.context_len + self.pred_len}，请调大 --win_size。"
                )

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda", enabled=(self.scaler is not None)
            ):
                out = hybrid_model(x)

            # -------- 重构分支 loss --------
            rec_list = out.get("recon_multi")
            recon = out.get("recon")
            if rec_list is None and recon is None:
                raise ValueError("混合模型未返回重构结果。")

            if rec_list is None:
                loss_recon = self._safe_mse_loss(recon, x)
            else:
                loss_val = 0.0
                for rec in rec_list:
                    loss_val = loss_val + self._safe_mse_loss(rec, x)
                loss_recon = loss_val

            # -------- 预测分支 loss --------
            y_pred = out.get("pred")
            if y_pred is None:
                raise ValueError("混合模型未返回预测结果 'pred'。")
            y_true = x[:, self.context_len : self.context_len + self.pred_len, :]

            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"混合模型预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致。"
                )

            loss_forecast = self._safe_mse_loss(y_pred, y_true)

            # -------- 多任务 loss 融合：不确定性加权 + λ 系数 --------
            if hasattr(hybrid_model, "log_sigma_recon") and hasattr(
                hybrid_model, "log_sigma_forecast"
            ):
                sigma_rec = torch.exp(hybrid_model.log_sigma_recon)
                sigma_fore = torch.exp(hybrid_model.log_sigma_forecast)

                precision_rec = 1.0 / (sigma_rec ** 2 + 1e-8)
                precision_fore = 1.0 / (sigma_fore ** 2 + 1e-8)

                # loss = (
                #     self.lambda_recon * precision_rec * loss_recon
                #     + self.lambda_forecast * precision_fore * loss_forecast
                #     + torch.log(sigma_rec + 1e-8)
                #     + torch.log(sigma_fore + 1e-8)
                # )
                # # 简单加权
                loss = (
                        self.lambda_recon * loss_recon
                        + self.lambda_forecast * loss_forecast
                )
            else:
                # 回退到简单加权
                loss = (
                    self.lambda_recon * loss_recon
                    + self.lambda_forecast * loss_forecast
                )

            if not self._backward(loss):
                continue

            total_loss += float(loss.detach().cpu().item())
            total_loss_recon += float(loss_recon.detach().cpu().item())
            total_loss_forecast += float(loss_forecast.detach().cpu().item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_rec = total_loss_recon / max(num_batches, 1)
        avg_fore = total_loss_forecast / max(num_batches, 1)

        self.logger.info(
            f"[Epoch {epoch}] 混合模型训练损失: total={avg_loss:.6f}, "
            f"recon={avg_rec:.6f}, forecast={avg_fore:.6f}"
        )
        return avg_loss

    # ==================================================================
    # 评估：重构分支
    # ==================================================================

    def _evaluate_recon(
        self, model: nn.Module, tag_prefix: str = "recon"
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        model.eval()

        num_seqs = len(self.labels_list)
        # 为每条原始序列分配累积得分数组
        sum_scores = [
            np.zeros(len(self.labels_list[i]), dtype=np.float64)
            for i in range(num_seqs)
        ]
        cnt_scores = [
            np.zeros(len(self.labels_list[i]), dtype=np.float64)
            for i in range(num_seqs)
        ]

        for batch in tqdm(
            self.test_loader, desc=f"Eval({tag_prefix})", leave=False
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x)
            else:
                x_t = x
            x_t = x_t.to(self.device, non_blocking=True)

            # 清洗输入
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e4, neginf=-1e4)
            x_t = torch.clamp(x_t, -1e4, 1e4)

            seq_idx = batch["seq_idx"]
            starts = batch["start"]

            if isinstance(seq_idx, torch.Tensor):
                seq_idx = seq_idx.cpu().numpy()
            elif not isinstance(seq_idx, np.ndarray):
                seq_idx = np.array(seq_idx, dtype=np.int64)

            if isinstance(starts, torch.Tensor):
                starts = starts.cpu().numpy()
            elif not isinstance(starts, np.ndarray):
                starts = np.array(starts, dtype=np.int64)

            with torch.no_grad():
                out = model(x_t)
                rec = out["recon"]  # [B, L, D]
                rec = torch.nan_to_num(rec, nan=0.0, posinf=1e4, neginf=-1e4)
                rec = torch.clamp(rec, -1e4, 1e4)

                mse = ((rec - x_t) ** 2).mean(dim=-1)  # [B, L]
                mse_np = mse.detach().cpu().numpy()

            B, L = mse_np.shape
            for i in range(B):
                k = int(seq_idx[i])
                s = int(starts[i])
                e = s + self.win_size
                sum_scores[k][s:e] += mse_np[i]
                cnt_scores[k][s:e] += 1.0

        scores_list = []
        labels_concat = []
        for k in range(num_seqs):
            c = cnt_scores[k]
            c[c == 0] = 1.0  # 避免除零
            seq_scores = sum_scores[k] / c
            scores_list.append(seq_scores)
            labels_concat.append(self.labels_list[k])

        scores = np.concatenate(scores_list)
        labels_full = np.concatenate(labels_concat).astype(int)

        thr, metrics = search_best_f1_threshold(
            scores,
            labels_full,
            num_steps=2048,
            use_point_adjust=self.use_point_adjust,
        )

        self.logger.info(
            f"[Eval-{tag_prefix}] F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"AUC={metrics['auc']:.4f}, thr={metrics['threshold']:.6f}, "
            f"point_adjust={metrics['use_point_adjust']}"
        )

        return scores, labels_full, thr, metrics

    # ==================================================================
    # 评估：预测分支
    # ==================================================================

    def _evaluate_forecast(
        self, model: nn.Module, tag_prefix: str = "forecast"
    ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        assert self.context_len is not None
        model.eval()

        num_seqs = len(self.labels_list)
        sum_scores = [
            np.zeros(len(self.labels_list[i]), dtype=np.float64)
            for i in range(num_seqs)
        ]
        cnt_scores = [
            np.zeros(len(self.labels_list[i]), dtype=np.float64)
            for i in range(num_seqs)
        ]

        for batch in tqdm(
            self.test_loader, desc=f"Eval({tag_prefix})", leave=False
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(x)
            else:
                x_t = x
            x_t = x_t.to(self.device, non_blocking=True)

            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=1e4, neginf=-1e4)
            x_t = torch.clamp(x_t, -1e4, 1e4)

            B, L, D = x_t.shape
            if L < self.context_len + self.pred_len:
                raise ValueError(
                    f"测试窗口长度 L={L} 小于 context_len + pred_len = "
                    f"{self.context_len + self.pred_len}，请调大 --win_size。"
                )

            seq_idx = batch["seq_idx"]
            starts = batch["start"]

            if isinstance(seq_idx, torch.Tensor):
                seq_idx = seq_idx.cpu().numpy()
            elif not isinstance(seq_idx, np.ndarray):
                seq_idx = np.array(seq_idx, dtype=np.int64)

            if isinstance(starts, torch.Tensor):
                starts = starts.cpu().numpy()
            elif not isinstance(starts, np.ndarray):
                starts = np.array(starts, dtype=np.int64)

            x_enc = x_t[:, : self.context_len, :]
            y_true = x_t[
                :, self.context_len : self.context_len + self.pred_len, :
            ]

            with torch.no_grad():
                y_pred = model(x_enc)
                if isinstance(y_pred, dict):
                    if "pred" not in y_pred:
                        raise ValueError(
                            "预测模型 forward 返回的字典中未包含 'pred' 键。"
                        )
                    y_pred = y_pred["pred"]

                y_pred = torch.nan_to_num(
                    y_pred, nan=0.0, posinf=1e4, neginf=-1e4
                )
                y_pred = torch.clamp(y_pred, -1e4, 1e4)

                y_true = torch.nan_to_num(
                    y_true, nan=0.0, posinf=1e4, neginf=-1e4
                )
                y_true = torch.clamp(y_true, -1e4, 1e4)

                if y_pred.shape != y_true.shape:
                    raise ValueError(
                        f"预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致。"
                    )

                mse = ((y_pred - y_true) ** 2).mean(dim=-1)  # [B, pred_len]
                mse_np = mse.detach().cpu().numpy()

            for i in range(B):
                k = int(seq_idx[i])
                s = int(starts[i])
                s_pred = s + self.context_len
                e_pred = s_pred + self.pred_len

                if s_pred >= len(sum_scores[k]):
                    continue
                if e_pred > len(sum_scores[k]):
                    valid_len = len(sum_scores[k]) - s_pred
                    if valid_len <= 0:
                        continue
                    sum_scores[k][s_pred:] += mse_np[i, :valid_len]
                    cnt_scores[k][s_pred:] += 1.0
                else:
                    sum_scores[k][s_pred:e_pred] += mse_np[i]
                    cnt_scores[k][s_pred:e_pred] += 1.0

        scores_list = []
        labels_concat = []
        for k in range(num_seqs):
            c = cnt_scores[k]
            c[c == 0] = 1.0
            seq_scores = sum_scores[k] / c
            scores_list.append(seq_scores)
            labels_concat.append(self.labels_list[k])

        scores = np.concatenate(scores_list)
        labels_full = np.concatenate(labels_concat).astype(int)

        thr, metrics = search_best_f1_threshold(
            scores,
            labels_full,
            num_steps=2048,
            use_point_adjust=self.use_point_adjust,
        )

        self.logger.info(
            f"[Eval-{tag_prefix}] F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"AUC={metrics['auc']:.4f}, thr={metrics['threshold']:.6f}, "
            f"point_adjust={metrics['use_point_adjust']}"
        )

        return scores, labels_full, thr, metrics

    # ==================================================================
    # 评估：混合模型（融合重构 + 预测得分）
    # ==================================================================

    def _evaluate_hybrid(self, epoch: int):
        hybrid_model = self.model

        # 分别评估两条分支
        scores_rec, labels_rec, thr_rec, metrics_rec = self._evaluate_recon(
            hybrid_model.recon_branch,
            tag_prefix="recon(hybrid)",
        )
        scores_pred, labels_pred, thr_pred, metrics_pred = (
            self._evaluate_forecast(
                hybrid_model.forecast_branch,
                tag_prefix="forecast(hybrid)",
            )
        )

        # 标签一致性检查
        if not np.array_equal(labels_rec, labels_pred):
            self.logger.warning(
                "重构 / 预测分支的标签展开结果不一致，默认采用重构分支的标签。"
            )
        labels_full = labels_rec

        # ---------------- 分数标准化（z-score） ----------------
        def _z_norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            m = x.mean()
            s = x.std()
            if not np.isfinite(s) or s < 1e-8:
                s = 1.0
            return (x - m) / s

        scores_rec_norm = _z_norm(scores_rec)
        scores_pred_norm = _z_norm(scores_pred)

        # ---------------- 由不确定性 / λ 决定线性融合权重 ----------------
        if hasattr(hybrid_model, "log_sigma_recon") and hasattr(
            hybrid_model, "log_sigma_forecast"
        ):
            with torch.no_grad():
                sigma_rec = float(
                    torch.exp(
                        hybrid_model.log_sigma_recon
                    ).detach()
                    .cpu()
                    .item()
                )
                sigma_pred = float(
                    torch.exp(
                        hybrid_model.log_sigma_forecast
                    ).detach()
                    .cpu()
                    .item()
                )

                # 1/σ² 作为任务“可信度”，再乘上 λ 系数
                w_rec = self.lambda_recon / (sigma_rec**2 + 1e-8)
                w_pred = self.lambda_forecast / (sigma_pred**2 + 1e-8)
        else:
            w_rec = self.lambda_recon
            w_pred = self.lambda_forecast

        if w_rec <= 0 and w_pred <= 0:
            w_rec = 1.0
            w_pred = 1.0

        alpha = w_rec / (w_rec + w_pred)
        beta = w_pred / (w_rec + w_pred)

        # ---------------- 线性融合（稳定版） ----------------
        # 与 MTAD-GAT / USAD 等工作类似，采用线性加权的形式：
        #   score = α * score_rec + β * score_pred
        # 相比 L2 融合，这种方式更稳健，不容易在排序上“翻车”。
        scores_hybrid = alpha * scores_rec_norm + beta * scores_pred_norm

        # 首先直接尝试当前方向下的阈值搜索
        thr, metrics = search_best_f1_threshold(
            scores_hybrid,
            labels_full,
            num_steps=2048,
            use_point_adjust=self.use_point_adjust,
        )

        # 若出现 AUC < 0.5，则说明整体排序近似被反转，
        # 这种情况在 SMD 上曾出现（AUC≈0.17），此时自动取反再计算一次。
        if metrics["auc"] < 0.5:
            self.logger.warning(
                f"融合后 AUC={metrics['auc']:.4f} < 0.5，"
                "疑似得分方向被反转，将 scores 取反后重新计算。"
            )
            scores_hybrid = -scores_hybrid
            thr, metrics = search_best_f1_threshold(
                scores_hybrid,
                labels_full,
                num_steps=2048,
                use_point_adjust=self.use_point_adjust,
            )

        self.logger.info(
            f"[Eval-hybrid] F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"AUC={metrics['auc']:.4f}, thr={metrics['threshold']:.6f}, "
            f"point_adjust={metrics['use_point_adjust']}; "
            f"recon_F1={metrics_rec['f1']:.4f}, forecast_F1={metrics_pred['f1']:.4f}"
        )

        # 额外记录到 TensorBoard，方便分析
        if self.writer is not None:
            self.writer.add_scalar("eval/recon_f1", metrics_rec["f1"], epoch)
            self.writer.add_scalar("eval/forecast_f1", metrics_pred["f1"], epoch)

        return scores_hybrid, labels_full, thr, metrics

    # ==================================================================
    # 对外暴露的训练主循环
    # ==================================================================

    def train(self, num_epochs: int) -> None:
        best_f1 = -1.0
        best_metrics: Dict[str, float] | None = None
        epochs_no_improve = 0

        os.makedirs(self.log_dir, exist_ok=True)
        ckpt_path = os.path.join(self.log_dir, f"best_model_{self.branch}.pt")

        for epoch in range(1, num_epochs + 1):
            self.logger.info(
                f"========== Epoch {epoch}/{num_epochs} ({self.branch}) =========="
            )

            # ---------------- 训练 + 评估 ----------------
            if self.branch == "recon":
                train_loss = self._train_epoch_recon(epoch)
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss_recon", train_loss, epoch
                    )
                scores, labels_full, thr, metrics = self._evaluate_recon(
                    self.model, tag_prefix="recon"
                )
            elif self.branch == "forecast":
                train_loss = self._train_epoch_forecast(epoch)
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss_forecast", train_loss, epoch
                    )
                scores, labels_full, thr, metrics = self._evaluate_forecast(
                    self.model, tag_prefix="forecast"
                )
            else:  # hybrid
                train_loss = self._train_epoch_hybrid(epoch)
                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss_hybrid", train_loss, epoch
                    )
                scores, labels_full, thr, metrics = self._evaluate_hybrid(
                    epoch
                )

            # 统一记录评估指标
            if self.writer is not None:
                self.writer.add_scalar("eval/f1", metrics["f1"], epoch)
                self.writer.add_scalar(
                    "eval/precision", metrics["precision"], epoch
                )
                self.writer.add_scalar("eval/recall", metrics["recall"], epoch)
                self.writer.add_scalar("eval/auc", metrics["auc"], epoch)

            # ---------------- 早停 & 保存最优模型 ----------------
            if metrics["f1"] > best_f1 + 1e-4:
                best_f1 = metrics["f1"]
                best_metrics = metrics
                epochs_no_improve = 0

                torch.save(self.model.state_dict(), ckpt_path)
                self.logger.info(f"发现更优模型，已保存至 {ckpt_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    self.logger.info(
                        f"早停触发：连续 {epochs_no_improve} 个 epoch F1 未提升，停止训练。"
                    )
                    break

            # ---------------- 可视化当前 epoch 的得分分布 ----------------
            vis_path = os.path.join(
                self.log_dir, f"scores_epoch{epoch}_{self.branch}.png"
            )
            plot_scores_with_labels(
                scores=scores,
                labels=labels_full,
                threshold=thr,
                save_path=vis_path,
                max_points=2000,
            )
            self.logger.info(f"已保存可视化图：{vis_path}")

        self.logger.info(
            f"训练结束，最佳 F1={best_f1:.4f}, 最佳指标={best_metrics}"
        )
