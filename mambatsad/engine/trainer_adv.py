# mambatsad/engine/trainer_adv.py
# -*- coding: utf-8 -*-
"""
TSADAdvTrainer: 在原有 TSADTrainer(hybrid) 的基础上引入输入级对抗训练。

设计原则：
- **不改动** 原始 TSADTrainer 的训练 / 评估逻辑；
- 通过继承 TSADTrainer，并覆写 `_train_epoch_hybrid`，实现对抗训练版本；
- 模型结构仍然来自 `mambatsad.models.hybrid.build_hybrid_model`，
  即使用你原本的重构分支 + 预测分支，保证与 main.py 一致；
- 只对 branch="hybrid_shared_adv" 开启该 Trainer，方便与普通 "hybrid" 做 ablation。

目前仅支持混合模型上的对抗训练（branch="hybrid_shared_adv"）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm

from .trainer import TSADTrainer


class TSADAdvTrainer(TSADTrainer):
    """在混合模型基础上加入 FGSM 风格输入对抗训练的 Trainer。"""

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
        # ---- 对抗训练相关超参数 ----
        adv_epsilon: float = 0.05,
        adv_beta: float = 0.5,
        adv_warmup_epochs: int = 5,
    ) -> None:
        branch = branch.lower()
        if branch != "hybrid_shared_adv":
            raise ValueError(
                f"TSADAdvTrainer 目前只支持 branch='hybrid_shared_adv'，收到 branch={branch!r}"
            )

        # 注意：这里 **故意** 用 branch='hybrid' 调用父类构造，
        # 让父类帮我们构建标准混合模型 (MambaTSADHybrid)，
        # 然后再把 self.branch 改成 'hybrid_shared_adv' 仅用于日志 / 模型文件名。
        super().__init__(
            branch="hybrid",
            device=device,
            input_dim=input_dim,
            win_size=win_size,
            pred_len=pred_len,
            train_loader=train_loader,
            test_loader=test_loader,
            labels_list=labels_list,
            entity_ids=entity_ids,
            logger=logger,
            writer=writer,
            log_dir=log_dir,
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            patience=patience,
            use_point_adjust=use_point_adjust,
            use_amp=use_amp,
            lambda_recon=lambda_recon,
            lambda_forecast=lambda_forecast,
        )

        # 覆盖 branch，保证 ckpt 名称等为 best_model_hybrid_shared_adv.pt
        self.branch = "hybrid_shared_adv"

        # 对抗训练配置
        self.adv_epsilon = float(adv_epsilon)
        self.adv_beta = float(adv_beta)
        self.adv_warmup_epochs = int(adv_warmup_epochs)

        self.logger.info(
            "[TSADAdvTrainer] 对抗训练配置: epsilon=%.4f, beta=%.4f, warmup_epochs=%d",
            self.adv_epsilon,
            self.adv_beta,
            self.adv_warmup_epochs,
        )

    # ------------------------------------------------------------------
    # 对抗样本生成（FGSM）
    # ------------------------------------------------------------------
    def _generate_adversarial_examples(self, x: torch.Tensor) -> torch.Tensor:
        """
        针对当前 batch 的输入窗口 x 生成对抗样本 x_adv。

        - 使用 FGSM: x_adv = x + epsilon * sign(∂L/∂x)
        - 这里的 L 与训练时相同：λ_rec * L_rec + λ_pred * L_pred
        - 只回传梯度到 x，不更新模型参数。
        """
        if self.adv_epsilon <= 0.0 or self.adv_beta <= 0.0:
            # 未启用对抗训练
            return x

        hybrid_model: nn.Module = self.model
        hybrid_model.eval()  # 生成对抗样本时关闭 dropout 等随机性

        x_adv = x.detach().clone().requires_grad_(True)
        self.optimizer.zero_grad(set_to_none=True)

        # 不启用 AMP，避免 half 精度下梯度过小或不稳定
        out = hybrid_model(x_adv)
        if not isinstance(out, dict):
            raise ValueError("混合模型 forward 必须返回 dict。")

        rec_list = out.get("recon_multi")
        recon = out.get("recon")
        y_pred = out.get("pred")
        if y_pred is None:
            raise ValueError("混合模型未返回 'pred'。")

        assert self.context_len is not None
        y_true = x_adv[:, self.context_len : self.context_len + self.pred_len, :]

        if rec_list is None and recon is None:
            raise ValueError("混合模型未返回重构结果。")

        if rec_list is None:
            loss_recon = self._safe_mse_loss(recon, x_adv)
        else:
            loss_val = 0.0
            for rec in rec_list:
                loss_val = loss_val + self._safe_mse_loss(rec, x_adv)
            loss_recon = loss_val

        loss_forecast = self._safe_mse_loss(y_pred, y_true)
        loss = self.lambda_recon * loss_recon + self.lambda_forecast * loss_forecast

        loss.backward()
        grad = x_adv.grad

        if grad is None or not torch.isfinite(grad).all():
            self.logger.warning(
                "[TSADAdvTrainer] 生成对抗样本时梯度为 None 或包含 NaN/Inf，跳过对抗扰动。"
            )
            hybrid_model.train()
            return x

        # FGSM：沿符号方向一步
        delta = self.adv_epsilon * grad.sign()
        x_adv = x_adv.detach() + delta
        x_adv = torch.nan_to_num(x_adv, nan=0.0, posinf=1e4, neginf=-1e4)
        x_adv = torch.clamp(x_adv, -1e4, 1e4)

        hybrid_model.train()  # 恢复训练模式
        return x_adv.detach()

    # ------------------------------------------------------------------
    # 覆写混合模型的训练一个 epoch：加入对抗分支
    # ------------------------------------------------------------------
    def _train_epoch_hybrid(self, epoch: int) -> float:  # type: ignore[override]
        assert self.context_len is not None
        hybrid_model: nn.Module = self.model
        hybrid_model.train()

        total_loss = 0.0
        total_loss_recon = 0.0
        total_loss_forecast = 0.0
        total_loss_adv = 0.0
        num_batches = 0

        use_adv = (
            (epoch + 1) >= self.adv_warmup_epochs
            and self.adv_epsilon > 0.0
            and self.adv_beta > 0.0
        )
        if use_adv:
            self.logger.info(
                "[Epoch %d] hybrid_shared_adv 启用对抗训练 (epsilon=%.4f, beta=%.4f)",
                epoch,
                self.adv_epsilon,
                self.adv_beta,
            )

        torch.autograd.set_detect_anomaly(False)

        for batch in tqdm(
            self.train_loader,
            desc=f"Train-Epoch{epoch}(hybrid_shared_adv)",
            leave=False,
        ):
            x = batch["window"]
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            x = x.to(self.device, non_blocking=True)
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
            x = torch.clamp(x, -1e4, 1e4)

            B, L, D = x.shape
            if L < self.context_len + self.pred_len:
                raise ValueError(
                    f"窗口长度 L={L} 小于 context_len + pred_len = "
                    f"{self.context_len + self.pred_len}，请调大 --win_size。"
                )

            # 先基于当前模型生成对抗样本（不更新参数）
            if use_adv:
                x_adv = self._generate_adversarial_examples(x)
            else:
                x_adv = None

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.scaler is not None)):
                # ---------------- 干净样本前向 ----------------
                out = hybrid_model(x)
                if not isinstance(out, dict):
                    raise ValueError("混合模型 forward 必须返回 dict。")

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

                y_pred = out.get("pred")
                if y_pred is None:
                    raise ValueError("混合模型未返回预测结果 'pred'。")

                y_true = x[:, self.context_len : self.context_len + self.pred_len, :]
                if y_pred.shape != y_true.shape:
                    raise ValueError(
                        f"混合模型预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致。"
                    )

                loss_forecast = self._safe_mse_loss(y_pred, y_true)
                clean_loss = (
                    self.lambda_recon * loss_recon
                    + self.lambda_forecast * loss_forecast
                )

                # ---------------- 对抗样本前向 ----------------
                adv_loss = torch.tensor(0.0, device=self.device)
                if use_adv and x_adv is not None:
                    out_adv = hybrid_model(x_adv)
                    if not isinstance(out_adv, dict):
                        raise ValueError("混合模型 forward 必须返回 dict。")

                    rec_list_adv = out_adv.get("recon_multi")
                    recon_adv = out_adv.get("recon")
                    if rec_list_adv is None and recon_adv is None:
                        raise ValueError("混合模型未返回对抗重构结果。")

                    if rec_list_adv is None:
                        loss_recon_adv = self._safe_mse_loss(recon_adv, x_adv)
                    else:
                        loss_val_adv = 0.0
                        for rec in rec_list_adv:
                            loss_val_adv = loss_val_adv + self._safe_mse_loss(
                                rec, x_adv
                            )
                        loss_recon_adv = loss_val_adv

                    y_pred_adv = out_adv.get("pred")
                    if y_pred_adv is None:
                        raise ValueError("混合模型未返回对抗预测结果 'pred'。")

                    y_true_adv = x_adv[
                        :, self.context_len : self.context_len + self.pred_len, :
                    ]
                    if y_pred_adv.shape != y_true_adv.shape:
                        raise ValueError(
                            "对抗样本预测输出形状与目标不一致："
                            f"{y_pred_adv.shape} vs {y_true_adv.shape}"
                        )

                    loss_forecast_adv = self._safe_mse_loss(y_pred_adv, y_true_adv)
                    adv_loss = (
                        self.lambda_recon * loss_recon_adv
                        + self.lambda_forecast * loss_forecast_adv
                    )

                # 总损失：干净样本 + β * 对抗样本
                loss = clean_loss + self.adv_beta * adv_loss

            if not self._backward(loss):
                continue

            total_loss += float(loss.detach().cpu().item())
            total_loss_recon += float(loss_recon.detach().cpu().item())
            total_loss_forecast += float(loss_forecast.detach().cpu().item())
            if use_adv:
                total_loss_adv += float(adv_loss.detach().cpu().item())
            num_batches += 1

        if num_batches == 0:
            return 0.0

        avg_loss = total_loss / num_batches
        avg_rec = total_loss_recon / num_batches
        avg_fore = total_loss_forecast / num_batches
        avg_adv = total_loss_adv / max(num_batches, 1) if use_adv else 0.0

        self.logger.info(
            "[Epoch %d] hybrid_shared_adv 训练损失: total=%.6f, recon=%.6f, "
            "forecast=%.6f, adv=%.6f",
            epoch,
            avg_loss,
            avg_rec,
            avg_fore,
            avg_adv,
        )

        if self.writer is not None:
            self.writer.add_scalar("train/loss_hybrid_adv/total", avg_loss, epoch)
            self.writer.add_scalar("train/loss_hybrid_adv/recon", avg_rec, epoch)
            self.writer.add_scalar("train/loss_hybrid_adv/forecast", avg_fore, epoch)
            if use_adv:
                self.writer.add_scalar("train/loss_hybrid_adv/adv", avg_adv, epoch)

        return avg_loss
