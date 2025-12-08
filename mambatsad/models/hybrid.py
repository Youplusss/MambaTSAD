# mambatsad/models/hybrid.py
# -*- coding: utf-8 -*-
"""
混合模型：同时包含重构分支和预测分支。

设计目标：
- 在一个统一的模型中封装两条子网络：
  * recon_branch: 负责重构任务，输出多尺度重构结果；
  * forecast_branch: 负责预测任务，输出未来 pred_len 步的预测结果；
- 方便在训练阶段做多任务联合优化；
- 在推理阶段可以同时得到重构误差和预测误差，并进一步融合为最终异常分数。

本次改动要点：
- 为混合模型增加两项“不确定性参数”：log_sigma_recon / log_sigma_forecast，
  供 Trainer 在多任务场景中进行自适应 loss 融合（Kendall 等人的做法），
  同时在 score 融合阶段使用 1/σ^2 作为权重；
- 保持两条分支结构上的独立性，便于与“只用重构 / 只用预测”的版本进行公平比较。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .recon import MambaTSAD, build_recon_model
from .forecast import MambaTSADForecast, build_forecast_model


class MambaTSADHybrid(nn.Module):
    """
    MambaTSAD 混合模型。

    参数
    ----
    input_dim: 输入特征维度 D。
    win_size:  滑动窗口长度 L。
    pred_len:  预测步数 T_pred。上下文长度将为 L - T_pred。
    """

    def __init__(
        self,
        input_dim: int,
        win_size: int,
        pred_len: int,
    ) -> None:
        super().__init__()

        if pred_len <= 0:
            raise ValueError("pred_len 必须为正整数。")
        if win_size <= pred_len:
            raise ValueError(
                f"混合模型要求 win_size ({win_size}) > pred_len ({pred_len})，"
                "否则无法同时做重构与预测。"
            )

        self.input_dim = input_dim
        self.win_size = win_size
        self.pred_len = pred_len
        self.context_len = win_size - pred_len

        # 重构分支：复用单独训练时的 MambaTSAD 结构
        self.recon_branch: MambaTSAD = build_recon_model(input_dim=input_dim)

        # 预测分支：复用单独训练时的 MambaTSADForecast 结构
        self.forecast_branch: MambaTSADForecast = build_forecast_model(
            input_dim=input_dim,
            seq_len=self.context_len,
            pred_len=pred_len,
        )

        # ---------------- 多任务不确定性参数 ----------------
        # 这里采用 Kendall 等人提出的多任务 loss 不确定性加权思路：
        # - log_sigma_* 为可学习参数（标量），在训练过程中自动调整；
        # - Trainer 端使用 1/sigma^2 作为每个任务的 loss 权重，同时加上 log(sigma) 正则；
        # - 在 score 融合阶段，同样可以使用 1/sigma^2 作为融合权重。
        #
        # 初始化为 0 => sigma = 1，此时权重相当于不做特殊偏置。
        self.log_sigma_recon = nn.Parameter(torch.zeros(1))
        self.log_sigma_forecast = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        """
        前向同时计算重构与预测。

        输入:
            x: [B, L, D]，其中 L 必须等于初始化时的 win_size。

        返回:
            {
                "recon":      [B, L, D],
                "recon_multi": List[[B, L, D]],
                "pred":       [B, pred_len, D],
            }
        """
        if x.dim() != 3:
            raise ValueError(
                f"MambaTSADHybrid 期望输入形状为 [B, L, D]，实际为 {x.shape}"
            )
        B, L, D = x.shape
        if L != self.win_size:
            raise ValueError(
                f"MambaTSADHybrid 配置的 win_size={self.win_size}，"
                f"但当前输入长度为 {L}。"
            )
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSADHybrid 配置的 input_dim={self.input_dim}，"
                f"但当前输入通道数为 {D}。"
            )

        # ---------------- 重构分支 ----------------
        recon_out = self.recon_branch(x)
        recon = recon_out.get("recon")
        recon_multi = recon_out.get("recon_multi")

        # ---------------- 预测分支 ----------------
        # 只使用前 context_len 步作为上下文
        x_enc = x[:, : self.context_len, :]
        pred = self.forecast_branch(x_enc)

        return {
            "recon": recon,
            "recon_multi": recon_multi,
            "pred": pred,
        }


def build_hybrid_model(input_dim: int, win_size: int, pred_len: int) -> MambaTSADHybrid:
    """
    混合模型的工厂函数。
    """
    model = MambaTSADHybrid(
        input_dim=input_dim,
        win_size=win_size,
        pred_len=pred_len,
    )
    return model
