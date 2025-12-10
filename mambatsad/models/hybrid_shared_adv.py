# mambatsad/models/hybrid_shared_adv.py
# -*- coding: utf-8 -*-
"""
实验版：共享 encoder + 对抗式混合模型（MambaTSADHybridSharedAdv）

与主仓库的 MambaTSADHybrid 最大不同：
- 这里使用一个共享的时间序列 encoder（BiMambaBlock 堆叠），
  同时服务于重构头和预测头，参数共享；
- 预测头不再使用 “倒置嵌入（变量为 token）” 的结构，而是更传统的
  「时间维建模 -> 最后一个时间步特征 -> 预测未来多步」；
- 为了配合 STAMP 式对抗训练，我们暴露了：
    * encode()       : 共享 encoder
    * decode_recon() : 重构头
    * decode_forecast() : 预测头
  方便 Trainer 在正/负样本上分别调用。

注意：
- 这是一个「实验模型」，建议放在单独分支中使用，不影响 master 主分支。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "使用共享 encoder 混合模型需要安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


class BiMambaBlock(nn.Module):
    """
    与重构分支类似的一维双向 Mamba Block：
    - 在时间维上建模时序依赖；
    - 内部包含前向 Mamba + 反向 Mamba + FFN。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        x_norm = self.norm(x)
        y_fwd = self.mamba_fwd(x_norm)
        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        y = (y_fwd + y_bwd) / 2.0
        x = x + self.dropout(y)

        y_ffn = self.ffn(x)
        out = x + self.dropout(y_ffn)
        return out


class SharedEncoder(nn.Module):
    """
    共享的时间序列 encoder：
    - 先做线性输入映射 input_dim -> d_model；
    - 再堆叠若干层 BiMambaBlock；
    - 最后做 LayerNorm。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.encoder_in = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, D] -> h: [B, L, d_model]
        """
        if x.dim() != 3:
            raise ValueError(f"SharedEncoder 期望输入形状为 [B, L, D]，实际为 {x.shape}")
        B, L, D = x.shape
        if D != self.input_dim:
            raise ValueError(
                f"SharedEncoder 配置的 input_dim={self.input_dim}，"
                f"但当前输入通道数为 {D}。"
            )

        h = self.encoder_in(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h


class ReconHead(nn.Module):
    """
    重构头：在共享编码的基础上，再堆叠少量 BiMambaBlock + 线性映射回原始维度。
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        num_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim

        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, d_model] -> x_hat: [B, L, D]
        """
        x = h
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x_hat = self.proj_out(x)
        return x_hat


class ForecastHead(nn.Module):
    """
    预测头：从 context 部分的编码特征预测未来 pred_len 步。

    设计：
    - 输入 h_c: [B, L_c, d_model]；
    - 通过若干层 BiMambaBlock 建模时间依赖；
    - 使用最后一个时间步的特征 h_last: [B, d_model] 作为 summary；
    - 线性层输出 [B, pred_len * D]，reshape 成 [B, pred_len, D]；
    - 可选：以最后一个真实观测值为 baseline，做残差预测。
    """

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        pred_len: int,
        num_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_last_residual: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.pred_len = pred_len
        self.use_last_residual = use_last_residual

        self.layers = nn.ModuleList(
            [
                BiMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, pred_len * output_dim)

    def forward(self, h_context: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        h_context : [B, L_c, d_model] 共享 encoder 后的上下文特征。
        x_context : [B, L_c, D] 原始上下文序列，用于残差预测时的 baseline。

        返回
        ----
        y_pred : [B, pred_len, D]
        """
        h = h_context
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        # 使用最后一个时间步的特征作为 summary
        h_last = h[:, -1, :]  # [B, d_model]
        y_flat = self.proj_out(h_last)  # [B, pred_len * D]
        B = h.shape[0]
        y = y_flat.view(B, self.pred_len, self.output_dim)  # [B, pred_len, D]

        if self.use_last_residual:
            last_step = x_context[:, -1:, :]  # [B, 1, D]
            baseline = last_step.repeat(1, self.pred_len, 1)
            y = y + baseline

        return y


class MambaTSADHybridSharedAdv(nn.Module):
    """
    共享 encoder + 对抗训练版混合模型。

    参数
    ----
    input_dim : 输入特征维度 D
    win_size  : 滑窗长度 L
    pred_len  : 预测步数 T_pred（上下文长度 L_c = L - T_pred）

    前向返回
    ----
    dict:
        {
            "recon": x_hat,  # [B, L, D] 正常重构
            "pred": y_pred,  # [B, pred_len, D]
        }
    """

    def __init__(
        self,
        input_dim: int,
        win_size: int,
        pred_len: int,
        d_model: int = 128,
        shared_layers: int = 2,
        recon_layers: int = 1,
        forecast_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_last_residual: bool = True,
    ) -> None:
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len 必须为正整数。")
        if win_size <= pred_len:
            raise ValueError(
                f"混合模型要求 win_size({win_size}) > pred_len({pred_len})，"
                "否则无法同时做重构与预测。"
            )

        self.input_dim = input_dim
        self.win_size = win_size
        self.pred_len = pred_len
        self.context_len = win_size - pred_len

        # 共享 encoder
        self.shared_encoder = SharedEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=shared_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # 重构头
        self.recon_head = ReconHead(
            d_model=d_model,
            output_dim=input_dim,
            num_layers=recon_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )

        # 预测头
        self.forecast_head = ForecastHead(
            d_model=d_model,
            output_dim=input_dim,
            pred_len=pred_len,
            num_layers=forecast_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_last_residual=use_last_residual,
        )

    # ------------- 若干子接口，方便 Trainer 使用 -------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        共享 encoder：x -> h
        """
        return self.shared_encoder(x)

    def decode_recon(self, h: torch.Tensor) -> torch.Tensor:
        """
        仅重构头
        """
        return self.recon_head(h)

    def decode_forecast(self, h_context: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        """
        仅预测头
        """
        return self.forecast_head(h_context, x_context)

    # ------------- 标准 forward（训练 / 推理） -------------

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, L, D]
        """
        if x.dim() != 3:
            raise ValueError(f"MambaTSADHybridSharedAdv 期望输入形状为 [B, L, D]，实际为 {x.shape}")
        B, L, D = x.shape
        if L != self.win_size:
            raise ValueError(
                f"MambaTSADHybridSharedAdv 配置的 win_size={self.win_size}，"
                f"但当前输入长度为 {L}。"
            )
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSADHybridSharedAdv 配置的 input_dim={self.input_dim}，"
                f"但当前输入通道数为 {D}。"
            )

        # 共享 encoder
        h = self.encode(x)                              # [B, L, d_model]
        x_hat = self.decode_recon(h)                    # [B, L, D]

        h_context = h[:, : self.context_len, :]         # [B, L_c, d_model]
        x_context = x[:, : self.context_len, :]         # [B, L_c, D]
        y_pred = self.decode_forecast(h_context, x_context)  # [B, pred_len, D]

        return {
            "recon": x_hat,
            "pred": y_pred,
        }


def build_hybrid_shared_adv_model(
    input_dim: int,
    win_size: int,
    pred_len: int,
) -> MambaTSADHybridSharedAdv:
    """
    工厂函数，使用一套比较稳妥的默认超参。
    """
    model = MambaTSADHybridSharedAdv(
        input_dim=input_dim,
        win_size=win_size,
        pred_len=pred_len,
        d_model=128,
        shared_layers=2,
        recon_layers=1,
        forecast_layers=1,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_last_residual=True,
    )
    return model
