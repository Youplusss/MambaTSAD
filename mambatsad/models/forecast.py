# mambatsad/models/forecast.py
# -*- coding: utf-8 -*-
"""
基于 Mamba 的时间序列预测分支（TSAD 预测模型）。

整体思路：
- 参考 S-D-Mamba（Is Mamba Effective for Time Series Forecasting?）中的
  "倒置嵌入" 结构，将每个变量视作一个 token；
- 对每个变量上一段时间窗口进行线性投影得到 token 表征，
  再在变量维上用 Mamba 做序列建模；
- 使用多步预测误差作为异常评分，类似 TranAD 等预测式异常检测模型。

本文件对应原仓库中的 mambatsad_pred.py。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "使用预测分支模型需要先安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


class InvertedTimeEmbedding(nn.Module):
    """简单版“倒置时间嵌入”。

    输入:
        x: [B, L, D]，其中 L 为时间长度，D 为变量数。

    做法:
        - 先把 x 转置为 [B, D, L]，把每个变量的一维时间序列视作一个 token；
        - 再通过线性层将长度 L 的时间序列投影到 d_model 维度；
        - 得到 [B, D, d_model] 的 token 表示。
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"InvertedTimeEmbedding 期望输入形状为 [B, L, D]，实际为 {x.shape}"
            )
        B, L, D = x.shape
        if L != self.seq_len:
            raise ValueError(
                f"InvertedTimeEmbedding 配置的 seq_len={self.seq_len}，"
                f"但当前 batch 的时间长度为 {L}，请确保 win_size - pred_len 与 seq_len 一致。"
            )

        # [B, L, D] -> [B, D, L]
        x_perm = x.permute(0, 2, 1).contiguous()
        x_flat = x_perm.view(B * D, L)  # [B*D, L]
        emb_flat = self.proj(x_flat)    # [B*D, d_model]
        emb = emb_flat.view(B, D, self.d_model)
        emb = self.dropout(emb)
        return emb


class BiMambaBlock1D(nn.Module):
    """一维双向 Mamba Block。

    与重构分支中的 BiMambaBlock 结构类似，但这里的“序列维”可以是
    时间维或通道维。预测分支中我们在变量维上建模变量间依赖。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
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
        # x: [B, L, C]
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


class MambaTSADForecast(nn.Module):
    """基于 Mamba 的时间序列预测模型（TSAD 预测分支）。

    输入:
        x: [B, L_c, D]，L_c = context_len = win_size - pred_len

    输出:
        y_pred: [B, pred_len, D]
    """

    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 128,
        e_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.use_norm = use_norm

        # 倒置时间嵌入：时间维 -> token 特征
        self.enc_embedding = InvertedTimeEmbedding(
            seq_len=seq_len,
            d_model=d_model,
            dropout=dropout,
        )

        # 在变量维上堆叠多层 BiMambaBlock1D
        self.encoder_layers = nn.ModuleList(
            [
                BiMambaBlock1D(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model)

        # 预测头：每个变量 token -> pred_len 个时间步
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"MambaTSADForecast 期望输入形状为 [B, L_c, D]，实际为 {x.shape}"
            )

        B, L, D = x.shape
        if L != self.seq_len:
            raise ValueError(
                f"MambaTSADForecast 配置的 seq_len={self.seq_len}，"
                f"但当前输入长度为 {L}。请检查 --win_size 与 --pred_len 的设置。"
            )
        if D != self.input_dim:
            raise ValueError(
                f"MambaTSADForecast 配置的 input_dim={self.input_dim}，"
                f"但当前输入通道数为 {D}。"
            )

        # 样本内归一化（沿时间维）
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x_centered = x - means
            var = torch.var(x_centered, dim=1, keepdim=True, unbiased=False)
            stdev = torch.sqrt(var + 1e-5)
            x_norm = x_centered / stdev
        else:
            means = None
            stdev = None
            x_norm = x

        # 倒置嵌入：时间 -> token 特征 [B, D, d_model]
        tokens = self.enc_embedding(x_norm)

        # 多层 BiMamba 编码
        h = tokens
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.enc_norm(h)  # [B, D, d_model]

        # 投影到 pred_len 个时间步
        y = self.projector(h)  # [B, D, pred_len]
        y = y.permute(0, 2, 1).contiguous()  # [B, pred_len, D]

        # 反归一化
        if self.use_norm:
            y = y * stdev[:, 0:1, :] + means[:, 0:1, :]

        return y


def build_forecast_model(input_dim: int, seq_len: int, pred_len: int) -> MambaTSADForecast:
    """预测分支的默认工厂函数。"""
    model = MambaTSADForecast(
        input_dim=input_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=128,
        e_layers=3,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        use_norm=True,
    )
    return model
