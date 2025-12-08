# mambatsad/models/mambatsad_pred.py
# -*- coding: utf-8 -*-
"""
基于 Mamba 的时间序列预测分支（TSAD 预测模型）

整体思路：
- 参考 S-D-Mamba（Is Mamba Effective for Time Series Forecasting?）中的 S-Mamba 结构，
  使用“倒置嵌入”：把每个变量 / 通道视作一个 token，在 token 维上做 Mamba 序列建模；
- 结合时间序列异常检测常见做法（如 Anomaly Transformer / TranAD 中的预测残差作为异常分数），
  使用多步预测误差作为异常评分；
- 保持与仓库中重构分支（MambaTSAD）一致的代码风格与接口，方便后续融合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 官方 mamba-ssm 库
    from mamba_ssm import Mamba
except ImportError as e:  # pragma: no cover - 环境错误时直接提示
    raise ImportError(
        "使用预测分支模型需要先安装 mamba-ssm 库，"
        "请先执行：pip install mamba-ssm"
    ) from e


class InvertedTimeEmbedding(nn.Module):
    """
    简单版“倒置时间嵌入”(DataEmbedding_inverted)：

    - 输入: x ∈ R^{B, L, D}
      * B: batch size
      * L: 时间长度（context_len）
      * D: 通道数（变量数）
    - 把每个通道上的时间序列 x[:, :, i] 看成一个 token：
      使用共享线性层 Linear(L → d_model) 进行投影。

    这样就得到了形状为 [B, D, d_model] 的 token 表示，
    后续在 D 这一维上使用 Mamba 进行序列建模。
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        # 对每个 token 的时间序列做线性投影：R^{L} -> R^{d_model}
        self.proj = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            形状 [B, L, D] 的输入时间序列

        返回
        ----
        emb : torch.Tensor
            形状 [B, D, d_model] 的 token 表示
        """
        if x.dim() != 3:
            raise ValueError(f"InvertedTimeEmbedding 期望输入形状为 [B, L, D]，实际为 {x.shape}")
        B, L, D = x.shape
        if L != self.seq_len:
            raise ValueError(
                f"InvertedTimeEmbedding 配置的 seq_len={self.seq_len}，"
                f"但当前 batch 的时间长度为 {L}，请确保 win_size - pred_len 与 seq_len 一致。"
            )

        # [B, L, D] -> [B, D, L]
        x_perm = x.permute(0, 2, 1).contiguous()
        # 合并 batch 和通道维度，便于一次性通过线性层
        x_flat = x_perm.view(B * D, L)  # [B*D, L]

        # 线性投影到 d_model 空间
        emb_flat = self.proj(x_flat)  # [B*D, d_model]
        emb = emb_flat.view(B, D, self.d_model)  # [B, D, d_model]
        emb = self.dropout(emb)
        return emb


class BiMambaBlock1D(nn.Module):
    """
    一维双向 Mamba Block（与重构分支中的 BiMambaBlock 思路相同）：

    - 先做 LayerNorm
    - 正向通过一个 Mamba 序列模型
    - 再对翻转后的序列做一次 Mamba（相当于“反向扫描”），然后再翻转回来
    - 将正向 / 反向结果平均后做一次残差连接
    - 再接一层前馈网络（FFN） + 残差

    这里的“序列维”可以是时间维，也可以是通道维：
    在预测分支中，我们在“通道维”（变量维）上做建模。
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

        # 正向 / 反向两个 Mamba 模块
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

        # FFN 前馈子层
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            形状 [B, L, C]，L 为“序列”长度（可为时间或通道），C 为通道维

        返回
        ----
        out : torch.Tensor
            与 x 同形状的输出
        """
        # 归一化
        x_norm = self.norm(x)

        # 正向 Mamba
        y_fwd = self.mamba_fwd(x_norm)  # [B, L, C]

        # 反向 Mamba：翻转序列，再翻回来
        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # 融合双向信息
        y = (y_fwd + y_bwd) / 2.0

        # 残差 1
        x = x + self.dropout(y)

        # FFN + 残差 2
        y_ffn = self.ffn(x)
        out = x + self.dropout(y_ffn)
        return out


class MambaTSADForecast(nn.Module):
    """
    基于 Mamba 的时间序列预测模型（TSAD 预测分支）。

    设计要点：
    1. 倒置嵌入（InvertedTimeEmbedding）：
       - 把每一个变量视作一个 token；
       - token 的特征由该变量上一段时间窗口经线性层投影得到；
       - Mamba 在“变量维”（token 序列）上建模变量间依赖。
    2. 编码器由若干层 BiMambaBlock1D 组成，支持双向信息流；
    3. 输出层为线性 projector：将每个 token 的隐藏表示映射为 pred_len 个时间步的预测；
    4. 使用样本内均值方差归一化（Non-stationary Transformer / S-D-Mamba 中常见），
       有助于缓解非平稳时间序列问题。

    输入 / 输出：
    - 输入 x: [B, L_c, D]，L_c = context_len = win_size - pred_len
    - 输出 y_pred: [B, pred_len, D]
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

        # 多层 BiMamba 编码器（在“变量维”上建模）
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

        # 预测头：将每个 token 的表征映射为 pred_len 个时间步
        # [B, D, d_model] -> [B, D, pred_len]
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : torch.Tensor
            形状 [B, L_c, D] 的上下文序列，L_c 必须等于初始化时的 seq_len

        返回
        ----
        y_pred : torch.Tensor
            预测的未来序列，形状 [B, pred_len, D]
        """
        if x.dim() != 3:
            raise ValueError(f"MambaTSADForecast 期望输入形状为 [B, L_c, D]，实际为 {x.shape}")
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

        # ------------------ 样本内归一化（可选） ------------------ #
        # 沿时间维计算每条样本的均值 / 标准差，shape: [B, 1, D]
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

        # ------------------ 倒置嵌入：时间 -> token 特征 ------------------ #
        # [B, L_c, D] -> [B, D, d_model]
        tokens = self.enc_embedding(x_norm)

        # ------------------ 多层 BiMamba 编码 ------------------ #
        h = tokens  # [B, D, d_model]
        for layer in self.encoder_layers:
            h = layer(h)
        h = self.enc_norm(h)  # [B, D, d_model]

        # ------------------ 线性投影得到未来 pred_len 步预测 ------------------ #
        # [B, D, d_model] -> [B, D, pred_len] -> [B, pred_len, D]
        y = self.projector(h)  # [B, D, pred_len]
        y = y.permute(0, 2, 1).contiguous()  # [B, pred_len, D]

        # ------------------ 反归一化（若做过样本内归一化） ------------------ #
        if self.use_norm:
            # means, stdev: [B, 1, D] -> 自动广播至 [B, pred_len, D]
            y = y * stdev[:, 0:1, :] + means[:, 0:1, :]

        return y


def mambatsad_ts_pred_base(
    input_dim: int,
    seq_len: int,
    pred_len: int,
) -> MambaTSADForecast:
    """
    预测分支的默认工厂函数，方便在 main.py 中直接调用。

    参数
    ----
    input_dim : int
        输入通道数 D。
    seq_len : int
        上下文长度 L_c，一般设置为 win_size - pred_len。
    pred_len : int
        需要预测的未来时间步数。

    返回
    ----
    model : MambaTSADForecast
        已构造好的预测模型实例。
    """
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
