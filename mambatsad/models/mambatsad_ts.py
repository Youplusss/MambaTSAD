# -*- coding: utf-8 -*-
"""
MambaTSAD: 基于 Mamba 的多维时间序列异常检测模型
核心思想：
1. 用 Mamba（State Space Model）做长序列全局建模，捕获长时间依赖
2. 用多核深度可分离卷积做局部模式建模（类似 MambaAD 的 LSS 模块）
3. 多时间尺度金字塔重构，多尺度重构误差作为异常评分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba  # 官方 mamba-ssm 库 :contentReference[oaicite:3]{index=3}


class BiMambaBlock(nn.Module):
    """
    双向 Mamba Block：
    - 沿时间维做前向 Mamba 扫描
    - 再对时间反转做一次 Mamba（相当于“反向扫描”）
    这对应了原论文 HSS 中“多方向扫描”的思想，不过在时间序列上使用
    正向/反向时间，而不是 Hilbert 曲线。:contentReference[oaicite:4]{index=4}
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # 正向、反向两个 Mamba 分支
        self.mamba_fwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # FFN 前馈网络，进一步提升表达力
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L, C]  B: batch, L: 时间长度, C: 特征维度
        """
        # ---- Mamba 全局建模部分 ----
        x_norm = self.norm(x)
        # 正向时间 Mamba
        y_fwd = self.mamba_fwd(x_norm)  # [B, L, C]
        # 反向时间 Mamba：先反转时间，再反转回来
        x_rev = torch.flip(x_norm, dims=[1])
        y_bwd = self.mamba_bwd(x_rev)
        y_bwd = torch.flip(y_bwd, dims=[1])

        # 融合双向信息，这里简单平均
        y = (y_fwd + y_bwd) / 2.0
        x = x + self.dropout(y)  # 残差连接

        # ---- FFN 层 ----
        y_ffn = self.ffn(x)
        out = x + self.dropout(y_ffn)  # 再加一层残差

        return out


class LocalConvBlock1D(nn.Module):
    """
    1D 局部卷积模块：
    - 结构：Conv1d(1x1) -> DWConv(k) -> Conv1d(1x1)
    - groups=channels 的 Conv1d 即深度可分离卷积（只在时间上卷积）
    这对应 MambaAD 中 LSS 的“多核卷积分支（k=5,7）”。:contentReference[oaicite:5]{index=5}
    """
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            # 1x1 卷积做通道间线性变换
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            # 深度可分离卷积（groups=channels）
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
            ),
            nn.InstanceNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x):
        """
        x: [B, L, C] -> 转成 [B, C, L] 做 1D 卷积
        """
        x_perm = x.transpose(1, 2)  # [B, C, L]
        y = self.block(x_perm)
        return y.transpose(1, 2)  # 再转回 [B, L, C]


class LSSBlockTS(nn.Module):
    """
    时间序列版 LSS 模块（Locality-Enhanced State Space）：
    - Global 分支：级联若干 BiMambaBlock，学习长时间依赖
    - Local 分支：两个不同 kernel_size 的 LocalConvBlock1D，学习局部形状
    - 输出：拼接 [Global, Local_k1, Local_k2] 后用 1x1 线性层降回 d_model，再加残差

    这严格对应原论文“LSS = HSS(Mamba) + 多核卷积”的设计，只是把 2D 特征图
    改成了 1D 时间序列。:contentReference[oaicite:6]{index=6}
    """
    def __init__(
        self,
        d_model,
        num_mamba_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        kernel_sizes=(5, 7),
        dropout=0.0,
    ):
        super().__init__()
        self.global_layers = nn.ModuleList([
            BiMambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_mamba_layers)
        ])

        self.local_branches = nn.ModuleList([
            LocalConvBlock1D(d_model, k) for k in kernel_sizes
        ])

        # 最终通道数：全局 1 份 + 局部 len(kernel_sizes) 份
        out_dim = d_model * (1 + len(kernel_sizes))
        self.proj = nn.Linear(out_dim, d_model)

    def forward(self, x):
        """
        x: [B, L, d_model]
        """
        # Global 分支：多层 BiMamba
        g = x
        for layer in self.global_layers:
            g = layer(g)

        # Local 分支：多核卷积
        locals_out = [branch(x) for branch in self.local_branches]

        # 拼接 Global + Local
        concat = torch.cat([g] + locals_out, dim=-1)  # [B, L, d_model * (1 + n_branch)]
        out = self.proj(concat)

        # 残差连接
        return out + x


class MambaTSAD(nn.Module):
    """
    整体 TSAD 模型：
    - 输入：多维时间序列窗口 [B, L, D_in]
    - Encoder：线性层把输入映射到 d_model
    - 多尺度金字塔：
        * level1: 原始分辨率 L
        * level2: 时间下采样 2 倍（L/2）
        * level3: 时间下采样 4 倍（L/4）
      每个尺度上堆叠若干 LSSBlockTS
    - Decoder：线性层从 d_model 重建回 D_in
    - 输出：多尺度重构结果，用于计算重构误差，作为异常评分
    """
    def __init__(
        self,
        input_dim,           # 原始通道数 D_in
        d_model=128,         # Mamba / 特征维度
        num_layers=(2, 2, 2),# 三个尺度上 LSSBlockTS 的层数
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 编码器：把原始维度映射到 d_model
        self.encoder = nn.Linear(input_dim, d_model)

        # 三个尺度的 LSS 堆叠
        self.level1 = nn.ModuleList([
            LSSBlockTS(
                d_model=d_model,
                num_mamba_layers=2,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers[0])
        ])

        self.level2 = nn.ModuleList([
            LSSBlockTS(
                d_model=d_model,
                num_mamba_layers=2,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers[1])
        ])

        self.level3 = nn.ModuleList([
            LSSBlockTS(
                d_model=d_model,
                num_mamba_layers=2,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(num_layers[2])
        ])

        # 解码器：从 d_model 映射回原始通道
        self.decoder = nn.Linear(d_model, input_dim)

    def forward_single_scale(self, x, blocks):
        """
        在单个尺度上串行通过若干 LSSBlockTS
        x: [B, L_s, d_model]
        """
        out = x
        for blk in blocks:
            out = blk(out)
        return out

    def forward(self, x):
        """
        前向：
        x: [B, L, D_in]
        返回：
        {
          "recon_multi": [rec1, rec2_upsampled, rec3_upsampled],
          "recon": 加权平均后的最终重构
        }
        """
        B, L, D = x.shape
        assert D == self.input_dim, "输入维度与模型 input_dim 不一致"

        # 编码
        x_embed = self.encoder(x)  # [B, L, d_model]

        # ----- Level 1: 原始分辨率 -----
        l1 = self.forward_single_scale(x_embed, self.level1)  # [B, L, d_model]

        # ----- Level 2: 下采样 2 倍 -----
        # 使用 avg_pool1d 实现时间下采样
        # 先转成 [B, d_model, L] 再 pool
        x2 = F.avg_pool1d(x_embed.transpose(1, 2), kernel_size=2, stride=2)
        l2 = self.forward_single_scale(x2.transpose(1, 2), self.level2)  # [B, L/2, d_model]

        # ----- Level 3: 下采样 4 倍 -----
        x3 = F.avg_pool1d(x_embed.transpose(1, 2), kernel_size=4, stride=4)
        l3 = self.forward_single_scale(x3.transpose(1, 2), self.level3)  # [B, L/4, d_model]

        # ----- 解码并上采样回原分辨率 -----
        rec1 = self.decoder(l1)  # [B, L, D_in]

        # level2 -> 上采样到 L
        l2_up = F.interpolate(
            l2.transpose(1, 2), size=L, mode="linear", align_corners=False
        ).transpose(1, 2)
        rec2 = self.decoder(l2_up)

        # level3 -> 上采样到 L
        l3_up = F.interpolate(
            l3.transpose(1, 2), size=L, mode="linear", align_corners=False
        ).transpose(1, 2)
        rec3 = self.decoder(l3_up)

        # 多尺度重构的简单平均，可以再加权
        recon = (rec1 + rec2 + rec3) / 3.0

        return {
            "recon_multi": [rec1, rec2, rec3],
            "recon": recon,
        }


def mambatsad_ts_base(input_dim: int) -> MambaTSAD:
    """
    一个默认配置的工厂函数，方便在 main 中直接调用。
    """
    model = MambaTSAD(
        input_dim=input_dim,
        d_model=128,
        num_layers=(2, 2, 2),
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    )
    return model