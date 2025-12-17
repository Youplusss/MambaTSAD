# models/MambaSTAMP.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mamba(x))


class MambaSTAMP(nn.Module):
    def __init__(self, win_size, enc_in, d_model=64, e_layers=3, dropout=0.1, device='cuda'):
        super(MambaSTAMP, self).__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.d_model = d_model

        # 1. 空间学习层 (Spatial Learning)
        # 模仿 STAMP，使用线性层捕获特征间的依赖，将原始特征维度映射到 d_model
        self.spatial_embedding = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. 时序学习层 (Temporal Learning - Shared Encoder)
        # 这是一个双向 Mamba 结构，预测和重构任务共享这个编码器
        self.encoder = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(e_layers)
        ])

        self.reverse_encoder = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(e_layers)
        ])

        # 3. 解码头 (Heads)
        # 预测头: 预测 t+1
        self.pred_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, enc_in)
        )

        # 重构头: 重构 t
        self.rec_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, enc_in)
        )

    def forward(self, x):
        # x shape: [Batch, Win_Size, Channels]

        # Spatial Embedding
        x_emb = self.spatial_embedding(x)  # [B, L, D]

        # Bi-Directional Mamba Encoding
        x_fwd = x_emb
        for layer in self.encoder:
            x_fwd = layer(x_fwd)

        x_bwd = torch.flip(x_emb, dims=[1])
        for layer in self.reverse_encoder:
            x_bwd = layer(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        # 拼接双向特征
        enc_out = torch.cat([x_fwd, x_bwd], dim=-1)  # [B, L, 2*D]

        # 输出两个分支的结果
        recon = self.rec_head(enc_out)
        pred = self.pred_head(enc_out)

        return recon, pred