
import math
from typing import Optional

import torch
from torch import nn


class MambaBlock(nn.Module):
    """
    非严格还原 S-D-Mamba，而是一个「时间维上的 Mamba 样式块」。
    如果环境中安装了 mamba-ssm，则优先使用；否则回退为简单的双向 GRU，
    这样在没有 mamba-ssm 的环境下也能跑通单元测试。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore
            self.use_mamba = True
            self.backbone = Mamba(
                d_model=d_model,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        except Exception:
            self.use_mamba = False
            self.backbone = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D)
        """
        residual = x
        if self.use_mamba:
            y = self.backbone(x)  # (B, L, D)
        else:
            y, _ = self.backbone(x)  # (B, L, D)
        x = residual + self.dropout(y)
        x = self.norm(x)
        x_ffn = self.ffn(x)
        x = x + self.dropout(x_ffn)
        x = self.norm(x)
        return x


class SharedEncoder(nn.Module):
    """
    多层 MambaBlock 堆叠得到共享表征。
    """

    def __init__(self, input_dim: int, d_model: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [MambaBlock(d_model=d_model, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, D_in)
        return: z (B, L, d_model)
        """
        z = self.input_proj(x)
        for layer in self.layers:
            z = layer(z)
        z = self.output_norm(z)
        return z


class ReconHead(nn.Module):
    """
    简单的时间卷积 + 线性层做重构。
    """

    def __init__(self, d_model: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, L, d_model)
        return: x_hat: (B, L, D_out)
        """
        x = z.transpose(1, 2)  # (B, d_model, L)
        x = self.conv(x)
        x = self.act(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, L, d_model)
        x_hat = self.proj(x)
        return x_hat


class ForecastHead(nn.Module):
    """
    把共享表征 z 的最后若干步作为上下文，通过 MLP 预测未来 pred_len 步。
    这里不使用自回归解码，而是一次性回归 pred_len * D_out。
    """

    def __init__(self, d_model: int, output_dim: int, max_pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        self.max_pred_len = max_pred_len

        self.pool = nn.AdaptiveAvgPool1d(1)  # 在时间维做平均池化
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, max_pred_len * output_dim),
        )

    def forward(self, z: torch.Tensor, pred_len: int) -> torch.Tensor:
        """
        z: (B, L, d_model)
        pred_len: 预测步数，<= max_pred_len
        return: y_hat: (B, pred_len, D_out)
        """
        if pred_len > self.max_pred_len:
            raise ValueError(f"pred_len={pred_len} > max_pred_len={self.max_pred_len}")

        # 池化得到全局上下文
        x = z.transpose(1, 2)  # (B, d_model, L)
        x = self.pool(x).squeeze(-1)  # (B, d_model)

        y = self.mlp(x)  # (B, max_pred_len * D_out)
        B = y.shape[0]
        y = y.view(B, self.max_pred_len, self.output_dim)
        y = y[:, :pred_len, :]
        return y


class HybridSharedAdvModel(nn.Module):
    """
    共享编码器 + 重构头 + 预测头的混合模型，
    为 shared_adv 对抗训练提供一个干净且稳定的实现。
    """

    def __init__(self,
                 input_dim: int,
                 d_model: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_pred_len: int = 50):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_pred_len = max_pred_len

        self.encoder = SharedEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.recon_head = ReconHead(
            d_model=d_model,
            output_dim=input_dim,
            dropout=dropout,
        )
        self.forecast_head = ForecastHead(
            d_model=d_model,
            output_dim=input_dim,
            max_pred_len=max_pred_len,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, pred_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, L, D_in)
        pred_len: 预测步数
        return:
            recon: (B, L, D_in)
            forecast: (B, pred_len, D_in) —— 对应于窗口末尾 pred_len 步的预测
        """
        if x.ndim != 3:
            raise ValueError(f"输入 x 维度应为 (B, L, D)，当前 shape={tuple(x.shape)}")
        z = self.encoder(x)           # (B, L, d_model)
        recon = self.recon_head(z)    # (B, L, D_in)
        forecast = self.forecast_head(z, pred_len=pred_len)  # (B, pred_len, D_in)
        return recon, forecast
