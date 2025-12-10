# mambatsad/engine/adv_loss.py
# -*- coding: utf-8 -*-
"""
对抗训练相关的损失函数。

这里给出一个相对稳定、数值上始终为正的 GAN 损失实现（基于 BCE with logits），
避免出现之前 total loss 为巨大负数的情况。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    """
    判别器损失：区分“真实样本 (label=1)” 与 “生成样本 (label=0)”。

    参数
    ----
    real_logits : 判别器在真实样本上的输出，shape [B, 1] 或 [B]；
    fake_logits : 判别器在生成样本上的输出。

    返回
    ----
    loss : 标量张量，越小越好（>=0）。
    """
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)

    loss_real = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)

    return 0.5 * (loss_real + loss_fake)


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    生成器（这里对应你的混合模型）希望“骗过”判别器：
    - 即让判别器把生成样本判成 1（正常）。

    参数
    ----
    fake_logits : 判别器在生成样本上的输出。

    返回
    ----
    loss : 标量张量，越小越好（>=0）。
    """
    target_labels = torch.ones_like(fake_logits)
    loss = F.binary_cross_entropy_with_logits(fake_logits, target_labels)
    return loss
