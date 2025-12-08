# -*- coding: utf-8 -*-
"""
随机种子相关的工具函数。
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: Optional[bool] = False) -> None:
    """固定所有能够控制到的随机种子。

    参数
    ----
    seed:
        基础随机种子。
    deterministic:
        是否开启 cudnn 的确定性选项（会略微降低速度，但结果完全可复现）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
