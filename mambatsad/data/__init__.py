# -*- coding: utf-8 -*-
"""
数据集工厂函数。

目前支持：
- "smd": Server Machine Dataset，多机版本；
- "msl": NASA MSL，多通道版本。

后续如需扩展新的数据集，只需要：
1. 在本包中新建 xxx.py，实现 build_xxx_multi_datasets；
2. 在 build_multi_entity_dataset 中注册一次即可。
"""
from __future__ import annotations

from typing import List, Tuple

from .smd import build_smd_multi_datasets
from .msl import build_msl_multi_datasets


def build_multi_entity_dataset(
    name: str,
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    """统一的数据集构建入口。

    返回
    ----
    train_ds, test_ds:
        训练 / 测试集 Dataset 对象。
    input_dim:
        每个时间步的特征维度。
    labels_list:
        按实体（机器 / 通道）切分的标签序列列表。
    entity_ids:
        实体 id 列表，例如 SMD 的 machine-1-1 等。
    """
    name_lower = name.lower()
    if name_lower == "smd":
        train_ds, test_ds, input_dim, labels_list, machine_ids = build_smd_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
        entity_ids = machine_ids
    elif name_lower == "msl":
        train_ds, test_ds, input_dim, labels_list, channel_ids = build_msl_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
        entity_ids = channel_ids
    else:
        raise ValueError(f"不支持的数据集名称：{name}")

    return train_ds, test_ds, input_dim, labels_list, entity_ids
