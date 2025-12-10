# mambatsad/data/__init__.py
# -*- coding: utf-8 -*-
"""
数据集统一构建入口。

目前支持的数据集：
- smd   : Server Machine Dataset，多机多变量；
- msl   : Mars Science Laboratory，多通道多变量；
- swat  : Secure Water Treatment，工业控制系统多变量；
- wadi  : Water Distribution，多变量。

每个数据集模块提供一个 build_xxx_multi_datasets 函数，
返回统一的五元组：
    train_ds, test_ds, input_dim, labels_list, entity_ids
"""

from __future__ import annotations

from typing import List, Tuple

from .smd import build_smd_multi_datasets, SMDMultiWindowDataset
from .msl import build_msl_multi_datasets, MSLMultiWindowDataset
from .swat import build_swat_multi_datasets
from .wadi import build_wadi_multi_datasets


def build_multi_entity_dataset(
    name: str,
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    """
    根据数据集名称构建统一的「多实体滑窗数据集」。

    参数
    ----
    name : 数据集名称，大小写不敏感，可选：
           "smd" / "msl" / "swat" / "wadi"
    processed_root : 预处理数据根目录（如 ./dataset/SMD）。
    win_size       : 滑动窗口长度。
    train_stride   : 训练集滑窗步长。
    test_stride    : 测试集滑窗步长。

    返回
    ----
    train_ds, test_ds, input_dim, labels_list, entity_ids
      其中：
      - train_ds, test_ds : torch.utils.data.Dataset
      - input_dim         : int，特征维度 D
      - labels_list       : List[np.ndarray]，每个实体一条 [T_test] 标签
      - entity_ids        : List[str]，实体 id 列表（机器 / 通道 / 工厂等）
    """
    name = name.lower()

    if name == "smd":
        return build_smd_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
    elif name == "msl":
        return build_msl_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
    elif name == "swat":
        return build_swat_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
    elif name == "wadi":
        return build_wadi_multi_datasets(
            processed_root=processed_root,
            win_size=win_size,
            train_stride=train_stride,
            test_stride=test_stride,
        )
    else:
        raise ValueError(f"暂不支持的数据集名称：{name}，可选 smd/msl/swat/wadi")
