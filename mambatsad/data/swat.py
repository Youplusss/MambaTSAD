# mambatsad/data/swat.py
# -*- coding: utf-8 -*-
"""
SWaT 预处理数据集加载。

目录结构（由 tools/preprocess_swat.py 生成）：
dataset/SWAT/
    entities.txt        # 仅一行：swat
    train/
        swat.npy        # [T_train, D]
    test/
        swat.npy        # [T_test, D]
    test_label/
        swat.npy        # [T_test]

这里复用 SMD 中的多机滑窗 Dataset，逻辑完全一致，只是实体 id 不同。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from torch.utils.data import Dataset

from .smd import (
    SMDMultiWindowDataset,
    compute_global_norm_stats,
    apply_norm_to_seqs,
    fill_sequence_with_own_feature_mean,
    fill_sequence_with_neighbor_mean,
)


def load_entity_ids(processed_root: str) -> List[str]:
    """
    读取 SWAT 实体 id 列表。

    当前预处理脚本只生成一个实体 "swat"，
    但这里仍按列表形式实现，方便以后扩展（例如多段 SWAT 数据）。
    """
    ent_txt = os.path.join(processed_root, "entities.txt")
    if os.path.exists(ent_txt):
        with open(ent_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return sorted(ids)

    # 兜底：根据 train 目录中的 .npy 文件名推断
    train_dir = os.path.join(processed_root, "train")
    ids: List[str] = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_swat_entity(processed_root: str, ent_id: str):
    """
    加载某个实体（通常只有 'swat'）的预处理结果。
    返回：
    - train: [T_train, D]
    - test: [T_test, D]
    - labels: [T_test]
    """
    train_path = os.path.join(processed_root, "train", f"{ent_id}.npy")
    test_path = os.path.join(processed_root, "test", f"{ent_id}.npy")
    label_path = os.path.join(processed_root, "test_label", f"{ent_id}.npy")

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(label_path)

    return train, test, labels


def build_swat_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[SMDMultiWindowDataset, SMDMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建 SWAT 的多实体（实际上通常只有一个）滑动窗口数据集。

    返回
    ----
    train_ds, test_ds : 训练 / 测试 Dataset
    input_dim         : 特征维度 D
    labels_list       : 每个实体对应的测试标签序列
    entity_ids        : 实体 id 列表
    """
    entity_ids = load_entity_ids(processed_root)

    # 与 SMD / MSL 一致的非有限值处理策略
    use_neighbor_fill = False

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for eid in entity_ids:
        train, test, labels = load_preprocessed_swat_entity(processed_root, eid)

        if use_neighbor_fill:
            train = fill_sequence_with_neighbor_mean(train)
            test = fill_sequence_with_neighbor_mean(test)
        else:
            train = fill_sequence_with_own_feature_mean(train)
            test = fill_sequence_with_own_feature_mean(test)

        labels = np.where(np.isfinite(labels), labels, 0).astype(np.int64)

        train_seqs.append(train.astype(np.float32))
        test_seqs.append(test.astype(np.float32))
        labels_list.append(labels.astype(np.int64))

    # 全局 z-score 归一化
    norm_stats = compute_global_norm_stats(train_seqs, method="zscore")
    train_seqs = apply_norm_to_seqs(train_seqs, norm_stats, method="zscore")
    test_seqs = apply_norm_to_seqs(test_seqs, norm_stats, method="zscore")

    input_dim = train_seqs[0].shape[1]

    train_ds = SMDMultiWindowDataset(
        sequences=train_seqs,
        labels_list=None,
        win_size=win_size,
        stride=train_stride,
        mode="train",
    )
    test_ds = SMDMultiWindowDataset(
        sequences=test_seqs,
        labels_list=labels_list,
        win_size=win_size,
        stride=test_stride,
        mode="test",
    )

    return train_ds, test_ds, input_dim, labels_list, entity_ids
