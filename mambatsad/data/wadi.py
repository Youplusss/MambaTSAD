# mambatsad/data/wadi.py
# -*- coding: utf-8 -*-
"""
WADI 预处理数据集加载。

目录结构：
dataset/WADI/
    entities.txt        # 通常只有 'wadi'
    train/
        wadi.npy        # [T_train, D]
    test/
        wadi.npy        # [T_test, D]
    test_label/
        wadi.npy        # [T_test]

与 SWAT 类似，复用 SMD 的多机滑窗 Dataset。
"""

from __future__ import annotations

import os
from typing import List, Tuple

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
    读取 WADI 实体 id，通常只有 'wadi'。
    """
    ent_txt = os.path.join(processed_root, "entities.txt")
    if os.path.exists(ent_txt):
        with open(ent_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return sorted(ids)

    train_dir = os.path.join(processed_root, "train")
    ids: List[str] = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_wadi_entity(processed_root: str, ent_id: str):
    """
    加载预处理后的 WADI 某实体数据。
    """
    train_path = os.path.join(processed_root, "train", f"{ent_id}.npy")
    test_path = os.path.join(processed_root, "test", f"{ent_id}.npy")
    label_path = os.path.join(processed_root, "test_label", f"{ent_id}.npy")

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(label_path)

    return train, test, labels


def build_wadi_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[SMDMultiWindowDataset, SMDMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建 WADI 多实体滑窗数据集（通常实体数=1）。

    返回
    ----
    train_ds, test_ds : 训练 / 测试 Dataset
    input_dim         : 特征维度 D
    labels_list       : 每个实体的测试标签序列
    entity_ids        : 实体 id 列表
    """
    entity_ids = load_entity_ids(processed_root)

    use_neighbor_fill = False

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for eid in entity_ids:
        train, test, labels = load_preprocessed_wadi_entity(processed_root, eid)

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
