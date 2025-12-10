# mambatsad/data/swat.py
# -*- coding: utf-8 -*-
"""
SWaT 预处理数据集加载。

目录结构（由 tools/preprocess_swat.py 生成）::
    dataset/SWAT/
        entities.txt
        train/ swat.npy
        test/  swat.npy
        test_label/ swat.npy

preprocess_swat.py 中已经使用 StandardScaler 基于训练集做了一次标准化，
本文件**不再做二次 z-score**，只做有限值清洗 + 滑窗。
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from .smd import SMDMultiWindowDataset


def _ensure_finite(arr: np.ndarray) -> np.ndarray:
    """与 wadi 一致的有限值兜底处理。"""
    if np.isfinite(arr).all():
        return arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def load_entity_ids(processed_root: str) -> List[str]:
    """读取 SWAT 实体 id 列表。通常只有 'swat'。"""
    ent_txt = os.path.join(processed_root, "entities.txt")
    if os.path.exists(ent_txt):
        with open(ent_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if ids:
            return sorted(ids)

    train_dir = os.path.join(processed_root, "train")
    ids: List[str] = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_swat_entity(processed_root: str, ent_id: str):
    """加载 SWAT 某实体数据。"""
    train_path = os.path.join(processed_root, "train", f"{ent_id}.npy")
    test_path = os.path.join(processed_root, "test", f"{ent_id}.npy")
    label_path = os.path.join(processed_root, "test_label", f"{ent_id}.npy")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"训练文件不存在: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"测试文件不存在: {test_path}")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"标签文件不存在: {label_path}")

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(label_path)

    return (
        _ensure_finite(train),
        _ensure_finite(test),
        np.where(np.isfinite(labels), labels, 0).astype(np.int64),
    )


def build_swat_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[SMDMultiWindowDataset, SMDMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建 SWAT 的滑窗数据集。

    与旧版本的区别：
    ---------------
    - 去掉 compute_global_norm_stats / apply_norm_to_seqs；
    - 只使用 preprocess 阶段的 StandardScaler + 有限值清洗。
    """
    entity_ids = load_entity_ids(processed_root)

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for eid in entity_ids:
        train, test, labels = load_preprocessed_swat_entity(processed_root, eid)
        train_seqs.append(train.astype(np.float32))
        test_seqs.append(test.astype(np.float32))
        labels_list.append(labels.astype(np.int64))

    if not train_seqs:
        raise RuntimeError(f"在 {processed_root} 中未找到任何 SWAT 实体数据")

    input_dim = int(train_seqs[0].shape[1])

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
