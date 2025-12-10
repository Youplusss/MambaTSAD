# mambatsad/data/wadi.py
# -*- coding: utf-8 -*-
"""
WADI 预处理数据集加载。

目录结构（由 tools/preprocess_wadi.py 生成）::
    dataset/WADI/
        entities.txt       # 通常只有一行 'wadi'
        train/ wadi.npy    # [T_train, D]，已在 preprocess 阶段做过 StandardScaler
        test/  wadi.npy    # [T_test, D]
        test_label/ wadi.npy  # [T_test]

设计要点
--------
1. **不再进行第二次 z-score 归一化**：
   - 在 preprocess_wadi.py 中已经使用 StandardScaler 基于训练集做过一次标准化；
   - 再做一次全局 z-score 容易把特征尺度压得过于均匀，削弱异常信号，导致分数曲线过于平滑。
2. 仅做：
   - 非有限值清洗（nan/inf -> 合理数值）；
   - 滑动窗口切片。
3. 复用 SMDMultiWindowDataset，实现与 SMD / SWaT 相同的滑窗接口。
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
from torch.utils.data import Dataset

from .smd import SMDMultiWindowDataset


def _ensure_finite(arr: np.ndarray) -> np.ndarray:
    """
    将数组中的 NaN / Inf 替换为有限值，避免后续计算异常。

    - NaN -> 0.0
    - +Inf -> 1e6
    - -Inf -> -1e6
    """
    if np.isfinite(arr).all():
        return arr.astype(np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def load_entity_ids(processed_root: str) -> List[str]:
    """读取 WADI 实体 id 列表，通常只有 'wadi'。"""
    ent_txt = os.path.join(processed_root, "entities.txt")
    if os.path.exists(ent_txt):
        with open(ent_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        if ids:
            return sorted(ids)

    # 兜底：从 train/ 目录中解析 .npy 名称
    train_dir = os.path.join(processed_root, "train")
    ids: List[str] = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_wadi_entity(processed_root: str, ent_id: str):
    """
    加载预处理后的 WADI 某实体数据。

    返回
    ----
    train : [T_train, D]，float32
    test  : [T_test, D]，float32
    labels: [T_test]，int64
    """
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


def build_wadi_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[SMDMultiWindowDataset, SMDMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建 WADI 的多实体滑窗数据集（一般只有一个实体 'wadi'）。

    与之前版本相比的**关键变化**：
    --------------------------------
    - 不再调用 compute_global_norm_stats / apply_norm_to_seqs 做二次 z-score；
    - 仅在 preprocess 中做一次 StandardScaler + 这里的有限值清洗，从而保留更丰富的异常幅度信息。

    返回
    ----
    train_ds, test_ds : 训练 / 测试 Dataset（SMDMultiWindowDataset）
    input_dim         : 特征维度 D
    labels_list       : 每个实体对应的完整测试标签序列
    entity_ids        : 实体 id 列表
    """
    entity_ids = load_entity_ids(processed_root)

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for eid in entity_ids:
        train, test, labels = load_preprocessed_wadi_entity(processed_root, eid)
        train_seqs.append(train.astype(np.float32))
        test_seqs.append(test.astype(np.float32))
        labels_list.append(labels.astype(np.int64))

    if not train_seqs:
        raise RuntimeError(f"在 {processed_root} 中未找到任何 WADI 实体数据")

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
