# mambatsad/data/smap.py
# -*- coding: utf-8 -*-
"""
SMAP 数据集构建工具。

假定你已通过 tools/preprocess_smap.py 将原始 NASA / telemanom 风格数据
转成以下结构::

    processed_root/
        train/      *.npy   # 每个通道一份 [T_train, D]
        test/       *.npy   # [T_test, D]
        test_label/ *.npy   # [T_test]
        channels.txt        # 通道 id 列表

本文件在原有滑窗基础上，增加了**全局 z-score 归一化**，
以避免各通道尺度差异导致训练困难。
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset

EPS = 1e-8


def compute_global_norm_stats(train_seqs: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    在所有训练序列上计算统一的均值 / 方差（z-score）。

    - train_seqs: 列表，每个元素形状 [T_train_i, D]
    """
    all_list = []
    for s in train_seqs:
        s = s.astype(np.float32)
        s = np.where(np.isfinite(s), s, np.nan)
        all_list.append(s)

    all_train = np.vstack(all_list).astype(np.float32)

    mean = np.nanmean(all_train, axis=0).astype(np.float32)
    std = np.nanstd(all_train, axis=0).astype(np.float32)

    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
    std = np.where(std < EPS, 1.0, std).astype(np.float32)

    return {"mean": mean, "std": std}


def apply_norm_to_seqs(
    seqs: List[np.ndarray],
    stats: Dict[str, np.ndarray],
) -> List[np.ndarray]:
    """对每个序列应用统一的 z-score 归一化。"""
    mean, std = stats["mean"], stats["std"]
    out: List[np.ndarray] = []
    for s in seqs:
        s = s.astype(np.float32, copy=True)
        s = (s - mean) / std
        out.append(s.astype(np.float32))
    return out


class SMAPMultiWindowDataset(Dataset):
    """多通道 SMAP 滑窗数据集，返回 dict 结构与 Trainer 对齐。"""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels_list: List[np.ndarray] | None,
        win_size: int,
        stride: int,
        mode: str = "train",
    ) -> None:
        assert mode in ("train", "test"), "mode 仅支持 train/test"

        self.sequences = sequences
        self.labels_list = labels_list
        self.win_size = int(win_size)
        self.stride = int(stride)
        self.mode = mode

        self.indices: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(sequences):
            T = seq.shape[0]
            if T < self.win_size:
                continue
            for start in range(0, T - self.win_size + 1, self.stride):
                self.indices.append((seq_idx, start))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.indices)

    def __getitem__(self, idx: int):  # type: ignore[override]
        seq_idx, start = self.indices[idx]
        seq = self.sequences[seq_idx]

        win = seq[start : start + self.win_size].astype(np.float32)
        if not np.isfinite(win).all():
            win = np.nan_to_num(win, nan=0.0, posinf=1e6, neginf=-1e6)

        item = {
            "window": win,
            "seq_idx": np.int64(seq_idx),
            "start": np.int64(start),
        }

        if self.mode == "test" and self.labels_list is not None:
            labels = self.labels_list[seq_idx]
            lab_win = labels[start : start + self.win_size].astype(np.int64)
            item["label"] = lab_win

        return item


def _load_smap_entities(
    processed_root: str,
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """读取 SMAP 预处理目录中的 train/test/label。"""
    train_root = os.path.join(processed_root, "train")
    test_root = os.path.join(processed_root, "test")
    label_root = os.path.join(processed_root, "test_label")
    channels_txt = os.path.join(processed_root, "channels.txt")

    if not os.path.exists(channels_txt):
        raise FileNotFoundError(f"找不到 SMAP 通道列表文件: {channels_txt}")

    with open(channels_txt, "r", encoding="utf-8") as f:
        entity_ids = [line.strip() for line in f.readlines() if line.strip()]

    if not entity_ids:
        raise ValueError(f"{channels_txt} 中未找到任何通道 id")

    train_series_list: List[np.ndarray] = []
    test_series_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for cid in entity_ids:
        train_path = os.path.join(train_root, f"{cid}.npy")
        test_path = os.path.join(test_root, f"{cid}.npy")
        label_path = os.path.join(label_root, f"{cid}.npy")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试文件不存在: {test_path}")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"标签文件不存在: {label_path}")

        train = np.load(train_path)
        test = np.load(test_path)
        labels = np.load(label_path)

        if train.ndim != 2 or test.ndim != 2:
            raise ValueError(
                f"期望 train/test 为二维数组 [T, D]，但 {cid} 得到 "
                f"train={train.shape}, test={test.shape}"
            )
        if labels.ndim != 1 or labels.shape[0] != test.shape[0]:
            raise ValueError(
                f"标签维度不匹配: {cid} labels.shape={labels.shape}, "
                f"test.shape[0]={test.shape[0]}"
            )

        train_series_list.append(train.astype(np.float32))
        test_series_list.append(test.astype(np.float32))
        labels_list.append(labels.astype(np.int64))

    return entity_ids, train_series_list, test_series_list, labels_list


def build_smap_dataset(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    """构建 SMAP 的训练 / 测试数据集，接口与其他数据集保持一致。"""
    entity_ids, train_list, test_list, labels_list = _load_smap_entities(processed_root)

    dims = {arr.shape[1] for arr in train_list}
    if len(dims) != 1:
        raise ValueError(f"SMAP 各通道特征维度不一致: {dims}")
    input_dim = dims.pop()

    # 先做有限值清洗
    def _ensure_finite(arr: np.ndarray) -> np.ndarray:
        if np.isfinite(arr).all():
            return arr.astype(np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

    train_seqs_raw = [_ensure_finite(a) for a in train_list]
    test_seqs_raw = [_ensure_finite(a) for a in test_list]

    # 在所有训练序列上计算全局 z-score，然后同时应用于 train/test
    norm_stats = compute_global_norm_stats(train_seqs_raw)
    train_seqs = apply_norm_to_seqs(train_seqs_raw, norm_stats)
    test_seqs = apply_norm_to_seqs(test_seqs_raw, norm_stats)

    labels_clean = [
        np.where(np.isfinite(lab), lab, 0).astype(np.int64) for lab in labels_list
    ]

    train_ds = SMAPMultiWindowDataset(
        sequences=train_seqs,
        labels_list=None,
        win_size=win_size,
        stride=train_stride,
        mode="train",
    )

    test_ds = SMAPMultiWindowDataset(
        sequences=test_seqs,
        labels_list=labels_clean,
        win_size=win_size,
        stride=test_stride,
        mode="test",
    )

    return train_ds, test_ds, input_dim, labels_clean, entity_ids
