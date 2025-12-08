# mambatsad/data/smd.py
# -*- coding: utf-8 -*-
"""
SMD 预处理数据集加载：

- 假设已经用 tools/preprocess_smd.py 预处理完毕

目录结构示例：
    dataset/SMD/
        machines.txt
        train/
            machine-1-1.npy
            ...
        test/
            machine-1-1.npy
            ...
        test_label/
            machine-1-1.npy
            ...

- 这里不在本文件中做标准化，只做滑动窗口切片 + 数值清洗，
  全局的 z-score 归一化会在 build_smd_multi_datasets 中统一完成。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

EPS = 1e-8


def fill_sequence_with_own_feature_mean(seq: np.ndarray) -> np.ndarray:
    """
    对单个序列 (T, D)：将非有限值替换成该序列每个特征维度的均值。

    - 首先忽略非有限值（nan/inf），按列计算均值；
    - 若某一列全为非有限，则该列均值设为 0.0；
    - 最终返回 float32 类型的新数组。
    """
    s = seq.astype(np.float32, copy=True)
    # 把非有限值视为 nan，便于使用 nanmean
    s_nan = np.where(np.isfinite(s), s, np.nan)
    feat_mean = np.nanmean(s_nan, axis=0)
    # 若某个特征维度的均值仍为 nan，则填 0
    feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0).astype(np.float32)
    # 用列均值替换所有非有限值
    s = np.where(np.isfinite(s), s, feat_mean)
    return s.astype(np.float32)


def fill_sequence_with_train_mean(train_seq: np.ndarray, seq_to_fill: np.ndarray) -> np.ndarray:
    """
    推荐做法：使用 train_seq 的特征均值去填充同一机器的其它序列（如 test）。
    """
    train = train_seq.astype(np.float32)
    train_nan = np.where(np.isfinite(train), train, np.nan)
    feat_mean = np.nanmean(train_nan, axis=0)
    feat_mean = np.where(np.isfinite(feat_mean), feat_mean, 0.0).astype(np.float32)

    s = seq_to_fill.astype(np.float32, copy=True)
    s = np.where(np.isfinite(s), s, feat_mean)
    return s.astype(np.float32)


def compute_global_norm_stats(train_seqs: List[np.ndarray], method: str = "zscore") -> Dict[str, np.ndarray]:
    """
    在所有训练序列上合并计算归一化统计量（推荐用于多机训练）。

    method:
        "zscore" 或 "minmax"。
    """
    all_train = np.vstack(
        [np.where(np.isfinite(s), s, np.nan) for s in train_seqs]
    ).astype(np.float32)

    if method == "zscore":
        mean = np.nanmean(all_train, axis=0).astype(np.float32)
        std = np.nanstd(all_train, axis=0).astype(np.float32)
        # 兜底处理：将 NaN 替为 0/1，并避免除零
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std), std, 1.0).astype(np.float32)
        std = np.where(std < EPS, 1.0, std).astype(np.float32)
        return {"mean": mean, "std": std}
    else:
        mn = np.nanmin(all_train, axis=0).astype(np.float32)
        mx = np.nanmax(all_train, axis=0).astype(np.float32)
        mn = np.where(np.isfinite(mn), mn, 0.0).astype(np.float32)
        mx = np.where(np.isfinite(mx), mx, 1.0).astype(np.float32)
        diff = (mx - mn).astype(np.float32)
        diff = np.where(~np.isfinite(diff) | (diff < EPS), 1.0, diff).astype(np.float32)
        return {"min": mn, "max": mx, "diff": diff}


def apply_norm_to_seqs(
    seqs: List[np.ndarray],
    stats: Dict[str, np.ndarray],
    method: str = "zscore",
) -> List[np.ndarray]:
    """将全局归一化统计量应用到每个序列上。"""
    out: List[np.ndarray] = []
    if method == "zscore":
        mean, std = stats["mean"], stats["std"]
        for s in seqs:
            s = s.astype(np.float32, copy=True)
            s = (s - mean) / std
            out.append(s.astype(np.float32))
    else:
        mn, diff = stats["min"], stats["diff"]
        for s in seqs:
            s = s.astype(np.float32, copy=True)
            s = (s - mn) / diff
            out.append(s.astype(np.float32))
    return out


def load_machine_ids(processed_root: str) -> List[str]:
    """从预处理目录中读取所有 machine id。"""
    machines_txt = os.path.join(processed_root, "machines.txt")
    if os.path.exists(machines_txt):
        with open(machines_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return sorted(ids)

    # 兜底：根据 train 目录推断
    train_dir = os.path.join(processed_root, "train")
    ids: List[str] = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_smd_machine(processed_root: str, machine_id: str):
    """
    加载某一台机器的预处理结果：
    - train:  [T_train, D]
    - test:   [T_test, D]
    - labels: [T_test]
    """
    train_path = os.path.join(processed_root, "train", f"{machine_id}.npy")
    test_path = os.path.join(processed_root, "test", f"{machine_id}.npy")
    label_path = os.path.join(processed_root, "test_label", f"{machine_id}.npy")

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(label_path)
    return train, test, labels


class SMDMultiWindowDataset(Dataset):
    """
    多机版本滑动窗口数据集。

    - 训练集：从所有机器的训练序列中滑窗采样；
    - 测试集：从所有机器的测试序列中滑窗采样，并记录来自哪一台机器、起始位置。
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels_list: Optional[List[np.ndarray]],
        win_size: int,
        stride: int,
        mode: str = "train",
    ) -> None:
        assert mode in ("train", "test")
        self.sequences = sequences
        self.labels_list = labels_list
        self.win_size = win_size
        self.stride = stride
        self.mode = mode

        # 预先构造所有窗口的索引：(seq_idx, start)
        self.indices: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(sequences):
            T = seq.shape[0]
            if T < win_size:
                continue
            for s in range(0, T - win_size + 1, stride):
                self.indices.append((seq_idx, s))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx, start = self.indices[idx]
        seq = self.sequences[seq_idx]
        win = seq[start : start + self.win_size].astype(np.float32)

        # 防守式数值清洗
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


def build_smd_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[SMDMultiWindowDataset, SMDMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建“多机合并”的 SMD 训练集和测试集。

    返回
    ----
    train_ds, test_ds:
        训练 / 测试集 Dataset。
    input_dim:
        特征维度 D。
    labels_list:
        每台机器的测试标签序列列表。
    machine_ids:
        机器 id 列表。
    """
    machine_ids = load_machine_ids(processed_root)
    use_train_mean_for_test = True

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for mid in machine_ids:
        train, test, labels = load_preprocessed_smd_machine(processed_root, mid)

        # 针对每个子序列单独均值填充 train
        train = fill_sequence_with_own_feature_mean(train)

        if use_train_mean_for_test:
            test = fill_sequence_with_train_mean(train, test)
        else:
            test = fill_sequence_with_own_feature_mean(test)

        # labels 理论上都是 0/1，如出现非有限值则用 0 填充
        labels = np.where(np.isfinite(labels), labels, 0).astype(np.int64)

        train_seqs.append(train.astype(np.float32))
        test_seqs.append(test.astype(np.float32))
        labels_list.append(labels.astype(np.int64))

    # 在所有训练序列上计算全局 z-score 统计量
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

    return train_ds, test_ds, input_dim, labels_list, machine_ids
