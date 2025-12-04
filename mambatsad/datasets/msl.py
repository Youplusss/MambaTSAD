# mambatsad/datasets/msl.py
# -*- coding: utf-8 -*-
"""
MSL 预处理数据集加载：
- 假设已经用 tools/preprocess_msl.py 预处理完毕
  目录结构示例：
    dataset/MSL/
      channels.txt
      train/
        M-1.npy
        M-2.npy
        ...
      test/
        M-1.npy
        M-2.npy
        ...
      test_label/
        M-1.npy
        M-2.npy
        ...

- 这里与 SMD 保持风格一致：
  * 不在此处做 StandardScaler，而是在所有训练序列上再算一遍全局 z-score；
  * 只做滑动窗口切片 + 防守式数值清洗。
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
from torch.utils.data import Dataset

EPS = 1e-8


def compute_global_norm_stats(
    train_seqs: List[np.ndarray], method: str = "zscore"
) -> Dict[str, np.ndarray]:
    """
    在所有训练序列上合并计算归一化统计量（推荐用于多通道训练）。
    method: "zscore" 或 "minmax"
    """
    all_list = []
    for s in train_seqs:
        s = s.astype(np.float32)
        s = np.where(np.isfinite(s), s, np.nan)
        all_list.append(s)
    all_train = np.vstack(all_list).astype(np.float32)

    if method == "zscore":
        mean = np.nanmean(all_train, axis=0).astype(np.float32)
        std = np.nanstd(all_train, axis=0).astype(np.float32)
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
    seqs: List[np.ndarray], stats: Dict[str, np.ndarray], method: str = "zscore"
) -> List[np.ndarray]:
    out = []
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


def load_channel_ids(processed_root: str) -> List[str]:
    """从预处理目录中读取所有 MSL 通道 id"""
    channels_txt = os.path.join(processed_root, "channels.txt")
    if os.path.exists(channels_txt):
        with open(channels_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return sorted(ids)

    # 兜底：根据 train 目录推断
    train_dir = os.path.join(processed_root, "train")
    ids = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    return sorted(ids)


def load_preprocessed_msl_channel(
    processed_root: str,
    chan_id: str,
):
    """
    加载某一通道的预处理结果：
    - train: [T_train, D]
    - test:  [T_test, D]
    - labels:[T_test]
    """
    train_path = os.path.join(processed_root, "train", f"{chan_id}.npy")
    test_path = os.path.join(processed_root, "test", f"{chan_id}.npy")
    label_path = os.path.join(processed_root, "test_label", f"{chan_id}.npy")

    train = np.load(train_path)
    test = np.load(test_path)
    labels = np.load(label_path)
    return train, test, labels


class MSLMultiWindowDataset(Dataset):
    """
    多通道版本滑动窗口数据集：
    - 训练集：从所有通道的训练序列中滑窗采样
    - 测试集：从所有通道的测试序列中滑窗采样，并记录来自哪一通道、起始位置
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels_list: Optional[List[np.ndarray]],
        win_size: int,
        stride: int,
        mode: str = "train",
    ):
        """
        参数：
        - sequences: N_seq 个 [T_i, D] 的数组
        - labels_list: N_seq 个 [T_i] 的 0/1 标签（训练集可为 None）
        - win_size: 窗口长度
        - stride: 滑动步长
        - mode: "train" 或 "test"
        """
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

    def __len__(self):
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


def build_msl_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[MSLMultiWindowDataset, MSLMultiWindowDataset, int, List[np.ndarray], List[str]]:
    """
    构建“多通道合并”的 MSL 训练集和测试集
    返回：
    - train_ds: 训练集 Dataset
    - test_ds:  测试集 Dataset
    - input_dim: 特征维度 D
    - labels_list: 每个通道的测试标签序列列表
    - channel_ids: 通道 id 列表
    """
    channel_ids = load_channel_ids(processed_root)

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for cid in channel_ids:
        train, test, labels = load_preprocessed_msl_channel(processed_root, cid)

        # 基础数值清洗
        train = np.nan_to_num(train, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        test = np.nan_to_num(test, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        labels = np.where(np.isfinite(labels), labels, 0).astype(np.int64)

        train_seqs.append(train)
        test_seqs.append(test)
        labels_list.append(labels)

    # 归一化：在所有训练序列上计算全局统计量，然后应用到 train/test
    norm_stats = compute_global_norm_stats(train_seqs, method="zscore")
    train_seqs = apply_norm_to_seqs(train_seqs, norm_stats, method="zscore")
    test_seqs = apply_norm_to_seqs(test_seqs, norm_stats, method="zscore")

    input_dim = train_seqs[0].shape[1]

    train_ds = MSLMultiWindowDataset(
        sequences=train_seqs,
        labels_list=None,
        win_size=win_size,
        stride=train_stride,
        mode="train",
    )
    test_ds = MSLMultiWindowDataset(
        sequences=test_seqs,
        labels_list=labels_list,
        win_size=win_size,
        stride=test_stride,
        mode="test",
    )
    return train_ds, test_ds, input_dim, labels_list, channel_ids
