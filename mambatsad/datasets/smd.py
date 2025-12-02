# mambatsad/datasets/smd.py
# -*- coding: utf-8 -*-
"""
SMD 预处理数据集加载：
- 假设已经用 tools/preprocess_smd.py 预处理完毕
  目录结构示例：
    data_processed/SMD/
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
- 这里不再做标准化，只做滑动窗口切片
"""

import os
from typing import List, Tuple, Optional, Dict

import numpy as np
from torch.utils.data import Dataset


def load_machine_ids(processed_root: str) -> List[str]:
    """
    从预处理目录中读取所有 machine id
    优先使用 machines.txt，否则根据 train/*.npy 猜测
    """
    machines_txt = os.path.join(processed_root, "machines.txt")
    if os.path.exists(machines_txt):
        with open(machines_txt, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        return sorted(ids)

    # 兜底：根据 train 目录推断
    train_dir = os.path.join(processed_root, "train")
    ids = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".npy"):
            ids.append(os.path.splitext(fn)[0])
    ids = sorted(ids)
    return ids


def load_preprocessed_smd_machine(
    processed_root: str,
    machine_id: str,
):
    """
    加载某一台机器的预处理结果：
    - train: [T_train, D]
    - test:  [T_test, D]
    - labels:[T_test]
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
    多机版本滑动窗口数据集：
    - 训练集：从所有机器的训练序列中滑窗采样
    - 测试集：从所有机器的测试序列中滑窗采样，并记录来自哪一台机器、起始位置
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
        - sequences: 长度为 N_seq 的列表，每个元素形状 [T_i, D]
        - labels_list: 测试模式下，对应的标签序列列表 [T_i]；训练模式下为 None
        - win_size: 时间窗口长度
        - stride: 滑动步长
        - mode: "train" 或 "test"
        """
        assert mode in ["train", "test"]
        self.sequences = sequences
        self.labels_list = labels_list
        self.win_size = win_size
        self.stride = stride
        self.mode = mode

        # 记录所有 (seq_idx, start) 组合，避免跨序列滑窗
        self.indices: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            T = seq.shape[0]
            for start in range(0, T - win_size + 1, stride):
                self.indices.append((seq_idx, start))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        seq_idx, start = self.indices[idx]
        seq = self.sequences[seq_idx]
        end = start + self.win_size
        window = seq[start:end]  # [L, D]

        if self.mode == "train":
            return {
                "window": window.astype(np.float32),
            }
        else:
            labels = self.labels_list[seq_idx]
            label_win = labels[start:end]
            return {
                "window": window.astype(np.float32),
                "label": label_win.astype(np.float32),
                "seq_idx": np.int64(seq_idx),
                "start": np.int64(start),
            }


def build_smd_multi_datasets(
    processed_root: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    """
    构建 SMD 多机训练 / 测试数据集：
    - 从 processed_root 下读取所有机器的 train/test/test_label
    - 训练集包含所有机器的窗口
    - 测试集也包含所有机器的窗口
    返回：
    - train_ds: SMDMultiWindowDataset（train 模式）
    - test_ds:  SMDMultiWindowDataset（test 模式）
    - input_dim: 特征维度 D
    - labels_list: 所有机器的完整标签序列列表（后续评估用）
    - machine_ids: 机器 ID 列表（与 labels_list 顺序一致）
    """
    machine_ids = load_machine_ids(processed_root)
    print(f"[SMD] 使用 {len(machine_ids)} 台机器：{machine_ids}")

    train_seqs: List[np.ndarray] = []
    test_seqs: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []

    for mid in machine_ids:
        train, test, labels = load_preprocessed_smd_machine(processed_root, mid)
        train_seqs.append(train)
        test_seqs.append(test)
        labels_list.append(labels.astype(int))

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
