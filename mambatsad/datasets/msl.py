# -*- coding: utf-8 -*-
"""
MSL 数据集处理（基于 Anomaly-Transformer 格式）
目录结构示例：
dataset/
  MSL/
    MSL_train.npy        [T_train, 55]
    MSL_test.npy         [T_test, 55]
    MSL_test_label.npy   [T_test]
"""

import os
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .smd import SMDWindowDataset  # 直接复用窗口 Dataset 结构


def build_msl_datasets(
    data_dir: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
) -> Tuple[Dataset, Dataset, int]:
    train_path = os.path.join(data_dir, "MSL_train.npy")
    test_path = os.path.join(data_dir, "MSL_test.npy")
    label_path = os.path.join(data_dir, "MSL_test_label.npy")

    train_raw = np.load(train_path)  # [T_train, D]
    test_raw = np.load(test_path)    # [T_test, D]
    labels = np.load(label_path)     # [T_test]

    input_dim = train_raw.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)

    train_ds = SMDWindowDataset(
        data=train_norm,
        labels=None,
        win_size=win_size,
        stride=train_stride,
        mode="train",
    )
    test_ds = SMDWindowDataset(
        data=test_norm,
        labels=labels,
        win_size=win_size,
        stride=test_stride,
        mode="test",
    )

    return train_ds, test_ds, input_dim
