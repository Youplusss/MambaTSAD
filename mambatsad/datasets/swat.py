# -*- coding: utf-8 -*-
"""
SWaT 数据集处理（典型 ICS 数据）
假设格式：
data/
  SWaT/
    train.csv     # 只包含正常数据
    test.csv      # 混合正常+攻击
    test_label.csv 或 测试文件中自带 "Normal/Attack" 列
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .smd import SMDWindowDataset


def load_swat(data_dir: str):
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # 假设测试集中有一列叫 "Normal/Attack"，Normal=0, Attack=1
    # 你也可以根据实际列名修改
    label_col_candidates = ["Normal/Attack", "label", "attack"]
    label_col = None
    for c in label_col_candidates:
        if c in df_test.columns:
            label_col = c
            break
    assert label_col is not None, "SWaT 测试集未找到标签列，请检查列名"

    labels = (df_test[label_col] != 0).astype(int).values  # [T_test]

    # 去掉 label 列，剩下全是特征
    df_test_feat = df_test.drop(columns=[label_col])

    # 一般还需要：
    # - 去掉时间戳列
    # - 处理 NaN（填充 / 删除）
    # 这里简单示例：删除非数值列，填充 NaN 为前向填充
    df_train_feat = df_train.select_dtypes(include=["float64", "int64"]).copy()
    df_test_feat = df_test_feat.select_dtypes(include=["float64", "int64"]).copy()
    df_train_feat = df_train_feat.fillna(method="ffill").fillna(method="bfill")
    df_test_feat = df_test_feat.fillna(method="ffill").fillna(method="bfill")

    train_raw = df_train_feat.values
    test_raw = df_test_feat.values

    return train_raw, test_raw, labels


def build_swat_datasets(
    data_dir: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    train_raw, test_raw, labels = load_swat(data_dir)
    input_dim = train_raw.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)

    train_ds = SMDWindowDataset(train_norm, None, win_size, train_stride, mode="train")
    test_ds = SMDWindowDataset(test_norm, labels, win_size, test_stride, mode="test")

    return train_ds, test_ds, input_dim
