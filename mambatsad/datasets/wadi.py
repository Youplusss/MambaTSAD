# -*- coding: utf-8 -*-
"""
WADI 数据集处理：
结构与 SWaT 类似，但更长且缺失值较多，
通常需要：
- 删除全 NaN / 大部分 NaN 的列
- 其余 NaN 用 0 或前向填充
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .smd import SMDWindowDataset


def load_wadi(data_dir: str):
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    # 假设有 label 列，1 表示攻击
    label_col_candidates = ["attack", "label", "Attack"]
    label_col = None
    for c in label_col_candidates:
        if c in df_test.columns:
            label_col = c
            break
    assert label_col is not None, "WADI 测试集未找到标签列，请检查列名"

    labels = (df_test[label_col] != 0).astype(int).values
    df_test_feat = df_test.drop(columns=[label_col])

    # 删除全 NaN 列
    df_train_feat = df_train.select_dtypes(include=["float64", "int64"]).copy()
    df_test_feat = df_test_feat.select_dtypes(include=["float64", "int64"]).copy()

    valid_cols = df_train_feat.columns[~df_train_feat.isna().all()]
    df_train_feat = df_train_feat[valid_cols]
    df_test_feat = df_test_feat[valid_cols]

    # 填充 NaN
    df_train_feat = df_train_feat.fillna(0.0)
    df_test_feat = df_test_feat.fillna(0.0)

    train_raw = df_train_feat.values
    test_raw = df_test_feat.values

    return train_raw, test_raw, labels


def build_wadi_datasets(
    data_dir: str,
    win_size: int,
    train_stride: int = 1,
    test_stride: int = 1,
):
    train_raw, test_raw, labels = load_wadi(data_dir)
    input_dim = train_raw.shape[1]

    scaler = StandardScaler()
    scaler.fit(train_raw)

    train_norm = scaler.transform(train_raw)
    test_norm = scaler.transform(test_raw)

    train_ds = SMDWindowDataset(train_norm, None, win_size, train_stride, mode="train")
    test_ds = SMDWindowDataset(test_norm, labels, win_size, test_stride, mode="test")

    return train_ds, test_ds, input_dim
