# tools/preprocess_wadi.py
# -*- coding: utf-8 -*-
"""
WADI 原始数据预处理脚本。

假定原始目录结构类似（你可根据实际情况调整参数）：
raw_data/WADI/
    WADI_14days.csv           # 训练（大部分为正常）
    WADI_attackdata.csv       # 测试（包含攻击）
    WADI_attacktimes.csv      # 攻击时间段列表（Start/End 日期时间）

本脚本将：
- 按 Date + Time 对齐 attacktimes 中的攻击区间，生成逐时刻的 0/1 标签；
- 丢弃非数值列（Date、Time 等）；
- 用训练集拟合 StandardScaler，并应用到 train/test；
- 按如下结构保存：

  dataset/WADI/
      entities.txt       # 仅一行：wadi
      train/
          wadi.npy
      test/
          wadi.npy
      test_label/
          wadi.npy
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _read_wadi_csv(path: str) -> pd.DataFrame:
    """读取 WADI 预处理后的 CSV，并做一些通用清洗（去列名两边空格）。"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_wadi(
    raw_root: str,
    out_root: str,
    train_csv: str = "WADI_train.csv",
    test_csv: str = "WADI_test.csv",
    label_col: str | None = None,
) -> None:
    """将已经初步处理好的 WADI CSV 转成与 SMD/MSL/SWaT 一致的 npy 结构。

    假设：
    - ``raw_root`` 目录下存在 ``train_csv`` 和 ``test_csv``，一般为
      ``../data/WADI/processing/WADI_train.csv`` 和 ``WADI_test.csv``；
    - 测试集 CSV 中包含标签列：
        * 若 ``label_col`` 为 None，则默认使用 **最后一列** 作为标签；
        * 标签值可以是 0/1 或 Normal/Attack（忽略大小写的字符串），会统一映射到 0/1；
    - 训练集不需要标签（即使有同名列也会丢弃）。

    输出目录结构：

    .. code-block:: text

        dataset/WADI/
            entities.txt       # 仅一行：wadi
            train/
                wadi.npy       # [T_train, D]
            test/
                wadi.npy       # [T_test, D]
            test_label/
                wadi.npy       # [T_test]
    """

    os.makedirs(out_root, exist_ok=True)
    train_path = os.path.join(raw_root, train_csv)
    test_path = os.path.join(raw_root, test_csv)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练文件：{train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试文件：{test_path}")

    df_train = _read_wadi_csv(train_path)
    df_test = _read_wadi_csv(test_path)

    # ---------- 处理标签列 ----------
    if label_col is None:
        label_col = df_test.columns[-1]

    if label_col not in df_test.columns:
        raise ValueError(f"测试文件中找不到标签列 '{label_col}'，请检查 CSV 列名。")

    # 将标签统一成 0/1：0 表示正常，1 表示攻击
    label_raw = df_test[label_col].astype(str).str.strip().str.lower()
    # 支持几种常见写法："0"/"1"、"normal"/"attack"，其余一律按攻击处理
    labels = np.where(
        (label_raw == "normal") | (label_raw == 0) | (label_raw == "0"),
        0,
        1,
    ).astype(np.int64)

    # 如果训练集中存在同名标签列，将其丢弃
    if label_col in df_train.columns:
        df_train = df_train.drop(columns=[label_col])

    # ---------- 特征列选择 ----------
    # 以测试集的列为基准：去掉标签列，再额外尝试去掉明显的时间列（含 time/date 的列名）
    feature_cols = df_test.columns.tolist()
    feature_cols.remove(label_col)

    # 去掉包含 time/date 字样的时间相关列
    feature_cols = [
        c
        for c in feature_cols
        if ("time" not in c.lower()) and ("date" not in c.lower())
    ]

    # 训练集只保留与 feature_cols 交集的列，避免多余列/缺失列导致对齐问题
    cols_train = [c for c in df_train.columns if c in feature_cols]

    # 将非数值列自动过滤掉（尝试 astype(float) 前先用 select_dtypes）
    df_train_feat = df_train[cols_train]
    df_test_feat = df_test[feature_cols]

    # 只保留数值列
    num_cols = df_train_feat.select_dtypes(include=["number"]).columns.tolist()
    df_train_feat = df_train_feat[num_cols]
    # 测试集按同样列顺序选择
    df_test_feat = df_test_feat[num_cols]

    x_train = df_train_feat.astype(float).values
    x_test = df_test_feat.astype(float).values

    # ---------- 标准化（只用训练集拟合） ----------
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train).astype(np.float32)
    x_test_norm = scaler.transform(x_test).astype(np.float32)

    # ---------- 保存到指定目录结构 ----------
    train_dir = os.path.join(out_root, "train")
    test_dir = os.path.join(out_root, "test")
    label_dir = os.path.join(out_root, "test_label")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    np.save(os.path.join(train_dir, "wadi.npy"), x_train_norm)
    np.save(os.path.join(test_dir, "wadi.npy"), x_test_norm)
    np.save(os.path.join(label_dir, "wadi.npy"), labels.astype(np.int64))

    # entities 列表（类似 SMD 的 machines.txt）
    with open(os.path.join(out_root, "entities.txt"), "w", encoding="utf-8") as f:
        f.write("wadi\n")

    print(f"[OK] WADI 预处理完成，保存至 {out_root}")
    print(
        f"  train shape = {x_train_norm.shape}, test shape = {x_test_norm.shape}, labels shape = {labels.shape}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WADI 预处理后的 CSV 转成 MambaTSAD 所需 npy 格式"
    )
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./data/WADI/processing",
        help="WADI 预处理 CSV 所在目录 (包含 WADI_train.csv / WADI_test.csv)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./dataset/WADI",
        help="预处理输出目录",
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="WADI_train.csv",
        help="训练 CSV 文件名 (默认: WADI_train.csv)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="WADI_test.csv",
        help="测试 CSV 文件名 (默认: WADI_test.csv)",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="attack",
        help="测试集标签列名，默认使用测试数据的最后一列",
    )
    args = parser.parse_args()

    preprocess_wadi(
        raw_root=args.raw_root,
        out_root=args.out_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
