# tools/preprocess_swat.py
# -*- coding: utf-8 -*-
"""
SWaT 原始数据预处理脚本。

目标：
- 读取官方发布的 SWaT 正常 / 攻击 CSV；
- 丢弃时间戳字段，只保留传感器 / 执行器数值列；
- 将 Attack/Normal 标签转成 0/1；
- 使用训练集做 StandardScaler 归一化；
- 存成与 SMD 类似的结构，便于复用现有 DataLoader：

  dataset/SWAT/
      entities.txt        # 仅一行：swat
      train/
          swat.npy        # [T_train, D]
      test/
          swat.npy        # [T_test, D]
      test_label/
          swat.npy        # [T_test]
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _read_swat_csv(path: str) -> pd.DataFrame:
    """
    读取 SWaT CSV，并做一些通用清洗（去两边空格等）。
    """
    df = pd.read_csv(path)
    # 去掉列名两端的空白，避免 " Normal/Attack " 这种问题
    df.columns = [c.strip() for c in df.columns]
    return df


def preprocess_swat(
    raw_root: str,
    out_root: str,
    train_csv: str = "SWaT_Dataset_Normal_v0.csv",
    test_csv: str = "SWaT_Dataset_Attack_v0.csv",
    label_col: str | None = None,
) -> None:
    """
    主入口：从原始 CSV 生成预处理后的 npy 文件。

    参数
    ----
    raw_root : 原始 CSV 所在目录。
    out_root : 预处理输出目录（如 ./dataset/SWAT）。
    train_csv : 训练（正常）文件名。
    test_csv : 测试（攻击）文件名。
    label_col : 标签列名。若为 None，则默认使用「最后一列」。
    """
    os.makedirs(out_root, exist_ok=True)
    train_path = os.path.join(raw_root, train_csv)
    test_path = os.path.join(raw_root, test_csv)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"找不到训练文件：{train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"找不到测试文件：{test_path}")

    df_train = _read_swat_csv(train_path)
    df_test = _read_swat_csv(test_path)

    # ---------- 处理标签列 ----------
    if label_col is None:
        label_col = df_test.columns[-1]

    if label_col not in df_test.columns:
        raise ValueError(f"测试文件中找不到标签列 '{label_col}'，请检查 CSV 列名。")

    # 有些版本的 Normal/Attack 列值形如 'Normal ', 'Attack', 'A ttack'
    label_raw = df_test[label_col].astype(str).str.strip().str.lower()
    labels = np.where((label_raw == "normal") | (label_raw == 0) | (label_raw == "0"), 0, 1).astype(np.int64)

    # ---------- 特征列选择 ----------
    # 一般 SWaT 第一列是时间戳，最后一列为标签；中间为传感器 / 执行器
    # 为稳妥起见：默认「去掉第一列 + 标签列」，其余全部视作数值特征。
    feature_cols = df_test.columns.tolist()

    # 去掉标签列
    feature_cols.remove(label_col)
    # 若第一列看起来像时间戳，则去掉
    first_col = feature_cols[0]
    if "time" in first_col.lower() or "date" in first_col.lower():
        feature_cols = feature_cols[1:]

    # 训练集可能包含同名的标签列（全 Normal），一并去掉
    cols_train = [c for c in df_train.columns if c in feature_cols]

    x_train = df_train[cols_train].astype(float).values
    x_test = df_test[feature_cols].astype(float).values

    # ---------- 标准化 ----------
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

    np.save(os.path.join(train_dir, "swat.npy"), x_train_norm)
    np.save(os.path.join(test_dir, "swat.npy"), x_test_norm)
    np.save(os.path.join(label_dir, "swat.npy"), labels.astype(np.int64))

    # entities 列表（类似 SMD 的 machines.txt）
    with open(os.path.join(out_root, "entities.txt"), "w", encoding="utf-8") as f:
        f.write("swat\n")

    print(f"[OK] 预处理完成，保存至 {out_root}")
    print(f"  train shape = {x_train_norm.shape}, test shape = {x_test_norm.shape}, labels shape = {labels.shape}")


def main():
    parser = argparse.ArgumentParser(description="SWaT 原始数据预处理为 MambaTSAD 所需格式")
    parser.add_argument("--raw_root", type=str, default="./data/SWaT",
                        help="SWaT 原始 CSV 所在目录")
    parser.add_argument("--out_root", type=str, default="./dataset/SWaT",
                        help="预处理输出目录")
    parser.add_argument("--train_csv", type=str, default="swat_train2.csv",
                        help="训练（正常）CSV 文件名")
    parser.add_argument("--test_csv", type=str, default="swat2.csv",
                        help="测试（攻击）CSV 文件名")
    parser.add_argument("--label_col", type=str, default=None,
                        help="标签列名，默认使用测试数据的最后一列")
    args = parser.parse_args()

    preprocess_swat(
        raw_root=args.raw_root,
        out_root=args.out_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        label_col=args.label_col,
    )


if __name__ == "__main__":
    main()
