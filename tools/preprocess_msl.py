# tools/preprocess_msl.py
# -*- coding: utf-8 -*-
"""
MSL 原始数据预处理脚本（NASA / telemanom 风格数据）
--------------------------------------------------
假设原始目录结构如下（telemanom 的 data 目录）：

raw_root/
  train/
    M-1.npy
    M-2.npy
    ...
  test/
    M-1.npy
    M-2.npy
    ...
  labeled_anomalies.csv

labeled_anomalies.csv 中包含字段：
  chan_id, spacecraft, anomaly_sequences, class, num_values

其中 spacecraft == 'MSL' 的行表示 MSL 的各个通道。
anomaly_sequences 为形如 [[start, end], [start, end], ...] 的字符串，
下标在 test 序列上，通常为闭区间 [start, end]。

本脚本将数据转为与 SMD 相同的预处理格式：

out_root/
  train/
    <chan_id>.npy       # 训练序列 (T_train, D)
  test/
    <chan_id>.npy       # 测试序列 (T_test, D)
  test_label/
    <chan_id>.npy       # 测试标注 (T_test,)
  channels.txt          # 所有 chan_id 列表

使用示例：
python tools/preprocess_msl.py \
    --raw_root ./data/MSL \
    --out_root ./dataset/MSL \
    --use_global_scaler
"""

import os
import ast
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_msl_meta(raw_root: str) -> pd.DataFrame:
    """读取 labeled_anomalies.csv 并只保留 spacecraft == 'MSL' 的通道。"""
    csv_path = os.path.join(raw_root, "labeled_anomalies.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到标注文件: {csv_path}")

    df = pd.read_csv(csv_path)
    # 统一成小写列名，兼容性更好
    df.columns = [c.strip().lower() for c in df.columns]

    required = {"chan_id", "spacecraft", "anomaly_sequences", "num_values"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"labeled_anomalies.csv 缺少必要列，期望包含: {required}，实际为: {df.columns}"
        )

    df_msl = df[df["spacecraft"].astype(str).str.upper() == "MSL"].copy()
    if df_msl.empty:
        raise ValueError("在 labeled_anomalies.csv 中未找到 spacecraft == 'MSL' 的行")

    return df_msl


def parse_anomaly_sequences(seq_str: str) -> List[Tuple[int, int]]:
    """
    解析 anomaly_sequences 字段：
    例如 "[[1850, 2030], [2670, 2790]]" -> [(1850, 2030), (2670, 2790)]
    """
    if isinstance(seq_str, float) and np.isnan(seq_str):
        return []
    seqs = ast.literal_eval(seq_str)
    out = []
    for seg in seqs:
        if len(seg) != 2:
            continue
        s, e = int(seg[0]), int(seg[1])
        out.append((s, e))
    return out


def preprocess_msl(
    raw_root: str,
    out_root: str,
    use_global_scaler: bool = False,
):
    os.makedirs(out_root, exist_ok=True)
    out_train = os.path.join(out_root, "train")
    out_test = os.path.join(out_root, "test")
    out_label = os.path.join(out_root, "test_label")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)

    train_dir = os.path.join(raw_root, "train")
    test_dir = os.path.join(raw_root, "test")

    meta_df = load_msl_meta(raw_root)
    chan_ids: List[str] = sorted(meta_df["chan_id"].astype(str).tolist())
    print(f"发现 {len(chan_ids)} 个 MSL 通道：{chan_ids}")

    # ------------------（可选）全局 StandardScaler ------------------
    global_scaler = None
    if use_global_scaler:
        print("使用【全局 StandardScaler】拟合所有通道的训练数据 ...")
        all_train = []
        for cid in chan_ids:
            train_path = os.path.join(train_dir, f"{cid}.npy")
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"训练文件不存在: {train_path}")
            train_raw = np.load(train_path)
            all_train.append(train_raw)
        all_train = np.concatenate(all_train, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(all_train)
        print("全局 scaler 拟合完成。")

    # 方便通过 chan_id 索引 meta 信息
    meta_df = meta_df.set_index("chan_id")

    # ------------------ 逐通道预处理并保存 ------------------
    for cid in chan_ids:
        print(f"[{cid}] 加载原始数据 ...")
        train_path = os.path.join(train_dir, f"{cid}.npy")
        test_path = os.path.join(test_dir, f"{cid}.npy")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"训练文件不存在: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"测试文件不存在: {test_path}")

        train_raw = np.load(train_path)
        test_raw = np.load(test_path)

        # 选择 scaler
        if use_global_scaler:
            scaler = global_scaler
        else:
            scaler = StandardScaler()
            scaler.fit(train_raw)

        train_norm = scaler.transform(train_raw).astype(np.float32)
        test_norm = scaler.transform(test_raw).astype(np.float32)

        row = meta_df.loc[cid]
        num_values = int(row["num_values"])
        if test_norm.shape[0] != num_values:
            print(
                f"[WARN] {cid} test 序列长度 {test_norm.shape[0]} 与 num_values={num_values} 不一致，"
                "以实际 test 长度为准生成标签。"
            )
            T = test_norm.shape[0]
        else:
            T = num_values

        labels = np.zeros(T, dtype=np.int64)
        seqs = parse_anomaly_sequences(row["anomaly_sequences"])
        for (s, e) in seqs:
            # 按照闭区间 [s, e] 生成 0/1 序列，并做边界裁剪
            s = max(0, int(s))
            e = min(T - 1, int(e))
            if s <= e:
                labels[s : e + 1] = 1

        np.save(os.path.join(out_train, f"{cid}.npy"), train_norm)
        np.save(os.path.join(out_test, f"{cid}.npy"), test_norm)
        np.save(os.path.join(out_label, f"{cid}.npy"), labels)

        print(
            f"[{cid}] 预处理完成：train {train_norm.shape}, "
            f"test {test_norm.shape}, label {labels.shape}（异常比例={labels.mean():.4f}）"
        )

    # 保存通道列表
    channels_txt = os.path.join(out_root, "channels.txt")
    with open(channels_txt, "w", encoding="utf-8") as f:
        for cid in chan_ids:
            f.write(str(cid) + "\n")
    print(f"已保存通道列表：{channels_txt}")


def parse_args():
    parser = argparse.ArgumentParser(description="预处理 MSL 原始数据")
    parser.add_argument(
        "--raw_root",
        type=str,
        default="./data/MSL",
        help="原始 MSL 数据根目录（包含 train/test/labeled_anomalies.csv）",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="./dataset/MSL",
        help="预处理后数据保存根目录，例如 ./dataset/MSL",
    )
    parser.add_argument(
        "--use_global_scaler",
        action="store_true",
        help="是否使用所有通道的训练数据拟合【统一】StandardScaler，"
        "默认：每个通道单独拟合 scaler",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_msl(
        raw_root=args.raw_root,
        out_root=args.out_root,
        use_global_scaler=args.use_global_scaler,
    )
