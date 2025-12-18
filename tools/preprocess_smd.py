# tools/preprocess_smd.py
# -*- coding: utf-8 -*-
"""
SMD 原始数据预处理脚本：
- 从原始 ServerMachineDataset 加载 train/test/test_label
- 对每个 machine 做标准化（默认：每台机器单独标准化，也可选择全局标准化）
- 保存为 .npy，供训练 / 测试脚本使用
使用示例：
python tools/preprocess_smd.py \
    --raw_root ./data/ServerMachineDataset \
    --out_root ./data_processed/SMD \
    --use_global_scaler
"""

import os
import argparse
from typing import List

import numpy as np
from sklearn.preprocessing import StandardScaler


def downsample_features(data: np.ndarray, factor: int) -> np.ndarray:
    """Median-pool time axis by the given factor; drop trailing remainder."""
    if factor <= 1 or data.shape[0] < factor:
        return data
    usable = (data.shape[0] // factor) * factor
    trimmed = data[:usable]
    reshaped = trimmed.reshape(-1, factor, data.shape[1])
    return np.median(reshaped, axis=1)


def downsample_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """Mark window anomalous if any inner label is anomalous."""
    if factor <= 1 or labels.shape[0] < factor:
        return labels
    usable = (labels.shape[0] // factor) * factor
    trimmed = labels[:usable]
    reshaped = trimmed.reshape(-1, factor)
    return np.max(reshaped, axis=1)


def list_machine_ids(raw_root: str) -> List[str]:
    train_dir = os.path.join(raw_root, "train")
    machine_ids = []
    for fn in os.listdir(train_dir):
        if fn.endswith(".txt"):
            machine_ids.append(os.path.splitext(fn)[0])
    machine_ids = sorted(machine_ids)
    return machine_ids


def load_raw_smd_machine(raw_root: str, machine_id: str):
    train_path = os.path.join(raw_root, "train", f"{machine_id}.txt")
    test_path = os.path.join(raw_root, "test", f"{machine_id}.txt")
    label_path = os.path.join(raw_root, "test_label", f"{machine_id}.txt")

    train = np.loadtxt(train_path, delimiter=",")
    test = np.loadtxt(test_path, delimiter=",")
    labels = np.loadtxt(label_path, delimiter=",")

    return train, test, labels


def skip_head_rows(arr: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or arr.shape[0] == 0:
        return arr
    if arr.shape[0] <= count:
        return arr[0:0]
    return arr[count:]


def preprocess_smd(
    raw_root: str,
    out_root: str,
    use_global_scaler: bool = False,
    downsample_factor: int = 1,
    skip_head: int = 2160,
):
    os.makedirs(out_root, exist_ok=True)
    out_train = os.path.join(out_root, "train")
    out_test = os.path.join(out_root, "test")
    out_label = os.path.join(out_root, "test_label")
    os.makedirs(out_train, exist_ok=True)
    os.makedirs(out_test, exist_ok=True)
    os.makedirs(out_label, exist_ok=True)

    machine_ids = list_machine_ids(raw_root)
    print(f"发现 {len(machine_ids)} 台机器：{machine_ids}")

    # ------------------（可选）先拟合全局 StandardScaler ------------------
    global_scaler = None
    if use_global_scaler:
        print("使用【全局 StandardScaler】拟合所有机器的训练数据 ...")
        all_train_list = []
        for mid in machine_ids:
            train_raw, _, _ = load_raw_smd_machine(raw_root, mid)
            all_train_list.append(train_raw)
        all_train = np.concatenate(all_train_list, axis=0)
        global_scaler = StandardScaler()
        global_scaler.fit(all_train)
        print("全局 scaler 拟合完成。")

    # ------------------ 逐机预处理并保存 ------------------
    for mid in machine_ids:
        print(f"[{mid}] 加载原始数据 ...")
        train_raw, test_raw, labels = load_raw_smd_machine(raw_root, mid)

        if use_global_scaler:
            scaler = global_scaler
        else:
            scaler = StandardScaler()
            scaler.fit(train_raw)

        train_norm = scaler.transform(train_raw).astype(np.float32)
        test_norm = scaler.transform(test_raw).astype(np.float32)
        labels = labels.astype(np.int64)

        if downsample_factor > 1:
            train_norm = downsample_features(train_norm, downsample_factor)
            test_norm = downsample_features(test_norm, downsample_factor)
            labels = downsample_labels(labels, downsample_factor)

        if skip_head > 0:
            train_norm = skip_head_rows(train_norm, skip_head)
            test_norm = skip_head_rows(test_norm, skip_head)
            labels = skip_head_rows(labels, skip_head)

        np.save(os.path.join(out_train, f"{mid}.npy"), train_norm)
        np.save(os.path.join(out_test, f"{mid}.npy"), test_norm)
        np.save(os.path.join(out_label, f"{mid}.npy"), labels)

        print(
            f"[{mid}] 预处理完成：train {train_norm.shape}, "
            f"test {test_norm.shape}, label {labels.shape}"
        )

    # 保存机器列表
    machines_txt = os.path.join(out_root, "machines.txt")
    with open(machines_txt, "w", encoding="utf-8") as f:
        for mid in machine_ids:
            f.write(mid + "\n")
    print(f"已保存机器列表：{machines_txt}")


def parse_args():
    parser = argparse.ArgumentParser(description="预处理 SMD 原始数据")
    parser.add_argument("--raw_root", type=str, default="./data/SMD",
                        help="原始 ServerMachineDataset 根目录（包含 train/test/test_label 子目录）")
    parser.add_argument("--out_root", type=str, default="./dataset/SMD",
                        help="预处理后数据保存根目录，例如 ./data_processed/SMD")
    parser.add_argument("--use_global_scaler", action="store_true",
                        help="是否使用所有机器的训练数据拟合【统一】StandardScaler，"
                             "默认：每台机器单独拟合 scaler")
    parser.add_argument("--downsample_factor", type=int, default=1,
                        help="可选下采样窗口长度（>1 则按窗口做中位数池化，并在标签上取最大值）")
    parser.add_argument("--skip_head", type=int, default=0,
                        help="跳过序列开头的样本数量，默认 2160（与 wadi.py 保持一致）；"
                             "设为 0 可关闭")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_smd(
        raw_root=args.raw_root,
        out_root=args.out_root,
        use_global_scaler=args.use_global_scaler,
        downsample_factor=args.downsample_factor,
        skip_head=args.skip_head,
    )
