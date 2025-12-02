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


def preprocess_smd(
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocess_smd(
        raw_root=args.raw_root,
        out_root=args.out_root,
        use_global_scaler=args.use_global_scaler,
    )
