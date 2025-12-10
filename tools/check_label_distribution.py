# tools/check_label_distribution.py
# -*- coding: utf-8 -*-
"""\
快速检查预处理后数据集的标签分布。

目前支持：
- dataset/WADI/test_label/wadi.npy
- dataset/SWAT/test_label/swat.npy

运行方式（在项目根目录）：

    python tools/check_label_distribution.py

可以用 --root 指定 dataset 根目录，例如：

    python tools/check_label_distribution.py --root ./dataset
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _load_labels(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"标签文件不存在: {path}")
    labels = np.load(path)
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    return labels.astype(int)


def _summarize_labels(name: str, labels: np.ndarray) -> None:
    total = labels.size
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    num_pos = int(dist.get(1, 0))
    num_neg = int(dist.get(0, 0))

    pos_ratio = num_pos / total if total > 0 else 0.0
    neg_ratio = num_neg / total if total > 0 else 0.0

    print(f"\n=== {name} 标签分布 ===")
    print(f"总样本数: {total}")
    print(f"取值统计: {dist}")
    print(f"正常(0): {num_neg} ({neg_ratio:.4%}), 攻击(1): {num_pos} ({pos_ratio:.4%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="检查 WADI / SWAT 标签分布")
    parser.add_argument(
        "--root",
        type=str,
        default="./dataset",
        help="dataset 根目录，默认 ./dataset",
    )
    args = parser.parse_args()

    root = args.root

    datasets = [
        ("WADI", os.path.join(root, "WADI", "test_label", "wadi.npy")),
        ("SWAT", os.path.join(root, "SWaT", "test_label", "swat.npy")),
        ("MSL", os.path.join(root, "MSL", "test_label", "C-1.npy")),
        ("SMD", os.path.join(root, "SMD", "test_label", "machine-1-1.npy")),
    ]

    for name, path in datasets:
        try:
            labels = _load_labels(path)
        except FileNotFoundError as e:
            print(f"\n=== {name} ===")
            print(f"未找到标签文件: {path}")
            continue
        _summarize_labels(name, labels)


if __name__ == "__main__":
    main()
