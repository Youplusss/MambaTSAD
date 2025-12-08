# -*- coding: utf-8 -*-
"""
简单可视化工具：
- 绘制一段时间上的 anomaly score + label + 阈值，
  用于快速观察模型检测效果。
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_scores_with_labels(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    save_path: str,
    max_points: Optional[int] = 5000,
) -> None:
    """
    绘制「异常分数 + 标签 + 阈值」曲线。

    参数
    ----
    scores:
        一维异常分数数组 [T]。
    labels:
        对应的一维 0/1 标签数组 [T]。
    threshold:
        阈值（绘制为一条水平线）。
    save_path:
        图片保存路径。
    max_points:
        最多绘制多少个时间点，以避免极长序列导致图像过密。
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    T = len(scores)

    if max_points is not None and T > max_points:
        scores = scores[:max_points]
        labels = labels[:max_points]
        T = max_points

    fig, ax1 = plt.subplots(figsize=(12, 4))

    ax1.plot(range(T), scores, label="Anomaly Score")
    ax1.axhline(threshold, linestyle="--", label=f"Threshold={threshold:.4f}")
    ax1.set_ylabel("Score")

    ax2 = ax1.twinx()
    ax2.plot(range(T), labels, color="g", alpha=0.3, label="Label (0/1)")
    ax2.set_ylabel("Label")

    ax1.set_xlabel("Time Index")
    ax1.set_title("Anomaly Scores & Labels")

    lines, labels_ = ax1.get_legend_handles_labels()
    lines2, labels2_ = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_ + labels2_, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
