# tools/plot_training_curves.py
# -*- coding: utf-8 -*-
"""
从训练日志（.log）中解析每个 epoch 的：
- 训练损失（total / recon / forecast）
- 各种 Eval F1 / P / R / AUC

并画成曲线，便于对比不同数据集、不同时间的训练过程。

用法示例：
python -u tools/plot_training_curves.py \
    --log_files ./logs/smd_hybrid/MambaTSAD_20251208_150536.log \
    --out_dir ./logs/vis_curves

也可以传多个日志文件，一张图中画多条曲线，便于对比：
python -u tools/plot_training_curves.py \
    --log_files log1.log log2.log \
    --out_dir ./logs/vis_curves
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


EPOCH_RE = re.compile(r"Epoch\s+(\d+)")
TRAIN_LOSS_RE = re.compile(
    r"混合模型训练损失: total=([0-9.eE+-]+), recon=([0-9.eE+-]+), forecast=([0-9.eE+-]+)"
)
TRAIN_LOSS_RE_SIMPLE = re.compile(
    r"本 epoch.*平均训练损失:\s*([0-9.eE+-]+)"
)
EVAL_RECON_RE = re.compile(
    r"\[Eval-recon.*\]\s*F1=([0-9.]+), P=([0-9.]+), R=([0-9.]+), AUC=([0-9.]+)"
)
EVAL_FORECAST_RE = re.compile(
    r"\[Eval-forecast.*\]\s*F1=([0-9.]+), P=([0-9.]+), R=([0-9.]+), AUC=([0-9.]+)"
)
EVAL_HYBRID_RE = re.compile(
    r"\[Eval-hybrid\]\s*F1=([0-9.]+), P=([0-9.]+), R=([0-9.]+), AUC=([0-9.]+)"
)


def parse_log(path: str) -> Dict[str, List[float]]:
    """
    从单个日志文件解析训练 / 测试指标。

    返回一个字典，key 例如：
    - 'epoch'
    - 'train_total_loss', 'train_recon_loss', 'train_forecast_loss'
    - 'recon_f1', 'recon_p', 'recon_r', 'recon_auc'
    - 'forecast_f1', ...
    - 'hybrid_f1', ...
    """
    metrics: Dict[str, List[float]] = {
        "epoch": [],
        "train_total_loss": [],
        "train_recon_loss": [],
        "train_forecast_loss": [],
        "recon_f1": [],
        "recon_p": [],
        "recon_r": [],
        "recon_auc": [],
        "forecast_f1": [],
        "forecast_p": [],
        "forecast_r": [],
        "forecast_auc": [],
        "hybrid_f1": [],
        "hybrid_p": [],
        "hybrid_r": [],
        "hybrid_auc": [],
    }

    current_epoch = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Epoch 行
            m_epoch = EPOCH_RE.search(line)
            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                if current_epoch not in metrics["epoch"]:
                    metrics["epoch"].append(current_epoch)

            # 训练损失（混合模型）
            m_train = TRAIN_LOSS_RE.search(line)
            if m_train:
                total = float(m_train.group(1))
                recon = float(m_train.group(2))
                forecast = float(m_train.group(3))
                metrics["train_total_loss"].append(total)
                metrics["train_recon_loss"].append(recon)
                metrics["train_forecast_loss"].append(forecast)

            # 仅预测分支的平均训练损失（forecast 单分支）
            m_train_simple = TRAIN_LOSS_RE_SIMPLE.search(line)
            if m_train_simple and not TRAIN_LOSS_RE.search(line):
                total = float(m_train_simple.group(1))
                metrics["train_total_loss"].append(total)

            # Eval-recon
            m_recon = EVAL_RECON_RE.search(line)
            if m_recon:
                metrics["recon_f1"].append(float(m_recon.group(1)))
                metrics["recon_p"].append(float(m_recon.group(2)))
                metrics["recon_r"].append(float(m_recon.group(3)))
                metrics["recon_auc"].append(float(m_recon.group(4)))

            # Eval-forecast
            m_fore = EVAL_FORECAST_RE.search(line)
            if m_fore:
                metrics["forecast_f1"].append(float(m_fore.group(1)))
                metrics["forecast_p"].append(float(m_fore.group(2)))
                metrics["forecast_r"].append(float(m_fore.group(3)))
                metrics["forecast_auc"].append(float(m_fore.group(4)))

            # Eval-hybrid
            m_hyb = EVAL_HYBRID_RE.search(line)
            if m_hyb:
                metrics["hybrid_f1"].append(float(m_hyb.group(1)))
                metrics["hybrid_p"].append(float(m_hyb.group(2)))
                metrics["hybrid_r"].append(float(m_hyb.group(3)))
                metrics["hybrid_auc"].append(float(m_hyb.group(4)))

    return metrics


def plot_curves_for_logs(log_files: List[str], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) 训练损失曲线
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))

    # 2) F1 曲线（重构 / 预测 / 混合）
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 4))

    for path in log_files:
        name = os.path.basename(path).replace(".log", "")
        metrics = parse_log(path)
        epochs = np.array(metrics["epoch"], dtype=np.int32)

        # ---------- loss ----------
        if metrics["train_total_loss"]:
            total = np.array(metrics["train_total_loss"])
            ax_loss.plot(epochs[: len(total)], total, label=f"{name}-total")
        if metrics["train_recon_loss"]:
            recon = np.array(metrics["train_recon_loss"])
            ax_loss.plot(epochs[: len(recon)], recon, linestyle="--", label=f"{name}-recon")
        if metrics["train_forecast_loss"]:
            fore = np.array(metrics["train_forecast_loss"])
            ax_loss.plot(epochs[: len(fore)], fore, linestyle=":", label=f"{name}-forecast")

        # ---------- F1 ----------
        if metrics["recon_f1"]:
            ax_f1.plot(
                epochs[: len(metrics["recon_f1"])],
                np.array(metrics["recon_f1"]),
                linestyle="--",
                label=f"{name}-reconF1",
            )
        if metrics["forecast_f1"]:
            ax_f1.plot(
                epochs[: len(metrics["forecast_f1"])],
                np.array(metrics["forecast_f1"]),
                linestyle=":",
                label=f"{name}-forecastF1",
            )
        if metrics["hybrid_f1"]:
            ax_f1.plot(
                epochs[: len(metrics["hybrid_f1"])],
                np.array(metrics["hybrid_f1"]),
                linestyle="-",
                label=f"{name}-hybridF1",
            )

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Train Loss")
    ax_loss.set_title("Training Loss Curves")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_f1.set_xlabel("Epoch")
    ax_f1.set_ylabel("F1 Score")
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.set_title("F1 Curves (recon / forecast / hybrid)")
    ax_f1.legend()
    ax_f1.grid(True, alpha=0.3)

    fig_loss.tight_layout()
    fig_f1.tight_layout()

    loss_path = os.path.join(out_dir, "loss_curves.png")
    f1_path = os.path.join(out_dir, "f1_curves.png")

    fig_loss.savefig(loss_path, dpi=200)
    fig_f1.savefig(f1_path, dpi=200)

    plt.close(fig_loss)
    plt.close(fig_f1)

    print(f"[OK] 已保存 loss 曲线到: {loss_path}")
    print(f"[OK] 已保存 F1 曲线到:   {f1_path}")


def main():
    parser = argparse.ArgumentParser(description="从训练日志中绘制 loss / F1 曲线")
    parser.add_argument(
        "--log_files",
        type=str,
        nargs="+",
        required=True,
        help="一个或多个 .log 文件路径",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./logs/vis_curves",
        help="输出图片目录",
    )
    args = parser.parse_args()

    plot_curves_for_logs(args.log_files, args.out_dir)


if __name__ == "__main__":
    main()
