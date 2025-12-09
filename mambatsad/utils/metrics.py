# mambatsad/utils/metrics.py
# -*- coding: utf-8 -*-
"""
评估指标与阈值搜索相关工具。

这里集中放置：
- point_adjust：时间序列异常检测中常用的段级别修正技巧；
- compute_roc_auc：不依赖 sklearn 的 ROC-AUC 计算；
- search_best_f1_threshold：在一组候选阈值上搜索 F1 最优阈值（带 P/R 平衡偏好）。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Point-adjust 技巧。

    对每一段连续的真实异常区间，如果模型在该区间内任意一点预测为异常，
    则把该区间全部视作预测异常。这样可以缓解“只命中一两个点却整段算错”的问题。

    参数
    ----
    pred : 一维 0/1 预测数组。
    labels : 一维 0/1 真实标签数组。

    返回
    ----
    new_pred : 应用 point-adjust 后的 0/1 数组。
    """
    pred = pred.astype(bool)
    labels = labels.astype(int)
    assert pred.shape == labels.shape

    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            # 向后找到这一段连续异常区间的结束位置 j（半开区间 [i, j)）
            j = i + 1
            while j < n and labels[j] == 1:
                j += 1
            # 若区间内任意一点被预测为异常，则整段都置为异常
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1

    return pred.astype(int)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    使用纯 numpy 计算 ROC-AUC，避免依赖 sklearn。

    实现依据 Mann–Whitney U 统计量的等价形式：
        AUC = (sum_ranks_pos - P*(P+1)/2) / (P*N)
    其中 P/N 分别为正/负样本数量。

    约定：scores 数值 **越大越异常**，labels 中 1 表示“异常”，0 表示“正常”。
    """
    labels = labels.astype(int)
    scores = scores.astype(float)
    assert labels.shape == scores.shape

    P = int(labels.sum())
    N = int(len(labels) - P)
    if P == 0 or N == 0:
        # 极端情况：全正或全负，AUC 定义不明，这里返回 0.5
        return 0.5

    # 根据得分从小到大排序
    order = np.argsort(scores)
    ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    ranks_pos = ranks[labels[order] == 1]
    sum_ranks_pos = ranks_pos.sum()

    auc = (sum_ranks_pos - P * (P + 1) / 2.0) / (P * N)
    return float(auc)


def search_best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    num_steps: int = 2048,
    use_point_adjust: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    在一组候选阈值上搜索 F1 最大的阈值，并附带返回 P/R/AUC 等指标。

    参数
    ----
    scores : 一维异常分数数组，**数值越大越异常**。
    labels : 对应的 0/1 标签数组（1=异常，0=正常）。
    num_steps : 最多使用多少个候选阈值（scores 唯一值过多时进行均匀采样）。
    use_point_adjust : 是否在计算 P/R/F1 之前应用 point_adjust 技巧。

    返回
    ----
    best_thr : F1 最优的阈值。
    metrics : dict，包含 "precision" / "recall" / "f1" / "auc" / "threshold" 等。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert scores.shape == labels.shape

    uniq_scores = np.unique(scores)

    # 若所有得分完全一致，则模型几乎没有区分能力
    if len(uniq_scores) == 1:
        best_thr = float(uniq_scores[0])
        pred = (scores >= best_thr).astype(int)
        if use_point_adjust:
            pred = point_adjust(pred, labels)

        tp = np.logical_and(pred == 1, labels == 1).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return best_thr, {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": 0.5,
            "threshold": best_thr,
            "use_point_adjust": use_point_adjust,
        }

    # 若唯一值过多，则在数轴上均匀抽样一部分作为候选阈值
    if len(uniq_scores) > num_steps:
        idxs = np.linspace(0, len(uniq_scores) - 1, num_steps).astype(int)
        cand_thrs = uniq_scores[idxs]
    else:
        cand_thrs = uniq_scores

    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    best_thr = float(cand_thrs[0])

    for thr in cand_thrs:
        pred = (scores >= thr).astype(int)
        if use_point_adjust:
            pred = point_adjust(pred, labels)

        tp = np.logical_and(pred == 1, labels == 1).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # ---------- 阈值选择策略 ----------
        # 1) 主指标为 F1；
        # 2) 当 F1 非常接近时，优先选择 P/R 更“平衡”的那个，
        #    这里使用 min(P, R) 作为平衡度度量，越大越好，
        #    这样可以避免出现“P 很高但 R 很低”或相反的极端情况。
        balance_curr = min(precision, recall)
        balance_best = min(best_p, best_r)

        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6
            and balance_curr > balance_best + 1e-6
        ):
            best_f1 = f1
            best_p = precision
            best_r = recall
            best_thr = float(thr)

    auc = compute_roc_auc(labels, scores)

    metrics = {
        "precision": float(best_p),
        "recall": float(best_r),
        "f1": float(best_f1),
        "auc": float(auc),
        "threshold": float(best_thr),
        "use_point_adjust": use_point_adjust,
    }
    return best_thr, metrics
