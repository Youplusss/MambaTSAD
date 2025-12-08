# -*- coding: utf-8 -*-
"""
评估指标与阈值搜索相关工具。

这里集中放置：
- point_adjust：时间序列异常检测中常用的段级别修正技巧；
- compute_roc_auc：不依赖 sklearn 的 ROC-AUC 计算；
- search_best_f1_threshold：在一组候选阈值上搜索 F1 最优阈值。
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Point-adjust 技巧。

    对每一段连续的真实异常区间，如果模型在该区间内任意一点预测为异常，
    则把该区间全部视作预测异常。这样可以缓解“只命中一两个点却整段算错”的问题。
    """
    pred = pred.astype(bool)
    labels = labels.astype(int)
    assert pred.shape == labels.shape

    n = len(labels)
    i = 0
    while i < n:
        if labels[i] == 1:
            j = i + 1
            while j < n and labels[j] == 1:
                j += 1
            # [i, j) 为一段连续的异常区间
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1

    return pred.astype(int)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """使用纯 numpy 计算 ROC-AUC，避免依赖 sklearn。

    实现依据 Mann–Whitney U 统计量的等价形式：
        AUC = (sum_ranks_pos - P*(P+1)/2) / (P*N)

    其中 P/N 分别为正/负样本数量。
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
    """在一组候选阈值上搜索 F1 最大的阈值。

    参数
    ----
    scores:
        一维异常分数数组，数值越大越异常。
    labels:
        对应的 0/1 标签数组。
    num_steps:
        最多使用多少个候选阈值（scores 唯一值过多时进行均匀采样）。
    use_point_adjust:
        是否在计算 P/R/F1 之前应用 point_adjust 技巧。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert scores.shape == labels.shape

    uniq_scores = np.unique(scores)
    if len(uniq_scores) == 1:
        # 所有得分完全一样，模型几乎没有区分能力
        best_thr = float(uniq_scores[0])
        pred = scores >= best_thr
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

        # 主指标为 F1，在 F1 非常接近时偏向 precision 更高的阈值
        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and precision > best_p
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
