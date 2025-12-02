# mambatsad/utils/metrics.py
# -*- coding: utf-8 -*-
"""
时间序列异常检测评估工具：
- 支持 AUC
- 支持基于 F1 的阈值搜索
- 支持 point_adjust
- 有意识地避免选择极端阈值（例如把所有点都判成异常）
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from .ts_postprocess import point_adjust


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """带异常捕获的 AUC 计算"""
    try:
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def search_best_f1_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    use_point_adjust: bool = True,
    num_thresholds: int = 200,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
):
    """
    在给定分数上搜索 F1 最优的阈值。

    注意：
    - 为了避免出现 "全部预测为异常" 这种极端解，
      我们只在 [low_percentile, high_percentile] 这个区间里搜索阈值。
      比如 [1%, 99%]，排除最极端的 0%/100% 部分。
    - 这是很多 TSAD 论文实际采用的做法（只不过没有明说）。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)

    assert scores.shape == labels.shape

    auc = _safe_auc(labels, scores)

    # 所有分数都一样 → 完全没有区分能力，直接返回一个退化结果
    if np.allclose(scores, scores[0]):
        thr = float(scores[0])
        preds = np.ones_like(labels)  # 或全 0，取决于你想怎么定义
        if use_point_adjust:
            preds = point_adjust(preds, labels)
        P = precision_score(labels, preds, zero_division=0)
        R = recall_score(labels, preds, zero_division=0)
        F1 = f1_score(labels, preds, zero_division=0)
        return {
            "best_thr": thr,
            "f1": float(F1),
            "precision": float(P),
            "recall": float(R),
            "auc": auc,
        }

    lo = float(np.percentile(scores, low_percentile))
    hi = float(np.percentile(scores, high_percentile))

    # 如果分布非常窄，就退化为 [min, max]
    if lo >= hi:
        lo = float(scores.min())
        hi = float(scores.max())

    thresholds = np.linspace(lo, hi, num=num_thresholds)

    best = {
        "best_thr": thresholds[0],
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "auc": auc,
    }

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        if use_point_adjust:
            preds = point_adjust(preds, labels)

        P = precision_score(labels, preds, zero_division=0)
        R = recall_score(labels, preds, zero_division=0)
        F1 = f1_score(labels, preds, zero_division=0)

        if F1 > best["f1"]:
            best.update(
                best_thr=float(thr),
                f1=float(F1),
                precision=float(P),
                recall=float(R),
            )

    return best
