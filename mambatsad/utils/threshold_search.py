# mambatsad/utils/threshold_search.py
# -*- coding: utf-8 -*-
"""
阈值搜索（带自动 score 方向修正）。

核心思路
--------
1. 先假设「分数越大越异常」，调用原来的 search_best_f1_threshold；
2. 如果此时 AUC < 0.5，说明整体排序有明显的反向趋势：
   - 在 score 取反空间 (-score) 上再跑一遍 search_best_f1_threshold；
   - 如果翻转后的 F1 更好，就采用翻转版本；
   - 同时在返回的 metrics 中记录:
       * metrics["flipped"] = True / False
       * metrics["cmp_op"] = ">=" or "<="   # 判定异常时用的比较方向
3. 最终返回的 threshold 始终是「原始 score 空间」上的阈值。
   - 若 flipped=False: 异常条件是 score >= threshold；
   - 若 flipped=True : 异常条件是 score <= threshold。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .metrics import search_best_f1_threshold


def search_best_f1_threshold_with_auto_flip(
    scores: np.ndarray,
    labels: np.ndarray,
    num_steps: int = 2048,
    use_point_adjust: bool = True,
    auto_flip: bool = True,
) -> Tuple[float, Dict[str, float]]:
    """
    带「自动翻转」能力的阈值搜索。

    参数
    ----
    scores : 一维异常分数数组，**原始方向**，不要求“越大越异常”。
    labels : 一维 0/1 标签数组。
    num_steps : 候选阈值个数上限，同 metrics.search_best_f1_threshold。
    use_point_adjust : 是否使用 point-adjust 技巧。
    auto_flip : 是否启用 score 方向自动翻转。

    返回
    ----
    best_thr : float
        在**原始 score 空间**上的阈值。
    metrics : dict
        在原有基础上增加：
        - "flipped": bool，是否采用了 score 取反方案；
        - "cmp_op": ">=" 或 "<="，判定异常时用的比较关系。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert scores.shape == labels.shape

    # 先按「分数越大越异常」跑一遍
    thr_raw, metrics_raw = search_best_f1_threshold(
        scores,
        labels,
        num_steps=num_steps,
        use_point_adjust=use_point_adjust,
    )
    metrics_raw = dict(metrics_raw)  # 拷贝一份，避免就地修改
    metrics_raw.setdefault("auc", 0.5)
    metrics_raw["flipped"] = False
    metrics_raw["cmp_op"] = ">="  # score >= thr 为异常

    if not auto_flip:
        return float(thr_raw), metrics_raw

    # 如果 AUC 明显小于 0.5，则尝试在取反空间重新搜索
    if metrics_raw["auc"] >= 0.5:
        # 排序方向正常，不需要翻转
        return float(thr_raw), metrics_raw

    # ---------- score 取反空间：small -> large ----------
    scores_flip = -scores
    thr_flip, metrics_flip = search_best_f1_threshold(
        scores_flip,
        labels,
        num_steps=num_steps,
        use_point_adjust=use_point_adjust,
    )
    metrics_flip = dict(metrics_flip)
    metrics_flip.setdefault("auc", 0.5)

    # 将阈值映射回原始 score 空间：
    # 在 scores_flip 空间中，异常条件是：scores_flip >= thr_flip
    # 即：-scores >= thr_flip  <=>  scores <= -thr_flip
    thr_flip_orig = -float(thr_flip)
    metrics_flip["threshold"] = thr_flip_orig
    metrics_flip["flipped"] = True
    metrics_flip["cmp_op"] = "<="  # score <= thr 为异常

    # 从 F1 角度选更好的那个方案
    if metrics_flip["f1"] > metrics_raw["f1"] + 1e-6:
        # 采用翻转版本；AUC 使用翻转空间计算结果（一般接近 1 - AUC_raw）
        return thr_flip_orig, metrics_flip
    else:
        # 维持原方向（虽然 AUC<0.5，但 F1 更好）
        return float(thr_raw), metrics_raw
