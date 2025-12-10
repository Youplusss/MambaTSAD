# mambatsad/utils/metrics.py
# -*- coding: utf-8 -*-
"""
评估指标与阈值搜索相关工具。

这里集中放置：
- point_adjust：时间序列异常检测中常用的段级别修正技巧；
- compute_roc_auc：不依赖 sklearn 的 ROC-AUC 计算；
- search_best_f1_threshold：在一组候选阈值上搜索 F1 最优阈值（带 P/R 平衡偏好）；
- search_best_f1_threshold_with_auto_flip：在此基础上自动判断分数方向是否需要翻转。
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

    约定：scores 数值 **越大越异常**，
    labels 中 1 表示“异常”，0 表示“正常”。
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


def _precision_recall_f1(
    pred: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    """内部小工具：给定 0/1 预测与标签，计算 P/R/F1。"""
    pred = pred.astype(int)
    labels = labels.astype(int)

    tp = np.logical_and(pred == 1, labels == 1).sum()
    fp = np.logical_and(pred == 1, labels == 0).sum()
    fn = np.logical_and(pred == 0, labels == 1).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return float(precision), float(recall), float(f1)


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

        precision, recall, f1 = _precision_recall_f1(pred, labels)

        return best_thr, {
            "precision": precision,
            "recall": recall,
            "f1": f1,
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

        precision, recall, f1 = _precision_recall_f1(pred, labels)

        # ---------- 阈值选择策略 ----------
        # 1) 主指标为 F1；
        # 2) 当 F1 非常接近时，优先选择 P/R 更“平衡”的那个，
        #    这里使用 min(P, R) 作为平衡度度量，越大越好，
        #    这样可以避免出现“P 很高但 R 很低”或相反的极端情况。
        balance_curr = min(precision, recall)
        balance_best = min(best_p, best_r)

        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and balance_curr > balance_best + 1e-6
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


def search_best_f1_threshold_with_auto_flip(
    scores: np.ndarray,
    labels: np.ndarray,
    num_steps: int = 2048,
    use_point_adjust: bool = True,
    f1_tolerance: float = 0.02,
) -> Tuple[float, Dict[str, float]]:
    """
    扩展版阈值搜索：自动判断“分数方向”是否需要翻转。

    背景
    ----
    - 有些模型产生的分数是「越大越异常」；
    - 但也可能因为取负 / 残差定义等原因，变成「越小越异常」；
    - 如果方向弄反，就会出现你在 SWAT 上看到的情况：
      AUC≈0.2 但 F1 / P / R 很高，看起来非常违和。

    做法
    ----
    1. 先假设「scores 越大越异常」，调用一次 search_best_f1_threshold；
    2. 再对 -scores 做同样的搜索，相当于假设「scores 越小越异常」；
    3. 默认优先选择 **AUC >= 0.5 的方向**（保证 ROC 语义正常），
       如果另一侧的 F1 高出非常多（超过 f1_tolerance），再让 F1“推翻”AUC 的选择；
    4. 返回最终方向下的阈值与指标，并在 metrics 中加入：
       - need_flip: 是否需要在外部对 scores 取负；
       - direction: "greater"（大于阈值为异常）或 "less"（小于阈值为异常）。

    注意
    ----
    外部在真正生成 0/1 预测时要遵循 metrics["direction"] 来做比较，
    或者在 need_flip 为 True 时先把 scores 取负，再用 >= threshold 判异常。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert scores.shape == labels.shape

    # 1) 原方向：越大越异常
    thr_pos, m_pos = search_best_f1_threshold(
        scores, labels, num_steps=num_steps, use_point_adjust=use_point_adjust
    )
    m_pos = dict(m_pos)  # 复制一份避免修改原字典
    auc_pos = float(m_pos.get("auc", 0.5))
    f1_pos = float(m_pos.get("f1", 0.0))

    # 2) 取负方向：越小越异常 <=> (-scores) 越大越异常
    thr_neg_on_neg, m_neg_raw = search_best_f1_threshold(
        -scores, labels, num_steps=num_steps, use_point_adjust=use_point_adjust
    )
    m_neg = dict(m_neg_raw)
    # 把“在 -scores 空间里的阈值”转换回原 scores 空间的阈值：
    thr_neg = -float(thr_neg_on_neg)
    m_neg["threshold"] = thr_neg
    auc_neg = float(m_neg.get("auc", 0.5))  # 这是“越小越异常”方向下的 AUC
    f1_neg = float(m_neg.get("f1", 0.0))

    # 默认：谁的 AUC >= 0.5（更远离 0.5），优先谁
    # 实际上 auc_neg = 1 - auc_pos（无 ties 时严格成立），
    # 所以通常只有一侧 >= 0.5
    choose_flip = False  # True 表示使用“越小越异常”的方向

    if (auc_neg >= 0.5 and auc_pos < 0.5) or (
        abs(auc_neg - 0.5) > abs(auc_pos - 0.5)
    ):
        choose_flip = True
    else:
        choose_flip = False

    # 如果某一侧 F1 明显更好，则允许 F1 推翻 AUC 的选择
    if choose_flip and f1_pos > f1_neg + f1_tolerance:
        choose_flip = False
    if (not choose_flip) and f1_neg > f1_pos + f1_tolerance:
        choose_flip = True

    # 补充方向信息
    m_pos["threshold"] = float(thr_pos)
    m_pos["need_flip"] = False
    m_pos["direction"] = "greater"

    m_neg["need_flip"] = True
    m_neg["direction"] = "less"

    if choose_flip:
        best_thr = thr_neg
        metrics = m_neg
    else:
        best_thr = thr_pos
        metrics = m_pos

    return float(best_thr), metrics
