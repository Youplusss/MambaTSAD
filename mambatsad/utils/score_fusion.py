# mambatsad/utils/score_fusion.py
# -*- coding: utf-8 -*-
"""
分数归一化与融合相关工具函数。

主要用于混合模型 (hybrid) 在评估阶段将重构分支与预测分支的异常分数进行
更稳定的融合，避免单一分支支配结果。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _zscore(x: np.ndarray) -> np.ndarray:
    """对一维分数做 z-score 归一化，方差为 0 时返回全零。"""
    x = np.asarray(x, dtype=np.float64)
    mu = float(x.mean())
    sigma = float(x.std())

    if sigma < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - mu) / sigma


def fuse_scores_by_zscore(
    recon_scores: np.ndarray,
    forecast_scores: np.ndarray,
    w_recon: float = 1.0,
    w_forecast: float = 1.0,
) -> np.ndarray:
    """
    使用 z-score + 线性加权的方式融合重构分支与预测分支的异常分数。

    参数
    ----
    recon_scores : 一维数组，重构分支的异常分数（越大越异常或待自动翻转）；
    forecast_scores : 一维数组，预测分支的异常分数；
    w_recon : 重构分支权重；
    w_forecast : 预测分支权重。

    返回
    ----
    fused : 一维融合分数数组。
    """
    recon_scores = np.asarray(recon_scores, dtype=np.float64)
    forecast_scores = np.asarray(forecast_scores, dtype=np.float64)
    assert recon_scores.shape == forecast_scores.shape

    z_rec = _zscore(recon_scores)
    z_fore = _zscore(forecast_scores)

    w_rec = float(w_recon)
    w_for = float(w_forecast)
    if w_rec < 0 or w_for < 0:
        raise ValueError("w_recon / w_forecast 必须为非负。")

    if (w_rec + w_for) <= 1e-8:
        # 退化情况：两边权重都接近 0，则直接返回重构分支
        return z_rec

    fused = (w_rec * z_rec + w_for * z_fore) / (w_rec + w_for)
    return fused
