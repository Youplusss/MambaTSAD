# mambatsad/utils/ts_postprocess.py
# -*- coding: utf-8 -*-
"""
时间序列异常检测常用的 point_adjust 实现

思想：
- 对每一段连续的真实异常区间 [s, e]：
  - 只要预测中在这段里“有任何一个点被判为异常”，
  - 就把整段 [s, e] 都置为预测异常 1。
- 这样能缓解“只打中一小段就被 F1 惩罚很狠”的问题，
  也是 OmniAnomaly、USAD、TranAD 等作品里常用的指标计算方式。
"""

import numpy as np


def point_adjust(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    """
    参数：
    - pred_labels: 预测标签，一维 0/1 数组
    - gt_labels:   真实标签，一维 0/1 数组
    返回：
    - 调整后的预测标签，一维 0/1 数组
    """
    pred = np.asarray(pred_labels, dtype=int).copy()
    gt = np.asarray(gt_labels, dtype=int)
    assert pred.shape == gt.shape, "pred_labels 与 gt_labels 形状必须一致"

    n = len(gt)
    i = 0
    while i < n:
        if gt[i] == 1:
            # 找到一段连续异常 [i, j)
            j = i + 1
            while j < n and gt[j] == 1:
                j += 1

            # 如果该区间内有任何一个预测为 1，则整段置为 1
            if pred[i:j].sum() > 0:
                pred[i:j] = 1

            i = j
        else:
            i += 1

    return pred
