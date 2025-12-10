# mambatsad/utils/pseudo_label.py
# -*- coding: utf-8 -*-
"""
伪标签生成工具。

思路：
- 在训练集滑窗数据上使用无监督异常检测算法（默认 IsolationForest）；
- 对每个滑窗打一个「置信正常」/「可疑」标记；
- 只保留置信正常的滑窗用于训练，从而把无监督学习转成「半监督」训练。

注意：
- 这里是以「窗口」为单位做伪标签；
- contamination 参数控制“认为异常”的比例，越大则过滤越激进。
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from torch.utils.data import Dataset


def collect_windows_from_dataset(
    dataset: Dataset,
    max_samples: int | None = None,
) -> np.ndarray:
    """
    从滑窗 Dataset 中采样部分窗口，并展平为 2D 特征向量 [N, L*D]。

    参数
    ----
    dataset    : 任何 __getitem__ 返回 dict 且含 "window" 的 Dataset（如 SMDMultiWindowDataset）。
    max_samples: 最多采样多少个窗口，None 表示全部。

    返回
    ----
    X : [N, L*D] 的 numpy 数组。
    """
    all_wins: List[np.ndarray] = []
    N = len(dataset)
    if max_samples is None or max_samples > N:
        max_samples = N

    idxs = np.random.permutation(N)[:max_samples]

    for idx in idxs:
        item = dataset[idx]
        win = item["window"]  # [L, D]
        win = np.asarray(win, dtype=np.float32)
        all_wins.append(win.reshape(-1))  # 展平为 1D

    X = np.stack(all_wins, axis=0)  # [N, L*D]
    return X


def generate_pseudo_label_mask(
    dataset: Dataset,
    contamination: float = 0.01,
    random_state: int = 42,
    max_samples_fit: int | None = 50000,
) -> np.ndarray:
    """
    使用 IsolationForest 在训练滑窗上生成伪标签 mask。

    返回
    ----
    mask : 长度为 len(dataset) 的布尔数组，True 表示「被认为是正常窗口」。
    """
    N = len(dataset)
    # 先抽样一部分窗口用于拟合模型
    X_fit = collect_windows_from_dataset(dataset, max_samples=max_samples_fit)

    clf = IsolationForest(
        n_estimators=200,
        max_samples=min(256, len(X_fit)),
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_fit)

    # 对全部窗口打分
    all_scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        win = np.asarray(dataset[i]["window"], dtype=np.float32).reshape(1, -1)
        # decision_function 越小越异常（负值为异常）
        score = clf.decision_function(win)[0]
        all_scores[i] = score

    # 根据分位数做阈值：得分最小的 contamination 比例视作异常
    thr = np.quantile(all_scores, contamination)
    mask = all_scores >= thr
    return mask
