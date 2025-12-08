# main.py
# -*- coding: utf-8 -*-
"""
MambaTSAD 统一训练 + 测试入口

当前支持数据集：
- SMD 多机：使用 mambatsad.datasets.smd.build_smd_multi_datasets
- MSL 多通道：使用 mambatsad.datasets.msl.build_msl_multi_datasets

当前支持任务类型：
- 重构分支（recon）：使用 mambatsad.models.mambatsad_ts.mambatsad_ts_base
- 预测分支（forecast）：使用 mambatsad.models.mambatsad_pred.mambatsad_ts_pred_base

注意：
- 不同数据集需要先用对应的 preprocess_xxx.py 做预处理，
  然后在本脚本中通过 --dataset 和 --processed_root 指定。
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mambatsad.datasets.smd import build_smd_multi_datasets
from mambatsad.datasets.msl import build_msl_multi_datasets
from mambatsad.models.mambatsad_ts import mambatsad_ts_base
from mambatsad.models.mambatsad_pred import mambatsad_ts_pred_base
from mambatsad.utils.logger import get_logger
from mambatsad.utils.visualization import plot_scores_with_labels


# ============================================================
# 一些基础工具函数
# ============================================================


def set_seed(seed: int = 42) -> None:
    """固定随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    point-adjust 技巧（时间序列异常检测常用）：

    - 对每一段连续的真实异常区间，如果模型在该区间内任意一点预测为异常，
      则把该区间全部置为预测异常。
    - 这样可以缓解“只命中一两个点但整段都算错”的问题。
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
            # [i, j) 为一段连续异常区间
            if pred[i:j].any():
                pred[i:j] = True
            i = j
        else:
            i += 1

    return pred.astype(int)


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    纯 numpy 实现的 ROC-AUC（不依赖 sklearn）

    使用 Mann–Whitney U 统计量公式计算：
        AUC = (sum_ranks_pos - P*(P+1)/2) / (P*N)
    """
    labels = labels.astype(int)
    scores = scores.astype(float)
    assert labels.shape == scores.shape

    P = int(labels.sum())
    N = int(len(labels) - P)
    if P == 0 or N == 0:
        # 极端情况：全为正或全为负，返回 0.5（无信息分类器）
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
    在一组候选阈值上搜索使 F1 最大的阈值。

    参数
    ----
    scores : np.ndarray
        形状 (T,) 的连续得分，数值越大越异常。
    labels : np.ndarray
        形状 (T,) 的 0/1 标签。
    num_steps : int
        最多使用多少个候选阈值（避免 scores 很多时计算量过大）。
    use_point_adjust : bool
        在计算 P/R/F1 之前是否使用 point-adjust 技巧。

    返回
    ----
    best_thr : float
        最优阈值。
    metrics : dict
        包含 precision / recall / f1 / auc / threshold / use_point_adjust
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    assert scores.shape == labels.shape

    uniq_scores = np.unique(scores)
    if len(uniq_scores) == 1:
        # 所有得分完全一样，模型几乎没有区分能力
        best_thr = uniq_scores[0]
        pred = scores >= best_thr
        if use_point_adjust:
            pred = point_adjust(pred, labels)

        tp = np.logical_and(pred == 1, labels == 1).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        auc = 0.5

        return best_thr, {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
            "threshold": float(best_thr),
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
    best_thr = cand_thrs[0]

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

        # 先比较 F1，如果 F1 很接近则倾向于 precision 更高的阈值
        if (f1 > best_f1 + 1e-6) or (
            abs(f1 - best_f1) <= 1e-6 and precision > best_p
        ):
            best_f1 = f1
            best_p = precision
            best_r = recall
            best_thr = thr

    auc = compute_roc_auc(labels, scores)
    metrics = {
        "precision": float(best_p),
        "recall": float(best_r),
        "f1": float(best_f1),
        "auc": float(auc),
        "threshold": float(best_thr),
        "use_point_adjust": use_point_adjust,
    }
    return float(best_thr), metrics


# ============================================================
# 训练 & 测试逻辑
# ============================================================


def train_one_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    logger=None,
) -> float:
    """
    单个 epoch 的训练循环（重构分支版本）：

    - 输入为滑动窗口 window ∈ R^{L, D};
    - 模型输出重构结果 out["recon"] 或 out["recon_multi"]；
    - 损失为重构 MSE。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    torch.autograd.set_detect_anomaly(False)

    for batch in tqdm(train_loader, desc="Train(Recon)", leave=False):
        x = batch["window"]
        x = (
            x.to(device, non_blocking=True)
            if isinstance(x, torch.Tensor)
            else torch.from_numpy(x).to(device, non_blocking=True)
        )

        # 防守式数值清洗
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, -1e6, 1e6)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            out = model(x)

            rec_list = out.get("recon_multi")
            if rec_list is None:
                recon = out.get("recon")
                recon = torch.nan_to_num(recon, nan=0.0, posinf=1e6, neginf=-1e6)
                recon = torch.clamp(recon, -1e6, 1e6)
                loss = F.mse_loss(recon, x, reduction="mean")
            else:
                loss_val = 0.0
                for rec in rec_list:
                    rec = torch.nan_to_num(rec, nan=0.0, posinf=1e6, neginf=-1e6)
                    rec = torch.clamp(rec, -1e6, 1e6)
                    loss_val = loss_val + F.mse_loss(rec, x, reduction="mean")
                loss = loss_val

        if (
            not isinstance(loss, torch.Tensor)
            or not loss.requires_grad
            or not torch.isfinite(loss)
        ):
            if logger is not None:
                logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if logger is not None:
        logger.info(f"本 epoch（重构分支）平均训练损失: {avg_loss:.6f}")
    return avg_loss


def train_one_epoch_forecast(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    max_grad_norm: float = 1.0,
    logger=None,
    context_len: int = 90,
    pred_len: int = 10,
) -> float:
    """
    单个 epoch 的训练循环（预测分支版本）：

    - 输入为滑动窗口 window ∈ R^{L, D}；
    - 前 context_len 步作为条件输入 x_enc；
    - 紧接的 pred_len 步作为预测目标 y_true；
    - 损失为预测值与真实未来序列的 MSE。
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    torch.autograd.set_detect_anomaly(False)

    for batch in tqdm(train_loader, desc="Train(Forecast)", leave=False):
        x = batch["window"]
        x = (
            x.to(device, non_blocking=True)
            if isinstance(x, torch.Tensor)
            else torch.from_numpy(x).to(device, non_blocking=True)
        )

        # 防守式数值清洗
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x = torch.clamp(x, -1e6, 1e6)

        B, L, D = x.shape
        if L < context_len + pred_len:
            raise ValueError(
                f"窗口长度 L={L} 小于 context_len + pred_len = "
                f"{context_len + pred_len}，请调大 --win_size。"
            )

        # 上下文与未来序列拆分
        x_enc = x[:, :context_len, :]
        y_true = x[:, context_len : context_len + pred_len, :]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(scaler is not None)):
            y_pred = model(x_enc)
            if isinstance(y_pred, dict):
                # 兼容可能返回 dict 的情况
                if "pred" not in y_pred:
                    raise ValueError("预测模型 forward 返回的字典中未包含 'pred' 键。")
                y_pred = y_pred["pred"]

            if y_pred.shape != y_true.shape:
                raise ValueError(
                    f"预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致，"
                    f"请检查 seq_len 与 pred_len 配置。"
                )

            loss = F.mse_loss(y_pred, y_true, reduction="mean")

        if (
            not isinstance(loss, torch.Tensor)
            or not loss.requires_grad
            or not torch.isfinite(loss)
        ):
            if logger is not None:
                logger.warning("遇到非有限 loss (NaN/Inf)，跳过该 batch")
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    if logger is not None:
        logger.info(f"本 epoch（预测分支）平均训练损失: {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def evaluate_multi(
    model: torch.nn.Module,
    test_loader: DataLoader,
    win_size: int,
    labels_list: List[np.ndarray],
    device: torch.device,
    use_point_adjust: bool = True,
    num_thresholds: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """
    在“多实体合并”的数据集上做评估（重构分支，SMD / MSL 通用）：

    - 将每个 window 对应的重构误差回填到原始时间轴上，在每个时间点做平均，
      得到逐点 anomaly score；
    - 在全局上搜索最优阈值（可选 point-adjust），计算 F1 / P / R / AUC。
    """
    model.eval()

    num_seqs = len(labels_list)
    # 为每条序列分配得分累积数组
    sum_scores = [
        np.zeros(len(labels_list[i]), dtype=np.float64) for i in range(num_seqs)
    ]
    cnt_scores = [
        np.zeros(len(labels_list[i]), dtype=np.float64) for i in range(num_seqs)
    ]

    for batch in tqdm(test_loader, desc="Eval(Recon)", leave=False):
        x = batch["window"]
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        else:
            x_t = x
        x_t = x_t.to(device, non_blocking=True)

        seq_idx = batch["seq_idx"]
        starts = batch["start"]

        if isinstance(seq_idx, torch.Tensor):
            seq_idx = seq_idx.cpu().numpy()
        elif not isinstance(seq_idx, np.ndarray):
            seq_idx = np.array(seq_idx, dtype=np.int64)

        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().numpy()
        elif not isinstance(starts, np.ndarray):
            starts = np.array(starts, dtype=np.int64)

        out = model(x_t)
        rec = out["recon"]  # [B, L, D]

        # 逐时间步重构误差 MSE: [B, L]
        mse = ((rec - x_t) ** 2).mean(dim=-1)
        mse_np = mse.detach().cpu().numpy()

        B, L = mse_np.shape
        for i in range(B):
            k = int(seq_idx[i])
            s = int(starts[i])
            e = s + win_size
            sum_scores[k][s:e] += mse_np[i]
            cnt_scores[k][s:e] += 1.0

    scores_list = []
    labels_concat = []

    for k in range(num_seqs):
        c = cnt_scores[k]
        c[c == 0] = 1.0  # 避免除零
        seq_scores = sum_scores[k] / c
        scores_list.append(seq_scores)
        labels_concat.append(labels_list[k])

    scores = np.concatenate(scores_list)
    labels_full = np.concatenate(labels_concat).astype(int)

    thr, metrics = search_best_f1_threshold(
        scores,
        labels_full,
        num_steps=num_thresholds,
        use_point_adjust=use_point_adjust,
    )
    return scores, labels_full, thr, metrics


@torch.no_grad()
def evaluate_multi_forecast(
    model: torch.nn.Module,
    test_loader: DataLoader,
    win_size: int,
    pred_len: int,
    labels_list: List[np.ndarray],
    device: torch.device,
    use_point_adjust: bool = True,
    num_thresholds: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """
    在“多实体合并”的数据集上做评估（预测分支，SMD / MSL 通用）：

    - 对每个滑动窗口 window:
        * 前 context_len = win_size - pred_len 步作为条件输入；
        * 紧接的 pred_len 步作为预测目标；
    - 计算逐时间步预测误差 MSE，回填到原始时间轴上并做平均；
    - 在全局上搜索最优阈值（可选 point-adjust），计算 F1 / P / R / AUC。
    """
    model.eval()

    context_len = win_size - pred_len
    if context_len <= 0:
        raise ValueError("win_size 必须大于 pred_len，才能进行预测式异常检测。")

    num_seqs = len(labels_list)
    sum_scores = [
        np.zeros(len(labels_list[i]), dtype=np.float64) for i in range(num_seqs)
    ]
    cnt_scores = [
        np.zeros(len(labels_list[i]), dtype=np.float64) for i in range(num_seqs)
    ]

    for batch in tqdm(test_loader, desc="Eval(Forecast)", leave=False):
        x = batch["window"]
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x)
        else:
            x_t = x
        x_t = x_t.to(device, non_blocking=True)

        B, L, D = x_t.shape
        if L < context_len + pred_len:
            raise ValueError(
                f"测试窗口长度 L={L} 小于 context_len + pred_len = "
                f"{context_len + pred_len}，请调大 --win_size。"
            )

        seq_idx = batch["seq_idx"]
        starts = batch["start"]

        if isinstance(seq_idx, torch.Tensor):
            seq_idx = seq_idx.cpu().numpy()
        elif not isinstance(seq_idx, np.ndarray):
            seq_idx = np.array(seq_idx, dtype=np.int64)

        if isinstance(starts, torch.Tensor):
            starts = starts.cpu().numpy()
        elif not isinstance(starts, np.ndarray):
            starts = np.array(starts, dtype=np.int64)

        # 拆分上下文与未来序列
        x_enc = x_t[:, :context_len, :]
        y_true = x_t[:, context_len : context_len + pred_len, :]

        y_pred = model(x_enc)
        if isinstance(y_pred, dict):
            if "pred" not in y_pred:
                raise ValueError("预测模型 forward 返回的字典中未包含 'pred' 键。")
            y_pred = y_pred["pred"]

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"预测输出形状 {y_pred.shape} 与目标形状 {y_true.shape} 不一致，"
                f"请检查 seq_len 与 pred_len 配置。"
            )

        # 逐时间步预测误差 MSE: [B, pred_len]
        mse = ((y_pred - y_true) ** 2).mean(dim=-1)
        mse_np = mse.detach().cpu().numpy()

        for i in range(B):
            k = int(seq_idx[i])
            s = int(starts[i])

            s_pred = s + context_len
            e_pred = s_pred + pred_len

            # 安全防护：防止越界（按设计一般不会越界）
            if s_pred >= len(sum_scores[k]):
                continue
            if e_pred > len(sum_scores[k]):
                valid_len = len(sum_scores[k]) - s_pred
                if valid_len <= 0:
                    continue
                sum_scores[k][s_pred:] += mse_np[i, :valid_len]
                cnt_scores[k][s_pred:] += 1.0
            else:
                sum_scores[k][s_pred:e_pred] += mse_np[i]
                cnt_scores[k][s_pred:e_pred] += 1.0

    scores_list = []
    labels_concat = []

    for k in range(num_seqs):
        c = cnt_scores[k]
        c[c == 0] = 1.0
        seq_scores = sum_scores[k] / c
        scores_list.append(seq_scores)
        labels_concat.append(labels_list[k])

    scores = np.concatenate(scores_list)
    labels_full = np.concatenate(labels_concat).astype(int)

    thr, metrics = search_best_f1_threshold(
        scores,
        labels_full,
        num_steps=num_thresholds,
        use_point_adjust=use_point_adjust,
    )
    return scores, labels_full, thr, metrics


# ============================================================
# 参数解析 & 主入口
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MambaTSAD on SMD / MSL")

    # 数据集相关
    parser.add_argument(
        "--dataset",
        type=str,
        default="smd",
        choices=["smd", "msl"],
        help="选择数据集：smd 或 msl",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="预处理数据根目录，例如 ./dataset/SMD 或 ./dataset/MSL",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/exp",
        help="日志与模型保存目录（不再按实体拆分）",
    )

    # 窗口 & 任务相关
    parser.add_argument(
        "--win_size",
        type=int,
        default=100,
        help="滑动窗口长度",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="recon",
        choices=["recon", "forecast"],
        help="任务类型：recon 为重构分支，forecast 为预测分支",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=10,
        help="预测分支时的预测步数（仅在 --task=forecast 时使用）",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="训练集滑动步长（一般为 1）",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="测试集滑动步长（可适当加大以加速）",
    )

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭混合精度训练",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值（防止梯度爆炸）",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="早停耐心轮数（基于 F1）",
    )
    parser.add_argument(
        "--no_point_adjust",
        action="store_true",
        help="关闭评估阶段的 point-adjust 策略（默认开启）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)

    logger = get_logger(args.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))

    logger.info(f"参数配置：{args}")

    # ------------ 构建多实体数据集（基于预处理结果） ------------ #
    dataset_name = args.dataset.lower()
    if dataset_name == "smd":
        train_ds, test_ds, input_dim, labels_list, entity_ids = build_smd_multi_datasets(
            processed_root=args.processed_root,
            win_size=args.win_size,
            train_stride=args.train_stride,
            test_stride=args.test_stride,
        )
        dataset_desc = "SMD 多机数据集"
    elif dataset_name == "msl":
        train_ds, test_ds, input_dim, labels_list, entity_ids = build_msl_multi_datasets(
            processed_root=args.processed_root,
            win_size=args.win_size,
            train_stride=args.train_stride,
            test_stride=args.test_stride,
        )
        dataset_desc = "MSL 多通道数据集"
    else:
        raise ValueError(f"不支持的数据集类型: {args.dataset}")

    logger.info(
        f"{dataset_desc} 构建完成：entities={entity_ids}, "
        f"input_dim={input_dim}, len(train)={len(train_ds)}, len(test)={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    # ------------ 构建模型（重构 / 预测分支） ------------ #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = args.task.lower()
    if task not in {"recon", "forecast"}:
        raise ValueError(f"不支持的任务类型: {args.task}")

    if task == "recon":
        model = mambatsad_ts_base(input_dim=input_dim)
        logger.info("任务类型：重构分支（recon）")
        context_len = None
    else:
        context_len = args.win_size - args.pred_len
        if context_len <= 0:
            raise ValueError(
                f"win_size={args.win_size} 必须大于 pred_len={args.pred_len} "
                "才能使用预测分支任务。"
            )
        model = mambatsad_ts_pred_base(
            input_dim=input_dim,
            seq_len=context_len,
            pred_len=args.pred_len,
        )
        logger.info(
            f"任务类型：预测分支（forecast），context_len={context_len}, pred_len={args.pred_len}"
        )

    model.to(device)
    logger.info(f"模型结构：\n{model}")

    # AMP GradScaler 使用新接口以去除 FutureWarning
    scaler: torch.amp.GradScaler | None = None
    if not args.no_amp and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 手工实现一个简单的早停逻辑（基于 F1）
    best_f1 = -1.0
    best_metrics: Dict[str, float] | None = None
    epochs_no_improve = 0
    best_ckpt_path = os.path.join(args.log_dir, "best_model.pt")

    use_point_adjust = not args.no_point_adjust

    # 选择对应的训练 / 评估函数
    if task == "recon":
        train_fn = train_one_epoch
        eval_fn = evaluate_multi
        train_kwargs = {}
        eval_kwargs = {
            "win_size": args.win_size,
            "labels_list": labels_list,
            "device": device,
            "use_point_adjust": use_point_adjust,
        }
    else:
        train_fn = train_one_epoch_forecast
        eval_fn = evaluate_multi_forecast
        train_kwargs = {
            "context_len": context_len,
            "pred_len": args.pred_len,
        }
        eval_kwargs = {
            "win_size": args.win_size,
            "pred_len": args.pred_len,
            "labels_list": labels_list,
            "device": device,
            "use_point_adjust": use_point_adjust,
        }

    # ------------ 训练 & 评估循环 ------------ #
    for epoch in range(1, args.epochs + 1):
        logger.info(f"========== Epoch {epoch}/{args.epochs} ==========")

        train_loss = train_fn(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm,
            logger=logger,
            **train_kwargs,
        )
        writer.add_scalar("train/loss", train_loss, epoch)

        scores, labels_full, thr, metrics = eval_fn(
            model,
            test_loader,
            num_thresholds=2048,
            **eval_kwargs,
        )

        logger.info(
            f"[Eval] F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"AUC={metrics['auc']:.4f}, thr={metrics['threshold']:.6f}, "
            f"point_adjust={metrics['use_point_adjust']}"
        )

        writer.add_scalar("eval/f1", metrics["f1"], epoch)
        writer.add_scalar("eval/precision", metrics["precision"], epoch)
        writer.add_scalar("eval/recall", metrics["recall"], epoch)
        writer.add_scalar("eval/auc", metrics["auc"], epoch)

        # 保存最优模型 & 早停
        if metrics["f1"] > best_f1 + 1e-4:
            best_f1 = metrics["f1"]
            best_metrics = metrics
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"发现更优模型，已保存至 {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(
                    f"早停触发：连续 {epochs_no_improve} 个 epoch F1 未提升，"
                    f"在第 {epoch} 个 epoch 停止训练。"
                )
                break

        # 保存一张当前 epoch 的分数与标签可视化图
        vis_path = os.path.join(args.log_dir, f"scores_epoch{epoch}.png")
        plot_scores_with_labels(
            scores=scores,
            labels=labels_full,
            threshold=thr,
            save_path=vis_path,
            max_points=2000,
        )
        logger.info(f"已保存可视化图：{vis_path}")

    logger.info(f"训练结束，最佳 F1={best_f1:.4f}, 最佳指标={best_metrics}")
    writer.close()


if __name__ == "__main__":
    main()
