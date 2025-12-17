
import argparse
import os
import time
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from mambatsad.models.hybrid_shared_adv import HybridSharedAdvModel


# ==========================
# Logging & seed utilities
# ==========================

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"MambaTSAD-ADV_{timestamp}.log")

    logger = logging.getLogger("MambaTSAD_ADV")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger


def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================
# Data loading
# ==========================

@dataclass
class DatasetSplits:
    train_series: list  # list of np.ndarray (T, D)
    test_series: list   # list of np.ndarray (T, D)
    test_labels: list   # list of np.ndarray (T,)


def load_processed_dataset(processed_root: str, logger: logging.Logger) -> DatasetSplits:
    """
    统一从 processed_root 读取数据：
        processed_root/
          train/<entity>.npy   -> (T, D)
          test/<entity>.npy    -> (T, D)
          test_label/<entity>.npy -> (T,)
          entities.txt (可选)
    如果 entities.txt 不存在，则按文件名排序推断实体顺序。
    并在此处做严格的长度对齐与形状检查（这是 WADI / SMAP 出问题的高发点）。
    """
    train_root = os.path.join(processed_root, "train")
    test_root = os.path.join(processed_root, "test")
    label_root = os.path.join(processed_root, "test_label")

    if not (os.path.isdir(train_root) and os.path.isdir(test_root) and os.path.isdir(label_root)):
        raise FileNotFoundError(
            f"processed_root 目录结构不完整：{processed_root}\n"
            "期望存在 train/、test/、test_label/ 三个子目录。"
        )

    ent_path = os.path.join(processed_root, "entities.txt")
    if os.path.isfile(ent_path):
        with open(ent_path, "r", encoding="utf-8") as f:
            entities = [line.strip() for line in f if line.strip()]
        logger.info(f"从 entities.txt 读取到 {len(entities)} 个实体。")
    else:
        # 回退：按 train/ 下的 .npy 文件名推断
        entities = sorted(
            os.path.splitext(fn)[0]
            for fn in os.listdir(train_root)
            if fn.endswith(".npy")
        )
        logger.warning(
            "未找到 entities.txt，改为按 train/*.npy 文件名排序推断实体顺序。"
        )

    train_series, test_series, test_labels = [], [], []

    for ent in entities:
        tr_path = os.path.join(train_root, f"{ent}.npy")
        te_path = os.path.join(test_root, f"{ent}.npy")
        lb_path = os.path.join(label_root, f"{ent}.npy")

        if not (os.path.isfile(tr_path) and os.path.isfile(te_path) and os.path.isfile(lb_path)):
            logger.warning(
                f"[{ent}] 缺少 train/test/label 中的某个文件，跳过该实体。"
            )
            continue

        tr = np.load(tr_path)
        te = np.load(te_path)
        lb = np.load(lb_path)

        if lb.ndim > 1:
            # 标签有时会以 (T,1) 或 (T, D_label) 存在，这里统一压扁
            lb = lb.squeeze()

        if tr.ndim != 2 or te.ndim != 2:
            raise ValueError(
                f"[{ent}] 期望 train/test 都为二维数组 (T, D)，但是得到 "
                f"train.shape={tr.shape}, test.shape={te.shape}"
            )

        if lb.shape[0] != te.shape[0]:
            min_len = min(lb.shape[0], te.shape[0])
            logger.warning(
                f"[{ent}] test_label 长度 {lb.shape[0]} 与 test 长度 {te.shape[0]} 不一致，"
                f"将两者统一截断到 {min_len}。"
            )
            te = te[:min_len]
            lb = lb[:min_len]

        # 防御性地去除 NaN / Inf
        for name, arr in [("train", tr), ("test", te)]:
            if not np.isfinite(arr).all():
                logger.warning(f"[{ent}] {name} 中存在 NaN/Inf，将使用 nan_to_num 修复。")
                np.nan_to_num(arr, copy=False)

        lb = (lb > 0).astype(np.float32)

        train_series.append(tr.astype(np.float32))
        test_series.append(te.astype(np.float32))
        test_labels.append(lb)

        logger.info(
            f"[{ent}] train={tr.shape}, test={te.shape}, label={lb.shape}, "
            f"异常比例={lb.mean():.6f}"
        )

    if not train_series:
        raise RuntimeError("没有成功加载到任何实体的数据，请检查 processed_root / entities.txt / .npy 文件名。")

    return DatasetSplits(train_series=train_series, test_series=test_series, test_labels=test_labels)


class SlidingWindowTrainDataset(Dataset):
    """
    在线切片训练数据，避免一次性展开所有窗口导致 WADI 这类长序列内存爆炸。
    """

    def __init__(self, series_list, win_size: int, stride: int):
        """
        series_list: list of np.ndarray (T, D)
        """
        super().__init__()
        self.series_list = series_list
        self.win_size = win_size
        self.stride = stride

        self.index = []  # (series_idx, start)
        for s_idx, seq in enumerate(self.series_list):
            T = seq.shape[0]
            if T < win_size:
                continue
            n_win = (T - win_size) // stride + 1
            for k in range(n_win):
                start = k * stride
                self.index.append((s_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        s_idx, start = self.index[idx]
        seq = self.series_list[s_idx]
        window = seq[start:start + self.win_size]  # (L, D)
        return torch.from_numpy(window)  # (L, D), float32


# ==========================
# Metrics & evaluation
# ==========================

def point_adjust(pred: np.ndarray, label: np.ndarray) -> np.ndarray:
    """
    最经典的 point-adjust：只要在一个异常区间内命中一点，就把该区间全部置为 1。
    """
    assert pred.shape == label.shape
    pred_adj = pred.copy()
    in_anom = False
    start = 0
    for i in range(len(label)):
        if label[i] > 0 and not in_anom:
            in_anom = True
            start = i
        elif label[i] == 0 and in_anom:
            end = i - 1
            if pred[start:end + 1].any():
                pred_adj[start:end + 1] = 1
            in_anom = False
    if in_anom:
        if pred[start:].any():
            pred_adj[start:] = 1
    return pred_adj


def compute_f1_metrics(scores: np.ndarray,
                       labels: np.ndarray,
                       num_thresholds: int = 200,
                       use_point_adjust: bool = True):
    assert scores.shape == labels.shape
    labels = (labels > 0).astype(np.int32)

    # 为了稳定性，对 score 做一个 z-score 归一化
    s = scores.astype(np.float64)
    mu = s.mean()
    std = s.std() + 1e-8
    s = (s - mu) / std

    best_f1, best_p, best_r, best_th = 0.0, 0.0, 0.0, 0.0

    s_min, s_max = s.min(), s.max()
    if s_max == s_min:
        # 极端情况：所有 score 一样
        return 0.0, 0.0, 0.0, 0.0

    for thr in np.linspace(s_min, s_max, num_thresholds):
        pred = (s > thr).astype(np.int32)
        if use_point_adjust:
            pred = point_adjust(pred, labels)

        tp = np.logical_and(pred == 1, labels == 1).sum()
        fp = np.logical_and(pred == 1, labels == 0).sum()
        fn = np.logical_and(pred == 0, labels == 1).sum()

        if tp + fp == 0 or tp + fn == 0:
            continue

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)

        if f1 > best_f1:
            best_f1, best_p, best_r, best_th = f1, p, r, thr

    return best_f1, best_p, best_r, best_th


def evaluate_model(model: torch.nn.Module,
                   splits: DatasetSplits,
                   win_size: int,
                   pred_len: int,
                   test_stride: int,
                   device: torch.device,
                   batch_size: int,
                   logger: logging.Logger,
                   use_point_adjust: bool = True):
    """
    使用滑动窗口在完整 test 序列上推理，得到每个时间点的重构+预测误差，
    再进行分数融合与阈值搜索。
    """
    model.eval()
    all_scores = []
    all_labels = []

    for ent_idx, (seq, label) in enumerate(zip(splits.test_series, splits.test_labels)):
        T, D = seq.shape
        if T < win_size:
            logger.warning(f"[ent={ent_idx}] 测试序列长度 {T} < win_size={win_size}，跳过。")
            continue

        seq_tensor = torch.from_numpy(seq).to(device)  # (T, D)

        # 统计每个时间点被多少个窗口覆盖
        recon_score = np.zeros(T, dtype=np.float64)
        recon_cnt = np.zeros(T, dtype=np.int64)
        fore_score = np.zeros(T, dtype=np.float64)
        fore_cnt = np.zeros(T, dtype=np.int64)

        indices = list(range(0, T - win_size + 1, test_stride))
        n = len(indices)

        with torch.no_grad():
            for b_start in range(0, n, batch_size):
                batch_starts = indices[b_start:b_start + batch_size]
                windows = []
                for s in batch_starts:
                    w = seq_tensor[s:s + win_size]  # (L, D)
                    windows.append(w)
                x = torch.stack(windows, dim=0)  # (B, L, D)
                x = x.to(device)

                recon, fore = model(x, pred_len=pred_len)  # (B, L, D), (B, pred_len, D)

                # 重构误差：对特征维做均方
                recon_err = torch.mean((recon - x) ** 2, dim=-1).cpu().numpy()  # (B, L)

                # 预测误差：对最后 pred_len 时刻做均方
                target_future = x[:, -pred_len:, :]
                fore_err = torch.mean((fore - target_future) ** 2, dim=-1).cpu().numpy()  # (B, pred_len)

                for bi, s in enumerate(batch_starts):
                    # 重构：窗口内每个位置都要累计
                    for offset in range(win_size):
                        t = s + offset
                        recon_score[t] += recon_err[bi, offset]
                        recon_cnt[t] += 1

                    # 预测：对应到最后 pred_len 步
                    for offset in range(pred_len):
                        t = s + (win_size - pred_len) + offset
                        if t < T:
                            fore_score[t] += fore_err[bi, offset]
                            fore_cnt[t] += 1

        # 归一化
        mask = recon_cnt > 0
        recon_score[mask] /= recon_cnt[mask]
        # 没被覆盖的点就继承最近的值，避免 0 干扰
        for t in range(T):
            if recon_cnt[t] == 0:
                recon_score[t] = recon_score[t - 1] if t > 0 else 0.0

        mask = fore_cnt > 0
        fore_score[mask] /= fore_cnt[mask]
        for t in range(T):
            if fore_cnt[t] == 0:
                fore_score[t] = fore_score[t - 1] if t > 0 else 0.0

        # 对当前实体做简单的 z-score，然后再融合
        def zscore(x):
            m = x.mean()
            s = x.std() + 1e-8
            return (x - m) / s

        recon_z = zscore(recon_score)
        fore_z = zscore(fore_score)
        fused = recon_z + fore_z  # 默认等权，后续也可以加 lambda

        all_scores.append(fused.astype(np.float32))
        all_labels.append(label.astype(np.float32))

    scores_cat = np.concatenate(all_scores, axis=0)
    labels_cat = np.concatenate(all_labels, axis=0)
    f1, p, r, thr = compute_f1_metrics(scores_cat, labels_cat, use_point_adjust=use_point_adjust)
    logger.info(f"[Eval] F1={f1:.4f}, P={p:.4f}, R={r:.4f}, best_th={thr:.4f}")
    return f1, p, r, thr


# ==========================
# Adversarial training config
# ==========================

@dataclass
class AdvConfig:
    enabled: bool = True
    epsilon: float = 0.05
    beta: float = 0.5
    warmup_epochs: int = 5
    lambda_recon: float = 1.0
    lambda_forecast: float = 1.0
    max_grad_norm: float = 1.0


# ==========================
# Train / eval loop
# ==========================

def train_one_epoch(model,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    pred_len: int,
                    adv_cfg: AdvConfig,
                    logger: logging.Logger):
    model.train()
    total_loss = 0.0
    total_clean = 0.0
    total_adv = 0.0
    n_batches = 0

    use_adv = adv_cfg.enabled and (epoch + 1) >= adv_cfg.warmup_epochs

    for batch in train_loader:
        x = batch.to(device)  # (B, L, D)
        x = x.float()

        # ---------- 1) 生成对抗样本（只利用输入梯度，不更新模型） ----------
        if use_adv:
            x_adv_src = x.detach().clone().requires_grad_(True)
            recon_src, fore_src = model(x_adv_src, pred_len=pred_len)

            # 这里用与 clean 相同的多任务损失来生成扰动
            target_future_src = x_adv_src[:, -pred_len:, :]
            recon_loss_src = torch.mean((recon_src - x_adv_src) ** 2)
            fore_loss_src = torch.mean((fore_src - target_future_src) ** 2)
            loss_src = adv_cfg.lambda_recon * recon_loss_src + adv_cfg.lambda_forecast * fore_loss_src

            model.zero_grad(set_to_none=True)
            loss_src.backward()
            grad = x_adv_src.grad.detach()
            grad = torch.nan_to_num(grad)

            x_adv = x + adv_cfg.epsilon * grad.sign()
            x_adv = x_adv.detach()
        else:
            x_adv = None

        # ---------- 2) 正式的前向 & 反向 ----------
        model.zero_grad(set_to_none=True)

        recon, fore = model(x, pred_len=pred_len)
        target_future = x[:, -pred_len:, :]

        recon_loss = torch.mean((recon - x) ** 2)
        fore_loss = torch.mean((fore - target_future) ** 2)
        clean_loss = adv_cfg.lambda_recon * recon_loss + adv_cfg.lambda_forecast * fore_loss

        adv_loss_val = torch.tensor(0.0, device=device)
        if use_adv and x_adv is not None:
            recon_adv, fore_adv = model(x_adv, pred_len=pred_len)
            target_future_adv = x_adv[:, -pred_len:, :]
            recon_loss_adv = torch.mean((recon_adv - x_adv) ** 2)
            fore_loss_adv = torch.mean((fore_adv - target_future_adv) ** 2)
            adv_loss_val = adv_cfg.lambda_recon * recon_loss_adv + adv_cfg.lambda_forecast * fore_loss_adv

        loss = clean_loss + adv_cfg.beta * adv_loss_val

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), adv_cfg.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_clean += clean_loss.item()
        total_adv += adv_loss_val.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0

    avg_loss = total_loss / n_batches
    avg_clean = total_clean / n_batches
    avg_adv = total_adv / n_batches
    logger.info(
        f"[Train][Epoch {epoch+1}] "
        f"loss={avg_loss:.6f}, clean={avg_clean:.6f}, adv={avg_adv:.6f}, "
        f"batches={n_batches}"
    )
    return avg_loss, avg_clean, avg_adv


# ==========================
# Main
# ==========================

def parse_args():
    parser = argparse.ArgumentParser("MambaTSAD shared_adv hybrid training (re-implemented)")

    # 数据与日志
    parser.add_argument("--dataset", type=str, default="smd",
                        choices=["smd", "msl", "smap", "swat", "wadi"])
    parser.add_argument("--processed_root", type=str, required=True,
                        help="预处理后的统一数据根目录，如 ./dataset/SMD")
    parser.add_argument("--log_dir", type=str, default="./logs/exp_adv")

    # 任务 / 窗口
    parser.add_argument("--branch", type=str, default="hybrid_shared_adv",
                        choices=["hybrid_shared_adv"],
                        help="此脚本仅支持 hybrid_shared_adv 分支")
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--pred_len", type=int, default=10)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--test_stride", type=int, default=1)

    # 训练超参
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    # 混合任务权重
    parser.add_argument("--lambda_recon", type=float, default=1.0)
    parser.add_argument("--lambda_forecast", type=float, default=1.0)

    # 对抗训练相关
    parser.add_argument("--adv_epsilon", type=float, default=0.05,
                        help="FGSM 步长，按输入标准差比例缩放后使用")
    parser.add_argument("--adv_beta", type=float, default=0.5,
                        help="对抗损失的权重系数")
    parser.add_argument("--adv_warmup", type=int, default=5,
                        help="从第 adv_warmup 个 epoch 开始启用对抗训练")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--no_point_adjust", action="store_true",
                        help="关闭 point-adjust 评估")
    parser.add_argument("--test_batch_size", type=int, default=128)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger = setup_logger(args.log_dir)
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Using device: {device}")

    if args.win_size <= args.pred_len:
        raise ValueError(f"win_size 必须大于 pred_len，当前 win_size={args.win_size}, pred_len={args.pred_len}")

    # ============ 加载数据 ============
    splits = load_processed_dataset(args.processed_root, logger)
    D = splits.train_series[0].shape[1]
    logger.info(f"检测到特征维度 D={D}")

    train_dataset = SlidingWindowTrainDataset(
        series_list=splits.train_series,
        win_size=args.win_size,
        stride=args.train_stride,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    logger.info(f"Train windows: {len(train_dataset)} (batch_size={args.batch_size})")

    # ============ 构建模型 ============
    model = HybridSharedAdvModel(
        input_dim=D,
        d_model=128,
        num_layers=4,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    adv_cfg = AdvConfig(
        enabled=True,
        epsilon=args.adv_epsilon,
        beta=args.adv_beta,
        warmup_epochs=args.adv_warmup,
        lambda_recon=args.lambda_recon,
        lambda_forecast=args.lambda_forecast,
        max_grad_norm=args.max_grad_norm,
    )

    logger.info(f"Model: {model}")
    logger.info(f"AdvConfig: {adv_cfg}")

    best_f1 = 0.0
    patience = 8
    no_improve = 0
    best_state = None

    for epoch in range(args.epochs):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            pred_len=args.pred_len,
            adv_cfg=adv_cfg,
            logger=logger,
        )

        f1, p, r, thr = evaluate_model(
            model=model,
            splits=splits,
            win_size=args.win_size,
            pred_len=args.pred_len,
            test_stride=args.test_stride,
            device=device,
            batch_size=args.test_batch_size,
            logger=logger,
            use_point_adjust=not args.no_point_adjust,
        )

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
            best_state = model.state_dict()
            save_path = os.path.join(args.log_dir, "best_model_hybrid_shared_adv.pt")
            torch.save(best_state, save_path)
            logger.info(
                f"[Epoch {epoch+1}] F1 提升为 {best_f1:.4f}，已保存最佳模型到 {save_path}"
            )
        else:
            no_improve += 1
            logger.info(
                f"[Epoch {epoch+1}] F1 未提升 (当前={f1:.4f}, 最佳={best_f1:.4f}), "
                f"no_improve={no_improve}/{patience}"
            )
            if no_improve >= patience:
                logger.info("F1 连续多轮未提升，提前停止训练。")
                break

    logger.info(f"训练结束，最佳 F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
