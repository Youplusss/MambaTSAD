# main_adv.py
# -*- coding: utf-8 -*-
"""
shared_adv 分支的入口脚本。

与 main.py 的设计保持一致：
- 通过 mambatsad.data.build_multi_entity_dataset 构建统一的数据加载流程；
- 训练 / 评估逻辑全部放在 mambatsad.engine.trainer_adv.TSADAdvTrainer 中；
- 只在 TSADTrainer 的混合模型 (hybrid) 基础上，额外加入输入级对抗训练，
  不再单独维护一套“脱离原有结构”的模型代码。

当前脚本仅支持 branch="hybrid_shared_adv"。
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mambatsad.data import build_multi_entity_dataset
from mambatsad.engine.trainer_adv import TSADAdvTrainer
from mambatsad.utils.logger import get_logger
from mambatsad.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MambaTSAD shared_adv：混合模型 + 对抗训练"
    )

    # ------------------ 数据集相关 ------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["smd", "msl", "swat", "wadi", "smap"],
        help="数据集名称",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="预处理后的数据根目录，例如 ./dataset/SMD / ./dataset/MSL / ./dataset/WADI 等。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/exp_adv",
        help="实验日志与模型权重保存目录。",
    )

    # ------------------ 窗口 / 分支配置 ------------------
    parser.add_argument(
        "--branch",
        type=str,
        default="hybrid_shared_adv",
        choices=["hybrid_shared_adv"],
        help="当前脚本仅支持 hybrid_shared_adv 分支（在原 hybrid 模型上叠加对抗训练）。",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=100,
        help="滑动窗口长度 L。",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=10,
        help="预测步数 T_pred，仅用于混合/预测分支。",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="训练集滑窗步长。",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="测试集滑窗步长。",
    )

    # ------------------ 训练超参数 ------------------
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭混合精度训练（默认开启）。",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="早停轮数（基于 F1）。",
    )

    # ------------------ 多任务损失权重 ------------------
    parser.add_argument(
        "--lambda_recon",
        type=float,
        default=1.0,
        help="重构分支损失权重。",
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=1.0,
        help="预测分支损失权重。",
    )

    # ------------------ 对抗训练超参数 ------------------
    parser.add_argument(
        "--adv_epsilon",
        type=float,
        default=0.05,
        help="FGSM 对抗扰动步长（相对于输入尺度）。",
    )
    parser.add_argument(
        "--adv_beta",
        type=float,
        default=0.5,
        help="对抗损失权重系数。",
    )
    parser.add_argument(
        "--adv_warmup_epochs",
        type=int,
        default=5,
        help="前多少个 epoch 仅用干净样本训练，不开启对抗训练。",
    )

    # ------------------ 评估配置 ------------------
    parser.add_argument(
        "--no_point_adjust",
        action="store_true",
        help="关闭 point-adjust 评估策略（默认开启）。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 后台线程数。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 固定随机种子
    set_global_seed(args.seed)

    # 日志 & TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))

    logger.info(f"[shared_adv] 命令行参数：{args}")

    if args.win_size <= args.pred_len:
        raise ValueError(
            f"win_size 必须大于 pred_len，当前 win_size={args.win_size}, pred_len={args.pred_len}"
        )

    # ------------------ 构建数据集 & DataLoader ------------------
    train_ds, test_ds, input_dim, labels_list, entity_ids = build_multi_entity_dataset(
        name=args.dataset,
        processed_root=args.processed_root,
        win_size=args.win_size,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
    )

    logger.info(
        "[shared_adv] 数据集 [%s] 构建完成，实体数量=%d，"
        "input_dim=%d，训练样本数=%d，测试样本数=%d",
        args.dataset,
        len(entity_ids),
        input_dim,
        len(train_ds),
        len(test_ds),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ------------------ 构建 Trainer 并启动训练 ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[shared_adv] 当前使用设备：{device}")

    trainer = TSADAdvTrainer(
        branch=args.branch,
        device=device,
        input_dim=input_dim,
        win_size=args.win_size,
        pred_len=args.pred_len,
        train_loader=train_loader,
        test_loader=test_loader,
        labels_list=labels_list,
        entity_ids=entity_ids,
        logger=logger,
        writer=writer,
        log_dir=args.log_dir,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        use_point_adjust=(not args.no_point_adjust),
        use_amp=(not args.no_amp),
        lambda_recon=args.lambda_recon,
        lambda_forecast=args.lambda_forecast,
        adv_epsilon=args.adv_epsilon,
        adv_beta=args.adv_beta,
        adv_warmup_epochs=args.adv_warmup_epochs,
    )

    trainer.train(num_epochs=args.epochs)

    writer.close()
    logger.info("[shared_adv] 训练流程已结束。")


if __name__ == "__main__":
    main()
