# main_adv.py
# -*- coding: utf-8 -*-
"""
实验入口：共享 encoder + 对抗训练 + 伪标签 的混合模型 TSAD。

与主分支 main.py 区别：
- 只支持 branch=hybrid_shared_adv 一种模式；
- 使用 MambaTSADHybridSharedAdv + AdvHybridTrainer；
- 支持以下实验开关：
    --use_adv_training   : 是否启用 STAMP 风格两阶段对抗训练；
    --use_pseudo_label   : 是否在训练前用 IsolationForest 做伪标签过滤；
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mambatsad.data import build_multi_entity_dataset
from mambatsad.engine.trainer_adv import AdvHybridTrainer
from mambatsad.models.hybrid_shared_adv import build_hybrid_shared_adv_model
from mambatsad.utils.logger import get_logger
from mambatsad.utils.seed import set_global_seed
from mambatsad.utils.pseudo_label import generate_pseudo_label_mask


def parse_args():
    parser = argparse.ArgumentParser(description="MambaTSAD 实验版：共享 encoder + 对抗训练 + 伪标签")

    parser.add_argument("--dataset", type=str, required=True,
                        choices=["smd", "msl", "swat", "wadi"],
                        help="数据集名称（与 master 保持一致）")
    parser.add_argument("--processed_root", type=str, required=True,
                        help="预处理数据根目录，如 ./dataset/SMD")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="日志与模型保存目录，如 ./logs/smd_adv_hybrid")

    parser.add_argument("--win_size", type=int, default=100,
                        help="滑动窗口长度")
    parser.add_argument("--pred_len", type=int, default=10,
                        help="预测步数")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--train_stride", type=int, default=1)
    parser.add_argument("--test_stride", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true",
                        help="关闭自动混合精度训练")

    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # 对抗训练相关
    parser.add_argument("--use_adv_training", action="store_true",
                        help="是否启用 STAMP 风格两阶段对抗训练")
    parser.add_argument("--adv_warmup_epochs", type=int, default=5,
                        help="对抗训练阶段1（联合优化）持续的 epoch 数")
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_pred", type=float, default=1.0)
    parser.add_argument("--lambda_adv1", type=float, default=0.5)
    parser.add_argument("--lambda_adv2", type=float, default=0.5)

    # 伪标签相关
    parser.add_argument("--use_pseudo_label", action="store_true",
                        help="是否在训练前使用 IsolationForest 生成伪标签，过滤可疑窗口")
    parser.add_argument("--pseudo_contamination", type=float, default=0.01,
                        help="伪标签阶段假定的异常比例，越大越激进")

    parser.add_argument("--no_point_adjust", action="store_true",
                        help="关闭 point-adjust 技巧")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir, name="MambaTSAD-ADV")

    logger.info(f"命令行参数：{args}")

    set_global_seed(args.seed, deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"当前使用的设备：{device}")

    # ---------------- 构建数据集 ----------------
    train_ds, test_ds, input_dim, labels_list, entity_ids = build_multi_entity_dataset(
        name=args.dataset,
        processed_root=args.processed_root,
        win_size=args.win_size,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
    )

    logger.info(
        f"数据集 [{args.dataset}] 构建完成，实体数量={len(entity_ids)}，"
        f"input_dim={input_dim}，训练样本数={len(train_ds)}，测试样本数={len(test_ds)}"
    )

    # 伪标签过滤：只保留「置信正常」窗口进行训练
    if args.use_pseudo_label:
        logger.info(
            f"开始使用 IsolationForest 生成伪标签，contamination={args.pseudo_contamination}"
        )
        mask = generate_pseudo_label_mask(
            dataset=train_ds,
            contamination=args.pseudo_contamination,
            random_state=args.seed,
        )
        keep_indices = np.where(mask)[0]
        logger.info(
            f"伪标签过滤结果：原训练窗口数={len(train_ds)}，"
            f"保留窗口数={len(keep_indices)} (比例={len(keep_indices)/len(train_ds):.3f})"
        )
        train_ds = Subset(train_ds, keep_indices)

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

    # ---------------- 构建模型 ----------------
    model = build_hybrid_shared_adv_model(
        input_dim=input_dim,
        win_size=args.win_size,
        pred_len=args.pred_len,
    )
    logger.info("模型结构：")
    logger.info(model)

    # ---------------- Trainer ----------------
    trainer = AdvHybridTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        labels_list=labels_list,
        log_dir=args.log_dir,
        win_size=args.win_size,
        pred_len=args.pred_len,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_amp=not args.no_amp,
        use_adv_training=args.use_adv_training,
        adv_warmup_epochs=args.adv_warmup_epochs,
        lambda_rec=args.lambda_rec,
        lambda_pred=args.lambda_pred,
        lambda_adv1=args.lambda_adv1,
        lambda_adv2=args.lambda_adv2,
        use_point_adjust=not args.no_point_adjust,
    )

    trainer.train(epochs=args.epochs, logger=logger)

    logger.info("实验版训练流程已结束。")


if __name__ == "__main__":
    main()
