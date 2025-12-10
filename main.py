# main.py
# -*- coding: utf-8 -*-
"""
MambaTSAD 统一训练 / 测试入口（重构版）

本文件只做三件事：
1. 解析命令行参数；
2. 构建数据集与数据加载器；
3. 调用 TSADTrainer 进行训练和评估。

真正的数据处理 / 模型定义 / 训练细节都在 mambatsad 包内部，
这样后续扩展数据集或模型时只需要改包内代码即可。
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mambatsad.data import build_multi_entity_dataset
from mambatsad.engine.trainer import TSADTrainer
from mambatsad.utils.logger import get_logger
from mambatsad.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这里尽量保持通用性：数据集、分支类型、训练超参都通过参数控制，
    方便后续扩展到新的数据集 / 模型。
    """
    parser = argparse.ArgumentParser(
        description="MambaTSAD：基于 Mamba 的时间序列异常检测（重构版）"
    )

    # ------------------ 数据集相关 ------------------
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default="smd",
    #     choices=["smd", "msl"],
    #     help="选择数据集名称，目前支持 smd / msl。",
    # )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["smd", "msl", "swat", "wadi", "smap"],
        help="数据集名称",
    )
    parser.add_argument(
        "--processed_root",
        type=str,
        required=True,
        help="预处理后的数据根目录，例如 ./dataset/SMD 或 ./dataset/MSL。",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/exp",
        help="实验日志与模型权重的保存目录。",
    )

    # ------------------ 窗口 / 任务配置 ------------------
    parser.add_argument(
        "--branch",
        type=str,
        default="recon",
        choices=["recon", "forecast", "hybrid"],
        help=(
            "训练分支类型："
            "recon=仅重构分支；forecast=仅预测分支；hybrid=重构+预测混合模型。"
        ),
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=100,
        help="滑动窗口长度 L（对所有分支统一）。",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=10,
        help="预测分支要预测的时间步数 T_pred（只在 forecast / hybrid 分支下生效）。",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="训练集滑动窗口步长（一般设为 1）。",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="测试集滑动窗口步长（可适当增大以加速推理）。",
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
        help="关闭混合精度训练（默认开启，如果显存充足建议保持开启）。",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值，防止梯度爆炸。",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="早停策略耐心轮数（基于 F1 分数）。",
    )

    # ------------------ 损失 / 评分系数（混合模型用） ------------------
    parser.add_argument(
        "--lambda_recon",
        type=float,
        default=1.0,
        help="混合模型中重构分支的损失权重（loss = λ_rec * L_rec + λ_pred * L_pred）。",
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=1.0,
        help="混合模型中预测分支的损失权重。",
    )

    # ------------------ 评估配置 ------------------
    parser.add_argument(
        "--no_point_adjust",
        action="store_true",
        help="关闭评估阶段的 point-adjust 技巧（默认开启）。",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 使用的后台线程数。",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 固定随机种子，保证实验可复现
    set_global_seed(args.seed)

    # 日志与 TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))

    logger.info(f"命令行参数：{args}")

    # ------------------ 构建数据集 & DataLoader ------------------
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
    logger.info(f"当前使用的设备：{device}")

    trainer = TSADTrainer(
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
    )

    trainer.train(num_epochs=args.epochs)

    writer.close()
    logger.info("训练流程已结束。")


if __name__ == "__main__":
    main()
