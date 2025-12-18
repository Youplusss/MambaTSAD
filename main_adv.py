# main_adv.py
# -*- coding: utf-8 -*-
"""
MambaTSAD 对抗训练入口（实验版，shared_adv 分支用）。

本脚本与 main.py 的设计保持一致：
1. 解析命令行参数；
2. 通过 mambatsad.data.build_multi_entity_dataset 构建数据集；
3. 调用 mambatsad.engine.trainer_adv.TSADAdvTrainer 进行「重构+预测」混合模型的
   对抗训练（仅支持 hybrid 分支）。

说明：
- 这里不再包含伪标签相关逻辑，只做 FGSM 风格的输入对抗扰动；
- 模型本身仍然复用原来的重构分支 / 预测分支（build_hybrid_model），
  只是训练时多了一条对抗样本的正则项。
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
        description="MambaTSAD：基于 Mamba 的时间序列异常检测（对抗训练实验版）"
    )

    # ================== 数据集相关 ==================
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
        help="预处理后的数据根目录，例如 ./dataset/SMD 或 ./dataset/MSL",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/exp_adv",
        help="实验日志与模型权重的保存目录",
    )

    # ================== 窗口 / 任务配置 ==================
    # 对抗训练版当前只支持混合模型（重构+预测）
    parser.add_argument(
        "--branch",
        type=str,
        default="hybrid",
        choices=["hybrid"],
        help="当前脚本仅支持 hybrid 分支（重构+预测混合模型）",
    )
    parser.add_argument(
        "--win_size",
        type=int,
        default=100,
        help="滑动窗口长度 L",
    )
    parser.add_argument(
        "--pred_len",
        type=int,
        default=10,
        help="预测步数 T_pred，仅 hybrid 分支中使用",
    )
    parser.add_argument(
        "--train_stride",
        type=int,
        default=1,
        help="训练集滑动窗口步长",
    )
    parser.add_argument(
        "--test_stride",
        type=int,
        default=1,
        help="测试集滑动窗口步长",
    )

    # ================== 训练超参数 ==================
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--no_amp",
        action="store_true",
        help="关闭混合精度训练（默认开启）",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="梯度裁剪阈值",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="早停轮数（基于 F1）",
    )

    # ================== 损失 / 评分系数（混合模型） ==================
    parser.add_argument(
        "--lambda_recon",
        type=float,
        default=1.0,
        help="重构分支损失权重",
    )
    parser.add_argument(
        "--lambda_forecast",
        type=float,
        default=1.0,
        help="预测分支损失权重",
    )

    # ================== 评估配置 ==================
    parser.add_argument(
        "--no_point_adjust",
        action="store_true",
        help="关闭 point-adjust 评估技巧（默认开启）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader 后台线程数",
    )

    # ================== 对抗训练相关 ==================
    parser.add_argument(
        "--use_adv_training",
        action="store_true",
        help="是否启用对抗训练（不加该开关则退化为普通混合模型训练）",
    )
    parser.add_argument(
        "--adv_epsilon",
        type=float,
        default=0.05,
        help="FGSM 对抗扰动步长（在输入空间，大致假设输入已 z-score 后方差~1）",
    )
    parser.add_argument(
        "--adv_beta",
        type=float,
        default=0.5,
        help="对抗损失在总 loss 中的权重系数 beta（loss = L_clean + beta * L_adv）",
    )
    parser.add_argument(
        "--adv_warmup_epochs",
        type=int,
        default=5,
        help="仅在该 epoch 之后才开始启用对抗训练（之前只做普通训练）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.win_size <= args.pred_len:
        raise ValueError(
            f"--win_size 必须大于 --pred_len，当前 win_size={args.win_size}, "
            f"pred_len={args.pred_len}"
        )

    os.makedirs(args.log_dir, exist_ok=True)
    logger = get_logger(args.log_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb_adv"))

    logger.info(f"[ADV] 命令行参数：{args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[ADV] 当前使用设备：{device}")

    # ============== 构建数据集 & DataLoader（与 main.py 一致） ==============
    train_ds, test_ds, input_dim, labels_list, entity_ids = build_multi_entity_dataset(
        name=args.dataset,
        processed_root=args.processed_root,
        win_size=args.win_size,
        train_stride=args.train_stride,
        test_stride=args.test_stride,
    )
    logger.info(
        f"[ADV] 数据集 [{args.dataset}] 构建完成，实体数={len(entity_ids)}，"
        f"input_dim={input_dim}，训练窗口数={len(train_ds)}，测试窗口数={len(test_ds)}"
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

    # ============== 构建对抗训练版 Trainer ==============
    trainer = TSADAdvTrainer(
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
        use_adv_training=args.use_adv_training,
        adv_epsilon=args.adv_epsilon,
        adv_beta=args.adv_beta,
        adv_warmup_epochs=args.adv_warmup_epochs,
    )

    trainer.train(num_epochs=args.epochs)
    writer.close()
    logger.info("[ADV] 训练流程结束。")


if __name__ == "__main__":
    main()