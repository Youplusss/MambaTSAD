#!/bin/bash
# 在 SMD 数据集上运行 实验版（共享 encoder + 对抗训练 + 伪标签）混合模型

DATA_ROOT=./dataset/WADI
LOG_DIR=./logs/wadi_adv_hybrid

CUDA_VISIBLE_DEVICES=3 python main_adv.py \
  --dataset wadi \
  --processed_root ${DATA_ROOT} \
  --log_dir ${LOG_DIR} \
  --branch hybrid_shared_adv \
  --win_size 100 \
  --pred_len 10 \
  --train_stride 1 \
  --test_stride 1 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --lambda_recon 1.0 \
  --lambda_forecast 1.0 \
  --adv_epsilon 0.05 \
  --adv_beta 0.5 \
  --adv_warmup_epochs 5
