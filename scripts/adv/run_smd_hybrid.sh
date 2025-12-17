#!/bin/bash
# 在 SMD 数据集上运行 实验版（共享 encoder + 对抗训练 + 伪标签）混合模型

DATA_ROOT=./dataset/SMD
LOG_DIR=./logs/smd_adv_hybrid

CUDA_VISIBLE_DEVICES=2 python main_adv.py \
  --dataset smd \
  --processed_root ${DATA_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --pred_len 10 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --adv_epsilon 0.05 \
  --adv_beta 0.5 \
  --adv_warmup 5

