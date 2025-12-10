#!/bin/bash
# 在 SMD 数据集上运行 实验版（共享 encoder + 对抗训练 + 伪标签）混合模型

DATA_ROOT=./dataset/SMD
LOG_DIR=./logs/smd_adv_hybrid

CUDA_VISIBLE_DEVICES=6 python -u main_adv.py \
  --dataset smd \
  --processed_root ${DATA_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --pred_len 10 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --train_stride 1 \
  --test_stride 1 \
  --use_adv_training \
  --adv_warmup_epochs 5 \
  --lambda_rec 1.0 \
  --lambda_pred 1.0 \
  --lambda_adv1 0.5 \
  --lambda_adv2 0.5 \
  --use_pseudo_label \
  --pseudo_contamination 0.01 \
  --seed 42
