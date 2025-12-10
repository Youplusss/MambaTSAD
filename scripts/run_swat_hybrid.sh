#!/bin/bash
# 在 SWAT 数据集上训练混合模型（重构 + 预测）

DATA_ROOT=./dataset/SWaT
LOG_DIR=./logs/swat_hybrid

CUDA_VISIBLE_DEVICES=7 python -u main.py \
  --dataset swat \
  --branch hybrid \
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
  --lambda_recon 1.0 \
  --lambda_forecast 1.0 \
  --seed 42
