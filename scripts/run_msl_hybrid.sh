#!/bin/bash
# 在 MSL 数据集上训练混合模型

DATA_ROOT=./dataset/MSL
LOG_DIR=./logs/msl_hybrid

CUDA_VISIBLE_DEVICES=6 python -u main.py \
  --dataset msl \
  --branch hybrid \
  --processed_root ${DATA_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --pred_len 10 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --train_stride 1 \
  --test_stride 1 \
  --lambda_recon 1.0 \
  --lambda_forecast 1.0 \
  --seed 42
