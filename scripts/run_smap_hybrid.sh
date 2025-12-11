#!/bin/bash
# SMAP 数据集 + 混合分支（hybrid：重构+预测）训练脚本

DATA_ROOT="./dataset/SMAP"
LOG_ROOT="./logs/smap_hybrid"

CUDA_VISIBLE_DEVICES=1 python main.py \
  --dataset smap \
  --processed_root "${DATA_ROOT}" \
  --branch hybrid \
  --win_size 100 \
  --pred_len 10 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-4 \
  --weight_decay 5e-4 \
  --train_stride 1 \
  --test_stride 1 \
  --log_dir "${LOG_ROOT}" \
  --lambda_recon 1.0 \
  --lambda_forecast 1.0 \
  --seed 42

