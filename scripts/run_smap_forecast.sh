#!/bin/bash
  --lambda_forecast 1.0
  --lambda_recon 1.0 \
  --log_dir "${LOG_ROOT}" \
  --test_stride 1 \
  --train_stride 1 \
  --weight_decay 1e-4 \
  --lr 1e-3 \
  --epochs 50 \
  --batch_size 64 \
  --pred_len 10 \
  --win_size 100 \
  --branch forecast \
  --processed_root "${DATA_ROOT}" \
  --dataset smap \
python main.py \

mkdir -p "${LOG_ROOT}"

LOG_ROOT="./logs/smap_forecast"
DATA_ROOT="./dataset/SMAP"

set -e

# SMAP 数据集 + 预测分支（forecast）训练脚本

