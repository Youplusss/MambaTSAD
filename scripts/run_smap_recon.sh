#!/bin/bash
# SMAP 数据集 + 重构分支（recon）训练脚本

set -e

DATA_ROOT="./dataset/SMAP"
LOG_ROOT="./logs/smap_recon"

mkdir -p "${LOG_ROOT}"

python main.py \
  --dataset smap \
  --processed_root "${DATA_ROOT}" \
  --branch recon \
  --win_size 100 \
  --pred_len 10 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --train_stride 1 \
  --test_stride 1 \
  --log_dir "${LOG_ROOT}" \
  --lambda_recon 1.0 \
  --lambda_forecast 1.0

