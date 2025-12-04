#!/usr/bin/env bash
# scripts/run_msl.sh
# 先确保已经预处理：
# python tools/preprocess_msl.py \
#   --raw_root ./data/MSL \
#   --out_root ./dataset/MSL \
#   --use_global_scaler

PROCESSED_ROOT=./dataset/MSL
LOG_DIR=./logs/msl

CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset msl \
  --processed_root ${PROCESSED_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --train_stride 1 \
  --test_stride 5 \
  --batch_size 64 \
  --epochs 50 \
  --lr 5e-5 \
  --weight_decay 5e-3 \
  --no_amp
