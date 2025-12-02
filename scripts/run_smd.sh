#!/usr/bin/env bash
# 先确保已经预处理：
# python tools/preprocess_smd.py \
#   --raw_root ./data/ServerMachineDataset \
#   --out_root ./data_processed/SMD \
#   --use_global_scaler

PROCESSED_ROOT=./dataset/SMD
LOG_DIR=./logs/smd_all

CUDA_VISIBLE_DEVICES=0 python main_smd.py \
  --processed_root ${PROCESSED_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --train_stride 1 \
  --test_stride 5 \
  --batch_size 64 \
  --epochs 50 \
  --lr 1e-3 \
  --weight_decay 1e-4
