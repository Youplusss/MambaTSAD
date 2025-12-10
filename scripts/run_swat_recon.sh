#!/bin/bash
# 在 SWAT 数据集上训练仅重构分支的 MambaTSAD 模型

DATA_ROOT=./dataset/SWAT
LOG_DIR=./logs/swat_recon

python -u main.py \
  --dataset swat \
  --branch recon \
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
  --seed 42
