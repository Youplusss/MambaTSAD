#!/usr/bin/env bash

PROCESSED_ROOT=./dataset/MSL
LOG_DIR=./logs/smd_forecast

CUDA_VISIBLE_DEVICES=2 python main_pred.py \
  --task forecast \
  --dataset smd \
  --processed_root ${PROCESSED_ROOT} \
  --log_dir ${LOG_DIR} \
  --win_size 100 \
  --pred_len 10
