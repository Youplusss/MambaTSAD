#!/bin/bash
# 在 SMD 数据集上运行：原 hybrid 模型 + 输入对抗训练

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
  --lambda_recon 1.0 \
  --lambda_forecast 1.0 \
  --use_adv_training \
  --adv_epsilon 0.05 \
  --adv_beta 0.5 \
  --adv_warmup_epochs 5
