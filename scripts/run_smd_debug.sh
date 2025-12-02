#!/usr/bin/env bash
# 快速小规模训练，验证环境是否正常

DATA_ROOT=./data/ServerMachineDataset
MACHINE_ID=machine-1-1
LOG_DIR=./logs/debug_smd_${MACHINE_ID}

CUDA_VISIBLE_DEVICES=0 python main_smd.py \
    --data_root ${DATA_ROOT} \
    --machine_id ${MACHINE_ID} \
    --log_dir ${LOG_DIR} \
    --win_size 50 \
    --train_stride 5 \
    --test_stride 10 \
    --batch_size 32 \
    --epochs 3 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --no_amp
