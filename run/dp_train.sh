#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

# NCCL_DEBUG=INFO \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# TORCH_NCCL_BLOCKING_WAIT=1 \
# NCCL_P2P_DISABLE=1 \
# NCCL_IB_DISABLE=1 \
# --ds_config ./configs/deepspeed/pretrain_zero2_2x48.json \
python trainer/train_pretrain.py \
--use_deepspeed 0 \
--model_config ./configs/model/pretrain_1b.json \
--tokenizer_path ../Qwen2.5-1.5B-tokenizer \
--data_path ../skypile/pretrain_skypile.jsonl \
--epochs 2 \
--max_seq_len 1024 \
--learning_rate 5e-4 \
--save_by epoch \
--save_interval 1 \
--log_interval 100 \
--use_wandb \
--wandb_project minigpt \
--batch_size 2 \
--accumulation_steps 8
