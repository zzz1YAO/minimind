#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

bsz=4

python trainer/train_pretrain.py \
--epochs 2 \
--model_config ./configs/model/pretrain_0_6b.json \
--max_seq_len 2048 \
--tokenizer_path ./model \
--learning_rate 5e-4 \
--save_by epoch \
--save_interval 1 \
--log_interval 100 \
--use_wandb \
--wandb_project minigpt_0_6b \
--batch_size "$bsz" \
--accumulation_steps 4 \
--data_path ../pretrain_t2t.jsonl
