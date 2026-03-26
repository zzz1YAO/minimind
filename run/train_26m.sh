#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

bsz=48
save_step=$((4000000 / (bsz * 2)))

python trainer/train_pretrain.py \
--epochs 2 \
--model_config ./configs/model/pretrain_26m.json \
--max_seq_len 2048 \
--tokenizer_path ./model \
--learning_rate 5e-4 \
--save_interval "$save_step" \
--log_interval 100 \
--use_wandb \
--wandb_project minigpt_26m \
--batch_size "$bsz" \
--accumulation_steps 4 \
--data_path ../pretrain_t2t.jsonl
