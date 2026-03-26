#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

TORCH_PATH="out/full_sft_1536.pth"
TRANSFORMERS_PATH="MiniMind2-0_6B"
MODEL_CONFIG="configs/model/pretrain_0_6b.json"
TOKENIZER_PATH="model"
EXPORT_FORMAT="auto"
DTYPE="float16"

python scripts/convert_model.py \
  --torch_path "${TORCH_PATH}" \
  --transformers_path "${TRANSFORMERS_PATH}" \
  --model_config "${MODEL_CONFIG}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --format "${EXPORT_FORMAT}" \
  --dtype "${DTYPE}"
