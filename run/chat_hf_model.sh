#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="MiniMind2-0_6B"
DEVICE="cuda"
MAX_NEW_TOKENS="1024"
TEMPERATURE="0.7"
TOP_P="0.9"
HISTORY_TURNS="0"
DO_SAMPLE="1"
PROMPT_MODE="pretrain"
DTYPE="float16"

python scripts/chat_hf_cli.py \
  --model_path "${MODEL_PATH}" \
  --device "${DEVICE}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --history_turns "${HISTORY_TURNS}" \
  --do_sample "${DO_SAMPLE}" \
  --prompt_mode "${PROMPT_MODE}" \
  --dtype "${DTYPE}"
