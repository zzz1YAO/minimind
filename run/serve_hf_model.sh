#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="MiniMind2-0_6B"

python scripts/serve_openai_api.py \
  --load_from "${MODEL_PATH}"
