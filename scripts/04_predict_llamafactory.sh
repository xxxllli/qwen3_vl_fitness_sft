#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}
ADAPTER=${2:-}
SPLIT=${3:-}

ARGS=(--config "$CONFIG")
if [ -n "$ADAPTER" ]; then
  ARGS+=(--adapter-path "$ADAPTER")
fi
if [ -n "$SPLIT" ]; then
  ARGS+=(--split "$SPLIT")
fi

python -m qwen3vl_trigger.infer.predict_llamafactory_lora "${ARGS[@]}"
