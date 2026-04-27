#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
PRED=${2:-}
if [ -z "$PRED" ]; then
  python -m qwen3vl_trigger.eval.evaluate --config "$CONFIG"
else
  python -m qwen3vl_trigger.eval.evaluate --config "$CONFIG" --predictions "$PRED"
fi
