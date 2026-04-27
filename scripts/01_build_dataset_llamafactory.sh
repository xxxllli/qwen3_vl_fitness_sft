#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/default.yaml}
FORMAT=${2:-conversations}

python -m qwen3vl_trigger.data.build_llamafactory_dataset --config "$CONFIG" --format "$FORMAT"
