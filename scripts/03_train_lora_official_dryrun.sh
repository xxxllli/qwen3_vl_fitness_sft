#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-configs/default.yaml}
python -m qwen3vl_trigger.training.run_official_train --config "$CONFIG" --dry-run
