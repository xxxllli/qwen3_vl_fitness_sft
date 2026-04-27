#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/llamafactory_qwen3vl_lora.yaml}

llamafactory-cli train "$CONFIG"
