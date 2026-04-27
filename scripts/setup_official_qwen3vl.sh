#!/usr/bin/env bash
set -euo pipefail
REPO_DIR=${1:-./third_party/Qwen3-VL}
if [ -d "$REPO_DIR/qwen-vl-finetune" ]; then
  echo "Qwen3-VL repo already exists: $REPO_DIR"
else
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone https://github.com/QwenLM/Qwen3-VL.git "$REPO_DIR"
fi
