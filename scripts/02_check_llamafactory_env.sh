#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/llamafactory_qwen3vl_lora.yaml}

if ! command -v llamafactory-cli >/dev/null 2>&1; then
  echo "ERROR: llamafactory-cli not found. Install LLaMA-Factory first." >&2
  echo 'Example: pip install -e "third_party/LLaMA-Factory[torch,metrics]"' >&2
  exit 1
fi

echo "llamafactory-cli: $(command -v llamafactory-cli)"
llamafactory-cli --help >/dev/null

python - "$CONFIG" <<'PY'
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
cfg = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}

def fail(message: str) -> None:
    print(f'ERROR: {message}', file=sys.stderr)
    raise SystemExit(1)

def version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        fail(f'cannot import {module_name}: {exc}')
    return getattr(mod, '__version__', 'unknown')

print(f"python: {sys.version.split()[0]}")
print(f"llamafactory: {version('llamafactory')}")
print(f"transformers: {version('transformers')}")

try:
    import torch
except Exception as exc:
    fail(f'cannot import torch: {exc}')
print(f"torch: {torch.__version__}")
print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device count: {torch.cuda.device_count()}")
    print(f"cuda device 0: {torch.cuda.get_device_name(0)}")

dataset_dir = Path(cfg.get('dataset_dir', './outputs/llamafactory_data'))
dataset = cfg.get('dataset')
dataset_info_path = dataset_dir / 'dataset_info.json'
if not dataset_info_path.exists():
    fail(f'dataset_info.json not found: {dataset_info_path}. Run scripts/01_build_dataset_llamafactory.sh first.')

dataset_info = json.loads(dataset_info_path.read_text(encoding='utf-8'))
if dataset not in dataset_info:
    fail(f'dataset {dataset!r} is not registered in {dataset_info_path}')

file_name = dataset_info[dataset].get('file_name')
if not file_name or not (dataset_dir / file_name).exists():
    fail(f'dataset file not found for {dataset!r}: {file_name}')
print(f"dataset registered: {dataset} -> {dataset_dir / file_name}")

template = cfg.get('template')
try:
    template_mod = importlib.import_module('llamafactory.data.template')
    templates = getattr(template_mod, 'TEMPLATES', None)
    if templates is not None and template not in templates:
        fail(f'template {template!r} not found in llamafactory.data.template.TEMPLATES')
    print(f"template available: {template}")
except SystemExit:
    raise
except Exception as exc:
    print(f"WARN: could not introspect LLaMA-Factory templates: {exc}")

model_name = str(cfg.get('model_name_or_path', ''))
if Path(model_name).exists():
    print(f"model path exists: {model_name}")
else:
    print(f"model will be resolved by transformers/huggingface: {model_name}")

for key in ('video_fps', 'video_maxlen'):
    if key not in cfg:
        fail(f'missing required multimodal argument in train config: {key}')
print('LLaMA-Factory environment check passed.')
PY
