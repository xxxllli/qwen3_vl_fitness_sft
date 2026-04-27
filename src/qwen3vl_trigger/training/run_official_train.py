from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from qwen3vl_trigger.training.register_official_dataset import register_dataset
from qwen3vl_trigger.utils.config import load_config


def _nproc_from_config(value) -> str:
    if str(value).lower() != 'auto':
        return str(value)
    try:
        out = subprocess.check_output(['nvidia-smi', '--list-gpus'], text=True)
        n = len([x for x in out.splitlines() if x.strip()])
        return str(max(n, 1))
    except Exception:
        return '1'


def _ensure_repo(cfg: dict) -> Path:
    repo = Path(cfg['paths']['qwen3vl_repo_dir']).resolve()
    if (repo / 'qwen-vl-finetune').exists():
        return repo
    repo.parent.mkdir(parents=True, exist_ok=True)
    print(f'Cloning Qwen3-VL into {repo} ...')
    subprocess.check_call(['git', 'clone', 'https://github.com/QwenLM/Qwen3-VL.git', str(repo)])
    return repo


def build_command(cfg: dict) -> tuple[Path, list[str]]:
    repo = _ensure_repo(cfg)
    register_dataset(cfg)
    ft_dir = repo / 'qwen-vl-finetune'
    tr = cfg['train']
    project_root = Path(__file__).resolve().parents[3]
    ds_cfg = str((project_root / tr.get('deepspeed_config', 'deepspeed/zero2.json')).resolve())
    output_dir = str(Path(cfg['model']['output_dir']).resolve())
    cache_dir = str(Path(cfg['model'].get('cache_dir', './cache')).resolve())
    nproc = _nproc_from_config(tr.get('nproc_per_node', 'auto'))
    master_port = os.environ.get('MASTER_PORT', '29517')

    cmd = [
        'torchrun',
        f'--nproc_per_node={nproc}',
        '--master_addr=127.0.0.1',
        f'--master_port={master_port}',
        'qwenvl/train/train_qwen.py',
        '--model_name_or_path', str(cfg['model']['model_name_or_path']),
        '--tune_mm_llm', str(bool(tr.get('tune_mm_llm', True))),
        '--tune_mm_vision', str(bool(tr.get('tune_mm_vision', False))),
        '--tune_mm_mlp', str(bool(tr.get('tune_mm_mlp', False))),
        '--dataset_use', 'fitness_trigger_train%100',
        '--output_dir', output_dir,
        '--cache_dir', cache_dir,
        '--per_device_train_batch_size', str(tr.get('per_device_train_batch_size', 1)),
        '--gradient_accumulation_steps', str(tr.get('gradient_accumulation_steps', 8)),
        '--learning_rate', str(tr.get('learning_rate', 1e-6)),
        '--warmup_ratio', str(tr.get('warmup_ratio', 0.03)),
        '--lr_scheduler_type', str(tr.get('lr_scheduler_type', 'cosine')),
        '--weight_decay', str(tr.get('weight_decay', 0.01)),
        '--model_max_length', str(tr.get('model_max_length', 4096)),
        '--data_flatten', str(bool(tr.get('data_flatten', False))),
        '--data_packing', str(bool(tr.get('data_packing', False))),
        '--max_pixels', str(tr.get('max_pixels', 576*28*28)),
        '--min_pixels', str(tr.get('min_pixels', 16*28*28)),
        '--video_fps', str(tr.get('video_fps', 2)),
        '--video_max_frames', str(tr.get('video_max_frames', 9)),
        '--video_min_frames', str(tr.get('video_min_frames', 4)),
        '--video_max_pixels', str(tr.get('video_max_pixels', 1664*28*28)),
        '--video_min_pixels', str(tr.get('video_min_pixels', 256*28*28)),
        '--num_train_epochs', str(tr.get('num_train_epochs', 2)),
        '--logging_steps', str(tr.get('logging_steps', 10)),
        '--save_steps', str(tr.get('save_steps', 500)),
        '--save_total_limit', str(tr.get('save_total_limit', 3)),
        '--lora_enable', str(bool(tr.get('lora_enable', True))),
        '--lora_r', str(tr.get('lora_r', 16)),
        '--lora_alpha', str(tr.get('lora_alpha', 32)),
        '--lora_dropout', str(tr.get('lora_dropout', 0.05)),
        '--deepspeed', ds_cfg,
    ]
    if bool(tr.get('bf16', True)):
        cmd.append('--bf16')
    return ft_dir, cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    cwd, cmd = build_command(cfg)
    print('Working directory:', cwd)
    print('Training command:')
    print(' '.join(cmd))
    if not args.dry_run:
        subprocess.check_call(cmd, cwd=str(cwd))


if __name__ == '__main__':
    main()
