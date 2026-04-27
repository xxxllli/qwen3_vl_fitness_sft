from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen3vl_trigger.data.frames import list_frame_files, abs_time_to_index
from qwen3vl_trigger.data.build_dataset import _find_annotation_files, _resolve_frame_dir, _assistant_items, _clip_abs_start, _load_overrides
from qwen3vl_trigger.utils.config import load_config, ensure_dir
from qwen3vl_trigger.utils.jsonio import read_json_any


def check_data(cfg: dict[str, Any]) -> Path:
    out_dir = ensure_dir(Path(cfg['paths']['output_dir']) / 'reports')
    ann_files = _find_annotation_files(cfg['paths']['annotation_glob'])
    overrides = _load_overrides(cfg['sample'].get('video_start_overrides_csv'))
    rows = []
    for ann_path in tqdm(ann_files, desc='checking'):
        for item in read_json_any(ann_path):
            conv = item.get('conversation') or item.get('conversations') or []
            video_uid = str(item.get('video_uid') or item.get('question_id') or '')
            frame_dir = _resolve_frame_dir(video_uid, cfg['paths'].get('frame_root', '')) if video_uid else Path('')
            exists = frame_dir.exists() if video_uid else False
            frames = list_frame_files(frame_dir, cfg['media'].get('image_exts', ['.jpg', '.png'])) if exists else []
            clip_start = _clip_abs_start(item, cfg, overrides) if video_uid else 0.0
            assistant_times = []
            out_of_range = 0
            for a in _assistant_items(conv):
                if a.get('time') is not None:
                    t = float(a['time'])
                    assistant_times.append(t)
                    idx = abs_time_to_index(t, float(cfg['sample']['fps']), clip_start)
                    if idx < 0 or idx >= len(frames):
                        out_of_range += 1
                span = a.get('timespan')
                if span and len(span) >= 2:
                    for t in (float(span[0]), float(span[-1])):
                        idx = abs_time_to_index(t, float(cfg['sample']['fps']), clip_start)
                        if idx < 0 or idx >= len(frames):
                            out_of_range += 1
            rows.append({
                'annotation': str(ann_path),
                'video_uid': video_uid,
                'frame_dir': str(frame_dir),
                'frame_dir_exists': exists,
                'num_frames': len(frames),
                'clip_abs_start_time': clip_start,
                'min_assistant_time': min(assistant_times) if assistant_times else None,
                'max_assistant_time': max(assistant_times) if assistant_times else None,
                'assistant_time_count': len(assistant_times),
                'out_of_range_time_count': out_of_range,
            })
    df = pd.DataFrame(rows)
    path = out_dir / 'data_check.xlsx'
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name='check')
        if not df.empty:
            df.groupby(['frame_dir_exists']).size().reset_index(name='count').to_excel(writer, index=False, sheet_name='frame_dir_exists')
    print(f'Wrote data check report: {path}')
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    check_data(load_config(args.config))


if __name__ == '__main__':
    main()
