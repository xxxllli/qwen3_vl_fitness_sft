from __future__ import annotations

import argparse
import glob
import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen3vl_trigger.data.frames import (
    list_frame_files,
    window_indices,
    select_window_frames,
    make_video_clip,
)
from qwen3vl_trigger.data.split import grouped_split
from qwen3vl_trigger.utils.config import load_config, ensure_dir
from qwen3vl_trigger.utils.jsonio import read_json_any, write_json, write_jsonl


def _find_annotation_files(pattern: str) -> list[Path]:
    p = Path(pattern)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(list(p.rglob('*.json')) + list(p.rglob('*.jsonl')))
    return [Path(x) for x in sorted(glob.glob(pattern, recursive=True))]


def _load_overrides(csv_path: str | None) -> dict[str, float]:
    if not csv_path:
        return {}
    df = pd.read_csv(csv_path)
    if 'video_uid' not in df.columns or 'clip_abs_start_time' not in df.columns:
        raise ValueError('video_start_overrides_csv must contain: video_uid, clip_abs_start_time')
    return {str(r['video_uid']): float(r['clip_abs_start_time']) for _, r in df.iterrows()}


def _safe_id(*parts: Any) -> str:
    raw = '||'.join(str(p) for p in parts)
    return hashlib.md5(raw.encode('utf-8')).hexdigest()[:16]


def _resolve_frame_dir(video_uid: str, frame_root: str = '') -> Path:
    p = Path(video_uid)
    if p.is_absolute() or not frame_root:
        return p
    return Path(frame_root) / p


def _assistant_items(conv: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [x for x in conv if x.get('role') == 'assistant']


def _user_content(conv: list[dict[str, Any]]) -> str:
    for x in conv:
        if x.get('role') == 'user':
            return str(x.get('content', '')).strip()
    return ''


def _clip_abs_start(item: dict[str, Any], cfg: dict[str, Any], overrides: dict[str, float]) -> float:
    video_uid = str(item.get('video_uid', ''))
    if video_uid in overrides:
        return float(overrides[video_uid])
    field = cfg['sample'].get('clip_abs_start_time_field')
    if field and field in item:
        return float(item[field])
    return float(cfg['sample'].get('default_clip_abs_start_time', 0.0))


def _cooldown_effective_positive_times(pos_times: list[float], cooldown: float) -> tuple[list[float], list[float]]:
    kept, ignored = [], []
    last = -10**12
    for t in sorted(pos_times):
        if t - last >= cooldown:
            kept.append(t)
            last = t
        else:
            ignored.append(t)
    return kept, ignored


def _make_prompt(media_mode: str, n_images: int, user_content: str, cfg: dict[str, Any]) -> str:
    body = cfg['prompt']['template'].format(
        user_content=user_content,
        window_sec=cfg['sample']['window_sec'],
    ).strip()
    if media_mode == 'multi_image':
        tags = ''.join('<image>\n' for _ in range(n_images))
        return tags + body
    return '<video>\n' + body


def _create_sample(
    *,
    video_uid: str,
    frame_dir: Path,
    frames: list[Path],
    t: float,
    label: int,
    source: str,
    user_content: str,
    cfg: dict[str, Any],
    clip_abs_start_time: float,
    out_media_dir: Path,
    desc: str = '',
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    fps = float(cfg['sample']['fps'])
    idxs = window_indices(
        abs_time=t,
        fps=fps,
        window_sec=float(cfg['sample']['window_sec']),
        clip_abs_start_time=clip_abs_start_time,
        max_frames=int(cfg['media'].get('max_window_frames') or 0) or None,
    )
    selected = select_window_frames(frames, idxs, cfg['media'].get('missing_frame_policy', 'skip_sample'))
    if selected is None or len(selected) < int(cfg['media'].get('min_window_frames', 1)):
        return None, None

    media_mode = cfg['media']['mode']
    sid = f"{_safe_id(video_uid, t, label, source)}_t{t:.1f}_y{label}"
    prompt = _make_prompt(media_mode, len(selected), user_content, cfg)
    official: dict[str, Any] = {
        'id': sid,
        'conversations': [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': str(label)},
        ],
    }
    if media_mode == 'video_clip':
        clip_path = out_media_dir / f'{sid}.mp4'
        if not clip_path.exists():
            make_video_clip(
                selected,
                clip_path,
                fps=float(cfg['media'].get('video_clip_fps', fps)),
                codec=str(cfg['media'].get('video_codec', 'libx264')),
                quality=int(cfg['media'].get('video_quality', 7)),
            )
        official['video'] = str(clip_path.resolve())
    elif media_mode == 'multi_image':
        official['image'] = [str(p.resolve()) for p in selected]
    else:
        raise ValueError(f"Unsupported media.mode: {media_mode}")

    meta = {
        'id': sid,
        'video_uid': video_uid,
        'frame_dir': str(frame_dir),
        'abs_time': float(t),
        'clip_abs_start_time': float(clip_abs_start_time),
        'local_time': float(t - clip_abs_start_time),
        'label': int(label),
        'source': source,
        'description': desc,
        'n_frames': len(selected),
        'first_frame': str(selected[0]),
        'last_frame': str(selected[-1]),
        'frame_indices': idxs,
        'media_mode': media_mode,
        'media_path': official.get('video') or '',
        'window_sec': float(cfg['sample']['window_sec']),
        'fps': fps,
    }
    return official, meta


def build_dataset(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = ensure_dir(cfg['paths']['output_dir'])
    official_dir = ensure_dir(out_dir / 'official_json')
    media_dir = ensure_dir(out_dir / 'video_clips')
    reports_dir = ensure_dir(out_dir / 'reports')

    ann_files = _find_annotation_files(cfg['paths']['annotation_glob'])
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found: {cfg['paths']['annotation_glob']}")
    overrides = _load_overrides(cfg['sample'].get('video_start_overrides_csv'))
    rows_official: list[dict[str, Any]] = []
    rows_meta: list[dict[str, Any]] = []
    ignored_meta: list[dict[str, Any]] = []
    skipped = []

    for ann_path in tqdm(ann_files, desc='reading annotations'):
        for item in read_json_any(ann_path):
            conv = item.get('conversation') or item.get('conversations') or []
            video_uid = str(item.get('video_uid') or item.get('question_id') or '')
            if not video_uid:
                skipped.append({'annotation': str(ann_path), 'reason': 'missing video_uid'})
                continue
            user_content = _user_content(conv)
            frame_dir = _resolve_frame_dir(video_uid, cfg['paths'].get('frame_root', ''))
            if not frame_dir.exists():
                skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'reason': f'frame_dir not found: {frame_dir}'})
                continue
            frames = list_frame_files(frame_dir, cfg['media'].get('image_exts', ['.jpg', '.png']))
            if not frames:
                skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'reason': 'no frames found'})
                continue
            clip_start = _clip_abs_start(item, cfg, overrides)
            assistants = _assistant_items(conv)
            pos_items = [x for x in assistants if int(x.get('resp_type', -1)) == 1]
            pos_times = [float(x.get('time')) for x in pos_items if x.get('time') is not None]
            pos_desc = {float(x.get('time')): str(x.get('description', '')) for x in pos_items if x.get('time') is not None}

            if cfg['sample'].get('positive_policy') == 'all_positive':
                keep_pos = sorted(pos_times)
                ignore_pos = []
            else:
                keep_pos, ignore_pos = _cooldown_effective_positive_times(pos_times, float(cfg['sample']['cooldown_sec']))

            for t in keep_pos:
                official, meta = _create_sample(
                    video_uid=video_uid,
                    frame_dir=frame_dir,
                    frames=frames,
                    t=t,
                    label=1,
                    source='positive_effective' if cfg['sample'].get('positive_policy') != 'all_positive' else 'positive_raw',
                    user_content=user_content,
                    cfg=cfg,
                    clip_abs_start_time=clip_start,
                    out_media_dir=media_dir,
                    desc=pos_desc.get(t, ''),
                )
                if official:
                    rows_official.append(official)
                    rows_meta.append(meta)  # type: ignore[arg-type]
                else:
                    skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'time': t, 'label': 1, 'reason': 'missing window frames'})

            for t in ignore_pos:
                ignored_meta.append({'video_uid': video_uid, 'abs_time': t, 'label': 1, 'source': 'positive_ignored_by_cooldown'})

            # Explicit negative spans only.
            neg_items = [x for x in assistants if int(x.get('resp_type', -1)) == 0]
            step = float(cfg['sample'].get('negative_sample_step_sec', cfg['sample']['step_sec']))
            for neg in neg_items:
                span = neg.get('timespan') or [neg.get('time'), neg.get('time')]
                if not span or span[0] is None:
                    continue
                start, end = float(span[0]), float(span[-1])
                t = start
                while t <= end + 1e-8:
                    official, meta = _create_sample(
                        video_uid=video_uid,
                        frame_dir=frame_dir,
                        frames=frames,
                        t=round(t, 3),
                        label=0,
                        source='negative_explicit_span',
                        user_content=user_content,
                        cfg=cfg,
                        clip_abs_start_time=clip_start,
                        out_media_dir=media_dir,
                        desc=str(neg.get('description', '')),
                    )
                    if official:
                        rows_official.append(official)
                        rows_meta.append(meta)  # type: ignore[arg-type]
                    else:
                        skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'time': t, 'label': 0, 'reason': 'missing window frames'})
                    t += step

    if not rows_meta:
        raise RuntimeError('No usable samples were produced. Check time/frame alignment and config paths.')

    # Split by group key; add split to meta and write official jsons.
    split_key = cfg['split'].get('group_key', 'video_uid')
    splits = grouped_split(
        rows_meta,
        group_key=split_key,
        train_ratio=float(cfg['split']['train_ratio']),
        val_ratio=float(cfg['split']['val_ratio']),
        test_ratio=float(cfg['split']['test_ratio']),
        seed=int(cfg['project'].get('seed', 42)),
    )
    meta_by_id = {m['id']: m for m in rows_meta}
    official_by_id = {o['id']: o for o in rows_official}
    summary = {}
    all_meta_with_split = []
    for split_name, meta_rows in splits.items():
        ids = {m['id'] for m in meta_rows}
        official_rows = [official_by_id[i] for i in official_by_id if i in ids]
        for m in meta_rows:
            m['split'] = split_name
        all_meta_with_split.extend(meta_rows)
        write_json(official_dir / f'{split_name}.json', official_rows)
        write_jsonl(out_dir / f'{split_name}_manifest.jsonl', meta_rows)
        summary[split_name] = {
            'samples': len(meta_rows),
            'positive': sum(1 for m in meta_rows if int(m['label']) == 1),
            'negative': sum(1 for m in meta_rows if int(m['label']) == 0),
            'videos': len(set(str(m['video_uid']) for m in meta_rows)),
        }

    write_jsonl(out_dir / 'manifest_all.jsonl', all_meta_with_split)
    write_jsonl(out_dir / 'ignored_manifest.jsonl', ignored_meta)
    write_jsonl(out_dir / 'skipped.jsonl', skipped)
    write_json(out_dir / 'dataset_summary.json', summary)

    df = pd.DataFrame(all_meta_with_split)
    with pd.ExcelWriter(reports_dir / 'dataset_stats.xlsx') as writer:
        pd.DataFrame([{'split': k, **v} for k, v in summary.items()]).to_excel(writer, index=False, sheet_name='summary')
        df.groupby(['split', 'label']).size().reset_index(name='count').to_excel(writer, index=False, sheet_name='label_by_split')
        df.groupby(['split', 'video_uid']).size().reset_index(name='samples').to_excel(writer, index=False, sheet_name='video_samples')
        pd.DataFrame(skipped).to_excel(writer, index=False, sheet_name='skipped')
        pd.DataFrame(ignored_meta).to_excel(writer, index=False, sheet_name='ignored')

    return {
        'output_dir': str(out_dir),
        'summary': summary,
        'official_json_dir': str(official_dir),
        'manifest': str(out_dir / 'manifest_all.jsonl'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = build_dataset(cfg)
    print('Dataset built successfully:')
    print(result)


if __name__ == '__main__':
    main()
