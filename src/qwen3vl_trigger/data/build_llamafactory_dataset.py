from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from qwen3vl_trigger.data.build_dataset import (
    _assistant_items,
    _clip_abs_start,
    _cooldown_effective_positive_times,
    _find_annotation_files,
    _load_overrides,
    _make_prompt,
    _resolve_frame_dir,
    _safe_id,
    _user_content,
)
from qwen3vl_trigger.data.frames import make_video_clip, select_window_frames, window_indices, list_frame_files
from qwen3vl_trigger.data.split import grouped_split
from qwen3vl_trigger.utils.config import ensure_dir, load_config
from qwen3vl_trigger.utils.jsonio import read_json_any, write_json, write_jsonl


DATASET_NAMES = {
    'train': 'fitness_trigger_train',
    'val': 'fitness_trigger_val',
    'test': 'fitness_trigger_test',
}


def _llamafactory_output_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path(cfg.get('paths', {}).get('llamafactory_output_dir') or './outputs/llamafactory_data')


def _relative_posix(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _metadata_fields(item: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key in ('question_id', 'video_uid', 'video_start_time', 'action_type', 'exercise', 'movement', 'subject_id'):
        if key in item:
            out[key] = item[key]
    return out


def _split_group_value(meta: dict[str, Any], cfg: dict[str, Any]) -> str:
    fields = cfg.get('split', {}).get('group_fields')
    if fields:
        return '||'.join(str(meta.get(k, '')) for k in fields)
    key = cfg.get('split', {}).get('group_key', 'video_uid')
    return str(meta.get(key) or meta.get('video_uid') or meta.get('id'))


def _copy_window_frames(selected: list[Path], sample_dir: Path, data_dir: Path) -> list[str]:
    ensure_dir(sample_dir)
    out = []
    for i, src in enumerate(selected):
        dst = sample_dir / f'{i:03d}{src.suffix.lower()}'
        if not dst.exists():
            shutil.copy2(src, dst)
        out.append(_relative_posix(dst, data_dir))
    return out


def _make_lf_sample(
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
    data_dir: Path,
    video_dir: Path,
    frame_media_dir: Path,
    sample_format: str,
    item_meta: dict[str, Any],
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

    media_mode = cfg['media'].get('mode', 'video_clip')
    sid = f"{_safe_id(video_uid, t, label, source)}_t{t:.1f}_y{label}"
    prompt = _make_prompt(media_mode, len(selected), user_content, cfg)

    sample: dict[str, Any] = {
        'id': sid,
        'video_uid': video_uid,
        'abs_time': float(t),
        'label': int(label),
        'source': source,
    }
    if sample_format == 'messages':
        sample['messages'] = [
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': str(label)},
        ]
    elif sample_format == 'conversations':
        sample['conversations'] = [
            {'from': 'human', 'value': prompt},
            {'from': 'gpt', 'value': str(label)},
        ]
    else:
        raise ValueError(f'Unsupported sample format: {sample_format}')

    if media_mode == 'video_clip':
        clip_path = video_dir / f'{sid}.mp4'
        if not clip_path.exists():
            make_video_clip(
                selected,
                clip_path,
                fps=float(cfg['media'].get('video_clip_fps', fps)),
                codec=str(cfg['media'].get('video_codec', 'libx264')),
                quality=int(cfg['media'].get('video_quality', 7)),
            )
        media_paths = [_relative_posix(clip_path, data_dir)]
        sample['videos'] = media_paths
    elif media_mode == 'multi_image':
        media_paths = _copy_window_frames(selected, frame_media_dir / sid, data_dir)
        sample['images'] = media_paths
    else:
        raise ValueError(f"Unsupported media.mode for LLaMA-Factory export: {media_mode}")

    meta = {
        **item_meta,
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
        'media_path': media_paths[0] if media_paths else '',
        'window_sec': float(cfg['sample']['window_sec']),
        'fps': fps,
    }
    meta['_split_group'] = _split_group_value(meta, cfg)
    return sample, meta


def _dataset_info(sample_format: str, media_mode: str) -> dict[str, Any]:
    media_column = 'videos' if media_mode == 'video_clip' else 'images'
    info = {}
    for split, name in DATASET_NAMES.items():
        entry: dict[str, Any] = {
            'file_name': f'{split}.json',
            'formatting': 'sharegpt',
            'columns': {
                'messages': 'messages' if sample_format == 'messages' else 'conversations',
                media_column: media_column,
            },
        }
        if sample_format == 'conversations':
            entry['tags'] = {
                'role_tag': 'from',
                'content_tag': 'value',
                'user_tag': 'human',
                'assistant_tag': 'gpt',
            }
        else:
            entry['tags'] = {
                'role_tag': 'role',
                'content_tag': 'content',
                'user_tag': 'user',
                'assistant_tag': 'assistant',
            }
        info[name] = entry
    return info


def build_llamafactory_dataset(
    cfg: dict[str, Any],
    output_dir: str | None = None,
    sample_format: str = 'conversations',
) -> dict[str, Any]:
    data_dir = ensure_dir(_llamafactory_output_dir(cfg, output_dir))
    media_root = ensure_dir(data_dir / 'media')
    video_dir = ensure_dir(media_root / 'video_clips')
    frame_media_dir = ensure_dir(media_root / 'frames')
    stats_dir = ensure_dir(data_dir / 'stats')

    ann_files = _find_annotation_files(cfg['paths']['annotation_glob'])
    if not ann_files:
        raise FileNotFoundError(f"No annotation files found: {cfg['paths']['annotation_glob']}")

    overrides = _load_overrides(cfg['sample'].get('video_start_overrides_csv'))
    rows_lf: list[dict[str, Any]] = []
    rows_meta: list[dict[str, Any]] = []
    ignored_meta: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

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
            item_meta = _metadata_fields(item)
            assistants = _assistant_items(conv)
            pos_items = [x for x in assistants if int(x.get('resp_type', -1)) == 1]
            pos_times = [float(x.get('time')) for x in pos_items if x.get('time') is not None]
            pos_desc = {float(x.get('time')): str(x.get('description', '')) for x in pos_items if x.get('time') is not None}

            if cfg['sample'].get('positive_policy') == 'all_positive':
                keep_pos = sorted(pos_times)
                ignore_pos: list[float] = []
            else:
                keep_pos, ignore_pos = _cooldown_effective_positive_times(pos_times, float(cfg['sample']['cooldown_sec']))

            for t in keep_pos:
                sample, meta = _make_lf_sample(
                    video_uid=video_uid,
                    frame_dir=frame_dir,
                    frames=frames,
                    t=t,
                    label=1,
                    source='positive_effective' if cfg['sample'].get('positive_policy') != 'all_positive' else 'positive_raw',
                    user_content=user_content,
                    cfg=cfg,
                    clip_abs_start_time=clip_start,
                    data_dir=data_dir,
                    video_dir=video_dir,
                    frame_media_dir=frame_media_dir,
                    sample_format=sample_format,
                    item_meta=item_meta,
                    desc=pos_desc.get(t, ''),
                )
                if sample:
                    rows_lf.append(sample)
                    rows_meta.append(meta)  # type: ignore[arg-type]
                else:
                    skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'time': t, 'label': 1, 'reason': 'missing window frames'})

            for t in ignore_pos:
                ignored_meta.append({
                    **item_meta,
                    'video_uid': video_uid,
                    'abs_time': float(t),
                    'label': 1,
                    'source': 'positive_ignored_by_cooldown',
                })

            # Explicit negative spans only. Unannotated time is intentionally not sampled.
            neg_items = [x for x in assistants if int(x.get('resp_type', -1)) == 0]
            step = float(cfg['sample'].get('negative_sample_step_sec', cfg['sample']['step_sec']))
            for neg in neg_items:
                span = neg.get('timespan') or [neg.get('time'), neg.get('time')]
                if not span or span[0] is None:
                    continue
                start, end = float(span[0]), float(span[-1])
                t = start
                while t <= end + 1e-8:
                    sample, meta = _make_lf_sample(
                        video_uid=video_uid,
                        frame_dir=frame_dir,
                        frames=frames,
                        t=round(t, 3),
                        label=0,
                        source='negative_explicit_span',
                        user_content=user_content,
                        cfg=cfg,
                        clip_abs_start_time=clip_start,
                        data_dir=data_dir,
                        video_dir=video_dir,
                        frame_media_dir=frame_media_dir,
                        sample_format=sample_format,
                        item_meta=item_meta,
                        desc=str(neg.get('description', '')),
                    )
                    if sample:
                        rows_lf.append(sample)
                        rows_meta.append(meta)  # type: ignore[arg-type]
                    else:
                        skipped.append({'annotation': str(ann_path), 'video_uid': video_uid, 'time': t, 'label': 0, 'reason': 'missing window frames'})
                    t += step

    if not rows_meta:
        raise RuntimeError('No usable LLaMA-Factory samples were produced. Check time/frame alignment and config paths.')

    splits = grouped_split(
        rows_meta,
        group_key='_split_group',
        train_ratio=float(cfg['split']['train_ratio']),
        val_ratio=float(cfg['split']['val_ratio']),
        test_ratio=float(cfg['split']['test_ratio']),
        seed=int(cfg['project'].get('seed', 42)),
    )

    sample_by_id = {x['id']: x for x in rows_lf}
    summary: dict[str, dict[str, int]] = {}
    all_meta_with_split = []
    for split_name, meta_rows in splits.items():
        ids = {m['id'] for m in meta_rows}
        samples = [sample_by_id[sid] for sid in sample_by_id if sid in ids]
        for m in meta_rows:
            m['split'] = split_name
        for s in samples:
            s['split'] = split_name
        all_meta_with_split.extend(meta_rows)
        write_json(data_dir / f'{split_name}.json', samples)
        write_jsonl(stats_dir / f'{split_name}_manifest.jsonl', meta_rows)
        summary[split_name] = {
            'samples': len(meta_rows),
            'positive': sum(1 for m in meta_rows if int(m['label']) == 1),
            'negative': sum(1 for m in meta_rows if int(m['label']) == 0),
            'videos': len(set(str(m['video_uid']) for m in meta_rows)),
            'groups': len(set(str(m['_split_group']) for m in meta_rows)),
        }

    write_json(data_dir / 'dataset_info.json', _dataset_info(sample_format, cfg['media'].get('mode', 'video_clip')))
    write_json(data_dir / 'dataset_summary.json', summary)
    write_jsonl(stats_dir / 'manifest_all.jsonl', all_meta_with_split)
    write_jsonl(stats_dir / 'ignored_samples.jsonl', ignored_meta)
    write_jsonl(stats_dir / 'skipped_samples.jsonl', skipped)

    df = pd.DataFrame(all_meta_with_split)
    with pd.ExcelWriter(stats_dir / 'dataset_stats.xlsx') as writer:
        pd.DataFrame([{'split': k, **v} for k, v in summary.items()]).to_excel(writer, index=False, sheet_name='summary')
        df.groupby(['split', 'label']).size().reset_index(name='count').to_excel(writer, index=False, sheet_name='label_by_split')
        df.groupby(['split', 'video_uid']).size().reset_index(name='samples').to_excel(writer, index=False, sheet_name='video_samples')
        df.groupby(['split', '_split_group']).size().reset_index(name='samples').to_excel(writer, index=False, sheet_name='groups')
        pd.DataFrame(skipped).to_excel(writer, index=False, sheet_name='skipped')
        pd.DataFrame(ignored_meta).to_excel(writer, index=False, sheet_name='ignored')

    return {
        'dataset_dir': str(data_dir),
        'dataset_info': str(data_dir / 'dataset_info.json'),
        'sample_format': sample_format,
        'summary': summary,
        'stats': str(stats_dir / 'dataset_stats.xlsx'),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--format', choices=['conversations', 'messages'], default='conversations')
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = build_llamafactory_dataset(cfg, output_dir=args.output_dir, sample_format=args.format)
    print('LLaMA-Factory dataset built successfully:')
    print(result)


if __name__ == '__main__':
    main()
