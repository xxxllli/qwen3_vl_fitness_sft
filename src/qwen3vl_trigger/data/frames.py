from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from PIL import Image


def _last_number_key(path: Path) -> tuple[int, str]:
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1]), path.name
    return -1, path.name


def list_frame_files(frame_dir: str | Path, exts: Sequence[str]) -> list[Path]:
    frame_dir = Path(frame_dir)
    lower_exts = {e.lower() for e in exts}
    files = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in lower_exts]
    files.sort(key=_last_number_key)
    return files


def abs_time_to_index(abs_time: float, fps: float, clip_abs_start_time: float = 0.0) -> int:
    local_t = abs_time - clip_abs_start_time
    return int(round(local_t * fps))


def window_indices(
    abs_time: float,
    fps: float,
    window_sec: float,
    clip_abs_start_time: float = 0.0,
    max_frames: int | None = None,
) -> list[int]:
    end_idx = abs_time_to_index(abs_time, fps, clip_abs_start_time)
    start_idx = max(0, abs_time_to_index(abs_time - window_sec, fps, clip_abs_start_time))
    idxs = list(range(start_idx, end_idx + 1))
    if max_frames is not None and len(idxs) > max_frames:
        idxs = idxs[-max_frames:]
    return idxs


def select_window_frames(
    frames: Sequence[Path],
    idxs: Sequence[int],
    missing_policy: str = 'skip_sample',
) -> list[Path] | None:
    out: list[Path] = []
    for idx in idxs:
        if idx < 0 or idx >= len(frames):
            if missing_policy == 'raise':
                raise IndexError(f'Frame index {idx} out of range 0..{len(frames)-1}')
            return None
        out.append(frames[idx])
    return out


def make_video_clip(
    frame_paths: Sequence[str | Path],
    out_path: str | Path,
    fps: float = 2.0,
    codec: str = 'libx264',
    quality: int = 7,
) -> Path:
    """Create a small mp4 clip from sampled frames.

    Uses imageio-ffmpeg. The output resolution follows the first frame; subsequent
    frames are resized to match to avoid encoder errors.
    """
    import imageio.v3 as iio
    import numpy as np

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = []
    base_size = None
    for p in frame_paths:
        img = Image.open(p).convert('RGB')
        if base_size is None:
            base_size = img.size
        elif img.size != base_size:
            img = img.resize(base_size)
        arrays.append(np.asarray(img))
    if not arrays:
        raise ValueError('Cannot create video clip with zero frames')
    iio.imwrite(out_path, arrays, fps=fps, codec=codec, quality=quality, macro_block_size=1)
    return out_path
