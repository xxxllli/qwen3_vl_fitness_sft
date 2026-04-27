from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) or {}
    cfg['_config_path'] = str(path.resolve())
    cfg['_project_root'] = str(Path(__file__).resolve().parents[3])
    return cfg


def deep_get(d: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = d
    for part in dotted.split('.'):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def as_path(path: str | os.PathLike, base: str | os.PathLike | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() or base is None:
        return p
    return Path(base) / p
