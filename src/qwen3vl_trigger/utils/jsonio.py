from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Any


def read_json_any(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON object, JSON list, or JSONL file and return a list of dicts."""
    path = Path(path)
    text = path.read_text(encoding='utf-8').strip()
    if not text:
        return []
    if path.suffix.lower() == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    obj = json.loads(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f'Unsupported JSON root type in {path}: {type(obj)}')


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
