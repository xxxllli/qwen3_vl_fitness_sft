from __future__ import annotations

import random
from collections import defaultdict
from typing import Any


def grouped_split(
    rows: list[dict[str, Any]],
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must be 1.0')
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        g = str(row.get(group_key) or row.get('video_uid') or row.get('id'))
        groups[g].append(row)
    keys = list(groups.keys())
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n = len(keys)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_keys = set(keys[:n_train])
    val_keys = set(keys[n_train:n_train + n_val])
    test_keys = set(keys[n_train + n_val:])
    out = {'train': [], 'val': [], 'test': []}
    for k, items in groups.items():
        if k in train_keys:
            out['train'].extend(items)
        elif k in val_keys:
            out['val'].extend(items)
        else:
            out['test'].extend(items)
    return out
