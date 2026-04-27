from __future__ import annotations

import argparse
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

from qwen3vl_trigger.utils.config import load_config, ensure_dir
from qwen3vl_trigger.utils.jsonio import read_jsonl, write_json


def point_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y_true = np.array([int(r['label']) for r in rows if r.get('label') is not None])
    scores = np.array([float(r['score']) for r in rows if r.get('label') is not None])
    y_pred = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'positive': int(y_true.sum()),
        'negative': int((1 - y_true).sum()),
    }
    if len(set(y_true.tolist())) == 2:
        out['roc_auc'] = roc_auc_score(y_true, scores)
        out['pr_auc'] = average_precision_score(y_true, scores)
    else:
        out['roc_auc'] = None
        out['pr_auc'] = None
    return out


def _match_events(gold: list[float], pred: list[float], tolerance: float) -> tuple[int, int, int, list[float]]:
    gold = sorted(gold)
    pred = sorted(pred)
    used = set()
    tp = 0
    delays = []
    for p in pred:
        best_i = None
        best_dist = 10**9
        for i, g in enumerate(gold):
            if i in used:
                continue
            dist = abs(p - g)
            if dist <= tolerance and dist < best_dist:
                best_i = i
                best_dist = dist
        if best_i is not None:
            used.add(best_i)
            tp += 1
            delays.append(p - gold[best_i])
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn, delays


def event_metrics(rows: list[dict[str, Any]], threshold: float, cooldown: float, tolerance: float) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_video[str(r.get('video_uid'))].append(r)
    total_tp = total_fp = total_fn = 0
    all_delays = []
    pred_events_total = 0
    gold_events_total = 0
    total_minutes = 0.0
    for _, items in by_video.items():
        items = sorted(items, key=lambda x: float(x.get('abs_time', 0.0)))
        if items:
            start = float(items[0].get('abs_time', 0.0))
            end = float(items[-1].get('abs_time', 0.0))
            total_minutes += max(end - start, 0.0) / 60.0
        gold_events = [float(x['abs_time']) for x in items if int(x.get('label', 0)) == 1]
        pred_events = []
        last = -10**12
        for x in items:
            t = float(x.get('abs_time', 0.0))
            if float(x.get('score', 0.0)) >= threshold and t - last >= cooldown:
                pred_events.append(t)
                last = t
        tp, fp, fn, delays = _match_events(gold_events, pred_events, tolerance)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_delays.extend(delays)
        pred_events_total += len(pred_events)
        gold_events_total += len(gold_events)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'threshold': threshold,
        'cooldown_sec': cooldown,
        'event_tolerance_sec': tolerance,
        'event_precision': precision,
        'event_recall': recall,
        'event_f1': f1,
        'event_tp': total_tp,
        'event_fp': total_fp,
        'event_fn': total_fn,
        'pred_events': pred_events_total,
        'gold_events': gold_events_total,
        'mean_delay_sec': float(np.mean(all_delays)) if all_delays else None,
        'median_delay_sec': float(np.median(all_delays)) if all_delays else None,
        'false_triggers_per_min': total_fp / total_minutes if total_minutes > 0 else None,
    }


def _rows_with_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        r = dict(row)
        if 'score' not in r and 'score_1' in r:
            r['score'] = r['score_1']
        out.append(r)
    return out


def _error_rows(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    errors = []
    for r in rows:
        pred = int(float(r['score']) >= threshold)
        if pred != int(r['label']):
            kind = 'false_positive' if pred == 1 else 'false_negative'
            errors.append({**r, 'pred_at_threshold': pred, 'threshold': threshold, 'error_type': kind})
    return errors


def _write_confusion_matrix_png(rows: list[dict[str, Any]], threshold: float, out_path: Path) -> None:
    from PIL import Image, ImageDraw

    y_true = [int(r['label']) for r in rows]
    y_pred = [int(float(r['score']) >= threshold) for r in rows]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cells = [[tn, fp], [fn, tp]]
    img = Image.new('RGB', (520, 420), 'white')
    draw = ImageDraw.Draw(img)
    left, top, size = 160, 90, 120
    labels_x = ['Pred 0', 'Pred 1']
    labels_y = ['Label 0', 'Label 1']
    draw.text((20, 20), f'Confusion Matrix @ threshold={threshold:.2f}', fill='black')
    for j, label in enumerate(labels_x):
        draw.text((left + j * size + 35, top - 30), label, fill='black')
    for i, label in enumerate(labels_y):
        draw.text((left - 90, top + i * size + 50), label, fill='black')
    max_cell = max(max(row) for row in cells) or 1
    for i in range(2):
        for j in range(2):
            value = cells[i][j]
            shade = 245 - int(125 * value / max_cell)
            fill = (shade, shade, 255)
            x0 = left + j * size
            y0 = top + i * size
            draw.rectangle((x0, y0, x0 + size, y0 + size), fill=fill, outline='black', width=2)
            draw.text((x0 + 50, y0 + 52), str(value), fill='black')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _write_timeline_html(rows: list[dict[str, Any]], threshold: float, out_path: Path) -> None:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_video[str(r.get('video_uid'))].append(r)
    parts = [
        '<!doctype html><meta charset="utf-8">',
        '<title>Fitness Trigger Timeline</title>',
        '<style>body{font-family:Arial,sans-serif;margin:24px;}table{border-collapse:collapse;width:100%;margin:12px 0 28px;}td,th{border:1px solid #ddd;padding:6px 8px;font-size:13px;}th{background:#f5f5f5}.fp{background:#ffe8e8}.fn{background:#fff5d8}.tp{background:#e8f7ed}</style>',
        f'<h1>Timeline Cases @ threshold={threshold:.2f}</h1>',
    ]
    for video_uid, items in sorted(by_video.items()):
        parts.append(f'<h2>{escape(video_uid)}</h2>')
        parts.append('<table><thead><tr><th>time</th><th>label</th><th>pred</th><th>score_1</th><th>raw_output</th><th>id</th></tr></thead><tbody>')
        for r in sorted(items, key=lambda x: float(x.get('abs_time', 0.0))):
            pred = int(float(r['score']) >= threshold)
            label = int(r['label'])
            klass = 'tp' if pred == label == 1 else 'fp' if pred == 1 and label == 0 else 'fn' if pred == 0 and label == 1 else ''
            parts.append(
                '<tr class="{klass}"><td>{time}</td><td>{label}</td><td>{pred}</td><td>{score:.4f}</td><td>{raw}</td><td>{sid}</td></tr>'.format(
                    klass=klass,
                    time=escape(str(r.get('abs_time', ''))),
                    label=label,
                    pred=pred,
                    score=float(r['score']),
                    raw=escape(str(r.get('raw_output') or r.get('generated') or '')),
                    sid=escape(str(r.get('id', ''))),
                )
            )
        parts.append('</tbody></table>')
    out_path.write_text('\n'.join(parts), encoding='utf-8')


def evaluate(cfg: dict[str, Any], predictions_path: str | None = None) -> Path:
    split = cfg['infer'].get('split', 'test')
    if predictions_path is None:
        predictions_path = str(Path(cfg['paths']['output_dir']) / 'predictions' / f'{split}_predictions.jsonl')
    rows = _rows_with_scores(read_jsonl(predictions_path))
    rows = [r for r in rows if r.get('label') is not None]
    thresholds = [float(x) for x in cfg['eval'].get('thresholds', [0.8])]
    cooldown = float(cfg['eval'].get('cooldown_sec', 4.0))
    tolerance = float(cfg['eval'].get('event_tolerance_sec', 1.0))
    point_rows = [point_metrics(rows, th) for th in thresholds]
    event_rows = [event_metrics(rows, th, cooldown, tolerance) for th in thresholds]

    out_dir = ensure_dir(Path(cfg['paths']['output_dir']) / 'reports')
    out_path = out_dir / f'eval_report_{split}.xlsx'
    threshold_sweep_path = out_dir / 'threshold_sweep.xlsx'
    th0 = float(cfg['infer'].get('threshold', 0.8))
    errors = _error_rows(rows, th0)

    with pd.ExcelWriter(out_path) as writer:
        pd.DataFrame(point_rows).to_excel(writer, index=False, sheet_name='point_metrics')
        pd.DataFrame(event_rows).to_excel(writer, index=False, sheet_name='event_metrics')
        pd.DataFrame(rows).to_excel(writer, index=False, sheet_name='predictions')
        pd.DataFrame(errors).to_excel(writer, index=False, sheet_name='errors')

    with pd.ExcelWriter(threshold_sweep_path) as writer:
        pd.DataFrame(point_rows).to_excel(writer, index=False, sheet_name='point_metrics')
        pd.DataFrame(event_rows).to_excel(writer, index=False, sheet_name='event_metrics')

    write_json(out_dir / 'point_metrics.json', point_rows)
    write_json(out_dir / f'event_metrics_cooldown{int(cooldown)}s.json', event_rows)
    pd.DataFrame(errors).to_excel(out_dir / 'error_cases.xlsx', index=False)
    _write_confusion_matrix_png(rows, th0, out_dir / 'confusion_matrix.png')
    _write_timeline_html(rows, th0, out_dir / 'timeline_cases.html')

    print(f'Wrote evaluation report: {out_path}')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--predictions', default=None)
    args = parser.parse_args()
    evaluate(load_config(args.config), args.predictions)


if __name__ == '__main__':
    main()
