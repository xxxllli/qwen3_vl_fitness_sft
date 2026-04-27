from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from qwen3vl_trigger.utils.config import load_config
from qwen3vl_trigger.utils.jsonio import read_jsonl, write_jsonl


def _strip_media_tags(prompt: str) -> str:
    return re.sub(r'(<image>\s*|<video>\s*)+', '', prompt).strip()


def _build_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = sample['conversations'][0]['value']
    text = _strip_media_tags(prompt)
    content: list[dict[str, Any]] = []
    if 'video' in sample:
        content.append({'type': 'video', 'video': sample['video']})
    elif 'image' in sample:
        for img in sample['image']:
            content.append({'type': 'image', 'image': img})
    else:
        raise ValueError(f"Sample {sample.get('id')} has no video/image field")
    content.append({'type': 'text', 'text': text})
    return [{'role': 'user', 'content': content}]


def _load_official_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def _score_next_token(model, processor, messages: list[dict[str, Any]]) -> float:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        return_tensors='pt',
    )
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
    tok = processor.tokenizer
    id0 = tok.encode('0', add_special_tokens=False)[0]
    id1 = tok.encode('1', add_special_tokens=False)[0]
    pair = torch.stack([logits[0, id0], logits[0, id1]])
    probs = torch.softmax(pair, dim=0)
    return float(probs[1].detach().cpu())


def _generate_label(model, processor, messages: list[dict[str, Any]]) -> str:
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, **video_kwargs, return_tensors='pt')
    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    gen_ids = ids[:, inputs['input_ids'].shape[1]:]
    text_out = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return '1' if text_out.startswith('1') else '0'


def predict(cfg: dict[str, Any]) -> Path:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    split = cfg['infer'].get('split', 'test')
    data_path = Path(cfg['paths']['output_dir']) / 'official_json' / f'{split}.json'
    manifest_path = Path(cfg['paths']['output_dir']) / f'{split}_manifest.jsonl'
    if not data_path.exists() or not manifest_path.exists():
        raise FileNotFoundError('Missing dataset json/manifest. Run build_dataset first.')
    samples = _load_official_json(data_path)
    meta = {m['id']: m for m in read_jsonl(manifest_path)}
    max_samples = cfg['infer'].get('max_samples')
    if max_samples:
        samples = samples[: int(max_samples)]

    model_name = cfg['model']['model_name_or_path']
    print(f'Loading processor/model: {model_name}')
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
    )
    adapter = cfg['infer'].get('adapter_path')
    if adapter and Path(adapter).exists():
        print(f'Loading LoRA adapter: {adapter}')
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    rows = []
    use_score = bool(cfg['infer'].get('use_score_from_logits', True))
    threshold = float(cfg['infer'].get('threshold', 0.8))
    for s in tqdm(samples, desc=f'predict {split}'):
        sid = s['id']
        messages = _build_messages(s)
        if use_score:
            score = _score_next_token(model, processor, messages)
            pred = int(score >= threshold)
            gen = str(pred)
        else:
            gen = _generate_label(model, processor, messages)
            pred = int(gen == '1')
            score = float(pred)
        m = meta.get(sid, {})
        rows.append({
            'id': sid,
            'score': score,
            'pred': pred,
            'generated': gen,
            'label': m.get('label'),
            'video_uid': m.get('video_uid'),
            'abs_time': m.get('abs_time'),
            'split': split,
        })
    out_dir = Path(cfg['paths']['output_dir']) / 'predictions'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{split}_predictions.jsonl'
    write_jsonl(out_path, rows)
    print(f'Wrote predictions: {out_path}')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    predict(load_config(args.config))


if __name__ == '__main__':
    main()
