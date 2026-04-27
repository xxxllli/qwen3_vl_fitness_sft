from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from qwen3vl_trigger.utils.config import load_config
from qwen3vl_trigger.utils.jsonio import read_jsonl, write_jsonl


def _dataset_dir(cfg: dict[str, Any], override: str | None = None) -> Path:
    if override:
        return Path(override)
    return Path(cfg.get('paths', {}).get('llamafactory_output_dir') or './outputs/llamafactory_data')


def _read_json(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding='utf-8'))


def _strip_media_tags(prompt: str) -> str:
    return re.sub(r'(<image>\s*|<video>\s*)+', '', prompt).strip()


def _prompt_from_sample(sample: dict[str, Any]) -> str:
    if sample.get('conversations'):
        return str(sample['conversations'][0]['value'])
    if sample.get('messages'):
        content = sample['messages'][0]['content']
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return ''.join(str(x.get('text', '')) for x in content if isinstance(x, dict) and x.get('type') == 'text')
    raise ValueError(f"Sample {sample.get('id')} has no conversations/messages prompt")


def _resolve_media(path: str, data_dir: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((data_dir / p).resolve())


def _media_from_sample(sample: dict[str, Any], data_dir: Path) -> tuple[str, list[str]]:
    if sample.get('videos'):
        return 'video', [_resolve_media(x, data_dir) for x in sample['videos']]
    if sample.get('video'):
        return 'video', [_resolve_media(str(sample['video']), data_dir)]
    if sample.get('images'):
        return 'image', [_resolve_media(x, data_dir) for x in sample['images']]
    if sample.get('image'):
        image = sample['image']
        if isinstance(image, list):
            return 'image', [_resolve_media(str(x), data_dir) for x in image]
        return 'image', [_resolve_media(str(image), data_dir)]
    raise ValueError(f"Sample {sample.get('id')} has no videos/images field")


def _build_messages(sample: dict[str, Any], data_dir: Path, fps: float) -> list[dict[str, Any]]:
    prompt = _strip_media_tags(_prompt_from_sample(sample))
    media_type, media_paths = _media_from_sample(sample, data_dir)
    content: list[dict[str, Any]] = []
    if media_type == 'video':
        for video_path in media_paths:
            content.append({'type': 'video', 'video': video_path, 'fps': fps})
    else:
        for image_path in media_paths:
            content.append({'type': 'image', 'image': image_path})
    content.append({'type': 'text', 'text': prompt})
    return [{'role': 'user', 'content': content}]


def _token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f'Tokenizer returned no ids for {text!r}')
    return int(ids[0])


def _prepare_inputs(model: Any, processor: Any, messages: list[dict[str, Any]]) -> dict[str, Any]:
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
    device = getattr(model, 'device', None)
    if device is None:
        device = next(model.parameters()).device
    return {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}


def _score_01(model: Any, processor: Any, messages: list[dict[str, Any]]) -> tuple[float, float]:
    import torch

    inputs = _prepare_inputs(model, processor, messages)
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[:, -1, :]
    tok = processor.tokenizer
    id0 = _token_id(tok, '0')
    id1 = _token_id(tok, '1')
    pair = torch.stack([logits[0, id0], logits[0, id1]])
    probs = torch.softmax(pair, dim=0).detach().cpu().tolist()
    return float(probs[0]), float(probs[1])


def _generate_raw(model: Any, processor: Any, messages: list[dict[str, Any]], max_new_tokens: int = 2) -> str:
    import torch

    inputs = _prepare_inputs(model, processor, messages)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def _load_model(model_name: str, adapter_path: str | None) -> tuple[Any, Any]:
    import torch
    import transformers
    from peft import PeftModel

    processor = transformers.AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model_cls = getattr(transformers, 'AutoModelForImageTextToText', None)
    if model_cls is None:
        model_cls = getattr(transformers, 'AutoModelForVision2Seq', None)
    if model_cls is None:
        raise ImportError('transformers must provide AutoModelForImageTextToText or AutoModelForVision2Seq')

    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )
    if adapter_path:
        adapter = Path(adapter_path)
        if not adapter.exists():
            raise FileNotFoundError(f'LoRA adapter path does not exist: {adapter_path}')
        print(f'Loading LLaMA-Factory LoRA adapter: {adapter_path}')
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, processor


def _default_adapter_path(cfg: dict[str, Any], override: str | None = None) -> str | None:
    if override:
        return override
    infer_cfg = cfg.get('infer', {})
    if infer_cfg.get('llamafactory_adapter_path'):
        return str(infer_cfg['llamafactory_adapter_path'])
    lf_default = Path('./outputs/qwen3vl_fitness_trigger_lora')
    if lf_default.exists():
        return str(lf_default)
    return infer_cfg.get('adapter_path')


def predict(
    cfg: dict[str, Any],
    dataset_dir: str | None = None,
    split: str | None = None,
    adapter_path: str | None = None,
    output_path: str | None = None,
) -> Path:
    data_dir = _dataset_dir(cfg, dataset_dir)
    split_name = split or cfg.get('infer', {}).get('split', 'test')
    data_path = data_dir / f'{split_name}.json'
    manifest_path = data_dir / 'stats' / f'{split_name}_manifest.jsonl'
    if not data_path.exists():
        raise FileNotFoundError(f'Missing LLaMA-Factory split file: {data_path}')

    samples = _read_json(data_path)
    meta = {m['id']: m for m in read_jsonl(manifest_path)} if manifest_path.exists() else {}
    max_samples = cfg.get('infer', {}).get('max_samples')
    if max_samples:
        samples = samples[: int(max_samples)]

    model_name = cfg.get('model', {}).get('model_name_or_path', 'Qwen/Qwen3-VL-8B-Instruct')
    threshold = float(cfg.get('infer', {}).get('threshold', 0.8))
    fps = float(cfg.get('infer', {}).get('video_fps') or cfg.get('sample', {}).get('fps', 2.0))
    generate_raw = bool(cfg.get('infer', {}).get('generate_raw_output', True))
    resolved_adapter = _default_adapter_path(cfg, adapter_path)

    print(f'Loading processor/model: {model_name}')
    model, processor = _load_model(model_name, resolved_adapter)

    rows = []
    for sample in tqdm(samples, desc=f'predict LLaMA-Factory {split_name}'):
        sid = sample['id']
        messages = _build_messages(sample, data_dir, fps=fps)
        score_0, score_1 = _score_01(model, processor, messages)
        pred = int(score_1 >= threshold)
        raw_output = _generate_raw(model, processor, messages) if generate_raw else str(pred)
        m = {**sample, **meta.get(sid, {})}
        rows.append({
            'id': sid,
            'video_uid': m.get('video_uid'),
            'abs_time': m.get('abs_time'),
            'label': m.get('label'),
            'pred': pred,
            'score': score_1,
            'score_1': score_1,
            'score_0': score_0,
            'raw_output': raw_output,
            'split': split_name,
            'media_path': m.get('media_path'),
        })

    out_path = Path(output_path) if output_path else Path(cfg['paths']['output_dir']) / 'predictions' / f'{split_name}_predictions.jsonl'
    write_jsonl(out_path, rows)
    print(f'Wrote predictions: {out_path}')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--split', default=None)
    parser.add_argument('--adapter-path', default=None)
    parser.add_argument('--output-path', default=None)
    args = parser.parse_args()
    predict(
        load_config(args.config),
        dataset_dir=args.dataset_dir,
        split=args.split,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
    )


if __name__ == '__main__':
    main()
