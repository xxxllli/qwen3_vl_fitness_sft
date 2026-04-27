# Qwen3-VL Fitness Trigger 增训工程

本工程用于把健身视频 1QnA 标注转换成 0.5 秒级 trigger 二分类样本，并微调 `Qwen/Qwen3-VL-8B-Instruct`。

trigger 模型只判断当前时刻是否应该触发一次健身指导：

```text
1 = 应该触发指导
0 = 不应该触发指导
```

推荐训练路线已经改为 LLaMA-Factory LoRA。原 `qwen-vl-finetune` 官方路线仍保留为 fallback。

## 关键规则

- 输入为用户观察任务 + 当前时刻之前最近 4 秒视频窗口。
- 2 fps 抽帧，线上判断频率为每 0.5 秒一次。
- 正样本来自 `resp_type=1` 的 `assistant.time`。
- 同一视频内正样本按 4 秒 cooldown 归一，cooldown 内重复正样本进入 ignore，不改成负样本。
- 负样本只来自显式 `resp_type=0` 的 `timespan`，未标注区间不默认当负样本。
- 数据切分按组完成，默认 `split.group_key: video_uid`，禁止窗口级随机切分。
- 训练目标严格为单字符 `0` 或 `1`，不训练 `description`。

## 目录

```text
configs/default.yaml                         # 数据、采样、推理、评测主配置
configs/llamafactory_qwen3vl_lora.yaml       # LLaMA-Factory LoRA 训练配置
src/qwen3vl_trigger/data/build_dataset.py    # fallback: qwen-vl-finetune 格式
src/qwen3vl_trigger/data/build_llamafactory_dataset.py
src/qwen3vl_trigger/infer/predict.py         # fallback 推理
src/qwen3vl_trigger/infer/predict_llamafactory_lora.py
src/qwen3vl_trigger/eval/evaluate.py
scripts/01_build_dataset_llamafactory.sh
scripts/02_check_llamafactory_env.sh
scripts/03_train_lora_llamafactory.sh
scripts/04_predict_llamafactory.sh
scripts/05_evaluate.sh
```

## 安装

建议在 Linux/WSL2 + NVIDIA GPU 环境训练。

```bash
cd qwen3vl_fitness_trigger
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements-train.txt
```

安装 LLaMA-Factory：

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git third_party/LLaMA-Factory
cd third_party/LLaMA-Factory
pip install -e ".[torch,metrics]"
cd ../..
```

## 配置数据

编辑：

```bash
configs/default.yaml
```

至少修改：

```yaml
paths:
  annotation_glob: /path/to/annotations/**/*.json
  frame_root: ""
  output_dir: ./outputs/fitness_trigger_qwen3vl
  llamafactory_output_dir: ./outputs/llamafactory_data

model:
  model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
```

如果 `assistant.time` 是原始长视频绝对时间，但 `clip_images` 是局部 clip 从 0 秒开始的帧目录，需要设置：

```yaml
sample:
  default_clip_abs_start_time: 45.0
```

或使用：

```yaml
sample:
  video_start_overrides_csv: /path/to/video_start_overrides.csv
```

CSV 两列：

```csv
video_uid,clip_abs_start_time
/path/to/clip_images,45.0
```

## 推荐路线：LLaMA-Factory

### 1. 检查原始数据

```bash
bash scripts/00_check_data.sh configs/default.yaml
```

### 2. 构建 LLaMA-Factory 数据集

```bash
bash scripts/01_build_dataset_llamafactory.sh configs/default.yaml
```

默认导出 ShareGPT `conversations` 格式。如果本地 LLaMA-Factory 版本更适合 OpenAI `messages` 字段，可以运行：

```bash
bash scripts/01_build_dataset_llamafactory.sh configs/default.yaml messages
```

输出：

```text
outputs/llamafactory_data/
├── train.json
├── val.json
├── test.json
├── dataset_info.json
├── media/
│   ├── video_clips/
│   └── frames/
└── stats/
    ├── dataset_stats.xlsx
    ├── ignored_samples.jsonl
    ├── skipped_samples.jsonl
    ├── train_manifest.jsonl
    ├── val_manifest.jsonl
    └── test_manifest.jsonl
```

样本核心形态：

```json
{
  "conversations": [
    {"from": "human", "value": "<video>\n...只输出一个数字..."},
    {"from": "gpt", "value": "1"}
  ],
  "videos": ["media/video_clips/xxx_t71.5_y1.mp4"]
}
```

### 3. 检查 LLaMA-Factory 环境

```bash
bash scripts/02_check_llamafactory_env.sh configs/llamafactory_qwen3vl_lora.yaml
```

检查内容包括：

- `llamafactory-cli` 是否存在
- `torch` / `transformers` / `llamafactory` 是否可导入
- CUDA 是否可用
- `dataset_info.json` 是否注册了 `fitness_trigger_train`
- `template: qwen3_vl` 是否能在当前 LLaMA-Factory 模板中找到
- `video_fps` / `video_maxlen` 是否已配置

### 4. 启动 LoRA 训练

```bash
bash scripts/03_train_lora_llamafactory.sh configs/llamafactory_qwen3vl_lora.yaml
```

默认关键参数：

```yaml
model_name_or_path: Qwen/Qwen3-VL-8B-Instruct
template: qwen3_vl
dataset: fitness_trigger_train
dataset_dir: ./outputs/llamafactory_data
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
learning_rate: 1.0e-5
num_train_epochs: 2
video_maxlen: 9
video_fps: 2
```

### 5. 推理

```bash
bash scripts/04_predict_llamafactory.sh configs/default.yaml
```

也可以显式指定 adapter 和 split：

```bash
bash scripts/04_predict_llamafactory.sh configs/default.yaml ./outputs/qwen3vl_fitness_trigger_lora test
```

输出：

```text
outputs/fitness_trigger_qwen3vl/predictions/test_predictions.jsonl
```

每行包含：

```text
id, video_uid, abs_time, label, pred, score, score_1, score_0, raw_output
```

其中 `score = score_1 = P("1") / (P("0") + P("1"))`，默认阈值来自 `infer.threshold`。

### 6. 评测

```bash
bash scripts/05_evaluate.sh configs/default.yaml
```

输出：

```text
outputs/fitness_trigger_qwen3vl/reports/
├── eval_report_test.xlsx
├── point_metrics.json
├── event_metrics_cooldown4s.json
├── threshold_sweep.xlsx
├── confusion_matrix.png
├── error_cases.xlsx
└── timeline_cases.html
```

评测包含点级 Accuracy / Precision / Recall / F1 / AUC / PR-AUC，以及带 4 秒 cooldown 的事件级 Precision / Recall / F1、平均触发延迟、每分钟误触发次数。

第一版建议优先看：

```text
Precision
false_triggers_per_min
event_precision
```

默认阈值扫描：

```yaml
eval:
  thresholds: [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
  cooldown_sec: 4.0
  event_tolerance_sec: 1.0
```

## fallback：官方 qwen-vl-finetune

原路线仍可使用：

```bash
bash scripts/01_build_dataset.sh configs/default.yaml
bash scripts/setup_official_qwen3vl.sh ./third_party/Qwen3-VL
bash scripts/03_train_lora_official_dryrun.sh configs/default.yaml
bash scripts/03_train_lora_official.sh configs/default.yaml
bash scripts/04_predict.sh configs/default.yaml
bash scripts/05_evaluate.sh configs/default.yaml
```

fallback 使用同一套核心数据规则：4 秒窗口、显式负样本、正样本 cooldown ignore、按视频分组切分、严格输出 `0/1`。

## 常见问题

### 为什么不训练 description？

这个模型只做 trigger，不负责生成指导内容。训练 `description` 会把“是否该说话”和“该说什么”混在一起，影响触发稳定性。

### 为什么未标注区间不当负样本？

当前标注是稀疏触发点。未标注不等于一定不该触发，强行作为负样本会污染训练目标。

### 为什么 cooldown 内重复正样本不改成 0？

cooldown 内的重复正样本语义上仍然“可以说”，只是产品规则不允许连续说。改成 0 会教坏模型，所以进入 ignore。
