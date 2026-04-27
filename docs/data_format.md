# 数据格式说明

## 原始输入标注

支持单个 JSON、JSON 数组、JSONL。核心字段：

```json
{
  "video_uid": "/path/to/clip_images",
  "conversation": [
    {"role": "user", "content": "用户观察要求", "time": 45.0},
    {"role": "assistant", "time": 51.0, "timespan": [51.0, 51.0], "resp_type": 1, "content": "单物体", "description": "..."},
    {"role": "assistant", "time": 80.0, "timespan": [80.0, 84.5], "resp_type": 0, "content": "负样本", "description": "无"}
  ]
}
```

## 输出 QwenVL official json

video_clip 模式：

```json
{
  "id": "xxx_t71.5_y1",
  "video": "/abs/path/to/window.mp4",
  "conversations": [
    {"from": "human", "value": "<video>\n你是健身实时指导系统中的 trigger 模型..."},
    {"from": "gpt", "value": "1"}
  ]
}
```

multi_image 模式：

```json
{
  "id": "xxx_t71.5_y1",
  "image": ["/abs/path/frame_0135.jpg", "..."],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n...你是健身实时指导系统中的 trigger 模型..."},
    {"from": "gpt", "value": "1"}
  ]
}
```
