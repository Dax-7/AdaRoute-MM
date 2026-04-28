# AdaRoute-MM

Adaptive Multimodal Routing System for Edge Inference.

AdaRoute-MM 是一个面向 Jetson Xavier NX 等边缘设备的多模态自适应路由推理框架。它通过 VLM 提取图像上下文，通过 Router 判断任务难度，再结合系统负载选择不同本地 Ollama LLM，并在回答失败或质量不足时自动 fallback。

## 架构

```text
Input(question, optional image)
  -> VLM caption, skipped for text-only
  -> Router difficulty: 简单 / 中等 / 困难
  -> Routing policy: always / random / difficulty / latency-aware
  -> LLM answer
  -> Quality check and fallback
  -> JSON result, logs, metrics
```

所有模型调用都走本地 Ollama `/api/generate`，没有云 API 依赖。模型名集中配置在 `configs/default.yaml`，代码只使用逻辑模型 key。

## 安装

```bash
conda create -n adaroute python=3.10
conda activate adaroute
pip install -r requirements.txt
```

开发测试可额外安装：

```bash
pip install pytest
```

## Ollama 准备

启动本地服务：

```bash
ollama serve
```

拉取默认模型：

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:1.5b-instruct-q4_1
ollama pull moondream
ollama pull phi3
ollama pull sam860/gemma3n:e2b-Q3_K_XL
```

检查模型：

```bash
python scripts/check_ollama_models.py --config configs/default.yaml
```

## 单条推理

纯文本：

```bash
python main.py --question "什么是边缘计算？"
```

图像问答：

```bash
python main.py --image data/inputs/example.jpg --question "图中有什么？"
```

指定策略：

```bash
python scripts/run_single.py --question "请解释边缘计算是什么" --policy latency_aware
```

完整 JSON 会保存到 `data/outputs/`。

## 批量评测

输入格式为 JSONL，每行一个样本：

```json
{"id":"text_001","image_path":null,"question":"什么是边缘计算？","answer":"边缘计算是在靠近数据源的位置进行计算。","task_type":"text_only"}
```

运行：

```bash
python scripts/run_batch.py --input data/datasets/demo.jsonl --output data/outputs/demo_results.jsonl --policy latency_aware
```

统计：

```bash
python scripts/evaluate_results.py --input data/outputs/demo_results.jsonl --output data/outputs/demo_summary.json
```

`data/datasets/demo.jsonl` 中包含一个图像样例路径 `data/inputs/example.jpg`。项目不自带图片，请自行放入测试图像；如果图片不存在，该样本会记录错误并继续处理。

## 路由策略

- `always_small`：所有样本使用 `qwen_small`
- `always_medium`：所有样本使用 `phi3_medium`
- `always_large`：所有样本使用 `gemma_large`
- `random`：从配置候选模型中随机选择
- `difficulty_based`：简单、中等、困难分别映射到小、中、大模型
- `latency_aware`：先按难度选择，系统过载时跳过大模型并选择轻量模型

AdaRoute-MM Full 可理解为：`latency_aware + fallback + cache + VLM`。

## 输出格式

结果 JSON 包含：

- `request_id`、`status`、`input`
- `caption_text`、`answer`
- `route.difficulty`、`route.policy`、`route.initial_model`、`route.final_model`
- `fallback.triggered`、`fallback.count`、`fallback.trace`
- `latency.total/vlm/router/llm`
- `system.cpu_percent/ram_percent/gpu_percent/temperature`
- `model_calls`
- `error`

失败时返回结构化错误，不会让批处理因为单条样本中断。

## Jetson Xavier NX 注意事项

- 默认串行推理，不建议多个 Ollama 模型并发。
- 建议开启 swap，并先测试小模型。
- `gemma_large` 首次加载可能较慢，内存压力也更高。
- 高温、高内存或高 GPU 占用时，`latency_aware` 会避免选择大模型。
- `tegrastats` 解析失败时会自动退化为 `psutil`。
- 批量测试建议开启 VLM cache，避免重复图像 caption。
- Jetson 配置可使用：

```bash
python main.py --override-config configs/jetson.yaml --question "什么是边缘计算？"
```

## 常见错误

- `OLLAMA_CONNECTION_ERROR`：确认 `ollama serve` 已启动，且 `configs/default.yaml` 中 `base_url` 正确。
- `MODEL_NOT_FOUND` 或模型缺失：运行 `python scripts/check_ollama_models.py`，按提示 `ollama pull`。
- `IMAGE_NOT_FOUND`：确认图像路径相对项目根目录存在。
- `ROUTER_PARSE_ERROR`：Router 输出不符合三个标签之一，系统会使用默认难度 `中等` 继续。
- 批量评测中断点续跑：默认会跳过输出 JSONL 中已有的 `sample_id`；使用 `--no-resume` 可关闭。

## 测试

```bash
python -m pytest tests
```

测试使用 mock，不需要真实调用 Ollama。
