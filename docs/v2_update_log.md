# AdaRoute-MM v2 更新日志与使用说明

本次 v2 更新聚焦“有 VLM 的图像 + 问题”系统，不再把 text-only 作为当前实验主线。第一版的单条推理、批量推理、路由、fallback 等入口尽量保留；v2 主要通过新增模块和脚本扩展实验框架。

## 1. 新增实验模式

新增 `adaroute/experiments/modes.py`，统一定义消融实验模式：

| mode | 路由策略 | Image caption cache | Fallback | 目的 |
| --- | --- | --- | --- | --- |
| `always_small` | 固定 small | 关闭 | 关闭 | 小模型基线 |
| `always_gemma` | 固定 Gemma | 关闭 | 关闭 | 大模型基线 |
| `random_routing` | 随机路由 | 关闭 | 关闭 | 随机策略基线 |
| `difficulty_routing` | 难度路由 | 关闭 | 关闭 | 核心路由能力 |
| `difficulty_cache` | 难度路由 | 开启 | 关闭 | 单独评估 image cache |
| `difficulty_fallback` | 难度路由 | 关闭 | 开启 | 单独评估 fallback |
| `latency_aware_routing` | 延迟感知路由 | 关闭 | 关闭 | 边缘负载场景对比 |
| `adaroute_mm_full` | 延迟感知路由 | 开启 | 开启 | 完整 AdaRoute-MM v2 |

这些模式都可以一条命令单独运行，也可以作为 suite 批量运行。

## 2. Image-level VLM Cache

修改了 `adaroute/modules/vlm.py`，新增 `vlm.caption_mode`：

- `question_aware`：保持第一版行为，cache key 包含 `image_path + question + prompt_version + model_name`。
- `image_caption`：v2 推荐模式，cache key 只和图片、caption prompt、VLM 模型有关，不包含 question。

v2 实验模式默认使用：

```yaml
vlm:
  enabled: true
  caption_mode: image_caption
```

这样同一张图片对应多个问题时，只生成一次 general caption，后续问题复用该 caption：

```text
image -> VLM -> general caption cache
caption + question -> LLM
```

cache 命中会体现在 `model_calls` 中：

```json
{"stage": "vlm", "cached": true}
```

评估 summary 中也会统计：

- `vlm_call_count`
- `vlm_cache_hit_count`
- `vlm_cache_hit_rate`
- `unique_image_count`
- `vlm_calls_per_sample`

## 3. v2 Prompt 文件

新增 `configs/prompts_v2.yaml`，避免继续依赖第一版中已经出现编码污染的 prompt 文本。

包含：

- `vlm`：question-aware caption prompt
- `vlm_general`：general image caption prompt，供 image-level cache 使用
- `router`：输出 `easy / medium / hard`
- `llm`：面向 VQA，yes/no 问题要求先输出 `yes` 或 `no`

同时更新了 difficulty 解析和策略映射，使系统兼容：

- 英文：`easy / medium / hard`
- 中文：`简单 / 中等 / 困难`
- 第一版遗留乱码标签

## 4. VQAv2 Yes/No 数据准备

新增脚本：

```bash
python scripts/prepare_vqav2_yesno.py \
  --source lmms-lab/VQAv2-FewShot \
  --config-name eval \
  --split validation \
  --limit 1000 \
  --output data/datasets/vqav2_yesno_1000.jsonl \
  --image-dir data/inputs/vqav2_yesno
```

脚本会：

1. 使用 Hugging Face `datasets` streaming 读取数据；
2. 默认只筛选 `answer_type == "yes/no"` 的样本，避免把答案碰巧为 `yes/no` 的非二分类问题混入评测；
3. 保存图片到 `data/inputs/vqav2_yesno/`；
4. 生成 AdaRoute-MM 可直接读取的 JSONL。

稳定性说明：

- `lmms-lab/VQAv2-FewShot` 有 `eval` / `full` subset，脚本默认使用 `--config-name eval`；
- 默认支持断点续跑：输出 JSONL 已存在时会跳过已有 `id` 并继续补齐到 `--limit`；
- 如需重新生成，显式加 `--overwrite`；
- 图片默认使用 `datasets.Image(decode=False)` 延迟解码，单张坏图会被跳过而不会中断整个任务；
- 如需兼容旧逻辑，保留答案为 `yes/no` 但 `answer_type` 不是 yes/no 的样本，可加 `--include-answer-only-yesno`。

如果你本地下载了 parquet，可以改用：

```bash
python scripts/prepare_vqav2_yesno.py \
  --local-parquet path/to/*.parquet \
  --split train \
  --limit 1000 \
  --output data/datasets/vqav2_yesno_1000.jsonl \
  --image-dir data/inputs/vqav2_yesno
```

输出 JSONL 字段包括：

```json
{
  "id": "vqav2_262148000",
  "image_path": "data/inputs/vqav2_yesno/262148000.jpg",
  "question": "Is this man a professional baseball player?",
  "answer": "yes",
  "multiple_choice_answer": "yes",
  "answers": ["yes", "yes", "yes"],
  "question_type": "is this",
  "answer_type": "yes/no",
  "category": "yes_no",
  "task_type": "image_qa",
  "source": "VQAv2",
  "image_id": 262148,
  "question_id": 262148000
}
```

依赖已加入 `requirements.txt`：

```text
datasets
pillow
pyarrow
```

## 5. 单模式实验运行

新增脚本：

```bash
python scripts/run_experiment.py \
  --mode difficulty_cache \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_v2
```

可选模式：

```text
always_small
always_gemma
random_routing
difficulty_routing
difficulty_cache
difficulty_fallback
latency_aware_routing
adaroute_mm_full
```

默认使用：

```text
configs/default.yaml
configs/prompts_v2.yaml
data/experiments/
```

输出目录示例：

```text
data/experiments/vqav2_yesno_v2/difficulty_cache/
  resolved_config.yaml
  results.jsonl
  summary.json
  requests/
  logs/
  cache/
```

## 6. 批量 Suite 运行

新增脚本：

```bash
python scripts/run_experiment_suite.py \
  --suite vqav2_yesno_ablation \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_v2
```

它会依次运行全部 v2 消融模式，并生成：

```text
data/experiments/vqav2_yesno_v2/manifest.json
```

`manifest.json` 会记录每个模式的 `results.jsonl` 和 `summary.json` 路径。

## 7. Yes/No 评估

新增：

- `adaroute/eval/yesno.py`
- `scripts/evaluate_yesno_results.py`

评估会从生成式回答中抽取 yes/no：

- `yes / yeah / true / correct` -> `yes`
- `no / nope / false / incorrect` -> `no`
- 空回答、含混回答、同时包含 yes 和 no -> `invalid`

主要指标：

- `accuracy`
- `accuracy_ci95`
- `macro_f1`
- `balanced_accuracy`
- `yes_precision / yes_recall / yes_f1`
- `no_precision / no_recall / no_f1`
- `invalid_rate`
- `vqa_soft_accuracy`
- `confusion_matrix`

单独评估已有结果：

```bash
python scripts/evaluate_yesno_results.py \
  --input data/experiments/vqav2_yesno_v2/difficulty_cache/results.jsonl \
  --output data/experiments/vqav2_yesno_v2/difficulty_cache/summary.json
```

原有 `scripts/evaluate_results.py` 现在也会在发现 yes/no 字段时附加 `yesno` 指标。

## 8. 结果保存与 Resume

实验 runner 默认 resume：

- 如果 `results.jsonl` 中已有某个 `sample_id`，再次运行会跳过。
- 若要强制重跑，加 `--no-resume`。

示例：

```bash
python scripts/run_experiment.py \
  --mode adaroute_mm_full \
  --dataset data/datasets/vqav2_yesno_1000.jsonl \
  --run-id vqav2_yesno_v2 \
  --no-resume
```

## 9. Git Ignore

新增忽略：

```text
data/experiments/
data/inputs/vqav2_yesno/
data/datasets/vqav2_yesno*.jsonl
```

避免把实验输出、缓存图片、生成的数据子集误提交。

## 10. 验证

本次更新后运行：

```bash
python -m pytest tests
```

结果：

```text
25 passed
```

还检查了新增脚本的 CLI：

```bash
python scripts/run_experiment.py --help
python scripts/run_experiment_suite.py --help
python scripts/prepare_vqav2_yesno.py --help
python scripts/evaluate_yesno_results.py --help
```

均可正常解析。
