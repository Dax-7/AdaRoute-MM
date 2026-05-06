# AdaRoute-MM v3 Text-Only Update Log

本版目标：先完全绕开图像/VLM，使用纯文本融合数据集验证路由、答案评估、模型成本和基础准确率。v3 尽量复用 v2 的 pipeline、policy、runner，只新增独立的 v3 数据、prompt、脚本和输出目录。

## 1. 新增文件

- `adaroute/v3/datasets.py`：Hugging Face streaming 数据集转换与融合构建接口。
- `adaroute/eval/text_answer.py`：文本答案抽取与评分，支持多选、数字、boolean、短文本。
- `configs/prompts_v3.yaml`：文本难度路由 prompt 和强制 `FINAL_ANSWER: <answer>` 输出 prompt。
- `configs/v3_text.yaml`：v3 文本实验覆盖配置，关闭 VLM、fallback、cache，并把短答案质量检查阈值降到 1。
- `scripts/v3_prepare_fusion_dataset.py`：生成五个 500 条组件数据集和 1000 条融合数据集。
- `scripts/v3_run_experiment_suite.py`：运行 v3 基础四组实验。
- `scripts/v3_select_dataset_after_baseline.py`：首轮基础实验后按模型表现筛选更适合路由研究的数据集。
- `data/experiments_v3/`：v3 默认实验输出目录，和 v2 的 `data/experiments/` 区分。

## 2. 数据集转换

数据源和主要字段参考 Hugging Face 数据集页：

- `mib-bench/arc_challenge`：`question`, `choices`, `label`/`answerKey`。来源：https://huggingface.co/datasets/mib-bench/arc_challenge
- `meta-math/GSM8K_zh`：`question_zh`, `answer_only`。来源：https://huggingface.co/datasets/meta-math/GSM8K_zh
- `TIGER-Lab/MMLU-Pro`：`question`, `options`, `answer`, `answer_index`, `category`。来源：https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro
- `lukaemon/bbh`：五个 config：`boolean_expressions`, `date_understanding`, `disambiguation_qa`, `object_counting`, `logical_deduction_three_objects`，字段为 `input`, `target`。来源：https://huggingface.co/datasets/lukaemon/bbh
- `ucinlp/drop`：`passage`, `question`, `answers_spans.spans`。来源：https://huggingface.co/datasets/ucinlp/drop

默认每个来源转换 500 条。BBH 为五类各 100 条。转换后保存到：

```text
data/datasets/v3_text_fusion/
  arc_challenge_500.jsonl
  gsm8k_zh_500.jsonl
  mmlu_pro_500.jsonl
  bbh_500.jsonl
  drop_500.jsonl
  fusion_1000_200-300-200-200-100.jsonl
  manifest.json
```

统一 JSONL 基础字段保持 v2 风格：

```json
{
  "id": "arc_challenge_0",
  "image_path": null,
  "question": "...",
  "answer": "A",
  "task_type": "text_only",
  "source": "mib-bench/arc_challenge",
  "category": "arc_challenge",
  "answer_type": "multiple_choice",
  "answer_format": "Return only one final answer as: FINAL_ANSWER: <answer>",
  "choices": ["...", "..."],
  "choice_labels": ["A", "B", "C", "D"],
  "metadata": {}
}
```

默认融合比例：

```text
arc_challenge=200
gsm8k_zh=300
mmlu_pro=200
bbh=200
drop=100
```

生成命令：

```bash
python scripts/v3_prepare_fusion_dataset.py
```

调整比例示例：

```bash
python scripts/v3_prepare_fusion_dataset.py --mix-counts arc_challenge=250,gsm8k_zh=250,mmlu_pro=200,bbh=200,drop=100
```

调整 split 示例：

```bash
python scripts/v3_prepare_fusion_dataset.py --mmlu-split test --drop-split validation
```

## 3. 答案评估方法

v3 要求模型在最后输出：

```text
FINAL_ANSWER: <answer>
```

评估器同时支持从不严格输出中兜底抽取：

- `multiple_choice`：抽取 `A` 到 `J`，也支持 `(A)`、`Answer: A`、完整选项文本匹配。
- `numeric`：抽取第一个数字，去掉千分位逗号，用 `Decimal` 精确比较，例如 `1,200` 等于 `1200`。
- `boolean`：抽取 `true` / `false`。
- `short_text`：大小写、标点、空白归一后做 exact/contains 弱匹配，主要服务 DROP span 和少量 BBH 非选择题。

实验 summary 中新增：

```json
"text_answer": {
  "accuracy": 0.0,
  "answer_parse_failure_rate": 0.0,
  "by_answer_type": {},
  "by_source": {}
}
```

## 4. 新增时间与 token 成本指标

每个 `model_calls[]` 会记录 Ollama 返回的 token 与 duration：

```json
{
  "prompt_eval_count": 100,
  "eval_count": 8,
  "timing": {
    "prompt_eval_duration_s": 0.12,
    "eval_duration_s": 0.08,
    "load_duration_s": 0.5,
    "inference_only_time_s": 0.2,
    "prefill_cost_per_token_s": 0.0012,
    "decode_cost_per_token_s": 0.01,
    "token_normalized_cost_s": 0.2
  }
}
```

定义：

- `inference_only_time_s = prompt_eval_duration + eval_duration`，不包含模型加载时间。
- `prefill_cost_per_token_s = prompt_eval_duration / prompt_eval_count`
- `decode_cost_per_token_s = eval_duration / eval_count`
- `token_normalized_cost_s = prompt_eval_count * prefill_cost_per_token_s + eval_count * decode_cost_per_token_s`

summary 中新增汇总字段：

- `all_model_calls_inference_only_time`
- `all_model_calls_token_normalized_cost`
- `llm_calls_inference_only_time`
- `llm_calls_token_normalized_cost`
- `prompt_eval_tokens`
- `decode_tokens`

## 5. v3 基础实验

基础 suite 只跑四组：

```text
always_small
always_gemma
difficulty_routing
random_routing
```

运行命令：

```bash
python scripts/v3_run_experiment_suite.py --dataset data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl --run-id v3_text_baseline_001
```

输出位置：

```text
data/experiments_v3/v3_text_baseline_001/
  always_small/results.jsonl
  always_small/summary.json
  always_gemma/results.jsonl
  always_gemma/summary.json
  difficulty_routing/results.jsonl
  difficulty_routing/summary.json
  random_routing/results.jsonl
  random_routing/summary.json
  manifest.json
```

单独重新汇总某个结果：

```bash
python scripts/evaluate_results.py --input data/experiments_v3/v3_text_baseline_001/always_small/results.jsonl --output data/experiments_v3/v3_text_baseline_001/always_small/summary.json
```

## 6. 首轮后筛选数据集

目标：优先保留 `small_wrong AND medium_or_large_correct`，控制全对、全错、解析失败和相似题。

默认目标占比：

```text
small_correct=30%
medium_large_better=45%
all_difficult=15%
robustness_challenge=10%
```

运行：

```bash
python scripts/v3_select_dataset_after_baseline.py --dataset data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl --run-dir data/experiments_v3/v3_text_baseline_001 --output data/datasets/v3_text_fusion/selected_after_baseline.jsonl
```

如果后续你补跑了纯 `always_medium` 或其他模型结果，可以显式传入多个结果文件：

```bash
python scripts/v3_select_dataset_after_baseline.py --dataset data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl --results data/experiments_v3/run/always_small/results.jsonl --results data/experiments_v3/run/always_medium/results.jsonl --results data/experiments_v3/run/always_gemma/results.jsonl --large-mode always_gemma --output data/datasets/v3_text_fusion/selected_after_baseline.jsonl
```

筛选脚本会输出：

- 新数据集 JSONL：默认 `selected_after_baseline.jsonl`
- 筛选报告：默认 `selected_after_baseline.report.json`

## 7. 建议操作顺序

1. 确认 Ollama 模型可用：

```bash
python scripts/check_ollama_models.py --config configs/default.yaml
```

2. 构建 v3 融合数据：

```bash
python scripts/v3_prepare_fusion_dataset.py
```

3. 先跑四组基础实验：

```bash
python scripts/v3_run_experiment_suite.py --run-id v3_text_baseline_001
```

4. 看每组 `summary.json` 的 `text_answer.accuracy`、`by_source`、`llm_calls_token_normalized_cost`。

5. 如果基础结果符合预期，再筛选下一轮数据：

```bash
python scripts/v3_select_dataset_after_baseline.py --dataset data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl --run-dir data/experiments_v3/v3_text_baseline_001
```

6. 用筛选后的数据重跑：

```bash
python scripts/v3_run_experiment_suite.py --dataset data/datasets/v3_text_fusion/selected_after_baseline.jsonl --run-id v3_text_selected_001
```

