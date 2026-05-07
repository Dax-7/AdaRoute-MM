# AdaRoute-MM v3_1 Text-Only Update Log

## 目标

v3_1 在 v3 text-only 分支上继续演进，但保持独立输出。核心变化是重构纯文本数据集、加入基于上一轮结果的 verified numeric 子集、修正 failed 时延对平均时延的污染，并用 source/answer_type 软先验缓解 router 向 hard 坍塌。

## 新增文件

- `configs/prompts_v3_1.yaml`：v3_1 router/LLM prompt。router 按“模型选择难度”而不是“学科难度”分类。
- `configs/v3_1_text.yaml`：v3_1 文本实验覆盖配置，关闭 VLM/fallback/cache，并开启 source prior。
- `scripts/v3_1_select_verified_numeric.py`：从 v3 baseline 结果中筛选 verified numeric 样本。
- `scripts/v3_1_prepare_fusion_dataset.py`：构建 v3_1 组件和最终 1000 条融合数据集。
- `scripts/v3_1_run_experiment_suite.py`：运行 v3_1 四组实验。
- `adaroute/v3/selection.py`：verified numeric 选择逻辑，供脚本复用。
- `docs/v3_1_update_log.md`：本操作日志。

## 数据集设计

组件默认输出到：

```text
data/datasets/v3_1_text_fusion/
```

最终融合文件：

```text
data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl
```

默认组件：

```text
arc_easy=500
sciq=500
arc_challenge=500
drop_span=500
verified_numeric=100
```

默认融合比例：

```text
arc_easy=200
sciq=200
arc_challenge=400
drop_span=100
verified_numeric=100
```

`verified_numeric` 默认读取上一轮 v3 结果：

```text
data/experiments_v3/v3_result_1/always_small/results_small.jsonl
data/experiments_v3/v3_result_1/always_gemma/results_gemma.jsonl
data/experiments_v3/v3_result_1/difficulty_routing/resultsdifficulty.jsonl
```

筛选规则：

```text
answer_type == numeric
AND always_small wrong
AND (always_gemma correct OR difficulty_routing correct)
```

如果 verified 样本不足 100 条，则从 `meta-math/GSM8K_zh` numeric 样本补足，并在 `selection_bucket` 中标记为 `gsm8k_numeric_filler`。

## Router v3_1

v3_1 router 使用新的提示词：

- 难度指模型选择难度，不是学科难度。
- 不把 science、reasoning、ARC-Challenge 自动判为 hard。
- hard 只用于 medium 也可能不可靠的样本。
- prompt 中加入 balanced benchmark 的期望分布约束，防止继续坍塌。

配置中的 source/answer_type prior：

```yaml
answer_type_priors:
  numeric: hard
source_priors:
  mib-bench/arc_easy: usually_easy
  allenai/sciq: easy_or_medium
  mib-bench/arc_challenge: medium_or_hard
  ucinlp/drop: medium_or_hard_span
```

其中 `numeric -> hard` 是直接规则；source prior 是软提示，router 可以根据题目内容覆盖。

## 时延口径

`summary.json` 中主平均时延现在只统计 `status == success` 的样本：

```json
{
  "average_latency": 0.0,
  "success_only_average_latency": 0.0,
  "average_latency_all_samples": 0.0,
  "failed_latency_excluded_count": 0
}
```

旧的全量口径保留在 `*_all_samples` 字段中，便于审计。

## 实验套件

v3_1 默认 suite：

```text
always_small
always_gemma
always_middle
difficulty_routing
```

`always_middle` 输出目录名是 `always_middle`，内部策略映射到现有 `always_medium -> phi3_medium`。

默认输出：

```text
data/experiments_v3_1/<run_id>/
```

## 推荐运行顺序

1. 先筛选 verified numeric：

```bash
python scripts/v3_1_select_verified_numeric.py
```

2. 构建 v3_1 数据集：

```bash
python scripts/v3_1_prepare_fusion_dataset.py
```

如果默认 v3 baseline 文件都存在，第 2 步也会自动补建 `verified_numeric_100.jsonl`。

3. 跑 v3_1 四组实验：

```bash
python scripts/v3_1_run_experiment_suite.py --run-id v3_1_result_001
```

4. 单独重算某个 summary：

```bash
python scripts/evaluate_results.py --input data/experiments_v3_1/v3_1_result_001/always_small/results.jsonl --output data/experiments_v3_1/v3_1_result_001/always_small/summary.json
```

5. 验证代码：

```bash
python -m pytest tests
python -m compileall adaroute scripts
python scripts/v3_1_prepare_fusion_dataset.py --help
python scripts/v3_1_select_verified_numeric.py --help
python scripts/v3_1_run_experiment_suite.py --help
```
