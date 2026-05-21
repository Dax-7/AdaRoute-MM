# AdaRoute-MM v3_2 更新日志

## 目标

v3_2 在 v3_1 text-only 工作流基础上做两类改动：

1. 将 small 模型改为 `qwen2.5:1.5b-instruct-q4_1`，与 router 使用同一个 Ollama 模型，减少 small 路径上的模型切换成本。
2. 将 `difficulty_routing` 的“三分类难度”思路改为风险分级/模型适配度路由，避免 v3_1 把大多数样本保守地送入 Gemma。

## v3_1 结果问题

本次设计参考了 `data/experiments_v3_1/v3_1_base4_20260508_020111` 下四组基线：

- `always_small`: accuracy 0.678，成功样本平均时延 0.302s。
- `always_middle`: accuracy 0.754，成功样本平均时延 0.257s。
- `always_gemma`: accuracy 0.805，成功样本平均时延 0.457s。
- `difficulty_routing`: accuracy 0.797，但路由分布为 Gemma 892 / middle 58 / small 50，成功样本平均时延 0.555s。

核心问题是：v3_1 的 router 虽然 prompt 写了“选最便宜且可能正确的模型”，但输出仍被绑定到 `easy/medium/hard`，实际运行时过度保守，导致 89.2% 样本走 Gemma，并且多了一次 router 调用后比 always_gemma 更慢。

## 新增文件

- `configs/v3_2_text.yaml`
- `configs/prompts_v3_2.yaml`
- `scripts/v3_2_run_experiment_suite.py`
- `tests/test_v3_2_text.py`

## 代码改动

- `adaroute/modules/router.py`
  - 支持解析 `small_ok` / `middle_ok` / `need_gemma` 风险标签，并映射回现有 `easy` / `medium` / `hard` policy 标签。
  - 增加 `risk_aware_static_gate`，用于 v3_2 的 dataset-aware/type-aware 风险门控。
  - v3_2 默认使用确定性风险门控，避免继续依赖当前过保守的三分类 router。

- `adaroute/policies/difficulty_policy.py`
  - 保持旧 `difficulty_based` policy，不新增复杂 policy。
  - 增补正常中文标签兼容，并继续支持历史编码污染标签。

- `adaroute/experiments/modes.py`
  - 新增 `risk_aware_routing` mode。
  - 新增 `text_fusion_v3_2_basic` suite：
    - `always_small`
    - `always_middle`
    - `always_gemma`
    - `risk_aware_routing`
    - `difficulty_routing`
    - `random_routing`
    - `adaroute_mm_full`
  - 新增 `text_fusion_v3_2_added_baselines` suite，用于在服务器上只补跑 v3_2.5 新增的两条 baseline：
    - `random_routing`
    - `adaroute_mm_full`

- `adaroute/eval/metrics.py`
  - 新增 `model_usage_by_source`
  - 新增 `difficulty_by_source`
  - 新增 `model_usage_by_answer_type`

## v3_2 路由规则

v3_2 默认规则来自 v3_1 四组实验的反事实分析：

- `answer_type == numeric` 直接走 Gemma。
- `source == ucinlp/drop` 直接走 Gemma。
- `source == meta-math/GSM8K_zh` 直接走 Gemma。
- `source == mib-bench/arc_challenge` 默认走 middle，不默认 Gemma。
- `source == allenai/sciq`：
  - 问题长度不超过 600 字符且没有 hard cue 时走 small。
  - 否则走 middle。
- `source == mib-bench/arc_easy`：
  - 问题长度不超过 600 字符且没有 hard cue 时走 small。
  - 否则走 middle。

hard cue 当前包括：

- `except`
- `not`
- `least likely`
- `best explains`
- `calculate`
- `infer from passage`
- `two-step`
- `according to the passage`

注意：v3_2 第一版没有为了降低 Gemma 占比而强行下放 DROP/GSM8K_zh/numeric，因为 v3_1 结果显示这些样本上 Gemma 的收益明确。

## 预期分布

基于 v3_1 已有结果做静态反事实估计，v3_2 `risk_aware_routing` 在当前 1000 条 fused dataset 上预计接近：

- small: 约 30%
- middle: 约 50%
- Gemma: 约 20%

这不是硬编码按比例分配，而是由 source/type/题面风险规则自然得到。验收时建议检查：

- Gemma 不应再接近 90%。
- middle 不应接近 80%。
- 如果 `model_usage_distribution` 单一模型超过 60%，需要重新调规则。

## 运行方式

## v3_2.5 新增 baseline

v3_2.5 不改变 v3_2 的模型、prompt、dataset、输出根目录和 `experiment_version`，只把两条已有实验模式加入 v3_2 text-only suite：

| mode | 含义 | 说明 |
| --- | --- | --- |
| `random_routing` | Random Routing | 每个样本随机选择一个模型，作为随机路由基线。 |
| `adaroute_mm_full` | AdaRoute-MM Full | 使用资源感知 `latency_aware` routing，并开启 fallback；在 v3_2 text-only 设置下仍关闭 VLM。 |

默认 suite `text_fusion_v3_2_basic` 现在会按顺序运行 7 个模式：

```text
always_small
always_middle
always_gemma
risk_aware_routing
difficulty_routing
random_routing
adaroute_mm_full
```

新增两个 baseline 的结果仍和 v3_2 结果并列保存在同一个 run 目录下，例如：

```text
data/experiments_v3_2/v3_2_5_added_baselines_001/random_routing/
data/experiments_v3_2/v3_2_5_added_baselines_001/adaroute_mm_full/
```

先确认本地或服务器 Ollama 已有模型：

```powershell
python scripts/check_ollama_models.py --config configs/default.yaml --override-config configs/v3_2_text.yaml
```

运行 v3_2 suite：

```powershell
python scripts/v3_2_run_experiment_suite.py --run-id v3_2_result_001
```

如果服务器上只想快捷补跑 v3_2.5 新增的两条 baseline：

```powershell
python scripts/v3_2_run_experiment_suite.py `
  --suite text_fusion_v3_2_added_baselines `
  --run-id v3_2_5_added_baselines_001
```

Linux / bash 写法：

```bash
python scripts/v3_2_run_experiment_suite.py \
  --suite text_fusion_v3_2_added_baselines \
  --run-id v3_2_5_added_baselines_001
```

默认输出：

```text
data/experiments_v3_2/v3_2_result_001/
```

如果只想跑 v3_2 风险路由，可继续复用底层实验入口：

```powershell
python scripts/run_experiment.py ^
  --mode risk_aware_routing ^
  --dataset data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl ^
  --experiments-dir data/experiments_v3_2 ^
  --run-id v3_2_risk_only ^
  --override-config configs/v3_2_text.yaml ^
  --prompts configs/prompts_v3_2.yaml ^
  --experiment-version v3_2_text
```

## 验证命令

本次代码侧验证建议至少执行：

```powershell
python -m pytest tests
python -m compileall adaroute scripts
python scripts/v3_2_run_experiment_suite.py --help
```

服务器跑完后重点查看：

- `summary.json` 中的 `text_answer.accuracy`
- `model_usage_distribution`
- `model_usage_by_source`
- `difficulty_by_source`
- `average_latency`
- `llm_calls_token_normalized_cost`
