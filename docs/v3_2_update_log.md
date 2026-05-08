# AdaRoute-MM v3_2 Update Log

## Goal

v3_2 builds on v3_1 without overwriting the v3_1 configs, scripts, or result directories. The main change is to move routing from difficulty classification to model adequacy risk routing:

- `small_ok` means the qwen 1.5B small model is expected to be enough.
- `middle_ok` means the middle model is expected to be enough.
- `large_required` means Gemma is needed because small and middle are high risk.

## What Changed

- Added `configs/v3_2_text.yaml`.
  - Sets both `router_small` and `qwen_small` to `qwen2.5:1.5b-instruct-q4_1`.
  - Enables `router.risk_gate`.
  - Keeps numeric, DROP, and GSM8K_zh on `large_required`.
  - Routes ARC Challenge to `middle_ok`.
  - Treats SciQ and ARC Easy as dynamic candidates for `small_ok` vs `middle_ok`; `risk_static_routing` uses their configured default `middle_ok`.

- Added `configs/prompts_v3_2.yaml`.
  - Router prompt asks for model adequacy labels instead of easy/medium/hard.
  - LLM prompt keeps the v3-family `FINAL_ANSWER: <answer>` format.

- Added `scripts/v3_2_run_experiment_suite.py`.
  - Default suite: `text_fusion_v3_2_basic`.
  - Default output root: `data/experiments_v3_2`.
  - Default dataset reuses the v3_1 fusion dataset:
    `data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl`.

- Extended routing outputs.
  - Each result route now includes `route_source`, `route_reason`, `static_gate`, and `dynamic_gate`.
  - This is intended for later route-by-source audit and paper figures.

- Extended summary metrics.
  - Adds model usage rates: `small_usage_rate`, `middle_usage_rate`, `gemma_usage_rate`.
  - Adds gate rates: `static_gate_rate`, `dynamic_gate_rate`.
  - Adds `model_switch_count` and `model_switch_rate`.
  - Adds `route_reason_distribution`.
  - Adds `text_answer.routing_by_source` with per-source model usage, label distribution, reason distribution, and accuracy.

## Suites

Default v3_2 suite:

```powershell
python scripts/v3_2_run_experiment_suite.py
```

Runs:

- `always_small`
- `always_middle`
- `always_gemma`
- `risk_static_routing`
- `risk_dynamic_routing`

Optional suite with the old difficulty router for same-version comparison:

```powershell
python scripts/v3_2_run_experiment_suite.py --suite text_fusion_v3_2_with_legacy
```

Runs the default modes plus `difficulty_routing`.

## Expected Behavior From v3_1 Base4 Analysis

The v3_1 base4 summaries showed:

- SciQ: middle and Gemma both around 0.98, so Gemma is not useful by default.
- ARC Challenge: middle was slightly above Gemma, so default Gemma is wasteful.
- ARC Easy: Gemma is higher than middle, but only by about 3 percentage points.
- DROP and GSM8K_zh: Gemma has clear gains and should stay the default.

For the first server run, compare:

- `risk_static_routing` against `always_gemma` for Gemma usage reduction.
- `risk_static_routing` against `always_middle` for DROP/GSM8K protection.
- `risk_dynamic_routing` against `risk_static_routing` to see whether dynamic small-vs-middle routing is worth the router call.

## Verification

Local validation commands:

```powershell
python -m pytest tests
python -m compileall adaroute scripts main.py
python scripts/v3_2_run_experiment_suite.py --help
```

Small smoke run:

```powershell
python scripts/v3_2_run_experiment_suite.py --run-id v3_2_smoke --no-resume
```

The smoke run uses the full default dataset unless you pass a smaller JSONL with `--dataset`.
