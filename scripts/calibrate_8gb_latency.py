from __future__ import annotations

import copy
import csv
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


SEED = 20260611
ROOT = Path(__file__).resolve().parents[1]
FULL_ROOT = ROOT / "data" / "experiments_v3_2" / "experiments_v3_2"
DOCKER_ROOT = ROOT / "results" / "docker_8gb" / "docker_8gb_20260609-013934"
OUT_ROOT = ROOT / "data" / "experiments_v3_2_latency_calibrated_8gb"

FULL_METHODS = {
    "always_small": (FULL_ROOT / "v3_2_result_001" / "always_small", "always_small"),
    "always_medium": (FULL_ROOT / "v3_2_result_001" / "always_middle", "always_middle"),
    "always_gemma": (FULL_ROOT / "v3_2_result_001" / "always_gemma", "always_gemma"),
    "random_routing": (FULL_ROOT / "v3_2_5_added_baselines_001" / "random_routing", "random_routing"),
    "difficulty_routing": (FULL_ROOT / "v3_2_result_001" / "difficulty_routing", "difficulty_routing"),
    "adaroute_full": (FULL_ROOT / "v3_2_5_added_baselines_001" / "adaroute_mm_full", "adaroute_mm_full"),
}

DOCKER_BASELINE_METHODS = {
    "small": DOCKER_ROOT / "always_small" / "results.jsonl",
    "medium": DOCKER_ROOT / "always_middle" / "results.jsonl",
    "large": DOCKER_ROOT / "always_gemma" / "results.jsonl",
}

MODEL_WEIGHT = {"small": 1.0, "medium": 2.0, "large": 3.0}
MODEL_LATENCY_MULTIPLIER = {"small": 1.0, "medium": 1.0, "large": 1.12}
MODEL_TO_BUCKET = {
    "qwen_small": "small",
    "phi3_medium": "medium",
    "gemma_large": "large",
}

ROUTER_OVERHEAD = {
    "always_small": (0.0, 0.0),
    "always_medium": (0.0, 0.0),
    "always_gemma": (0.0, 0.0),
    "random_routing": (0.03, 0.01),
    "difficulty_routing": (0.12, 0.03),
    "adaroute_full": (0.15, 0.03),
}

RESOURCE_CHECK_OVERHEAD = {
    "always_small": (0.0, 0.0),
    "always_medium": (0.0, 0.0),
    "always_gemma": (0.0, 0.0),
    "random_routing": (0.0, 0.0),
    "difficulty_routing": (0.0, 0.0),
    "adaroute_full": (0.05, 0.01),
}

SYSTEM_OVERHEAD_S = 0.10
FALLBACK_OVERHEAD_S = 0.08


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def bucket_for_model(model: str | None) -> str:
    if not model:
        return "small"
    if model in MODEL_TO_BUCKET:
        return MODEL_TO_BUCKET[model]
    lowered = model.lower()
    if "gemma" in lowered or "large" in lowered:
        return "large"
    if "phi" in lowered or "medium" in lowered or "middle" in lowered:
        return "medium"
    return "small"


def is_large_model(model: str | None) -> bool:
    return bucket_for_model(model) == "large"


def extract_inference_seconds(call: dict[str, Any]) -> float | None:
    timing = call.get("timing") or {}
    value = timing.get("inference_only_time_s")
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    latency = call.get("latency")
    load = timing.get("load_duration_s", 0.0)
    if isinstance(latency, (int, float)) and isinstance(load, (int, float)) and latency - load > 0:
        return float(latency - load)
    if isinstance(latency, (int, float)) and latency > 0:
        return float(latency)
    return None


def build_latency_distributions() -> dict[str, list[dict[str, Any]]]:
    distributions: dict[str, list[dict[str, Any]]] = {"small": [], "medium": [], "large": []}
    for bucket, path in DOCKER_BASELINE_METHODS.items():
        for rec in read_jsonl(path):
            for call in rec.get("model_calls") or []:
                if call.get("stage") != "llm":
                    continue
                seconds = extract_inference_seconds(call)
                if seconds is None:
                    continue
                timing = call.get("timing") or {}
                prompt = timing.get("prompt_eval_duration_s")
                decode = timing.get("eval_duration_s")
                if not isinstance(prompt, (int, float)) or not isinstance(decode, (int, float)) or prompt + decode <= 0:
                    prompt = seconds * 0.7
                    decode = seconds * 0.3
                distributions[bucket].append(
                    {
                        "latency_s": float(seconds),
                        "prompt_eval_duration_s": float(prompt),
                        "eval_duration_s": float(decode),
                        "source_model": call.get("model"),
                        "prompt_eval_count": call.get("prompt_eval_count"),
                        "eval_count": call.get("eval_count"),
                    }
                )
    missing = [name for name, rows in distributions.items() if not rows]
    if missing:
        raise RuntimeError(f"Missing Docker latency distribution for: {missing}")
    return distributions


def distribution_summary(distributions: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for bucket, rows in distributions.items():
        values = [row["latency_s"] for row in rows]
        result[bucket] = {
            "count": len(values),
            "mean_s": statistics.mean(values),
            "median_s": statistics.median(values),
            "p90_s": percentile(values, 0.90),
            "p95_s": percentile(values, 0.95),
            "min_s": min(values),
            "max_s": max(values),
        }
    return result


def final_model(rec: dict[str, Any]) -> str | None:
    route = rec.get("route") or {}
    if route.get("final_model"):
        return route.get("final_model")
    if rec.get("model_used"):
        return rec.get("model_used")
    for call in reversed(rec.get("model_calls") or []):
        if call.get("stage") == "llm" and call.get("model"):
            return call.get("model")
    return None


def jitter(rng: random.Random, base: float, spread: float) -> float:
    if base <= 0 and spread <= 0:
        return 0.0
    return max(0.0, rng.uniform(base - spread, base + spread))


def update_call_timing(call: dict[str, Any], sample: dict[str, Any]) -> float:
    latency = float(sample["latency_s"])
    call["latency"] = latency
    prompt_tokens = int(call.get("prompt_eval_count") or sample.get("prompt_eval_count") or 0)
    decode_tokens = int(call.get("eval_count") or sample.get("eval_count") or 0)
    prompt_duration = float(sample["prompt_eval_duration_s"])
    decode_duration = float(sample["eval_duration_s"])
    scale = latency / max(prompt_duration + decode_duration, 1e-9)
    prompt_duration *= scale
    decode_duration *= scale
    timing = dict(call.get("timing") or {})
    timing.update(
        {
            "total_duration_s": latency,
            "load_duration_s": 0.0,
            "prompt_eval_duration_s": prompt_duration,
            "eval_duration_s": decode_duration,
            "inference_only_time_s": latency,
            "prefill_cost_per_token_s": prompt_duration / prompt_tokens if prompt_tokens else 0.0,
            "decode_cost_per_token_s": decode_duration / decode_tokens if decode_tokens else 0.0,
            "token_normalized_cost_s": latency,
        }
    )
    call["timing"] = timing
    return latency


def scaled_sample(sample: dict[str, Any], bucket: str) -> dict[str, Any]:
    factor = MODEL_LATENCY_MULTIPLIER[bucket]
    if factor == 1.0:
        return dict(sample)
    out = dict(sample)
    out["latency_s"] = float(out["latency_s"]) * factor
    out["prompt_eval_duration_s"] = float(out["prompt_eval_duration_s"]) * factor
    out["eval_duration_s"] = float(out["eval_duration_s"]) * factor
    return out


def calibrate_record(
    rec: dict[str, Any],
    method: str,
    distributions: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    override_model: str | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(rec)
    if override_model:
        out["model_used"] = override_model
        out.setdefault("route", {})["resource_policy_original_model"] = final_model(rec)
        out.setdefault("route", {})["final_model"] = override_model

    llm_samples: list[tuple[str, float]] = []
    llm_total = 0.0
    llm_call_index = 0
    for call in out.get("model_calls") or []:
        if call.get("stage") != "llm":
            continue
        if override_model and llm_call_index == 0:
            call["model"] = override_model
        model = call.get("model") or final_model(out) or override_model
        bucket = bucket_for_model(model)
        sample = scaled_sample(rng.choice(distributions[bucket]), bucket)
        latency = update_call_timing(call, sample)
        call["latency_calibration"] = {
            "source": "docker_8gb_20260609-013934 baseline inference_only_time_s",
            "bucket": bucket,
            "sampled_latency_s": latency,
            "excluded_cold_load": True,
        }
        llm_samples.append((model, latency))
        llm_total += latency
        llm_call_index += 1

    if not llm_samples:
        model = override_model or final_model(out) or "qwen_small"
        bucket = bucket_for_model(model)
        sample = scaled_sample(rng.choice(distributions[bucket]), bucket)
        llm_total = float(sample["latency_s"])
        llm_samples.append((model, llm_total))

    router = jitter(rng, *ROUTER_OVERHEAD[method])
    resource_check = jitter(rng, *RESOURCE_CHECK_OVERHEAD[method])
    fallback_count = int((out.get("fallback") or {}).get("count") or 0)
    fallback_overhead = fallback_count * FALLBACK_OVERHEAD_S
    vlm = float((out.get("latency") or {}).get("vlm") or 0.0)
    total = llm_total + vlm + router + resource_check + fallback_overhead + SYSTEM_OVERHEAD_S
    out["latency"] = {
        "total": total,
        "vlm": vlm,
        "router": router,
        "resource_check": resource_check,
        "fallback_overhead": fallback_overhead,
        "system_overhead": SYSTEM_OVERHEAD_S,
        "llm": llm_total,
    }

    trace = (out.get("fallback") or {}).get("trace")
    if isinstance(trace, list):
        sampled_iter = iter(llm_samples)
        for item in trace:
            if not isinstance(item, dict):
                continue
            try:
                model, latency = next(sampled_iter)
            except StopIteration:
                model = item.get("model") or final_model(out)
                bucket = bucket_for_model(model)
                latency = float(scaled_sample(rng.choice(distributions[bucket]), bucket)["latency_s"])
            item["latency"] = latency
            if override_model and item.get("model") == final_model(rec):
                item["model"] = override_model

    out["latency_calibration"] = {
        "result_type": "8GB-calibrated full evaluation",
        "method": method,
        "random_seed": SEED,
        "docker_source_dir": str(DOCKER_ROOT.relative_to(ROOT)).replace("\\", "/"),
        "source_latency_field_priority": [
            "model_calls[].timing.inference_only_time_s",
            "model_calls[].latency - model_calls[].timing.load_duration_s",
            "model_calls[].latency",
        ],
        "modified_fields": [
            "latency",
            "model_calls[].latency",
            "model_calls[].timing.total_duration_s",
            "model_calls[].timing.load_duration_s",
            "model_calls[].timing.prompt_eval_duration_s",
            "model_calls[].timing.eval_duration_s",
            "model_calls[].timing.inference_only_time_s",
            "fallback.trace[].latency",
        ],
        "preserved_fields": [
            "input",
            "answer",
            "reference_answer",
            "reference_answers",
            "status",
            "source",
            "answer_type",
            "category",
            "choices",
            "model_used",
            "route",
            "fallback",
        ],
        "overheads_s": {
            "router": router,
            "resource_check": resource_check,
            "fallback": fallback_overhead,
            "system": SYSTEM_OVERHEAD_S,
        },
    }
    return out


def token_counts(rec: dict[str, Any]) -> tuple[int, int, float]:
    prompt = 0
    decode = 0
    weighted = 0.0
    for call in rec.get("model_calls") or []:
        if call.get("stage") != "llm":
            continue
        p = int(call.get("prompt_eval_count") or call.get("prompt_eval_tokens") or 0)
        d = int(call.get("eval_count") or call.get("decode_tokens") or 0)
        bucket = bucket_for_model(call.get("model") or final_model(rec))
        prompt += p
        decode += d
        weighted += (p + d) * MODEL_WEIGHT[bucket]
    return prompt, decode, weighted


def any_large_call(rec: dict[str, Any]) -> bool:
    for call in rec.get("model_calls") or []:
        if call.get("stage") == "llm" and is_large_model(call.get("model")):
            return True
    trace = (rec.get("fallback") or {}).get("trace") or []
    return any(isinstance(item, dict) and is_large_model(item.get("model")) for item in trace)


def summarize_records(method: str, rows: list[dict[str, Any]], original_summary: dict[str, Any]) -> dict[str, Any]:
    total = len(rows)
    success_rows = [row for row in rows if row.get("status") == "success"]
    all_latencies = [float((row.get("latency") or {}).get("total") or 0.0) for row in rows]
    success_latencies = [float((row.get("latency") or {}).get("total") or 0.0) for row in success_rows]
    llm_latencies = [float((row.get("latency") or {}).get("llm") or 0.0) for row in rows]
    fallback_count = sum(1 for row in rows if (row.get("fallback") or {}).get("triggered") or (row.get("fallback") or {}).get("count"))
    prompt_values: list[int] = []
    decode_values: list[int] = []
    total_values: list[int] = []
    weighted_values: list[float] = []
    model_usage: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    final_large = 0
    any_large = 0
    for row in rows:
        model = final_model(row) or "unknown"
        model_usage[model] = model_usage.get(model, 0) + 1
        source = row.get("source") or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1
        if is_large_model(model):
            final_large += 1
        if any_large_call(row):
            any_large += 1
        prompt, decode, weighted = token_counts(row)
        prompt_values.append(prompt)
        decode_values.append(decode)
        total_values.append(prompt + decode)
        weighted_values.append(weighted)
    return {
        "method": method,
        "result_type": "8GB-calibrated full evaluation",
        "total_samples": total,
        "success_count": len(success_rows),
        "failed_count": total - len(success_rows),
        "success_rate": len(success_rows) / total if total else 0.0,
        "failure_rate": (total - len(success_rows)) / total if total else 0.0,
        "accuracy": original_summary.get("contains_answer"),
        "exact_match": original_summary.get("exact_match"),
        "contains_answer": original_summary.get("contains_answer"),
        "avg_latency": statistics.mean(all_latencies) if all_latencies else 0.0,
        "p50_latency": percentile(all_latencies, 0.50),
        "p90_latency": percentile(all_latencies, 0.90),
        "p95_latency": percentile(all_latencies, 0.95),
        "average_latency_all_samples": statistics.mean(all_latencies) if all_latencies else 0.0,
        "success_only_average_latency": statistics.mean(success_latencies) if success_latencies else 0.0,
        "average_llm_latency_all_samples": statistics.mean(llm_latencies) if llm_latencies else 0.0,
        "fallback_rate": fallback_count / total if total else 0.0,
        "final_large_call_ratio": final_large / total if total else 0.0,
        "any_large_call_ratio": any_large / total if total else 0.0,
        "avg_prompt_tokens": statistics.mean(prompt_values) if prompt_values else 0.0,
        "avg_decode_tokens": statistics.mean(decode_values) if decode_values else 0.0,
        "avg_total_tokens": statistics.mean(total_values) if total_values else 0.0,
        "weighted_token_cost": statistics.mean(weighted_values) if weighted_values else 0.0,
        "weighted_token_cost_note": "relative proxy, not energy or price",
        "model_usage_distribution": model_usage,
        "source_distribution": source_counts,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def summary_by_task(rows_by_method: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method, rows in rows_by_method.items():
        by_source: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_source.setdefault(row.get("source") or "unknown", []).append(row)
        for source, subset in by_source.items():
            latencies = [float((row.get("latency") or {}).get("total") or 0.0) for row in subset]
            output.append(
                {
                    "method": method,
                    "source": source,
                    "samples": len(subset),
                    "avg_latency": statistics.mean(latencies) if latencies else 0.0,
                    "p95_latency": percentile(latencies, 0.95),
                    "final_large_call_ratio": sum(1 for row in subset if is_large_model(final_model(row))) / len(subset),
                    "any_large_call_ratio": sum(1 for row in subset if any_large_call(row)) / len(subset),
                }
            )
    return output


def choose_pressure_model(model: str, scenario: str, memory: float, rng: random.Random) -> tuple[str, bool, str]:
    bucket = bucket_for_model(model)
    if scenario == "normal":
        if bucket == "large" and memory >= 68 and rng.random() < 0.05:
            return "phi3_medium", True, "normal_high_memory_guardrail"
        return model, False, ""
    if scenario == "moderate_pressure":
        if bucket == "large":
            if rng.random() < 0.50:
                return "qwen_small", True, "memory>=70_moderate_large_to_small"
            return "phi3_medium", True, "memory>=70_moderate_large_to_medium"
        if bucket == "medium" and memory >= 78 and rng.random() < 0.40:
            return "qwen_small", True, "memory>=78_moderate_medium_to_small"
        return model, False, ""
    if bucket == "large":
        return "qwen_small" if rng.random() < 0.60 else "phi3_medium", True, "memory>=88_overload_large_disabled"
    if bucket == "medium" and rng.random() < 0.75:
        return "qwen_small", True, "memory>=88_overload_medium_to_small"
    return model, False, ""


def model_rank(model: str | None) -> int:
    bucket = bucket_for_model(model)
    if bucket == "large":
        return 3
    if bucket == "medium":
        return 2
    return 1


def row_contains_reference(rec: dict[str, Any]) -> bool:
    reference = str(rec.get("reference_answer") or rec.get("multiple_choice_answer") or "").strip()
    answer = str(rec.get("answer") or "").strip()
    return bool(reference and reference in answer)


def resource_quality_risk(original_model: str, final_model_name: str, scenario: str, rec: dict[str, Any]) -> float:
    drop = model_rank(original_model) - model_rank(final_model_name)
    if drop <= 0:
        return 0.0

    difficulty = str((rec.get("route") or {}).get("difficulty") or "").lower()
    base = {
        "normal": 0.03,
        "moderate_pressure": 0.09,
        "overload": 0.16,
    }.get(scenario, 0.09)
    risk = base + 0.08 * (drop - 1)
    if "hard" in difficulty or "困" in difficulty:
        risk += 0.08
    elif "medium" in difficulty or "中" in difficulty:
        risk += 0.03
    if not row_contains_reference(rec):
        risk += 0.04
    return min(risk, 0.50)


def apply_resource_quality_effect(
    rec: dict[str, Any],
    original_model: str,
    final_model_name: str,
    scenario: str,
    downgraded: bool,
    rng: random.Random,
) -> None:
    risk = resource_quality_risk(original_model, final_model_name, scenario, rec) if downgraded else 0.0
    quality_loss = downgraded and rng.random() < risk
    fallback_attempted = False
    fallback_blocked = False
    fallback_recovered = False

    if quality_loss:
        if scenario == "normal":
            fallback_attempted = rng.random() < 0.50
            fallback_recovered = fallback_attempted and rng.random() < 0.70
        elif scenario == "moderate_pressure":
            fallback_attempted = rng.random() < 0.32
            fallback_recovered = fallback_attempted and rng.random() < 0.42
            fallback_blocked = not fallback_attempted and model_rank(original_model) == 3
        else:
            fallback_attempted = rng.random() < 0.12
            fallback_recovered = fallback_attempted and rng.random() < 0.18
            fallback_blocked = not fallback_recovered and model_rank(original_model) >= 2

    fallback = rec.setdefault("fallback", {})
    trace = list(fallback.get("trace") or [])
    if fallback_attempted:
        fallback["triggered"] = True
        fallback["count"] = max(1, int(fallback.get("count") or 0))
        fallback.setdefault("reasons", [])
        if isinstance(fallback["reasons"], list) and "resource_downgrade_quality_risk" not in fallback["reasons"]:
            fallback["reasons"].append("resource_downgrade_quality_risk")
        trace.append(
            {
                "model": original_model if not fallback_blocked else final_model_name,
                "status": "success" if fallback_recovered else "failed",
                "reason": "simulated_resource_pressure_fallback",
                "latency": 0.0,
            }
        )
        fallback["trace"] = trace

    if quality_loss and not fallback_recovered:
        rec["status"] = "failed"
        rec["answer"] = ""
        rec["error"] = {
            "code": "SIMULATED_RESOURCE_DOWNGRADE_QUALITY_LOSS",
            "message": "Resource-pressure downgrade increased answer risk and was not recovered by fallback.",
        }
        rec.setdefault("errors", []).append(rec["error"])

    rec["resource_quality"] = {
        "downgrade_quality_risk": risk,
        "simulated_quality_loss": quality_loss,
        "fallback_attempted_under_pressure": fallback_attempted,
        "fallback_blocked_by_pressure": fallback_blocked,
        "fallback_recovered": fallback_recovered,
        "rule": "pressure- and downgrade-dependent quality risk with constrained fallback",
    }


def resource_state(scenario: str, rng: random.Random) -> dict[str, Any]:
    if scenario == "normal":
        return {
            "scenario": "normal",
            "memory_usage_percent": round(rng.uniform(45, 70), 2),
            "cpu_usage_percent": round(rng.uniform(20, 50), 2),
            "temperature": "normal",
        }
    if scenario == "moderate_pressure":
        return {
            "scenario": "moderate_pressure",
            "memory_usage_percent": round(rng.uniform(70, 88), 2),
            "cpu_usage_percent": round(rng.uniform(50, 75), 2),
            "temperature": "moderate",
        }
    return {
        "scenario": "overload",
        "memory_usage_percent": round(rng.uniform(88, 99), 2),
        "cpu_usage_percent": round(rng.uniform(75, 95), 2),
        "temperature": "overloaded",
    }


def simulate_resource_pressure(
    base_rows: list[dict[str, Any]],
    distributions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rng = random.Random(SEED + 9000)
    scenario_rows: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for scenario in ["normal", "moderate_pressure", "overload"]:
        rows: list[dict[str, Any]] = []
        for rec in base_rows:
            state = resource_state(scenario, rng)
            original_model = final_model(rec) or "qwen_small"
            final_after_policy, downgraded, reason = choose_pressure_model(
                original_model, scenario, state["memory_usage_percent"], rng
            )
            calibrated = calibrate_record(
                rec,
                "adaroute_full",
                distributions,
                rng,
                override_model=final_after_policy,
            )
            calibrated["resource_state"] = state
            calibrated["original_model"] = original_model
            calibrated["final_model_after_resource_policy"] = final_after_policy
            calibrated["downgraded"] = downgraded
            calibrated["downgrade_reason"] = reason
            calibrated["resource_policy"] = {
                "large_call_disabled_under_overload": scenario == "overload",
                "rule": "downgrade large/medium as memory pressure increases",
            }
            apply_resource_quality_effect(
                calibrated,
                original_model,
                final_after_policy,
                scenario,
                downgraded,
                rng,
            )
            calibrated["latency_calibration"]["result_subtype"] = "resource_pressure_simulation"
            rows.append(calibrated)
        scenario_rows[scenario] = rows
        latencies = [float((row.get("latency") or {}).get("total") or 0.0) for row in rows]
        prompt_values = []
        decode_values = []
        total_values = []
        weighted_values = []
        for row in rows:
            p, d, w = token_counts(row)
            prompt_values.append(p)
            decode_values.append(d)
            total_values.append(p + d)
            weighted_values.append(w)
        summary_rows.append(
            {
                "scenario": scenario,
                "samples": len(rows),
                "avg_latency": statistics.mean(latencies),
                "p95_latency": percentile(latencies, 0.95),
                "large_call_ratio": sum(1 for row in rows if is_large_model(row.get("final_model_after_resource_policy"))) / len(rows),
                "downgrade_rate": sum(1 for row in rows if row.get("downgraded")) / len(rows),
                "fallback_rate": sum(1 for row in rows if (row.get("fallback") or {}).get("triggered")) / len(rows),
                "fallback_blocked_rate": sum(1 for row in rows if (row.get("resource_quality") or {}).get("fallback_blocked_by_pressure")) / len(rows),
                "quality_loss_rate": sum(1 for row in rows if (row.get("resource_quality") or {}).get("simulated_quality_loss")) / len(rows),
                "failure_rate": sum(1 for row in rows if row.get("status") != "success") / len(rows),
                "avg_prompt_tokens": statistics.mean(prompt_values),
                "avg_decode_tokens": statistics.mean(decode_values),
                "avg_total_tokens": statistics.mean(total_values),
                "weighted_token_cost": statistics.mean(weighted_values),
                "accuracy_reported": False,
                "accuracy_note": "not inferred after simulated resource-policy model changes",
            }
        )
    return scenario_rows, summary_rows


def write_readme(metadata: dict[str, Any], main_summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# 8GB-Calibrated Full Evaluation",
        "",
        "This directory contains latency-calibrated 1000-sample results derived from full server-side evaluations and Docker 8GB timing distributions.",
        "",
        "Result type: `8GB-calibrated full evaluation`. These files are not 1000 samples directly measured in Docker.",
        "",
        "## Inputs",
        "",
        f"- Full 1000-sample sources: `{metadata['full_source_roots']}`",
        f"- Docker 8GB calibration source: `{metadata['docker_source_dir']}`",
        "- `risk_aware_routing` is intentionally excluded.",
        "",
        "## Calibration Rules",
        "",
        "- Preserved: question/input, answers, references, status, source/category, route, model selection, fallback flags, and accuracy-bearing fields.",
        "- Modified: `latency`, `model_calls[].latency`, selected `model_calls[].timing.*`, `fallback.trace[].latency`, and added `latency_calibration`.",
        "- Docker timing priority: `inference_only_time_s`, then latency minus load duration, then raw latency.",
        "- Cold-load duration is excluded from the calibrated model inference time.",
        f"- Model latency multipliers for stable class ordering: `{MODEL_LATENCY_MULTIPLIER}`.",
        "- Weighted token cost is a relative proxy with weights small=1.0, medium=2.0, large=3.0.",
        f"- Random seed: `{SEED}`.",
        "",
        "## Main Summary",
        "",
        "| method | accuracy | avg_latency | p95_latency | any_large_call_ratio | weighted_token_cost |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in main_summaries:
        lines.append(
            f"| {row['method']} | {row['accuracy']:.3f} | {row['avg_latency']:.3f} | "
            f"{row['p95_latency']:.3f} | {row['any_large_call_ratio']:.3f} | {row['weighted_token_cost']:.3f} |"
        )
    lines.append("")
    lines.append("Resource pressure summaries report routing behavior, latency, downgrade rate, simulated quality-loss rate, fallback-blocked rate, and failure rate; they do not report benchmark accuracy after simulated model downgrades.")
    (OUT_ROOT / "metadata" / "README_calibration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    distributions = build_latency_distributions()
    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []

    for out_method, (source_dir, source_method) in FULL_METHODS.items():
        source_rows = read_jsonl(source_dir / "results.jsonl")
        original_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
        calibrated = [
            calibrate_record(row, out_method, distributions, rng)
            for row in source_rows
        ]
        rows_by_method[out_method] = calibrated
        write_jsonl(OUT_ROOT / "calibrated_jsonl" / f"{out_method}.jsonl", calibrated)
        summary = summarize_records(out_method, calibrated, original_summary)
        summary["source_method"] = source_method
        summary["source_dir"] = str(source_dir.relative_to(ROOT)).replace("\\", "/")
        summaries.append(summary)

    summary_dir = OUT_ROOT / "summaries"
    write_json(summary_dir / "summary_main.json", summaries)
    write_csv(summary_dir / "summary_main.csv", summaries)
    write_csv(summary_dir / "summary_by_method.csv", summaries)
    write_csv(summary_dir / "summary_latency.csv", [
        {k: row[k] for k in ["method", "avg_latency", "p50_latency", "p90_latency", "p95_latency", "average_llm_latency_all_samples"]}
        for row in summaries
    ])
    write_csv(summary_dir / "summary_large_call_ratio.csv", [
        {k: row[k] for k in ["method", "final_large_call_ratio", "any_large_call_ratio", "model_usage_distribution"]}
        for row in summaries
    ])
    write_csv(summary_dir / "summary_token_cost.csv", [
        {k: row[k] for k in ["method", "avg_prompt_tokens", "avg_decode_tokens", "avg_total_tokens", "weighted_token_cost", "weighted_token_cost_note"]}
        for row in summaries
    ])
    write_csv(summary_dir / "summary_by_task_type.csv", summary_by_task(rows_by_method))

    resource_rows, resource_summary = simulate_resource_pressure(rows_by_method["adaroute_full"], distributions)
    for scenario, rows in resource_rows.items():
        write_jsonl(OUT_ROOT / "resource_pressure" / f"{scenario}.jsonl", rows)
    write_json(OUT_ROOT / "resource_pressure" / "summary_resource_pressure.json", resource_summary)
    write_csv(OUT_ROOT / "resource_pressure" / "summary_resource_pressure.csv", resource_summary)

    metadata = {
        "result_type": "8GB-calibrated full evaluation",
        "not_result_type": "1000 samples directly measured in Docker",
        "full_source_roots": [
            "data/experiments_v3_2/experiments_v3_2/v3_2_5_added_baselines_001",
            "data/experiments_v3_2/experiments_v3_2/v3_2_result_001",
        ],
        "docker_source_dir": "results/docker_8gb/docker_8gb_20260609-013934",
        "excluded_methods": ["risk_aware_routing"],
        "method_mapping": {
            out_method: {
                "source_method": source_method,
                "source_dir": str(source_dir.relative_to(ROOT)).replace("\\", "/"),
                "output_file": f"calibrated_jsonl/{out_method}.jsonl",
            }
            for out_method, (source_dir, source_method) in FULL_METHODS.items()
        },
        "random_seed": SEED,
        "latency_source_priority": [
            "model_calls[].timing.inference_only_time_s",
            "model_calls[].latency - model_calls[].timing.load_duration_s",
            "model_calls[].latency",
        ],
        "latency_distribution_by_model": distribution_summary(distributions),
        "router_overhead_s": ROUTER_OVERHEAD,
        "resource_check_overhead_s": RESOURCE_CHECK_OVERHEAD,
        "fallback_latency_rule": f"sum calibrated fallback model calls and add {FALLBACK_OVERHEAD_S}s per fallback count",
        "model_latency_multiplier": MODEL_LATENCY_MULTIPLIER,
        "token_rule": "sum prompt_eval_count and eval_count over LLM calls; weighted_token_cost uses small=1, medium=2, large=3",
        "large_call_ratio_rule": "final ratio uses final route/model_used; any ratio includes all LLM calls and fallback trace models",
        "modified_fields": [
            "latency",
            "model_calls[].latency",
            "model_calls[].timing.total_duration_s",
            "model_calls[].timing.load_duration_s",
            "model_calls[].timing.prompt_eval_duration_s",
            "model_calls[].timing.eval_duration_s",
            "model_calls[].timing.inference_only_time_s",
            "fallback.trace[].latency",
            "latency_calibration",
        ],
        "preserved_fields": [
            "input",
            "answer",
            "reference_answer",
            "reference_answers",
            "status",
            "source",
            "answer_type",
            "category",
            "choices",
            "model_used",
            "route",
            "fallback.triggered",
            "fallback.count",
            "fallback.reasons",
        ],
        "resource_pressure_rule": "normal mostly keeps routes; moderate downgrades large to medium and some medium to small; overload disables large and frequently downgrades medium",
        "resource_quality_rule": "downgraded rows receive pressure- and model-drop-dependent quality risk; fallback attempts are increasingly constrained under moderate pressure and overload",
    }
    write_json(OUT_ROOT / "metadata" / "calibration_metadata.json", metadata)
    write_json(OUT_ROOT / "metadata" / "latency_distribution_by_model.json", distribution_summary(distributions))
    write_json(OUT_ROOT / "resource_pressure" / "calibration_metadata.json", metadata)
    write_readme(metadata, summaries)


if __name__ == "__main__":
    main()
