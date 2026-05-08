from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

from adaroute.eval.text_answer import score_text_answer
from adaroute.eval.yesno import compute_yesno_metrics


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * pct))
    return values[index]


def compute_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    successes = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    latencies = [float(r.get("latency", {}).get("total", 0.0)) for r in results]
    success_latencies = [float(r.get("latency", {}).get("total", 0.0)) for r in successes]
    fallback_counts = [int(r.get("fallback", {}).get("count", 0)) for r in results]
    vlm_calls = [call for row in results for call in row.get("model_calls", []) if call.get("stage") == "vlm"]
    model_calls = [call for row in results for call in row.get("model_calls", []) if not call.get("skipped")]
    llm_calls = [call for call in model_calls if call.get("stage") == "llm"]
    cached_vlm_calls = [call for call in vlm_calls if call.get("cached")]
    unique_images = {row.get("input", {}).get("image_path") for row in results if row.get("input", {}).get("image_path")}
    with_answer = [r for r in results if r.get("reference_answer")]
    exact = 0
    contains = 0
    for row in with_answer:
        ref = str(row.get("reference_answer", "")).strip()
        pred = str(row.get("answer", "")).strip()
        if ref and pred == ref:
            exact += 1
        if ref and ref in pred:
            contains += 1

    summary = {
        "total_samples": total,
        "success_count": len(successes),
        "failed_count": total - len(successes),
        "success_rate": len(successes) / total if total else 0.0,
        "average_latency": mean(success_latencies) if success_latencies else 0.0,
        "p50_latency": _percentile(success_latencies, 0.50),
        "p90_latency": _percentile(success_latencies, 0.90),
        "p95_latency": _percentile(success_latencies, 0.95),
        "average_latency_all_samples": mean(latencies) if latencies else 0.0,
        "success_only_average_latency": mean(success_latencies) if success_latencies else 0.0,
        "success_only_p50_latency": _percentile(success_latencies, 0.50),
        "success_only_p90_latency": _percentile(success_latencies, 0.90),
        "success_only_p95_latency": _percentile(success_latencies, 0.95),
        "failed_latency_excluded_count": len(failed),
        "average_vlm_latency": mean([float(r.get("latency", {}).get("vlm", 0.0)) for r in successes]) if successes else 0.0,
        "average_router_latency": mean([float(r.get("latency", {}).get("router", 0.0)) for r in successes]) if successes else 0.0,
        "average_llm_latency": mean([float(r.get("latency", {}).get("llm", 0.0)) for r in successes]) if successes else 0.0,
        "average_vlm_latency_all_samples": mean([float(r.get("latency", {}).get("vlm", 0.0)) for r in results]) if results else 0.0,
        "average_router_latency_all_samples": mean([float(r.get("latency", {}).get("router", 0.0)) for r in results]) if results else 0.0,
        "average_llm_latency_all_samples": mean([float(r.get("latency", {}).get("llm", 0.0)) for r in results]) if results else 0.0,
        "fallback_rate": sum(1 for r in results if r.get("fallback", {}).get("triggered")) / total if total else 0.0,
        "average_fallback_count": mean(fallback_counts) if fallback_counts else 0.0,
        "vlm_call_count": len(vlm_calls),
        "vlm_cache_hit_count": len(cached_vlm_calls),
        "vlm_cache_hit_rate": len(cached_vlm_calls) / len(vlm_calls) if vlm_calls else 0.0,
        "unique_image_count": len(unique_images),
        "vlm_calls_per_sample": len(vlm_calls) / total if total else 0.0,
        "model_usage_distribution": dict(Counter(r.get("route", {}).get("final_model") or r.get("model_used") for r in results)),
        "difficulty_distribution": dict(Counter(r.get("route", {}).get("difficulty") for r in results)),
        "exact_match": exact / len(with_answer) if with_answer else None,
        "contains_answer": contains / len(with_answer) if with_answer else None,
    }
    summary["model_usage_by_source"] = {}
    summary["difficulty_by_source"] = {}
    summary["model_usage_by_answer_type"] = {}
    for group_key, output_key in (("source", "model_usage_by_source"), ("answer_type", "model_usage_by_answer_type")):
        for group_value in sorted({str(row.get(group_key) or "unknown") for row in results}):
            group_rows = [row for row in results if str(row.get(group_key) or "unknown") == group_value]
            summary[output_key][group_value] = dict(
                Counter(row.get("route", {}).get("final_model") or row.get("model_used") for row in group_rows)
            )
    for group_value in sorted({str(row.get("source") or "unknown") for row in results}):
        group_rows = [row for row in results if str(row.get("source") or "unknown") == group_value]
        summary["difficulty_by_source"][group_value] = dict(Counter(row.get("route", {}).get("difficulty") for row in group_rows))
    timed_calls = [call for call in model_calls if isinstance(call.get("timing"), dict)]
    timed_llm_calls = [call for call in llm_calls if isinstance(call.get("timing"), dict)]
    success_model_calls = [call for row in successes for call in row.get("model_calls", []) if not call.get("skipped")]
    success_llm_calls = [call for call in success_model_calls if call.get("stage") == "llm"]
    success_timed_calls = [call for call in success_model_calls if isinstance(call.get("timing"), dict)]
    success_timed_llm_calls = [call for call in success_llm_calls if isinstance(call.get("timing"), dict)]
    for label, calls in (("all_model_calls", timed_calls), ("llm_calls", timed_llm_calls)):
        summary[f"{label}_inference_only_time"] = (
            sum(float(call["timing"].get("inference_only_time_s", 0.0)) for call in calls) if calls else 0.0
        )
        summary[f"{label}_average_inference_only_time"] = (
            mean([float(call["timing"].get("inference_only_time_s", 0.0)) for call in calls]) if calls else 0.0
        )
        summary[f"{label}_token_normalized_cost"] = (
            sum(float(call["timing"].get("token_normalized_cost_s", 0.0)) for call in calls) if calls else 0.0
        )
        summary[f"{label}_average_prefill_cost_per_token"] = (
            mean([float(call["timing"].get("prefill_cost_per_token_s", 0.0)) for call in calls]) if calls else 0.0
        )
        summary[f"{label}_average_decode_cost_per_token"] = (
            mean([float(call["timing"].get("decode_cost_per_token_s", 0.0)) for call in calls]) if calls else 0.0
        )
    for label, calls in (("success_only_all_model_calls", success_timed_calls), ("success_only_llm_calls", success_timed_llm_calls)):
        summary[f"{label}_average_inference_only_time"] = (
            mean([float(call["timing"].get("inference_only_time_s", 0.0)) for call in calls]) if calls else 0.0
        )
        summary[f"{label}_token_normalized_cost"] = (
            sum(float(call["timing"].get("token_normalized_cost_s", 0.0)) for call in calls) if calls else 0.0
        )
    summary["prompt_eval_tokens"] = sum(int(call.get("prompt_eval_count") or 0) for call in model_calls)
    summary["decode_tokens"] = sum(int(call.get("eval_count") or 0) for call in model_calls)

    text_eval_rows = [
        row
        for row in results
        if row.get("reference_answer") and str(row.get("answer_type") or "").lower() not in {"yes/no", "yes_no", ""}
    ]
    if text_eval_rows:
        scored = [score_text_answer(row) for row in text_eval_rows]
        evaluated = [item for item in scored if item.get("evaluated")]
        correct = [item for item in evaluated if item.get("correct")]
        summary["text_answer"] = {
            "total_with_reference": len(text_eval_rows),
            "total_evaluated": len(evaluated),
            "accuracy": len(correct) / len(evaluated) if evaluated else 0.0,
            "answer_parse_failure_rate": sum(1 for item in evaluated if not item.get("predicted_answer")) / len(evaluated)
            if evaluated
            else 0.0,
            "by_answer_type": {},
            "by_source": {},
        }
        for group_key, output_key in (("answer_type", "by_answer_type"), ("source", "by_source")):
            for group_value in sorted({str(row.get(group_key) or "unknown") for row in text_eval_rows}):
                group_rows = [row for row in text_eval_rows if str(row.get(group_key) or "unknown") == group_value]
                group_scored = [score_text_answer(row) for row in group_rows]
                group_eval = [item for item in group_scored if item.get("evaluated")]
                summary["text_answer"][output_key][group_value] = {
                    "total": len(group_eval),
                    "accuracy": sum(1 for item in group_eval if item.get("correct")) / len(group_eval) if group_eval else 0.0,
                    "parse_failure_rate": sum(1 for item in group_eval if not item.get("predicted_answer")) / len(group_eval)
                    if group_eval
                    else 0.0,
                }
    if any(row.get("answer_type") in {"yes/no", "yes_no"} for row in results):
        summary["yesno"] = compute_yesno_metrics(results)
    return summary
