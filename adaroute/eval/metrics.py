from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any

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
    latencies = [float(r.get("latency", {}).get("total", 0.0)) for r in results]
    fallback_counts = [int(r.get("fallback", {}).get("count", 0)) for r in results]
    vlm_calls = [call for row in results for call in row.get("model_calls", []) if call.get("stage") == "vlm"]
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
        "average_latency": mean(latencies) if latencies else 0.0,
        "p50_latency": _percentile(latencies, 0.50),
        "p90_latency": _percentile(latencies, 0.90),
        "p95_latency": _percentile(latencies, 0.95),
        "average_vlm_latency": mean([float(r.get("latency", {}).get("vlm", 0.0)) for r in results]) if results else 0.0,
        "average_router_latency": mean([float(r.get("latency", {}).get("router", 0.0)) for r in results]) if results else 0.0,
        "average_llm_latency": mean([float(r.get("latency", {}).get("llm", 0.0)) for r in results]) if results else 0.0,
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
    if any(row.get("answer_type") in {"yes/no", "yes_no"} for row in results):
        summary["yesno"] = compute_yesno_metrics(results)
    return summary
