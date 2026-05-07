from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from adaroute.eval.text_answer import score_text_answer
from adaroute.utils.io import ensure_dir, read_jsonl, write_json


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    import json

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_result_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id") or row.get("request_id") or "")
        if not sample_id:
            continue
        scored = score_text_answer(row)
        scores[sample_id] = {
            "correct": bool(scored.get("correct")),
            "parsed": bool(scored.get("predicted_answer")),
            "predicted_answer": scored.get("predicted_answer"),
            "status": row.get("status"),
            "model_used": row.get("model_used") or row.get("route", {}).get("final_model"),
        }
    return scores


def select_verified_numeric(
    *,
    dataset_path: str | Path,
    small_results: str | Path,
    gemma_results: str | Path,
    route_results: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    target_size: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    rng = random.Random(seed)
    dataset_rows = {str(row["id"]): row for row in read_jsonl(dataset_path)}
    small_scores = _load_result_scores(small_results)
    gemma_scores = _load_result_scores(gemma_results)
    route_scores = _load_result_scores(route_results)

    verified: list[dict[str, Any]] = []
    filler: list[dict[str, Any]] = []
    removed_reasons: Counter[str] = Counter()
    for sample_id, row in dataset_rows.items():
        if row.get("answer_type") != "numeric":
            continue
        small = small_scores.get(sample_id)
        gemma = gemma_scores.get(sample_id)
        route = route_scores.get(sample_id)
        if not small or not gemma or not route:
            removed_reasons["missing_baseline_result"] += 1
            continue
        if not small.get("parsed"):
            removed_reasons["small_parse_failure"] += 1
            continue
        if (not small.get("correct")) and (gemma.get("correct") or route.get("correct")):
            enriched = dict(row)
            enriched["selection_bucket"] = "verified_numeric"
            enriched["selection_reason"] = "small_wrong_and_gemma_or_route_correct"
            enriched["baseline_scores"] = {
                "always_small": small,
                "always_gemma": gemma,
                "difficulty_routing": route,
            }
            verified.append(enriched)
        elif row.get("source") == "meta-math/GSM8K_zh":
            enriched = dict(row)
            enriched["selection_bucket"] = "gsm8k_numeric_filler"
            enriched["selection_reason"] = "numeric_filler_after_verified_shortfall"
            enriched["baseline_scores"] = {
                "always_small": small,
                "always_gemma": gemma,
                "difficulty_routing": route,
            }
            filler.append(enriched)
        else:
            removed_reasons["not_verified_numeric"] += 1

    rng.shuffle(verified)
    rng.shuffle(filler)
    selected = verified[:target_size]
    filler_needed = max(0, target_size - len(selected))
    selected.extend(filler[:filler_needed])
    selected = selected[:target_size]
    rng.shuffle(selected)
    write_jsonl(output_path, selected)

    report = {
        "dataset": str(dataset_path),
        "small_results": str(small_results),
        "gemma_results": str(gemma_results),
        "route_results": str(route_results),
        "output": str(output_path),
        "target_size": target_size,
        "selected_size": len(selected),
        "verified_available": len(verified),
        "filler_available": len(filler),
        "selected_bucket_counts": dict(Counter(row.get("selection_bucket") for row in selected)),
        "selected_source_counts": dict(Counter(row.get("source") for row in selected)),
        "removed_reason_counts": dict(removed_reasons),
    }
    report_target = Path(report_path) if report_path else Path(output_path).with_suffix(".report.json")
    write_json(report_target, report)
    report["report_path"] = str(report_target)
    return report
