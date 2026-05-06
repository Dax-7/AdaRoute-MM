from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.eval.text_answer import normalize_text, score_text_answer
from adaroute.utils.io import ensure_dir, read_jsonl


DEFAULT_BUCKET_RATIOS = {
    "small_correct": 0.30,
    "medium_large_better": 0.45,
    "all_difficult": 0.15,
    "robustness_challenge": 0.10,
}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    import json

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _result_paths(run_dir: Path, explicit: list[str]) -> list[Path]:
    if explicit:
        return [Path(path) for path in explicit]
    return sorted(run_dir.glob("*/results.jsonl"))


def _mode_name(path: Path) -> str:
    return path.parent.name


def _load_scores(paths: list[Path]) -> dict[str, dict[str, dict[str, Any]]]:
    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        mode = _mode_name(path)
        for row in read_jsonl(path):
            sample_id = str(row.get("sample_id") or row.get("request_id") or "")
            if not sample_id:
                continue
            scored = score_text_answer(row)
            by_sample[sample_id][mode] = {
                "mode": mode,
                "correct": bool(scored.get("correct")),
                "parsed": bool(scored.get("predicted_answer")),
                "predicted_answer": scored.get("predicted_answer"),
                "model_used": row.get("model_used") or row.get("route", {}).get("final_model"),
            }
    return by_sample


def _similar_key(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("question", ""))
    question = re.sub(r"\d+", "#", question)
    answer = normalize_text(row.get("answer", ""))
    return f"{row.get('source')}|{row.get('answer_type')}|{answer}|{question[:160]}"


def _classify(
    sample_scores: dict[str, dict[str, Any]],
    small_mode: str,
    large_modes: set[str],
) -> tuple[str | None, str]:
    if not sample_scores:
        return None, "missing_results"
    if any(not score.get("parsed") for score in sample_scores.values()):
        return "robustness_challenge", "answer_parse_failure"

    small = sample_scores.get(small_mode)
    non_small = [score for mode, score in sample_scores.items() if mode != small_mode]
    preferred_large = [score for mode, score in sample_scores.items() if mode in large_modes]
    comparison_pool = preferred_large or non_small
    all_scores = list(sample_scores.values())

    if small and not small["correct"] and any(score["correct"] for score in comparison_pool):
        return "medium_large_better", "small_wrong_non_small_correct"
    if all(score["correct"] for score in all_scores):
        return None, "all_correct_removed"
    if all(not score["correct"] for score in all_scores):
        return "all_difficult", "all_wrong_kept_limited"
    if small and small["correct"]:
        return "small_correct", "small_correct"
    return "robustness_challenge", "mixed_or_unstable"


def _parse_ratios(text: str) -> dict[str, float]:
    ratios = dict(DEFAULT_BUCKET_RATIOS)
    if not text:
        return ratios
    for part in text.split(","):
        key, value = part.split("=", 1)
        ratios[key.strip()] = float(value.strip())
    total = sum(ratios.values())
    return {key: value / total for key, value in ratios.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a cleaner v3 text fusion dataset after baseline model results.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--results", action="append", default=[], help="Explicit results.jsonl path. Can be repeated.")
    parser.add_argument("--output", default="data/datasets/v3_text_fusion/selected_after_baseline.jsonl")
    parser.add_argument("--report", default="")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--small-mode", default="always_small")
    parser.add_argument("--large-mode", action="append", default=["always_gemma"])
    parser.add_argument("--ratios", default="", help="Comma list like small_correct=0.30,medium_large_better=0.45")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else Path()
    paths = _result_paths(run_dir, args.results)
    if not paths:
        raise SystemExit("No results.jsonl files found. Provide --run-dir or repeated --results.")

    rng = random.Random(args.seed)
    dataset_rows = {str(row["id"]): row for row in read_jsonl(args.dataset)}
    scores = _load_scores(paths)
    ratios = _parse_ratios(args.ratios)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    seen_similar: set[str] = set()
    for sample_id, row in dataset_rows.items():
        bucket, reason = _classify(scores.get(sample_id, {}), args.small_mode, set(args.large_mode))
        if bucket is None:
            audit.append({"id": sample_id, "kept": False, "reason": reason})
            continue
        similar_key = _similar_key(row)
        if similar_key in seen_similar:
            audit.append({"id": sample_id, "kept": False, "reason": "similar_question_removed"})
            continue
        seen_similar.add(similar_key)
        enriched = dict(row)
        enriched["selection_bucket"] = bucket
        enriched["selection_reason"] = reason
        enriched["baseline_scores"] = scores.get(sample_id, {})
        buckets[bucket].append(enriched)

    selected: list[dict[str, Any]] = []
    for bucket, ratio in ratios.items():
        rows = list(buckets.get(bucket, []))
        rng.shuffle(rows)
        selected.extend(rows[: round(args.target_size * ratio)])
    if len(selected) < args.target_size:
        remainder = [row for bucket_rows in buckets.values() for row in bucket_rows if row not in selected]
        rng.shuffle(remainder)
        selected.extend(remainder[: args.target_size - len(selected)])
    selected = selected[: args.target_size]
    rng.shuffle(selected)

    output_path = Path(args.output)
    _write_jsonl(output_path, selected)

    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")
    from adaroute.utils.io import write_json

    write_json(
        report_path,
        {
            "dataset": args.dataset,
            "results": [str(path) for path in paths],
            "output": str(output_path),
            "target_size": args.target_size,
            "selected_size": len(selected),
            "bucket_available": {key: len(value) for key, value in buckets.items()},
            "bucket_selected": dict(Counter(row["selection_bucket"] for row in selected)),
            "ratios": ratios,
            "removed_count": len(audit),
            "removed_reason_counts": {
                reason: sum(1 for item in audit if item["reason"] == reason) for reason in sorted({item["reason"] for item in audit})
            },
        },
    )
    print(f"Selected dataset: {output_path}")
    print(f"Selection report: {report_path}")


if __name__ == "__main__":
    main()
