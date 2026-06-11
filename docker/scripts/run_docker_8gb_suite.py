from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.experiments.runner import default_run_id, run_experiment
from adaroute.utils.io import ensure_dir, iter_jsonl, write_json


DEFAULT_MODES = [
    "always_small",
    "always_middle",
    "always_gemma",
    "difficulty_routing",
    "random_routing",
    "adaroute_mm_full",
]


def _stratum(row: dict[str, Any], field: str) -> str:
    return str(row.get(field) or row.get("category") or row.get("answer_type") or "unknown")


def _allocation(groups: dict[str, list[tuple[int, dict[str, Any]]]], sample_size: int) -> dict[str, int]:
    total = sum(len(rows) for rows in groups.values())
    if sample_size >= total:
        return {name: len(rows) for name, rows in groups.items()}

    exact: dict[str, float] = {name: len(rows) * sample_size / total for name, rows in groups.items()}
    counts = {name: min(len(groups[name]), int(value)) for name, value in exact.items()}
    remaining = sample_size - sum(counts.values())
    ranked = sorted(groups, key=lambda name: (exact[name] - counts[name], len(groups[name]), name), reverse=True)
    while remaining > 0:
        progressed = False
        for name in ranked:
            if counts[name] < len(groups[name]):
                counts[name] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break
    return counts


def _even_positions(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count >= total:
        return list(range(total))
    if count == 1:
        return [total // 2]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def write_stratified_sample(
    dataset_path: str | Path,
    output_path: str | Path,
    sample_size: int,
    stratify_by: str,
) -> dict[str, Any]:
    rows = list(iter_jsonl(dataset_path))
    if not rows:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[_stratum(row, stratify_by)].append((index, row))

    counts = _allocation(groups, min(sample_size, len(rows)))
    selected_indices: set[int] = set()
    for name, entries in groups.items():
        for position in _even_positions(len(entries), counts.get(name, 0)):
            selected_indices.add(entries[position][0])

    selected = [row for index, row in enumerate(rows) if index in selected_indices]
    target = Path(output_path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "source_dataset": str(dataset_path),
        "sampled_dataset": str(target),
        "stratify_by": stratify_by,
        "requested_sample_size": sample_size,
        "source_count": len(rows),
        "sample_count": len(selected),
        "source_distribution": {name: len(entries) for name, entries in sorted(groups.items())},
        "sample_distribution": {name: counts[name] for name in sorted(counts)},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Docker 8GB AdaRoute-MM v3_2 six-mode experiment.")
    parser.add_argument("--dataset", default="data/datasets/v3_1_text_fusion/fusion_v3_1_1000_200-200-400-100-100.jsonl")
    parser.add_argument("--results-dir", default="results/docker_8gb")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--stratify-by", default="source")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default="configs/v3_2_text.yaml")
    parser.add_argument("--prompts", default="configs/prompts_v3_2.yaml")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or default_run_id()
    run_dir = ensure_dir(Path(args.results_dir) / run_id)
    sampled_dataset = run_dir / f"sampled_dataset_{args.sample_size}.jsonl"
    sample_manifest = write_stratified_sample(args.dataset, sampled_dataset, args.sample_size, args.stratify_by)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    completed = []
    for mode in modes:
        result = run_experiment(
            mode=mode,
            dataset_path=str(sampled_dataset),
            run_id=run_id,
            config_path=args.config,
            override_config_path=args.override_config,
            prompts_path=args.prompts,
            experiments_dir=args.results_dir,
            resume=not args.no_resume,
            experiment_version="v3_2_text",
        )
        completed.append({key: result[key] for key in ("mode", "results_path", "summary_path", "processed")})
        print(f"[{mode}] processed={result['processed']} summary={result['summary_path']}")

    write_json(
        run_dir / "manifest.json",
        {
            "suite": "docker_8gb_v3_2_six_modes",
            "version": "v3_2_text",
            "run_id": run_id,
            "memory_limit": "8g",
            "sample": sample_manifest,
            "modes": completed,
        },
    )
    print(f"Docker 8GB run complete: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
