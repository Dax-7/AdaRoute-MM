from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.v3.selection import select_verified_numeric


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select AdaRoute-MM v3_1 verified numeric samples from v3 baseline results.")
    parser.add_argument("--dataset", default="data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl")
    parser.add_argument("--small-results", default="data/experiments_v3/v3_result_1/always_small/results_small.jsonl")
    parser.add_argument("--gemma-results", default="data/experiments_v3/v3_result_1/always_gemma/results_gemma.jsonl")
    parser.add_argument("--route-results", default="data/experiments_v3/v3_result_1/difficulty_routing/resultsdifficulty.jsonl")
    parser.add_argument("--output", default="data/datasets/v3_1_text_fusion/verified_numeric_100.jsonl")
    parser.add_argument("--report", default="")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = select_verified_numeric(
        dataset_path=args.dataset,
        small_results=args.small_results,
        gemma_results=args.gemma_results,
        route_results=args.route_results,
        output_path=args.output,
        report_path=args.report or None,
        target_size=args.target_size,
        seed=args.seed,
    )
    print(f"Verified numeric dataset: {args.output}")
    print(f"Selected: {report['selected_size']} / {report['target_size']}")
    print(f"Buckets: {report['selected_bucket_counts']}")
    print(f"Report: {report['report_path']}")


if __name__ == "__main__":
    main()
