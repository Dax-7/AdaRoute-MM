from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.v3.datasets import (
    DEFAULT_V3_1_COMPONENT_COUNTS,
    DEFAULT_V3_1_MIX_COUNTS,
    DatasetBuildConfigV31,
    build_fusion_dataset_v3_1,
)
from adaroute.v3.selection import select_verified_numeric


def _parse_counts(text: str, defaults: dict[str, int]) -> dict[str, int]:
    counts = dict(defaults)
    if not text:
        return counts
    for part in text.split(","):
        name, value = part.split("=", 1)
        counts[name.strip()] = int(value.strip())
    return counts


def _path_exists(path: str) -> bool:
    return bool(path) and Path(path).exists()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AdaRoute-MM v3_1 text fusion JSONL datasets.")
    parser.add_argument("--output-dir", default="data/datasets/v3_1_text_fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--component-counts", default="", help="Comma list such as arc_easy=500,sciq=500")
    parser.add_argument("--mix-counts", default="", help="Comma list such as arc_easy=200,sciq=200,arc_challenge=400,drop_span=100,verified_numeric=100")
    parser.add_argument("--arc-easy-split", default="train")
    parser.add_argument("--arc-challenge-split", default="train")
    parser.add_argument("--sciq-split", default="train")
    parser.add_argument("--drop-split", default="train")
    parser.add_argument("--no-streaming", action="store_true")
    parser.add_argument("--verified-numeric", default="", help="Existing verified numeric JSONL. Defaults to output-dir/verified_numeric_100.jsonl")
    parser.add_argument("--v3-dataset", default="data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl")
    parser.add_argument("--small-results", default="data/experiments_v3/v3_result_1/always_small/results_small.jsonl")
    parser.add_argument("--gemma-results", default="data/experiments_v3/v3_result_1/always_gemma/results_gemma.jsonl")
    parser.add_argument("--route-results", default="data/experiments_v3/v3_result_1/difficulty_routing/resultsdifficulty.jsonl")
    parser.add_argument("--skip-verified-build", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    verified_numeric = Path(args.verified_numeric) if args.verified_numeric else output_dir / "verified_numeric_100.jsonl"
    if not args.skip_verified_build and not verified_numeric.exists():
        if all(_path_exists(path) for path in (args.v3_dataset, args.small_results, args.gemma_results, args.route_results)):
            report = select_verified_numeric(
                dataset_path=args.v3_dataset,
                small_results=args.small_results,
                gemma_results=args.gemma_results,
                route_results=args.route_results,
                output_path=verified_numeric,
                report_path=output_dir / "verified_numeric_selection.report.json",
                seed=args.seed,
            )
            print(f"Built verified numeric component: {verified_numeric} ({report['selected_size']} rows)")
        else:
            print("Verified numeric inputs are incomplete; expecting an existing --verified-numeric file.")

    manifest = build_fusion_dataset_v3_1(
        DatasetBuildConfigV31(
            output_dir=args.output_dir,
            seed=args.seed,
            component_counts=_parse_counts(args.component_counts, DEFAULT_V3_1_COMPONENT_COUNTS),
            mix_counts=_parse_counts(args.mix_counts, DEFAULT_V3_1_MIX_COUNTS),
            arc_easy_split=args.arc_easy_split,
            arc_challenge_split=args.arc_challenge_split,
            sciq_split=args.sciq_split,
            drop_split=args.drop_split,
            streaming=not args.no_streaming,
            verified_numeric_path=str(verified_numeric),
        )
    )
    print(f"v3_1 fusion dataset: {manifest['fusion_path']}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
