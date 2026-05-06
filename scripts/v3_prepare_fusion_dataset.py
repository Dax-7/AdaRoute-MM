from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.v3.datasets import DEFAULT_COMPONENT_COUNTS, DEFAULT_MIX_COUNTS, DatasetBuildConfig, build_fusion_dataset


def _parse_counts(text: str, defaults: dict[str, int]) -> dict[str, int]:
    counts = dict(defaults)
    if not text:
        return counts
    for part in text.split(","):
        name, value = part.split("=", 1)
        counts[name.strip()] = int(value.strip())
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare AdaRoute-MM v3 text fusion JSONL datasets from Hugging Face streaming datasets.")
    parser.add_argument("--output-dir", default="data/datasets/v3_text_fusion")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--component-counts", default="", help="Comma list such as arc_challenge=500,gsm8k_zh=500")
    parser.add_argument("--mix-counts", default="", help="Comma list such as arc_challenge=200,gsm8k_zh=300,mmlu_pro=200,bbh=200,drop=100")
    parser.add_argument("--arc-split", default="train")
    parser.add_argument("--gsm-split", default="train")
    parser.add_argument("--mmlu-split", default="test")
    parser.add_argument("--bbh-split", default="test")
    parser.add_argument("--drop-split", default="train")
    parser.add_argument("--no-streaming", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_fusion_dataset(
        DatasetBuildConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            component_counts=_parse_counts(args.component_counts, DEFAULT_COMPONENT_COUNTS),
            mix_counts=_parse_counts(args.mix_counts, DEFAULT_MIX_COUNTS),
            arc_split=args.arc_split,
            gsm_split=args.gsm_split,
            mmlu_split=args.mmlu_split,
            bbh_split=args.bbh_split,
            drop_split=args.drop_split,
            streaming=not args.no_streaming,
        )
    )
    print(f"Fusion dataset: {manifest['fusion_path']}")
    print(f"Manifest: {Path(args.output_dir) / 'manifest.json'}")


if __name__ == "__main__":
    main()

