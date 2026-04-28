from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.eval.runners import run_batch
from adaroute.utils.io import load_config, load_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AdaRoute-MM batch evaluation.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--prompts", default="configs/prompts.yaml")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = AdaRoutePipeline(load_config(args.config, args.override_config), load_prompts(args.prompts))
    results = run_batch(pipeline, args.input, args.output, policy=args.policy, resume=not args.no_resume)
    print(f"Processed {len(results)} samples. Saved to: {args.output}")


if __name__ == "__main__":
    main()
