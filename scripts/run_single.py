from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.core.types import InferenceInput
from adaroute.utils.io import load_config, load_prompts, output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one AdaRoute-MM inference request.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--image", dest="image_path", default=None)
    parser.add_argument("--task-type", default="auto")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--prompts", default="configs/prompts.yaml")
    return parser


def run_from_args(args: argparse.Namespace) -> dict:
    config = load_config(args.config, args.override_config)
    prompts = load_prompts(args.prompts)
    pipeline = AdaRoutePipeline(config, prompts)
    result = pipeline.run(
        InferenceInput(
            question=args.question,
            image_path=args.image_path,
            task_type=args.task_type,
            request_id=args.request_id,
        ),
        policy_name=args.policy,
    )
    saved = output_path(config.get("paths", {}).get("output_dir", "data/outputs"), result["request_id"])
    print("\nAnswer:")
    print(result.get("answer", ""))
    print(f"\nModel used: {result.get('route', {}).get('final_model')}")
    print(f"Difficulty: {result.get('route', {}).get('difficulty')}")
    print(f"Fallback: {str(result.get('fallback', {}).get('triggered')).lower()}")
    print(f"Latency: {result.get('latency', {}).get('total', 0.0):.2f}s")
    print(f"Saved to: {Path(saved)}")
    return result


def main() -> None:
    run_from_args(build_parser().parse_args())


if __name__ == "__main__":
    main()
