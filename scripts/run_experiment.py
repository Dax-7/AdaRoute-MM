from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.experiments.modes import available_modes
from adaroute.experiments.runner import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one AdaRoute-MM v2 experiment mode.")
    parser.add_argument("--mode", required=True, choices=available_modes())
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiments-dir", default="data/experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default=None)
    parser.add_argument("--prompts", default="configs/prompts_v2.yaml")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(
        mode=args.mode,
        dataset_path=args.dataset,
        run_id=args.run_id,
        config_path=args.config,
        override_config_path=args.override_config,
        prompts_path=args.prompts,
        experiments_dir=args.experiments_dir,
        resume=not args.no_resume,
    )
    print(f"Mode: {result['mode']}")
    print(f"Processed: {result['processed']}")
    print(f"Results: {result['results_path']}")
    print(f"Summary: {result['summary_path']}")


if __name__ == "__main__":
    main()
