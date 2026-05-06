from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.experiments.modes import suite_modes
from adaroute.experiments.runner import default_run_id, run_experiment
from adaroute.utils.io import ensure_dir, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AdaRoute-MM v3 text-only experiment suite.")
    parser.add_argument("--suite", default="text_fusion_v3_basic")
    parser.add_argument("--dataset", default="data/datasets/v3_text_fusion/fusion_1000_200-300-200-200-100.jsonl")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--experiments-dir", default="data/experiments_v3")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override-config", default="configs/v3_text.yaml")
    parser.add_argument("--prompts", default="configs/prompts_v3.yaml")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or default_run_id()
    completed = []
    for mode in suite_modes(args.suite):
        result = run_experiment(
            mode=mode,
            dataset_path=args.dataset,
            run_id=run_id,
            config_path=args.config,
            override_config_path=args.override_config,
            prompts_path=args.prompts,
            experiments_dir=args.experiments_dir,
            resume=not args.no_resume,
            experiment_version="v3_text",
        )
        completed.append({key: result[key] for key in ("mode", "results_path", "summary_path", "processed")})
        print(f"[{mode}] processed={result['processed']} summary={result['summary_path']}")

    run_dir = ensure_dir(Path(args.experiments_dir) / run_id)
    write_json(
        run_dir / "manifest.json",
        {
            "suite": args.suite,
            "version": "v3_text",
            "run_id": run_id,
            "dataset": args.dataset,
            "modes": completed,
        },
    )
    print(f"Manifest: {run_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

