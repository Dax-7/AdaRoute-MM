from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.eval.metrics import compute_metrics
from adaroute.eval.runners import run_batch
from adaroute.experiments.modes import resolve_mode_config
from adaroute.utils.io import ensure_dir, load_config, load_prompts, read_jsonl, write_json, write_yaml


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_experiment(
    *,
    mode: str,
    dataset_path: str,
    run_id: str | None = None,
    config_path: str = "configs/default.yaml",
    override_config_path: str | None = None,
    prompts_path: str = "configs/prompts_v2.yaml",
    experiments_dir: str = "data/experiments",
    resume: bool = True,
    experiment_version: str = "v2",
) -> dict[str, Any]:
    run_name = run_id or default_run_id()
    mode_dir = ensure_dir(Path(experiments_dir) / run_name / mode)
    output_path = mode_dir / "results.jsonl"
    summary_path = mode_dir / "summary.json"

    base_config = load_config(config_path, override_config_path)
    config = resolve_mode_config(
        base_config,
        mode,
        output_dir=str(mode_dir).replace("\\", "/"),
        experiment_version=experiment_version,
    )
    prompts = load_prompts(prompts_path)
    write_yaml(mode_dir / "resolved_config.yaml", config)

    pipeline = AdaRoutePipeline(config, prompts)
    processed = run_batch(
        pipeline,
        dataset_path,
        str(output_path),
        policy=config.get("routing", {}).get("default_policy"),
        resume=resume,
    )
    all_results = read_jsonl(output_path) if output_path.exists() else []
    summary = compute_metrics(all_results)
    summary["experiment"] = {
        "mode": mode,
        "version": experiment_version,
        "run_id": run_name,
        "dataset": dataset_path,
        "results_path": str(output_path),
        "processed_this_run": len(processed),
        "resume": resume,
    }
    write_json(summary_path, summary)
    return {
        "mode": mode,
        "run_id": run_name,
        "mode_dir": str(mode_dir),
        "results_path": str(output_path),
        "summary_path": str(summary_path),
        "processed": len(processed),
        "summary": summary,
    }
