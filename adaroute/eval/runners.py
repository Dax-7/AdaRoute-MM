from __future__ import annotations

from typing import Any

from tqdm import tqdm

from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.core.types import InferenceInput
from adaroute.eval.dataset_loader import load_jsonl_dataset
from adaroute.utils.io import append_jsonl, read_jsonl


def existing_ids(output_path: str) -> set[str]:
    try:
        return {str(row.get("sample_id") or row.get("request_id")) for row in read_jsonl(output_path)}
    except FileNotFoundError:
        return set()


def run_batch(
    pipeline: AdaRoutePipeline,
    input_path: str,
    output_path: str,
    policy: str | None = None,
    resume: bool = True,
) -> list[dict[str, Any]]:
    rows = load_jsonl_dataset(input_path)
    seen = existing_ids(output_path) if resume else set()
    results: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="AdaRoute-MM batch"):
        sample_id = str(row["id"])
        if resume and sample_id in seen:
            continue
        try:
            result = pipeline.run(
                InferenceInput(
                    question=row["question"],
                    image_path=row.get("image_path"),
                    task_type=row.get("task_type", "auto"),
                    request_id=sample_id,
                ),
                policy_name=policy,
            )
            result["sample_id"] = sample_id
            result["reference_answer"] = row.get("answer")
        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "request_id": sample_id,
                "status": "failed",
                "answer": "",
                "reference_answer": row.get("answer"),
                "error": {"code": "UNKNOWN_ERROR", "message": str(exc)},
            }
        append_jsonl(output_path, result)
        results.append(result)
    return results
