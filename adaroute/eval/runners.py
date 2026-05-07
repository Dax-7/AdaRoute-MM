from __future__ import annotations

from typing import Any

from tqdm import tqdm

from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.core.types import InferenceInput
from adaroute.eval.dataset_loader import iter_jsonl_dataset
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
    seen = existing_ids(output_path) if resume else set()
    results: list[dict[str, Any]] = []
    for row in tqdm(iter_jsonl_dataset(input_path), desc="AdaRoute-MM batch"):
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
                    source=row.get("source"),
                    answer_type=row.get("answer_type"),
                    category=row.get("category"),
                    metadata=row.get("metadata", {}),
                ),
                policy_name=policy,
            )
            result["sample_id"] = sample_id
            result["reference_answer"] = row.get("answer")
            result["multiple_choice_answer"] = row.get("multiple_choice_answer", row.get("answer"))
            result["reference_answers"] = row.get("answers", [])
            result["answer_type"] = row.get("answer_type")
            result["question_type"] = row.get("question_type")
            result["category"] = row.get("category")
            result["source"] = row.get("source")
            result["choices"] = row.get("choices")
            result["choice_labels"] = row.get("choice_labels")
            result["answer_format"] = row.get("answer_format")
            result["metadata"] = row.get("metadata", {})
            result["image_id"] = row.get("image_id")
            result["question_id"] = row.get("question_id")
        except Exception as exc:
            result = {
                "sample_id": sample_id,
                "request_id": sample_id,
                "status": "failed",
                "answer": "",
                "reference_answer": row.get("answer"),
                "multiple_choice_answer": row.get("multiple_choice_answer", row.get("answer")),
                "reference_answers": row.get("answers", []),
                "answer_type": row.get("answer_type"),
                "question_type": row.get("question_type"),
                "category": row.get("category"),
                "source": row.get("source"),
                "choices": row.get("choices"),
                "choice_labels": row.get("choice_labels"),
                "answer_format": row.get("answer_format"),
                "metadata": row.get("metadata", {}),
                "image_id": row.get("image_id"),
                "question_id": row.get("question_id"),
                "error": {"code": "UNKNOWN_ERROR", "message": str(exc)},
            }
        append_jsonl(output_path, result)
        results.append(result)
    return results
