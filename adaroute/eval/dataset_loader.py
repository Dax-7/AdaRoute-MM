from __future__ import annotations

from typing import Any

from adaroute.utils.io import read_jsonl


class DatasetFormatError(ValueError):
    pass


def load_jsonl_dataset(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for idx, row in enumerate(rows, start=1):
        if "id" not in row:
            raise DatasetFormatError(f"DATASET_FORMAT_ERROR: line {idx} missing id")
        if "question" not in row or not row["question"]:
            raise DatasetFormatError(f"DATASET_FORMAT_ERROR: line {idx} missing question")
        row.setdefault("image_path", None)
        row.setdefault("task_type", "auto")
    return rows
