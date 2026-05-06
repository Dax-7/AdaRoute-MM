from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset

from adaroute.utils.io import ensure_dir, write_json


DEFAULT_COMPONENT_COUNTS = {
    "arc_challenge": 500,
    "gsm8k_zh": 500,
    "mmlu_pro": 500,
    "bbh": 500,
    "drop": 500,
}

DEFAULT_MIX_COUNTS = {
    "arc_challenge": 200,
    "gsm8k_zh": 300,
    "mmlu_pro": 200,
    "bbh": 200,
    "drop": 100,
}

BBH_TASKS = [
    "boolean_expressions",
    "date_understanding",
    "disambiguation_qa",
    "object_counting",
    "logical_deduction_three_objects",
]

MC_LABELS = [chr(ord("A") + idx) for idx in range(10)]


@dataclass(frozen=True)
class DatasetBuildConfig:
    output_dir: str = "data/datasets/v3_text_fusion"
    seed: int = 42
    component_counts: dict[str, int] | None = None
    mix_counts: dict[str, int] | None = None
    arc_split: str = "train"
    gsm_split: str = "train"
    mmlu_split: str = "test"
    bbh_split: str = "test"
    drop_split: str = "train"
    streaming: bool = True


def _jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            import json

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _take(rows: Iterable[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _format_options(options: list[str], labels: list[str] | None = None) -> str:
    labels = labels or MC_LABELS[: len(options)]
    return "\n".join(f"{label}. {text}" for label, text in zip(labels, options))


def _base_row(
    *,
    sample_id: str,
    source: str,
    question: str,
    answer: str,
    answer_type: str,
    category: str | None = None,
    choices: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "id": sample_id,
        "image_path": None,
        "question": question.strip(),
        "answer": str(answer).strip(),
        "task_type": "text_only",
        "source": source,
        "category": category,
        "answer_type": answer_type,
        "answer_format": "Return only one final answer as: FINAL_ANSWER: <answer>",
    }
    if choices is not None:
        row["choices"] = choices
        row["choice_labels"] = MC_LABELS[: len(choices)]
    if metadata:
        row["metadata"] = metadata
    return row


def _choice_texts(raw_choices: Any) -> tuple[list[str], list[str] | None]:
    if isinstance(raw_choices, dict):
        texts = raw_choices.get("text") or raw_choices.get("choices") or raw_choices.get("option")
        labels = raw_choices.get("label")
        if texts:
            return [str(item) for item in texts], [str(label) for label in labels] if labels else None
    if isinstance(raw_choices, list):
        return [str(item) for item in raw_choices], None
    return [], None


def convert_arc_challenge(split: str = "train", limit: int = 500, streaming: bool = True) -> list[dict[str, Any]]:
    dataset = load_dataset("mib-bench/arc_challenge", split=split, streaming=streaming)

    def rows() -> Iterable[dict[str, Any]]:
        for idx, item in enumerate(dataset):
            choices, labels = _choice_texts(item.get("choices"))
            if not choices:
                continue
            answer_key = item.get("label")
            if answer_key is None and item.get("answerKey") is not None:
                answer_key = MC_LABELS[int(item["answerKey"])]
            if answer_key is None:
                continue
            label_map = {str(label): MC_LABELS[i] for i, label in enumerate(labels or MC_LABELS[: len(choices)])}
            answer = label_map.get(str(answer_key), str(answer_key))
            question = (
                f"{item.get('question', '').strip()}\n\nOptions:\n{_format_options(choices)}\n\n"
                "Answer with the option letter only."
            )
            yield _base_row(
                sample_id=f"arc_challenge_{item.get('idx', idx)}",
                source="mib-bench/arc_challenge",
                category="arc_challenge",
                question=question,
                answer=answer,
                answer_type="multiple_choice",
                choices=choices,
                metadata={"arc_id": item.get("arc_id")},
            )

    return _take(rows(), limit)


def convert_gsm8k_zh(split: str = "train", limit: int = 500, streaming: bool = True) -> list[dict[str, Any]]:
    dataset = load_dataset("meta-math/GSM8K_zh", split=split, streaming=streaming)

    def rows() -> Iterable[dict[str, Any]]:
        for idx, item in enumerate(dataset):
            question_text = item.get("question_zh") or item.get("question") or ""
            answer = item.get("answer_only")
            if answer is None:
                answer = item.get("answer_zh") or item.get("answer")
            if not question_text or answer is None:
                continue
            question = f"{question_text.strip()}\n\nAnswer with the final number only."
            yield _base_row(
                sample_id=f"gsm8k_zh_{idx}",
                source="meta-math/GSM8K_zh",
                category="gsm8k_zh",
                question=question,
                answer=str(answer),
                answer_type="numeric",
                metadata={"split": item.get("split")},
            )

    return _take(rows(), limit)


def convert_mmlu_pro(split: str = "test", limit: int = 500, streaming: bool = True) -> list[dict[str, Any]]:
    dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split, streaming=streaming)

    def rows() -> Iterable[dict[str, Any]]:
        for idx, item in enumerate(dataset):
            choices = [str(choice) for choice in item.get("options", [])]
            if not choices:
                continue
            answer = item.get("answer")
            if answer is None and item.get("answer_index") is not None:
                answer = MC_LABELS[int(item["answer_index"])]
            if answer is None:
                continue
            question = (
                f"{item.get('question', '').strip()}\n\nOptions:\n{_format_options(choices)}\n\n"
                "Answer with the option letter only."
            )
            yield _base_row(
                sample_id=f"mmlu_pro_{item.get('question_id', idx)}",
                source="TIGER-Lab/MMLU-Pro",
                category=item.get("category") or "mmlu_pro",
                question=question,
                answer=str(answer).strip(),
                answer_type="multiple_choice",
                choices=choices,
                metadata={"src": item.get("src"), "answer_index": item.get("answer_index")},
            )

    return _take(rows(), limit)


def _bbh_answer_type(task: str, target: str) -> str:
    if task == "boolean_expressions" or target.strip().lower() in {"true", "false"}:
        return "boolean"
    if target.strip().startswith("(") and len(target.strip()) >= 3:
        return "multiple_choice"
    if target.strip().replace(",", "").isdigit():
        return "numeric"
    return "short_text"


def convert_bbh(
    tasks: list[str] | None = None,
    per_task_limit: int = 100,
    split: str = "test",
    streaming: bool = True,
) -> list[dict[str, Any]]:
    tasks = tasks or BBH_TASKS
    converted: list[dict[str, Any]] = []
    for task in tasks:
        dataset = load_dataset("lukaemon/bbh", task, split=split, streaming=streaming)

        def rows() -> Iterable[dict[str, Any]]:
            for idx, item in enumerate(dataset):
                target = str(item.get("target", "")).strip()
                if not target:
                    continue
                answer_type = _bbh_answer_type(task, target)
                instruction = "Answer with the exact final answer."
                if answer_type == "boolean":
                    instruction = "Answer with True or False only."
                question = f"{str(item.get('input', '')).strip()}\n\n{instruction}"
                yield _base_row(
                    sample_id=f"bbh_{task}_{idx}",
                    source="lukaemon/bbh",
                    category=task,
                    question=question,
                    answer=target,
                    answer_type=answer_type,
                )

        converted.extend(_take(rows(), per_task_limit))
    return converted


def convert_drop(split: str = "train", limit: int = 500, streaming: bool = True) -> list[dict[str, Any]]:
    dataset = load_dataset("ucinlp/drop", split=split, streaming=streaming)

    def rows() -> Iterable[dict[str, Any]]:
        for idx, item in enumerate(dataset):
            spans = (item.get("answers_spans") or {}).get("spans") or []
            if not spans:
                continue
            answer = str(spans[0])
            answer_type = "numeric" if answer.replace(",", "").replace(".", "", 1).isdigit() else "short_text"
            question = (
                f"Passage:\n{item.get('passage', '').strip()}\n\n"
                f"Question:\n{item.get('question', '').strip()}\n\n"
                "Answer with the shortest exact answer."
            )
            yield _base_row(
                sample_id=f"drop_{item.get('query_id', idx)}",
                source="ucinlp/drop",
                category=item.get("section_id"),
                question=question,
                answer=answer,
                answer_type=answer_type,
                metadata={"all_spans": [str(span) for span in spans]},
            )

    return _take(rows(), limit)


def build_components(config: DatasetBuildConfig) -> dict[str, list[dict[str, Any]]]:
    counts = config.component_counts or DEFAULT_COMPONENT_COUNTS
    bbh_per_task = max(1, counts.get("bbh", 500) // len(BBH_TASKS))
    return {
        "arc_challenge": convert_arc_challenge(config.arc_split, counts.get("arc_challenge", 500), config.streaming),
        "gsm8k_zh": convert_gsm8k_zh(config.gsm_split, counts.get("gsm8k_zh", 500), config.streaming),
        "mmlu_pro": convert_mmlu_pro(config.mmlu_split, counts.get("mmlu_pro", 500), config.streaming),
        "bbh": convert_bbh(BBH_TASKS, bbh_per_task, config.bbh_split, config.streaming),
        "drop": convert_drop(config.drop_split, counts.get("drop", 500), config.streaming),
    }


def build_fusion_dataset(config: DatasetBuildConfig) -> dict[str, Any]:
    output_dir = ensure_dir(config.output_dir)
    rng = random.Random(config.seed)
    components = build_components(config)
    mix_counts = config.mix_counts or DEFAULT_MIX_COUNTS

    component_paths: dict[str, str] = {}
    for name, rows in components.items():
        path = output_dir / f"{name}_500.jsonl"
        _jsonl_write(path, rows)
        component_paths[name] = str(path)

    mixed_rows: list[dict[str, Any]] = []
    for name, count in mix_counts.items():
        rows = list(components.get(name, []))
        if len(rows) < count:
            raise ValueError(f"Dataset component '{name}' has {len(rows)} rows, below requested mix count {count}")
        mixed_rows.extend(rows[:count])
    rng.shuffle(mixed_rows)

    fusion_path = output_dir / "fusion_1000_200-300-200-200-100.jsonl"
    _jsonl_write(fusion_path, mixed_rows)

    manifest = {
        "version": "v3_text_fusion",
        "seed": config.seed,
        "streaming": config.streaming,
        "component_counts": {name: len(rows) for name, rows in components.items()},
        "mix_counts": mix_counts,
        "component_paths": component_paths,
        "fusion_path": str(fusion_path),
        "bbh_tasks": BBH_TASKS,
        "format": {
            "required": ["id", "question", "answer", "image_path", "task_type"],
            "v3_added": ["source", "category", "answer_type", "answer_format", "choices", "choice_labels", "metadata"],
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest

