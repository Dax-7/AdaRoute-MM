from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_consensus(row: dict[str, Any]) -> float:
    answer = str(row.get("multiple_choice_answer") or row.get("answer") or "").strip().lower()
    answers = row.get("answers") or []
    normalized = [str(value).strip().lower() for value in answers if str(value).strip()]
    if not answer or not normalized:
        return 0.0
    return sum(1 for value in normalized if value == answer) / len(normalized)


def select_balanced_100x3(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int, str, list[dict[str, Any]]]] = []
    for image_id, rows in groups.items():
        if len(rows) < 3:
            continue
        ordered = sorted(rows, key=lambda row: (-answer_consensus(row), int(row.get("question_id") or 0)))
        chosen = ordered[:3]
        candidates.append((mean(answer_consensus(row) for row in chosen), len(rows), -int(image_id), image_id, chosen))
    candidates.sort(reverse=True)
    if len(candidates) < 100:
        raise SystemExit(f"Need 100 images with at least 3 questions, found {len(candidates)}.")
    selected: list[dict[str, Any]] = []
    for _, _, _, _, rows in candidates[:100]:
        selected.extend(sorted(rows, key=lambda row: int(row.get("question_id") or 0)))
    return selected


def select_average_100x4(groups: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int, str, list[dict[str, Any]]]] = []
    for image_id, rows in groups.items():
        if len(rows) < 3:
            continue
        ordered = sorted(rows, key=lambda row: (-answer_consensus(row), int(row.get("question_id") or 0)))
        candidates.append((mean(answer_consensus(row) for row in ordered[: min(4, len(ordered))]), len(rows), -int(image_id), image_id, ordered))
    candidates.sort(reverse=True)
    if len(candidates) < 100:
        raise SystemExit(f"Need 100 images with at least 3 questions, found {len(candidates)}.")

    selected_groups = candidates[:100]
    allocation = {image_id: 3 for _, _, _, image_id, _ in selected_groups}
    extra_needed = 100
    while extra_needed:
        changed = False
        for _, _, _, image_id, rows in selected_groups:
            if extra_needed == 0:
                break
            if allocation[image_id] < len(rows):
                allocation[image_id] += 1
                extra_needed -= 1
                changed = True
        if not changed:
            raise SystemExit("Selected 100 images do not contain enough questions to average 4 per image.")

    selected: list[dict[str, Any]] = []
    for _, _, _, image_id, rows in selected_groups:
        selected.extend(sorted(rows[: allocation[image_id]], key=lambda row: int(row.get("question_id") or 0)))
    return selected


def canonicalize_images(rows: list[dict[str, Any]], image_dir: Path) -> list[dict[str, Any]]:
    image_dir.mkdir(parents=True, exist_ok=True)
    first_source_by_image: dict[str, Path] = {}
    for row in rows:
        image_id = str(row["image_id"])
        first_source_by_image.setdefault(image_id, Path(row["image_path"]))

    for image_id, source in first_source_by_image.items():
        target = image_dir / f"{image_id}.jpg"
        if not target.exists():
            shutil.copy2(source, target)

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["image_path"] = str((image_dir / f"{row['image_id']}.jpg").as_posix())
        item["cache_subset"] = {
            "canonical_image_id": row["image_id"],
            "selection": "high_consensus_question_subset",
            "image_level_cache_expected": True,
        }
        output_rows.append(item)
    output_rows.sort(key=lambda row: (int(row["image_id"]), int(row["question_id"])))
    return output_rows


def build_manifest(name: str, source: Path, rows: list[dict[str, Any]], image_dir: Path) -> dict[str, Any]:
    image_counts = Counter(str(row["image_id"]) for row in rows)
    answers = Counter(str(row.get("answer") or row.get("multiple_choice_answer") or "").lower() for row in rows)
    consensus_values = [answer_consensus(row) for row in rows]
    repeated_rows = sum(count - 1 for count in image_counts.values())
    return {
        "name": name,
        "source_dataset": source.as_posix(),
        "rows": len(rows),
        "unique_images": len(image_counts),
        "questions_per_image_min": min(image_counts.values()),
        "questions_per_image_max": max(image_counts.values()),
        "questions_per_image_mean": len(rows) / len(image_counts),
        "repeated_image_rows": repeated_rows,
        "theoretical_image_cache_hit_rate": repeated_rows / len(rows),
        "answer_distribution": dict(sorted(answers.items())),
        "mean_answer_consensus": mean(consensus_values) if consensus_values else 0.0,
        "canonical_image_dir": image_dir.as_posix(),
        "selection_rule": (
            "Group VQAv2 yes/no rows by image_id, keep images with at least 3 questions, "
            "rank candidate questions by answer consensus, and rewrite image_path to one canonical file per image_id."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VQAv2 image-level cache evaluation subsets.")
    parser.add_argument("--source", default="data/datasets/vqav2_yesno_1000.jsonl")
    parser.add_argument("--output-dir", default="data/datasets/vqav2_cache_eval")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output_dir)
    image_dir = Path("data/inputs/vqav2_cache_eval")
    rows = read_jsonl(source)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["image_id"])].append(row)

    balanced = canonicalize_images(select_balanced_100x3(groups), image_dir)
    average = canonicalize_images(select_average_100x4(groups), image_dir)

    balanced_path = output_dir / "vqav2_cache_100img_3q_300.jsonl"
    average_path = output_dir / "vqav2_cache_100img_avg4q_400.jsonl"
    write_jsonl(balanced_path, balanced)
    write_jsonl(average_path, average)

    manifest = {
        "source_rows": len(rows),
        "source_unique_images": len(groups),
        "source_images_with_at_least_3_questions": sum(1 for value in groups.values() if len(value) >= 3),
        "source_images_with_at_least_4_questions": sum(1 for value in groups.values() if len(value) >= 4),
        "datasets": [
            build_manifest("vqav2_cache_100img_3q_300", source, balanced, image_dir),
            build_manifest("vqav2_cache_100img_avg4q_400", source, average, image_dir),
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {balanced_path}")
    print(f"Wrote {average_path}")
    print(f"Wrote {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
