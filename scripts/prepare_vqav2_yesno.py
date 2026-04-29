from __future__ import annotations

import argparse
import importlib.util
import io
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.utils.io import append_jsonl, ensure_dir


def _require_dependencies() -> None:
    packages = {
        "datasets": "datasets",
        "PIL": "pillow",
        "pyarrow": "pyarrow",
    }
    missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        install = " ".join(sorted(missing))
        raise SystemExit(
            "Missing dependency: install required packages first:\n"
            f"  python -m pip install {install}\n"
            "or run:\n"
            "  python -m pip install -r requirements.txt"
        )


def _config_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null", "-"}:
        return None
    return normalized


def _load_dataset(args: argparse.Namespace):
    _require_dependencies()
    from datasets import Image as DatasetImage
    from datasets import load_dataset

    if args.local_parquet:
        dataset = load_dataset("parquet", data_files=args.local_parquet, split=args.split, streaming=True)
    else:
        config_name = _config_name(args.config_name)
        if config_name:
            dataset = load_dataset(args.source, config_name, split=args.split, streaming=True)
        else:
            dataset = load_dataset(args.source, split=args.split, streaming=True)

    if not args.decode_images:
        try:
            dataset = dataset.cast_column("image", DatasetImage(decode=False))
        except (KeyError, ValueError, TypeError):
            pass
    return dataset


def _answer_list(answers: Any, fallback: str) -> list[str]:
    if isinstance(answers, list):
        values = []
        for item in answers:
            if isinstance(item, dict):
                values.append(str(item.get("answer", "")).strip())
            else:
                values.append(str(item).strip())
        return [value for value in values if value]
    return [fallback] if fallback else []


def _save_image(image: Any, image_path: Path) -> bool:
    ensure_dir(image_path.parent)
    if image_path.exists() and image_path.stat().st_size > 0:
        return True
    if image is None:
        return False
    if hasattr(image, "save"):
        image.convert("RGB").save(image_path, format="JPEG", quality=92)
        return True
    if isinstance(image, dict):
        if image.get("bytes"):
            from PIL import Image

            with Image.open(io.BytesIO(image["bytes"])) as pil_image:
                pil_image.convert("RGB").save(image_path, format="JPEG", quality=92)
            return True
        if image.get("path") and Path(image["path"]).exists():
            from PIL import Image

            with Image.open(image["path"]) as pil_image:
                pil_image.convert("RGB").save(image_path, format="JPEG", quality=92)
            return True
    return False


def _load_existing_ids(output: Path) -> set[str]:
    if not output.exists():
        return set()

    existing: set[str] = set()
    with output.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                import json

                row = json.loads(text)
            except Exception as exc:
                raise SystemExit(f"Cannot resume from invalid JSONL at {output}:{line_no}: {exc}") from exc
            row_id = row.get("id")
            if row_id:
                existing.add(str(row_id))
    return existing


def _is_yesno_row(row: dict[str, Any], include_answer_only: bool = False) -> bool:
    answer_type = str(row.get("answer_type") or "").lower()
    answer = str(row.get("multiple_choice_answer") or row.get("answer") or "").strip().lower()
    if answer_type in {"yes/no", "yes_no"}:
        return True
    return include_answer_only and answer in {"yes", "no"}


def _row_id(row: dict[str, Any], fallback_index: int) -> str:
    question_id = row.get("question_id") or row.get("id") or f"stream_{fallback_index}"
    return f"vqav2_{question_id}"


def _format_row(row: dict[str, Any], image_path: str | None, fallback_index: int) -> dict[str, Any]:
    question_id = row.get("question_id") or row.get("id") or fallback_index
    image_id = row.get("image_id") or fallback_index
    answer = str(row.get("multiple_choice_answer") or row.get("answer") or "").strip().lower()
    answers = _answer_list(row.get("answers"), answer)
    return {
        "id": f"vqav2_{question_id}",
        "image_path": image_path,
        "question": str(row.get("question") or "").replace("[QUESTION]", "").strip(),
        "answer": answer,
        "multiple_choice_answer": answer,
        "answers": answers,
        "question_type": row.get("question_type"),
        "answer_type": "yes/no",
        "category": "yes_no",
        "task_type": "image_qa",
        "source": "VQAv2",
        "image_id": image_id if isinstance(image_id, int) else str(image_id),
        "question_id": question_id if isinstance(question_id, int) else str(question_id),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream and save a VQAv2 yes/no JSONL subset.")
    parser.add_argument("--source", default="lmms-lab/VQAv2-FewShot")
    parser.add_argument(
        "--config-name",
        default="eval",
        help="Hugging Face dataset config/subset name. Use 'none' for datasets without a config.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--local-parquet", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", default="data/datasets/vqav2_yesno_1000.jsonl")
    parser.add_argument("--image-dir", default="data/inputs/vqav2_yesno")
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument(
        "--include-answer-only-yesno",
        action="store_true",
        help="Also keep rows whose answer is yes/no even when answer_type is not yes/no.",
    )
    parser.add_argument("--decode-images", action="store_true", help="Let datasets decode images eagerly.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output and start over.")
    parser.add_argument("--retries", type=int, default=5, help="Retry full streaming pass after network errors.")
    parser.add_argument("--retry-wait", type=float, default=10.0, help="Seconds to wait before the first retry.")
    parser.add_argument("--max-scan", type=int, default=0, help="Stop after scanning this many rows. 0 means no cap.")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N saved rows. 0 disables.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = Path(args.output)
    if args.limit <= 0:
        raise SystemExit("--limit must be greater than 0")
    if args.retries < 0:
        raise SystemExit("--retries must be 0 or greater")
    if args.overwrite and output.exists():
        output.unlink()

    image_dir = ensure_dir(args.image_dir)
    existing_ids = _load_existing_ids(output)
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} existing rows in {output}")

    saved = 0
    scanned_total = 0
    skipped_errors = 0
    target_saved = max(args.limit - len(existing_ids), 0)

    for attempt in range(args.retries + 1):
        if saved >= target_saved:
            break

        try:
            dataset = _load_dataset(args)
            seen_this_pass = 0
            for row in dataset:
                seen_this_pass += 1
                scanned_total += 1
                if args.max_scan and seen_this_pass > args.max_scan:
                    break
                if not _is_yesno_row(row, args.include_answer_only_yesno):
                    continue

                formatted_id = _row_id(row, seen_this_pass)
                if formatted_id in existing_ids:
                    continue

                question_id = row.get("question_id") or row.get("id") or f"stream_{seen_this_pass}"
                image_path = image_dir / f"{question_id}.jpg"
                try:
                    has_image = _save_image(row.get("image"), image_path)
                except Exception as exc:
                    skipped_errors += 1
                    print(f"Skip row {formatted_id}: failed to save image: {exc}", file=sys.stderr)
                    continue
                if not has_image and not args.allow_missing_images:
                    continue

                formatted = _format_row(row, str(image_path).replace("\\", "/") if has_image else None, saved)
                append_jsonl(output, formatted)
                existing_ids.add(formatted["id"])
                saved += 1

                if args.progress_every and saved % args.progress_every == 0:
                    print(f"Saved this run: {saved}/{target_saved} (total output rows: {len(existing_ids)})")
                if saved >= target_saved:
                    break

            if saved < target_saved and args.max_scan:
                break
            if saved < target_saved and attempt < args.retries:
                print(f"Only saved {saved}/{target_saved}; restarting stream to continue.")
            else:
                break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if attempt >= args.retries:
                raise SystemExit(f"Failed after {args.retries + 1} streaming attempt(s): {exc}") from exc
            wait = args.retry_wait * (2**attempt)
            print(f"Streaming error on attempt {attempt + 1}/{args.retries + 1}: {exc}", file=sys.stderr)
            print(f"Retrying in {wait:.1f}s; already saved {len(existing_ids)} rows.", file=sys.stderr)
            time.sleep(wait)

    print(f"Scanned: {scanned_total}")
    print(f"Saved this run: {saved}")
    print(f"Output rows: {len(existing_ids)}")
    print(f"Skipped image/save errors: {skipped_errors}")
    print(f"Dataset: {output}")
    print(f"Images: {image_dir}")
    if len(existing_ids) < args.limit:
        raise SystemExit(f"Requested {args.limit} rows but only prepared {len(existing_ids)} rows.")


if __name__ == "__main__":
    main()
