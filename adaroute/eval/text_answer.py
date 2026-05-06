from __future__ import annotations

import re
import string
from decimal import Decimal, InvalidOperation
from typing import Any

MC_LABELS = [chr(ord("A") + idx) for idx in range(10)]


def extract_final_answer(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"FINAL_ANSWER\s*[:：]\s*(.+)",
        r"final answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
        r"答\s*[:：]\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    return text.strip().splitlines()[-1].strip()


def normalize_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.translate(table).split())


def extract_choice(text: str, choices: list[str] | None = None) -> str | None:
    answer = extract_final_answer(text) or ""
    normalized = answer.strip()
    match = re.search(r"\b([A-J])\b", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"[\(\[]\s*([A-J])\s*[\)\]]", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    if choices:
        pred_norm = normalize_text(normalized)
        for idx, choice in enumerate(choices):
            if pred_norm and pred_norm == normalize_text(choice):
                return MC_LABELS[idx]
    return None


def _decimal(value: str) -> Decimal | None:
    cleaned = str(value).strip().replace(",", "")
    cleaned = cleaned.rstrip("%")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def extract_number(text: str) -> str | None:
    answer = extract_final_answer(text) or ""
    value = _decimal(answer)
    return str(value.normalize()) if value is not None else None


def extract_boolean(text: str) -> str | None:
    answer = (extract_final_answer(text) or "").strip().lower()
    if re.search(r"\btrue\b", answer):
        return "true"
    if re.search(r"\bfalse\b", answer):
        return "false"
    return None


def score_text_answer(row: dict[str, Any]) -> dict[str, Any]:
    answer_type = str(row.get("answer_type") or "").lower()
    reference = str(row.get("reference_answer") or row.get("answer") or "").strip()
    prediction_text = str(row.get("answer") or row.get("prediction") or "")
    choices = row.get("choices") if isinstance(row.get("choices"), list) else None

    if not reference:
        return {"evaluated": False, "correct": False, "predicted_answer": None, "reason": "missing_reference"}

    if answer_type == "multiple_choice":
        pred = extract_choice(prediction_text, choices)
        ref = extract_choice(reference, choices) or reference.strip().upper().strip("().")
        return {"evaluated": True, "correct": bool(pred and pred == ref), "predicted_answer": pred, "reference_answer": ref}

    if answer_type == "numeric":
        pred_decimal = _decimal(extract_final_answer(prediction_text) or prediction_text)
        ref_decimal = _decimal(reference)
        correct = pred_decimal is not None and ref_decimal is not None and pred_decimal == ref_decimal
        return {
            "evaluated": True,
            "correct": correct,
            "predicted_answer": str(pred_decimal.normalize()) if pred_decimal is not None else None,
            "reference_answer": str(ref_decimal.normalize()) if ref_decimal is not None else reference,
        }

    if answer_type == "boolean":
        pred = extract_boolean(prediction_text)
        ref = extract_boolean(reference) or reference.strip().lower()
        return {"evaluated": True, "correct": bool(pred and pred == ref), "predicted_answer": pred, "reference_answer": ref}

    pred_text = normalize_text(extract_final_answer(prediction_text) or prediction_text)
    ref_text = normalize_text(reference)
    correct = bool(pred_text and (pred_text == ref_text or pred_text in ref_text or ref_text in pred_text))
    return {"evaluated": True, "correct": correct, "predicted_answer": pred_text or None, "reference_answer": ref_text}

