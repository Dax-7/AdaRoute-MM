from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Any


YES_WORDS = {"yes", "yeah", "yep", "true", "correct"}
NO_WORDS = {"no", "nope", "false", "incorrect"}


def normalize_answer(text: str | None) -> str:
    value = (text or "").strip().lower()
    value = re.sub(r"[^a-z0-9'\s:]", " ", value)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_yes_no(text: str | None) -> str:
    value = normalize_answer(text)
    if not value:
        return "invalid"
    tokens = set(value.split())
    yes = bool(tokens & YES_WORDS)
    no = bool(tokens & NO_WORDS)
    if yes and not no:
        return "yes"
    if no and not yes:
        return "no"
    if value.startswith("yes"):
        return "yes"
    if value.startswith("no"):
        return "no"
    return "invalid"


def vqa_soft_accuracy(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0
    normalized_answers = [normalize_answer(answer) for answer in answers]
    matches = sum(1 for answer in normalized_answers if answer == pred)
    return min(matches / 3.0, 1.0)


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _binary_prf(labels: list[str], preds: list[str], positive: str) -> dict[str, float]:
    tp = sum(1 for label, pred in zip(labels, preds) if label == positive and pred == positive)
    fp = sum(1 for label, pred in zip(labels, preds) if label != positive and pred == positive)
    fn = sum(1 for label, pred in zip(labels, preds) if label == positive and pred != positive)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * pct))
    return ordered[index]


def bootstrap_ci(values: list[float], iterations: int = 1000, seed: int = 42) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iterations):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    return {"mean": _mean(values), "low": _percentile(means, 0.025), "high": _percentile(means, 0.975)}


def compute_yesno_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in results if str(row.get("answer_type") or row.get("input", {}).get("answer_type") or "").lower() in {"yes/no", "yes_no", ""}]
    labels: list[str] = []
    preds: list[str] = []
    correctness: list[float] = []
    vqa_scores: list[float] = []
    invalid = 0

    for row in rows:
        reference = row.get("multiple_choice_answer") or row.get("reference_answer") or row.get("answer")
        label = extract_yes_no(str(reference))
        pred = extract_yes_no(str(row.get("prediction") or row.get("answer") or ""))
        if label not in {"yes", "no"}:
            continue
        labels.append(label)
        preds.append(pred)
        if pred == "invalid":
            invalid += 1
        score = 1.0 if pred == label else 0.0
        correctness.append(score)
        answers = row.get("answers") or row.get("reference_answers") or []
        if answers and isinstance(answers[0], dict):
            answers = [str(item.get("answer", "")) for item in answers]
        vqa_scores.append(vqa_soft_accuracy(pred if pred in {"yes", "no"} else "", [str(answer) for answer in answers]) if answers else score)

    yes = _binary_prf(labels, preds, "yes")
    no = _binary_prf(labels, preds, "no")
    recalls = [yes["recall"], no["recall"]]
    macro_f1 = (yes["f1"] + no["f1"]) / 2 if labels else 0.0
    accuracy_ci = bootstrap_ci(correctness) if correctness else {"mean": 0.0, "low": 0.0, "high": 0.0}

    confusion = Counter(f"{label}->{pred}" for label, pred in zip(labels, preds))
    return {
        "total_evaluated": len(labels),
        "label_distribution": dict(Counter(labels)),
        "prediction_distribution": dict(Counter(preds)),
        "confusion_matrix": dict(confusion),
        "accuracy": _mean(correctness),
        "accuracy_ci95": accuracy_ci,
        "macro_f1": macro_f1,
        "balanced_accuracy": _mean(recalls),
        "yes_precision": yes["precision"],
        "yes_recall": yes["recall"],
        "yes_f1": yes["f1"],
        "no_precision": no["precision"],
        "no_recall": no["recall"],
        "no_f1": no["f1"],
        "invalid_count": invalid,
        "invalid_rate": _safe_div(invalid, len(labels)),
        "vqa_soft_accuracy": _mean(vqa_scores),
        "standard_error": math.sqrt(_safe_div(_mean(correctness) * (1 - _mean(correctness)), len(correctness))) if correctness else 0.0,
    }

