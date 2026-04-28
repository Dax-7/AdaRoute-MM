from __future__ import annotations


def exact_match(prediction: str, reference: str) -> bool:
    return prediction.strip() == reference.strip()


def contains_answer(prediction: str, reference: str) -> bool:
    ref = reference.strip()
    return bool(ref) and ref in prediction
