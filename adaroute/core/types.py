from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class InferenceInput:
    question: str
    image_path: Optional[str] = None
    task_type: str = "auto"
    request_id: Optional[str] = None


@dataclass
class ModelResponse:
    ok: bool
    model: str
    text: str
    latency: float
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    raw: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class RouteDecision:
    difficulty: str
    policy: str
    selected_model: str
    reason: Optional[str] = None
    overloaded: bool = False


def to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
