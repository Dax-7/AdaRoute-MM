from __future__ import annotations

from typing import Any, Callable

from adaroute.core.types import ModelResponse


def check_response_quality(text: str | None, config: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = config.get("fallback", {}).get("quality_checks", {})
    reasons: list[str] = []
    normalized = (text or "").strip()
    if not normalized:
        reasons.append("empty_response")
    if normalized and len(normalized) < int(checks.get("min_chars", 20)):
        reasons.append("too_short")
    lowered = normalized.lower()
    for keyword in checks.get("uncertainty_keywords", []):
        if str(keyword).lower() in lowered:
            reasons.append("uncertainty")
            break
    abnormal_markers = ["error", "none", "null", "你是一个严谨的ai助手", "用户问题：", "视觉上下文："]
    if any(marker in lowered for marker in abnormal_markers):
        reasons.append("abnormal_text")
    return not reasons, reasons


def fallback_candidates(initial_model: str, config: dict[str, Any], system_state: dict[str, Any]) -> list[str]:
    order = list(config.get("fallback", {}).get("upgrade_order", []))
    if initial_model in order:
        start = order.index(initial_model) + 1
        candidates = order[start:]
    else:
        candidates = order

    if system_state.get("is_overloaded"):
        hard_model = config.get("routing", {}).get("policies", {}).get("latency_aware", {}).get("hard")
        candidates = [model for model in candidates if model != hard_model]
        if initial_model == hard_model:
            candidates = [model for model in reversed(order) if model != hard_model]
    return candidates


class FallbackManager:
    def __init__(self, config: dict[str, Any], llm_runner: Callable[[str], ModelResponse]):
        self.config = config
        self.llm_runner = llm_runner

    def run(self, initial_model: str, initial_response: ModelResponse, system_state: dict[str, Any]) -> tuple[ModelResponse, dict[str, Any]]:
        valid, reasons = check_response_quality(initial_response.text if initial_response.ok else "", self.config)
        trace = [
            {
                "model": initial_model,
                "status": "success" if initial_response.ok and valid else "failed",
                "reason": ",".join(reasons) if reasons else initial_response.error,
                "latency": initial_response.latency,
            }
        ]
        if valid and initial_response.ok:
            return initial_response, {"triggered": False, "count": 0, "reasons": [], "trace": trace}

        if not self.config.get("fallback", {}).get("enabled", True):
            return initial_response, {"triggered": False, "count": 0, "reasons": reasons, "trace": trace}

        max_attempts = int(self.config.get("fallback", {}).get("max_attempts", 3))
        all_reasons = list(reasons or [initial_response.error or "model_failed"])
        final_response = initial_response
        attempts = 0
        for model_key in fallback_candidates(initial_model, self.config, system_state):
            if attempts >= max_attempts - 1:
                break
            attempts += 1
            response = self.llm_runner(model_key)
            valid, next_reasons = check_response_quality(response.text if response.ok else "", self.config)
            trace.append(
                {
                    "model": model_key,
                    "status": "success" if response.ok and valid else "failed",
                    "reason": ",".join(next_reasons) if next_reasons else response.error,
                    "latency": response.latency,
                }
            )
            final_response = response
            if response.ok and valid:
                return response, {"triggered": True, "count": attempts, "reasons": all_reasons, "trace": trace}
            all_reasons.extend(next_reasons or [response.error or "model_failed"])
        return final_response, {"triggered": True, "count": attempts, "reasons": all_reasons, "trace": trace}
