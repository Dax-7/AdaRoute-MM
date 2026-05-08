from __future__ import annotations

from pathlib import Path
from typing import Any

from adaroute.core.types import ModelResponse
from adaroute.utils.io import cache_key, read_cache, write_cache


LEGACY_EASY = "莽禄聽芒鈥毬┞嶁€?"
LEGACY_MEDIUM = "忙露鈥溍€β犆р€溾€?"
LEGACY_HARD = "茅聧楼盲录麓忙炉娄"

RISK_LABEL_TO_DIFFICULTY = {
    "small_ok": "easy",
    "route_small": "easy",
    "use_small": "easy",
    "middle_ok": "medium",
    "medium_ok": "medium",
    "route_middle": "medium",
    "route_medium": "medium",
    "use_middle": "medium",
    "use_medium": "medium",
    "need_gemma": "hard",
    "route_gemma": "hard",
    "use_gemma": "hard",
    "need_large": "hard",
    "high_risk": "hard",
}


def parse_difficulty(text: str, default: str = LEGACY_MEDIUM) -> tuple[str, str | None]:
    normalized = (text or "").strip().lower()
    first_line = normalized.splitlines()[0].strip(" .,:;") if normalized else ""
    for label, difficulty in RISK_LABEL_TO_DIFFICULTY.items():
        if first_line == label or label in normalized:
            return difficulty, None
    for label in ("easy", "medium", "hard"):
        if label in normalized:
            return label, None
    for label in ("简单", "中等", "困难", "绠€鍗?", "涓瓑", "鍥伴毦"):
        if label in (text or ""):
            return label, None
    for label in (LEGACY_EASY, LEGACY_MEDIUM, LEGACY_HARD):
        if label in (text or ""):
            return label, None
    return default, "ROUTER_PARSE_ERROR"


def _prior_text(source: str | None, answer_type: str | None, category: str | None, router_cfg: dict[str, Any]) -> str:
    if not router_cfg.get("use_source_prior", False):
        return ""
    source_priors = router_cfg.get("source_priors", {})
    prior = source_priors.get(source or "") or source_priors.get(category or "")
    if not prior:
        return ""
    return (
        "Sample metadata prior:\n"
        f"- source: {source or 'unknown'}\n"
        f"- answer_type: {answer_type or 'unknown'}\n"
        f"- prior: {prior}\n"
        "Use this as a soft prior. Override it when the question clearly needs a different model difficulty."
    )


def _direct_prior(answer_type: str | None, router_cfg: dict[str, Any]) -> str | None:
    if not router_cfg.get("use_source_prior", False):
        return None
    answer_type_priors = router_cfg.get("answer_type_priors", {})
    label = answer_type_priors.get(str(answer_type or "").lower())
    if label in {"easy", "medium", "hard"}:
        return label
    return None


def _contains_any(text: str, cues: list[str]) -> bool:
    normalized = f" {text.lower()} "
    return any(str(cue).lower() in normalized for cue in cues)


def _source_rule(source: str | None, category: str | None, gate_cfg: dict[str, Any]) -> dict[str, Any] | None:
    rules = gate_cfg.get("source_rules", {})
    return rules.get(source or "") or rules.get(category or "")


def _risk_static_gate(
    question: str,
    source: str | None,
    answer_type: str | None,
    category: str | None,
    router_cfg: dict[str, Any],
) -> str | None:
    gate_cfg = router_cfg.get("risk_aware_static_gate", {})
    if not gate_cfg.get("enabled", False):
        return None

    label = gate_cfg.get("direct_answer_type_priors", {}).get(str(answer_type or "").lower())
    if label in {"easy", "medium", "hard"}:
        return label

    source_priors = gate_cfg.get("direct_source_priors", {})
    label = source_priors.get(source or "") or source_priors.get(category or "")
    if label in {"easy", "medium", "hard"}:
        return label

    rule = _source_rule(source, category, gate_cfg)
    if not rule:
        fallback = gate_cfg.get("fallback_difficulty")
        return fallback if fallback in {"easy", "medium", "hard"} else None

    question_text = question or ""
    hard_cues = list(rule.get("hard_cues", gate_cfg.get("hard_cues", [])))
    has_hard_cue = _contains_any(question_text, hard_cues)
    if rule.get("allow_hard_on_cue", False) and has_hard_cue:
        return rule.get("hard_difficulty", "hard")

    max_simple_chars = int(rule.get("max_simple_chars", 0) or 0)
    simple_difficulty = rule.get("simple_difficulty")
    if simple_difficulty and max_simple_chars and len(question_text) <= max_simple_chars and not has_hard_cue:
        return simple_difficulty

    difficulty = rule.get("default_difficulty", gate_cfg.get("fallback_difficulty", "medium"))
    return difficulty if difficulty in {"easy", "medium", "hard"} else "medium"


class RouterModule:
    def __init__(self, config: dict[str, Any], prompts: dict[str, Any], client: Any):
        self.config = config
        self.prompts = prompts
        self.client = client

    def run(
        self,
        question: str,
        caption_text: str,
        source: str | None = None,
        answer_type: str | None = None,
        category: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str, ModelResponse | None, str | None]:
        router_cfg = self.config.get("router", {})
        default = router_cfg.get("default_difficulty", LEGACY_MEDIUM)
        if not router_cfg.get("enabled", True):
            return default, None, None

        risk_label = _risk_static_gate(question, source, answer_type, category, router_cfg)
        gate_cfg = router_cfg.get("risk_aware_static_gate", {})
        if risk_label and not gate_cfg.get("model_gate_enabled", False):
            return risk_label, None, None

        prior_label = risk_label or _direct_prior(answer_type, router_cfg)
        if prior_label and not gate_cfg.get("model_gate_enabled", False):
            return prior_label, None, None

        model_key = router_cfg.get("model", "router_small")
        model_cfg = self.config["models"][model_key]
        prompt_cfg = self.prompts["router"]
        routing_prior = _prior_text(source, answer_type, category, router_cfg)
        if risk_label:
            routing_prior = (
                f"{routing_prior}\n"
                "Risk-aware static gate:\n"
                f"- initial_candidate: {risk_label}\n"
                "- Judge only whether this candidate should be upgraded."
            ).strip()
        prompt = prompt_cfg["template"].format(
            question=question,
            caption_text=caption_text,
            source=source or "unknown",
            answer_type=answer_type or "unknown",
            category=category or "unknown",
            metadata=metadata or {},
            routing_prior=routing_prior,
        )

        cache_cfg = self.config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True) and cache_cfg.get("cache_router", True)
        key = cache_key([question, caption_text, source, answer_type, category, routing_prior, prompt_cfg.get("version"), model_cfg["model_name"]])
        cache_dir = Path(cache_cfg.get("cache_dir", "data/cache")) / "router"
        if cache_enabled:
            cached = read_cache(cache_dir, key)
            if cached:
                response = ModelResponse(True, model_cfg["model_name"], cached.get("raw_text", ""), 0.0, raw={"cached": True})
                return cached.get("difficulty", default), response, cached.get("parse_error")

        response = self.client.call_model(model_cfg["model_name"], prompt, timeout=model_cfg.get("timeout"))
        if not response.ok:
            return default, response, response.error or "MODEL_CALL_FAILED"
        difficulty, parse_error = parse_difficulty(response.text, default)
        if cache_enabled:
            write_cache(cache_dir, key, {"difficulty": difficulty, "raw_text": response.text, "parse_error": parse_error})
        return difficulty, response, parse_error
