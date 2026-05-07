from __future__ import annotations

from pathlib import Path
from typing import Any

from adaroute.core.types import ModelResponse
from adaroute.utils.io import cache_key, read_cache, write_cache


LEGACY_EASY = "ç» â‚¬é—?"
LEGACY_MEDIUM = "æ¶“î… ç“‘"
LEGACY_HARD = "é¥ä¼´æ¯¦"


def parse_difficulty(text: str, default: str = LEGACY_MEDIUM) -> tuple[str, str | None]:
    normalized = (text or "").strip().lower()
    for label in ("easy", "medium", "hard"):
        if label in normalized:
            return label, None
    for label in ("简单", "中等", "困难"):
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
        prior_label = _direct_prior(answer_type, router_cfg)
        if prior_label:
            return prior_label, None, None

        model_key = router_cfg.get("model", "router_small")
        model_cfg = self.config["models"][model_key]
        prompt_cfg = self.prompts["router"]
        routing_prior = _prior_text(source, answer_type, category, router_cfg)
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
