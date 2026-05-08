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
    for label in ("small_ok", "middle_ok", "large_required"):
        if label in normalized:
            return label, None
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


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower()


def _risk_route(
    question: str,
    source: str | None,
    answer_type: str | None,
    category: str | None,
    router_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    risk_cfg = router_cfg.get("risk_gate", {})
    if not risk_cfg.get("enabled", False):
        return None

    default_label = risk_cfg.get("default_label", "middle_ok")
    answer_type_routes = risk_cfg.get("answer_type_routes", {})
    source_routes = risk_cfg.get("source_routes", {})
    category_routes = risk_cfg.get("category_routes", {})
    dynamic_sources = risk_cfg.get("dynamic_sources", {})
    dynamic_categories = risk_cfg.get("dynamic_categories", {})

    answer_type_key = _normalize(answer_type)
    if answer_type_key in answer_type_routes:
        label = answer_type_routes[answer_type_key]
        return {
            "label": label,
            "source": "static_gate",
            "reason": f"answer_type:{answer_type_key}->{label}",
            "static_gate": True,
            "dynamic_gate": False,
        }

    source_key = source or ""
    if source_key in source_routes:
        label = source_routes[source_key]
        return {
            "label": label,
            "source": "static_gate",
            "reason": f"source:{source_key}->{label}",
            "static_gate": True,
            "dynamic_gate": False,
        }

    category_key = category or ""
    if category_key in category_routes:
        label = category_routes[category_key]
        return {
            "label": label,
            "source": "static_gate",
            "reason": f"category:{category_key}->{label}",
            "static_gate": True,
            "dynamic_gate": False,
        }

    dynamic_cfg = dynamic_sources.get(source_key) or dynamic_categories.get(category_key)
    if dynamic_cfg:
        if risk_cfg.get("static_only", False):
            label = dynamic_cfg.get("default_label", default_label)
            return {
                "label": label,
                "source": "static_gate",
                "reason": f"dynamic_default:{source_key or category_key}->{label}",
                "static_gate": True,
                "dynamic_gate": False,
            }
        return {
            "label": dynamic_cfg.get("default_label", default_label),
            "source": "dynamic_gate",
            "reason": f"dynamic_candidate:{source_key or category_key}",
            "static_gate": False,
            "dynamic_gate": True,
            "candidate_labels": dynamic_cfg.get("allowed_labels", risk_cfg.get("allowed_labels", [])),
        }

    if risk_cfg.get("static_only", False):
        return {
            "label": default_label,
            "source": "static_gate",
            "reason": f"default->{default_label}",
            "static_gate": True,
            "dynamic_gate": False,
        }
    return {
        "label": default_label,
        "source": "dynamic_gate",
        "reason": "dynamic_default",
        "static_gate": False,
        "dynamic_gate": True,
        "candidate_labels": risk_cfg.get("allowed_labels", []),
    }


def _candidate_text(candidate_labels: list[str] | None) -> str:
    if not candidate_labels:
        return "small_ok, middle_ok, or large_required"
    return ", ".join(candidate_labels)


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
    ) -> tuple[str, ModelResponse | None, str | None, dict[str, Any]]:
        router_cfg = self.config.get("router", {})
        default = router_cfg.get("default_difficulty", LEGACY_MEDIUM)
        if not router_cfg.get("enabled", True):
            return default, None, None, {
                "route_source": "router_disabled",
                "route_reason": "router_disabled",
                "static_gate": False,
                "dynamic_gate": False,
            }

        risk_route = _risk_route(question, source, answer_type, category, router_cfg)
        if risk_route and not risk_route.get("dynamic_gate"):
            return risk_route["label"], None, None, {
                "route_source": risk_route["source"],
                "route_reason": risk_route["reason"],
                "static_gate": bool(risk_route.get("static_gate")),
                "dynamic_gate": False,
            }

        prior_label = _direct_prior(answer_type, router_cfg)
        if prior_label:
            return prior_label, None, None, {
                "route_source": "direct_prior",
                "route_reason": f"answer_type:{_normalize(answer_type)}->{prior_label}",
                "static_gate": True,
                "dynamic_gate": False,
            }

        model_key = router_cfg.get("model", "router_small")
        model_cfg = self.config["models"][model_key]
        prompt_cfg = self.prompts["router"]
        routing_prior = _prior_text(source, answer_type, category, router_cfg)
        candidate_labels = risk_route.get("candidate_labels") if risk_route else None
        prompt = prompt_cfg["template"].format(
            question=question,
            caption_text=caption_text,
            source=source or "unknown",
            answer_type=answer_type or "unknown",
            category=category or "unknown",
            metadata=metadata or {},
            routing_prior=routing_prior,
            candidate_labels=_candidate_text(candidate_labels),
        )

        cache_cfg = self.config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True) and cache_cfg.get("cache_router", True)
        key = cache_key(
            [
                question,
                caption_text,
                source,
                answer_type,
                category,
                routing_prior,
                candidate_labels,
                prompt_cfg.get("version"),
                model_cfg["model_name"],
            ]
        )
        cache_dir = Path(cache_cfg.get("cache_dir", "data/cache")) / "router"
        if cache_enabled:
            cached = read_cache(cache_dir, key)
            if cached:
                response = ModelResponse(True, model_cfg["model_name"], cached.get("raw_text", ""), 0.0, raw={"cached": True})
                return cached.get("difficulty", default), response, cached.get("parse_error"), {
                    "route_source": "dynamic_gate" if risk_route else "router_model",
                    "route_reason": (risk_route or {}).get("reason", "cached_router_model"),
                    "static_gate": False,
                    "dynamic_gate": bool(risk_route),
                }

        response = self.client.call_model(model_cfg["model_name"], prompt, timeout=model_cfg.get("timeout"))
        if not response.ok:
            fallback_label = (risk_route or {}).get("label", default)
            return fallback_label, response, response.error or "MODEL_CALL_FAILED", {
                "route_source": "dynamic_gate" if risk_route else "router_model",
                "route_reason": f"{(risk_route or {}).get('reason', 'router_model')}:fallback->{fallback_label}",
                "static_gate": False,
                "dynamic_gate": bool(risk_route),
            }
        difficulty, parse_error = parse_difficulty(response.text, default)
        if candidate_labels and difficulty not in set(candidate_labels):
            difficulty = (risk_route or {}).get("label", default)
            parse_error = "ROUTER_PARSE_ERROR"
        if cache_enabled:
            write_cache(cache_dir, key, {"difficulty": difficulty, "raw_text": response.text, "parse_error": parse_error})
        return difficulty, response, parse_error, {
            "route_source": "dynamic_gate" if risk_route else "router_model",
            "route_reason": (risk_route or {}).get("reason", "router_model"),
            "static_gate": False,
            "dynamic_gate": bool(risk_route),
        }
