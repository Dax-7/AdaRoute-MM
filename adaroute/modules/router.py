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


class RouterModule:
    def __init__(self, config: dict[str, Any], prompts: dict[str, Any], client: Any):
        self.config = config
        self.prompts = prompts
        self.client = client

    def run(self, question: str, caption_text: str) -> tuple[str, ModelResponse | None, str | None]:
        router_cfg = self.config.get("router", {})
        default = router_cfg.get("default_difficulty", LEGACY_MEDIUM)
        if not router_cfg.get("enabled", True):
            return default, None, None

        model_key = router_cfg.get("model", "router_small")
        model_cfg = self.config["models"][model_key]
        prompt_cfg = self.prompts["router"]
        prompt = prompt_cfg["template"].format(question=question, caption_text=caption_text)

        cache_cfg = self.config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True) and cache_cfg.get("cache_router", True)
        key = cache_key([question, caption_text, prompt_cfg.get("version"), model_cfg["model_name"]])
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
