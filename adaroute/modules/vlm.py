from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from adaroute.core.types import ModelResponse
from adaroute.utils.io import cache_key, read_cache, write_cache


class VLMModule:
    def __init__(self, config: dict[str, Any], prompts: dict[str, Any], client: Any):
        self.config = config
        self.prompts = prompts
        self.client = client

    def run(self, question: str, image_path: str | None) -> tuple[str, ModelResponse | None]:
        if not image_path or not self.config.get("vlm", {}).get("enabled", True):
            return "", None
        if not Path(image_path).exists():
            return "", ModelResponse(False, self.config["vlm"].get("model", "moondream_vlm"), "", 0.0, error="IMAGE_NOT_FOUND")

        model_key = "moondream_vlm"
        model_cfg = self.config["models"][model_key]
        prompt_cfg = self.prompts["vlm"]
        prompt = prompt_cfg["template"].format(question=question)

        cache_cfg = self.config.get("cache", {})
        cache_enabled = cache_cfg.get("enabled", True) and cache_cfg.get("cache_vlm", True)
        key = cache_key([image_path, question, prompt_cfg.get("version"), model_cfg["model_name"]])
        cache_dir = Path(cache_cfg.get("cache_dir", "data/cache")) / "vlm"
        if cache_enabled:
            cached = read_cache(cache_dir, key)
            cached_caption = (cached or {}).get("caption_text", "").strip()
            if cached_caption:
                response = ModelResponse(True, model_cfg["model_name"], cached_caption, 0.0, raw={"cached": True})
                return response.text, response

        response = self.client.call_model(
            model_name=model_cfg["model_name"],
            prompt=prompt,
            images=[image_path],
            timeout=model_cfg.get("timeout"),
        )
        if response.ok and not response.text.strip():
            retry_response = self.client.call_model(
                model_name=model_cfg["model_name"],
                prompt="Describe this image in detail.",
                images=[image_path],
                timeout=model_cfg.get("timeout"),
            )
            total_latency = response.latency + retry_response.latency
            retry_response.latency = total_latency
            if retry_response.ok and retry_response.text.strip():
                retry_response.raw = {
                    "retried": True,
                    "retry_reason": "EMPTY_VLM_RESPONSE",
                    "initial_raw": response.raw,
                    "retry_raw": retry_response.raw,
                }
                response = retry_response
            else:
                response.raw = {
                    "retried": True,
                    "retry_reason": "EMPTY_VLM_RESPONSE",
                    "initial_raw": response.raw,
                    "retry_raw": retry_response.raw,
                }
                response.latency = total_latency
                response.error = retry_response.error or "EMPTY_VLM_RESPONSE"
                response.ok = False

        if response.ok and not response.text.strip():
            response = ModelResponse(
                False,
                response.model,
                "",
                response.latency,
                response.prompt_eval_count,
                response.eval_count,
                response.raw,
                error="EMPTY_VLM_RESPONSE",
            )
        if response.ok and cache_enabled:
            write_cache(cache_dir, key, {"caption_text": response.text, "created_at": time.time()})
        return response.text if response.ok else "", response
