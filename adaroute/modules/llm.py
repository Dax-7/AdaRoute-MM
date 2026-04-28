from __future__ import annotations

from typing import Any

from adaroute.core.types import ModelResponse


class LLMModule:
    def __init__(self, config: dict[str, Any], prompts: dict[str, Any], client: Any):
        self.config = config
        self.prompts = prompts
        self.client = client

    def run(self, question: str, caption_text: str, model_key: str) -> ModelResponse:
        model_cfg = self.config["models"][model_key]
        prompt = self.prompts["llm"]["template"].format(question=question, caption_text=caption_text)
        return self.client.call_model(model_cfg["model_name"], prompt, timeout=model_cfg.get("timeout"))
