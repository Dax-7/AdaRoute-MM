from __future__ import annotations

from typing import Optional, Protocol

from adaroute.core.types import ModelResponse


class ModelClient(Protocol):
    def call_model(
        self,
        model_name: str,
        prompt: str,
        images: Optional[list[str]] = None,
        options: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> ModelResponse:
        ...
