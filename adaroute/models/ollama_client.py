from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

import requests

from adaroute.core.types import ModelResponse


class OllamaClient:
    def __init__(self, base_url: str, api_generate: str = "/api/generate", default_timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_generate = api_generate
        self.default_timeout = default_timeout

    def _encode_image(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        with path.open("rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def call_model(
        self,
        model_name: str,
        prompt: str,
        images: Optional[list[str]] = None,
        options: Optional[dict] = None,
        timeout: Optional[int] = None,
    ) -> ModelResponse:
        start = time.perf_counter()
        try:
            payload = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            }
            if options:
                payload["options"] = options
            if images:
                payload["images"] = [self._encode_image(image) for image in images]

            response = requests.post(
                f"{self.base_url}{self.api_generate}",
                json=payload,
                timeout=timeout or self.default_timeout,
            )
            latency = time.perf_counter() - start
            response.raise_for_status()
            raw = response.json()
            text = raw.get("response", "") or ""
            return ModelResponse(
                ok=True,
                model=model_name,
                text=text.strip(),
                latency=latency,
                prompt_eval_count=raw.get("prompt_eval_count"),
                eval_count=raw.get("eval_count"),
                raw=raw,
            )
        except requests.Timeout as exc:
            return ModelResponse(False, model_name, "", time.perf_counter() - start, error=f"OLLAMA_TIMEOUT: {exc}")
        except requests.ConnectionError as exc:
            return ModelResponse(False, model_name, "", time.perf_counter() - start, error=f"OLLAMA_CONNECTION_ERROR: {exc}")
        except FileNotFoundError as exc:
            return ModelResponse(False, model_name, "", time.perf_counter() - start, error=f"IMAGE_NOT_FOUND: {exc}")
        except Exception as exc:
            return ModelResponse(False, model_name, "", time.perf_counter() - start, error=f"MODEL_CALL_FAILED: {exc}")
