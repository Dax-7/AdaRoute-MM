from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from adaroute.core.registry import get_policy
from adaroute.core.types import InferenceInput, ModelResponse
from adaroute.models.ollama_client import OllamaClient
from adaroute.modules.fallback import FallbackManager, check_response_quality
from adaroute.modules.llm import LLMModule
from adaroute.modules.router import RouterModule
from adaroute.modules.vlm import VLMModule
from adaroute.utils.io import output_path, write_json
from adaroute.utils.logger import setup_logger
from adaroute.utils.system_monitor import get_system_state


class AdaRoutePipeline:
    def __init__(self, config: dict[str, Any], prompts: dict[str, Any]):
        self.config = config
        self.prompts = prompts
        ollama_cfg = config.get("ollama", {})
        self.client = OllamaClient(
            ollama_cfg.get("base_url", "http://localhost:11434"),
            ollama_cfg.get("api_generate", "/api/generate"),
            ollama_cfg.get("default_timeout", 120),
        )
        self.logger = setup_logger(config.get("paths", {}).get("log_dir", "data/outputs/logs"))
        self.vlm = VLMModule(config, prompts, self.client)
        self.router = RouterModule(config, prompts, self.client)
        self.llm = LLMModule(config, prompts, self.client)

    def _call_record(self, stage: str, model_key: str, response: ModelResponse | None) -> dict[str, Any]:
        if response is None:
            return {"stage": stage, "model": model_key, "latency": 0.0, "ok": True, "skipped": True}
        record = {"stage": stage, "model": model_key, "latency": response.latency, "ok": response.ok, "error": response.error}
        if isinstance(response.raw, dict) and response.raw.get("cached"):
            record["cached"] = True
        if isinstance(response.raw, dict) and response.raw.get("retried"):
            record["retried"] = True
            record["retry_reason"] = response.raw.get("retry_reason")
        return record

    def run(self, inference_input: InferenceInput, policy_name: str | None = None) -> dict[str, Any]:
        request_id = inference_input.request_id or f"sample_{uuid.uuid4().hex[:8]}"
        start = time.perf_counter()
        model_calls: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        latency = {"total": 0.0, "vlm": 0.0, "router": 0.0, "llm": 0.0}

        system_state = get_system_state(self.config)
        caption_text = ""
        try:
            caption_text, vlm_response = self.vlm.run(inference_input.question, inference_input.image_path)
            if vlm_response is not None:
                latency["vlm"] = vlm_response.latency
                model_calls.append(self._call_record("vlm", "moondream_vlm", vlm_response))
                self.logger.info("request_id=%s stage=vlm latency=%.2f ok=%s", request_id, vlm_response.latency, vlm_response.ok)
                if not vlm_response.ok:
                    errors.append({"code": "IMAGE_NOT_FOUND" if "IMAGE_NOT_FOUND" in (vlm_response.error or "") else "MODEL_CALL_FAILED", "message": vlm_response.error or ""})
            else:
                model_calls.append(self._call_record("vlm", "moondream_vlm", None))
        except Exception as exc:
            errors.append({"code": "UNKNOWN_ERROR", "message": f"VLM failed: {exc}"})

        difficulty = self.config.get("router", {}).get("default_difficulty", "中等")
        router_error = None
        try:
            difficulty, router_response, router_error = self.router.run(inference_input.question, caption_text)
            if router_response is not None:
                latency["router"] = router_response.latency
                model_calls.append(self._call_record("router", self.config.get("router", {}).get("model", "router_small"), router_response))
            if router_error:
                errors.append({"code": "ROUTER_PARSE_ERROR" if router_error == "ROUTER_PARSE_ERROR" else "MODEL_CALL_FAILED", "message": router_error})
            self.logger.info("request_id=%s stage=router difficulty=%s latency=%.2f", request_id, difficulty, latency["router"])
        except Exception as exc:
            errors.append({"code": "UNKNOWN_ERROR", "message": f"Router failed: {exc}"})

        selected_policy = policy_name or self.config.get("routing", {}).get("default_policy", "latency_aware")
        try:
            decision = get_policy(selected_policy).select_model(difficulty, system_state, self.config)
        except Exception as exc:
            errors.append({"code": "CONFIG_ERROR", "message": str(exc)})
            decision = get_policy("latency_aware").select_model(difficulty, system_state, self.config)
        self.logger.info(
            "request_id=%s stage=routing policy=%s selected_model=%s overloaded=%s",
            request_id,
            decision.policy,
            decision.selected_model,
            decision.overloaded,
        )

        def run_llm(model_key: str) -> ModelResponse:
            response = self.llm.run(inference_input.question, caption_text, model_key)
            model_calls.append(self._call_record("llm", model_key, response))
            return response

        initial_response = run_llm(decision.selected_model)
        latency["llm"] = initial_response.latency
        fallback = FallbackManager(self.config, run_llm)
        final_response, fallback_info = fallback.run(decision.selected_model, initial_response, system_state)
        if final_response is not initial_response:
            latency["llm"] = sum(call.get("latency", 0.0) for call in model_calls if call.get("stage") == "llm")
        valid, quality_reasons = check_response_quality(final_response.text if final_response.ok else "", self.config)

        status = "success" if final_response.ok and valid else "failed"
        final_model_key = fallback_info["trace"][-1]["model"] if fallback_info.get("trace") else decision.selected_model
        if status == "failed" and not errors:
            errors.append({"code": "LOW_QUALITY_RESPONSE", "message": ",".join(quality_reasons) or final_response.error or "LLM failed"})

        latency["total"] = time.perf_counter() - start
        result = {
            "request_id": request_id,
            "status": status,
            "input": asdict(inference_input),
            "caption_text": caption_text,
            "answer": final_response.text if final_response.ok else "",
            "model_used": final_model_key,
            "route": {
                "difficulty": difficulty,
                "policy": decision.policy,
                "initial_model": decision.selected_model,
                "final_model": final_model_key,
            },
            "fallback": fallback_info,
            "latency": latency,
            "system": {
                "cpu_percent": system_state.get("cpu_percent"),
                "ram_percent": system_state.get("ram_percent"),
                "gpu_percent": system_state.get("gpu_percent"),
                "temperature": system_state.get("temperature"),
                "is_overloaded": system_state.get("is_overloaded"),
                "backend": system_state.get("backend"),
            },
            "model_calls": model_calls,
            "error": None if status == "success" else (errors[-1] if errors else {"code": "UNKNOWN_ERROR", "message": "Unknown failure"}),
            "errors": errors,
        }
        write_json(output_path(self.config.get("paths", {}).get("output_dir", "data/outputs"), request_id), result)
        self.logger.info("request_id=%s stage=done total_latency=%.2f fallback=%s", request_id, latency["total"], fallback_info["triggered"])
        return result
