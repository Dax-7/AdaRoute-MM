from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.core.types import InferenceInput, ModelResponse


def cfg(tmp_path):
    return {
        "ollama": {"base_url": "http://localhost:11434", "api_generate": "/api/generate", "default_timeout": 1},
        "models": {
            "moondream_vlm": {"model_name": "moondream", "timeout": 1},
            "router_small": {"model_name": "router", "timeout": 1},
            "qwen_small": {"model_name": "qwen", "timeout": 1},
            "phi3_medium": {"model_name": "phi3", "timeout": 1},
            "gemma_large": {"model_name": "gemma", "timeout": 1},
        },
        "vlm": {"enabled": True},
        "router": {"enabled": True, "model": "router_small", "default_difficulty": "中等"},
        "routing": {
            "default_policy": "latency_aware",
            "policies": {
                "latency_aware": {
                    "simple": "qwen_small",
                    "medium": "phi3_medium",
                    "hard": "gemma_large",
                    "overload_model": "qwen_small",
                    "skip_large_when_overloaded": True,
                }
            },
        },
        "fallback": {
            "enabled": True,
            "max_attempts": 3,
            "upgrade_order": ["qwen_small", "phi3_medium", "gemma_large"],
            "quality_checks": {"min_chars": 5, "uncertainty_keywords": ["不确定"]},
        },
        "system": {"monitor_backend": "psutil", "overload_policy": {"cpu_percent": 100, "ram_percent": 100}},
        "cache": {"enabled": False, "cache_dir": str(tmp_path / "cache")},
        "paths": {"output_dir": str(tmp_path / "outputs"), "log_dir": str(tmp_path / "logs")},
    }


def prompts():
    return {
        "vlm": {"version": "v1", "template": "{question}"},
        "router": {"version": "v1", "template": "{question}\n{caption_text}"},
        "llm": {"version": "v1", "template": "{question}\n{caption_text}"},
    }


class FakeClient:
    def __init__(self, router_text="简单", llm_ok=True):
        self.router_text = router_text
        self.llm_ok = llm_ok

    def call_model(self, model_name, prompt, images=None, options=None, timeout=None):
        if model_name == "router":
            return ModelResponse(True, model_name, self.router_text, 0.01)
        if images:
            return ModelResponse(True, model_name, "图像描述", 0.01)
        if not self.llm_ok:
            return ModelResponse(False, model_name, "", 0.01, error="MODEL_CALL_FAILED")
        return ModelResponse(True, model_name, "这是一个中文回答。", 0.01)


def pipeline(tmp_path, client):
    pipe = AdaRoutePipeline(cfg(tmp_path), prompts())
    pipe.client = client
    pipe.vlm.client = client
    pipe.router.client = client
    pipe.llm.client = client
    return pipe


def test_text_input_runs(tmp_path):
    result = pipeline(tmp_path, FakeClient()).run(InferenceInput(question="什么是边缘计算？", request_id="t1"))
    assert result["status"] == "success"
    assert result["caption_text"] == ""


def test_no_image_skips_vlm(tmp_path):
    result = pipeline(tmp_path, FakeClient()).run(InferenceInput(question="hello", request_id="t2"))
    assert result["model_calls"][0]["stage"] == "vlm"
    assert result["model_calls"][0]["skipped"]


def test_router_parse_failure_uses_default(tmp_path):
    result = pipeline(tmp_path, FakeClient(router_text="unknown")).run(InferenceInput(question="hello", request_id="t3"))
    assert result["route"]["difficulty"] == "中等"


def test_llm_failure_returns_failed(tmp_path):
    result = pipeline(tmp_path, FakeClient(llm_ok=False)).run(InferenceInput(question="hello", request_id="t4"))
    assert result["status"] == "failed"


def test_output_contains_required_fields(tmp_path):
    result = pipeline(tmp_path, FakeClient()).run(InferenceInput(question="hello", request_id="t5"))
    for key in ["request_id", "status", "input", "route", "fallback", "latency", "system", "model_calls", "error"]:
        assert key in result
