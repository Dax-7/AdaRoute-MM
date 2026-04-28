from adaroute.core.types import ModelResponse
from adaroute.modules.fallback import FallbackManager, check_response_quality, fallback_candidates


def cfg():
    return {
        "fallback": {
            "enabled": True,
            "max_attempts": 3,
            "upgrade_order": ["qwen_small", "phi3_medium", "gemma_large"],
            "quality_checks": {"min_chars": 20, "uncertainty_keywords": ["不确定"]},
        },
        "routing": {"policies": {"latency_aware": {"hard": "gemma_large"}}},
    }


def test_empty_response_triggers_fallback():
    valid, reasons = check_response_quality("", cfg())
    assert not valid
    assert "empty_response" in reasons


def test_short_response_triggers_fallback():
    valid, reasons = check_response_quality("太短", cfg())
    assert not valid
    assert "too_short" in reasons


def test_uncertainty_triggers_fallback():
    valid, reasons = check_response_quality("我不确定这个问题的答案是什么。", cfg())
    assert not valid
    assert "uncertainty" in reasons


def test_normal_response_is_valid():
    valid, reasons = check_response_quality("边缘计算是在靠近数据源的位置进行计算，以降低延迟并减少带宽消耗。", cfg())
    assert valid
    assert reasons == []


def test_fallback_order():
    assert fallback_candidates("qwen_small", cfg(), {"is_overloaded": False}) == ["phi3_medium", "gemma_large"]


def test_fallback_manager_uses_next_model():
    calls = []

    def runner(model_key):
        calls.append(model_key)
        return ModelResponse(True, model_key, "这是一个足够长且明确的中文回答，用于通过质量检查。", 0.1)

    manager = FallbackManager(cfg(), runner)
    final, info = manager.run("qwen_small", ModelResponse(True, "qwen_small", "", 0.1), {"is_overloaded": False})
    assert final.model == "phi3_medium"
    assert info["triggered"]
    assert calls == ["phi3_medium"]
