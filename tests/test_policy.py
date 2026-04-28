from adaroute.core.registry import get_policy


def cfg():
    return {
        "routing": {
            "policies": {
                "difficulty_based": {"simple": "qwen_small", "medium": "phi3_medium", "hard": "gemma_large"},
                "always_small": {"default": "qwen_small"},
                "always_medium": {"default": "phi3_medium"},
                "always_large": {"default": "gemma_large"},
                "random": {"candidates": ["qwen_small", "phi3_medium", "gemma_large"], "seed": 42},
                "latency_aware": {
                    "simple": "qwen_small",
                    "medium": "phi3_medium",
                    "hard": "gemma_large",
                    "overload_model": "qwen_small",
                    "skip_large_when_overloaded": True,
                },
            }
        }
    }


def test_difficulty_based_policy():
    decision = get_policy("difficulty_based").select_model("困难", {}, cfg())
    assert decision.selected_model == "gemma_large"


def test_always_small_policy():
    decision = get_policy("always_small").select_model("困难", {}, cfg())
    assert decision.selected_model == "qwen_small"


def test_random_policy_candidate():
    decision = get_policy("random").select_model("简单", {}, cfg())
    assert decision.selected_model in {"qwen_small", "phi3_medium", "gemma_large"}


def test_latency_aware_overload_returns_small():
    decision = get_policy("latency_aware").select_model("困难", {"is_overloaded": True}, cfg())
    assert decision.selected_model == "qwen_small"
    assert decision.overloaded
