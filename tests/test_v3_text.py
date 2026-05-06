from adaroute.core.pipeline import AdaRoutePipeline
from adaroute.core.types import InferenceInput, ModelResponse
from adaroute.eval.metrics import compute_metrics
from adaroute.eval.text_answer import score_text_answer
from adaroute.experiments.modes import resolve_mode_config, suite_modes


def test_v3_suite_is_text_only(tmp_path):
    config = resolve_mode_config(
        {
            "routing": {"policies": {"always_small": {"default": "qwen_small"}}},
            "paths": {},
            "cache": {},
        },
        "always_small",
        output_dir=str(tmp_path / "run"),
        experiment_version="v3_text",
    )

    assert suite_modes("text_fusion_v3_basic") == ["always_small", "always_gemma", "difficulty_routing", "random_routing"]
    assert config["vlm"]["enabled"] is False
    assert config["fallback"]["enabled"] is False
    assert config["runtime"]["experiment_version"] == "v3_text"


def test_text_answer_scoring_multiple_choice_and_numeric():
    assert score_text_answer({"answer": "FINAL_ANSWER: C", "reference_answer": "C", "answer_type": "multiple_choice"})[
        "correct"
    ]
    assert score_text_answer({"answer": "Therefore 1,200.", "reference_answer": "1200", "answer_type": "numeric"})[
        "correct"
    ]
    assert not score_text_answer({"answer": "FINAL_ANSWER: B", "reference_answer": "C", "answer_type": "multiple_choice"})[
        "correct"
    ]


def test_metrics_include_v3_accuracy_and_token_cost():
    summary = compute_metrics(
        [
            {
                "status": "success",
                "answer": "FINAL_ANSWER: A",
                "reference_answer": "A",
                "answer_type": "multiple_choice",
                "source": "demo",
                "model_calls": [
                    {
                        "stage": "llm",
                        "model": "qwen_small",
                        "prompt_eval_count": 10,
                        "eval_count": 2,
                        "timing": {
                            "inference_only_time_s": 0.3,
                            "token_normalized_cost_s": 0.3,
                            "prefill_cost_per_token_s": 0.02,
                            "decode_cost_per_token_s": 0.05,
                        },
                    }
                ],
            }
        ]
    )

    assert summary["text_answer"]["accuracy"] == 1.0
    assert summary["llm_calls_token_normalized_cost"] == 0.3
    assert summary["prompt_eval_tokens"] == 10


def test_pipeline_records_ollama_token_timing(tmp_path):
    config = {
        "ollama": {"base_url": "http://localhost:11434", "api_generate": "/api/generate", "default_timeout": 1},
        "models": {
            "moondream_vlm": {"model_name": "moondream", "timeout": 1},
            "router_small": {"model_name": "router", "timeout": 1},
            "qwen_small": {"model_name": "qwen", "timeout": 1},
            "phi3_medium": {"model_name": "phi3", "timeout": 1},
            "gemma_large": {"model_name": "gemma", "timeout": 1},
        },
        "vlm": {"enabled": False},
        "router": {"enabled": False, "model": "router_small", "default_difficulty": "medium"},
        "routing": {"default_policy": "always_small", "policies": {"always_small": {"default": "qwen_small"}}},
        "fallback": {"enabled": False, "quality_checks": {"min_chars": 1, "uncertainty_keywords": []}},
        "system": {"monitor_backend": "psutil", "overload_policy": {"cpu_percent": 100, "ram_percent": 100}},
        "cache": {"enabled": False, "cache_dir": str(tmp_path / "cache")},
        "paths": {"output_dir": str(tmp_path / "outputs"), "log_dir": str(tmp_path / "logs")},
    }
    prompts = {"router": {"version": "v1", "template": "{question}"}, "llm": {"version": "v1", "template": "{question}"}}

    class Client:
        def call_model(self, model_name, prompt, images=None, options=None, timeout=None):
            return ModelResponse(
                True,
                model_name,
                "FINAL_ANSWER: A",
                0.4,
                prompt_eval_count=10,
                eval_count=2,
                raw={"prompt_eval_duration": 200_000_000, "eval_duration": 100_000_000, "load_duration": 50_000_000},
            )

    pipe = AdaRoutePipeline(config, prompts)
    pipe.client = Client()
    pipe.llm.client = pipe.client
    result = pipe.run(InferenceInput(question="Question?", request_id="v3_timing"))
    timing = [call for call in result["model_calls"] if call["stage"] == "llm"][0]["timing"]

    assert timing["inference_only_time_s"] == 0.30000000000000004
    assert timing["prefill_cost_per_token_s"] == 0.02
    assert timing["decode_cost_per_token_s"] == 0.05
