from adaroute.core.types import ModelResponse
from adaroute.eval.metrics import compute_metrics
from adaroute.experiments.modes import resolve_mode_config, suite_modes
from adaroute.modules.router import RouterModule


def test_v3_2_suite_and_model_override(tmp_path):
    base = {
        "models": {
            "router_small": {"model_name": "old_router"},
            "qwen_small": {"model_name": "old_small"},
        },
        "routing": {"policies": {"always_medium": {"default": "phi3_medium"}}},
        "paths": {},
        "cache": {},
    }
    config = resolve_mode_config(base, "risk_static_routing", output_dir=str(tmp_path / "run"), experiment_version="v3_2_text")

    assert suite_modes("text_fusion_v3_2_basic") == [
        "always_small",
        "always_middle",
        "always_gemma",
        "risk_static_routing",
        "risk_dynamic_routing",
    ]
    assert config["runtime"]["experiment_version"] == "v3_2_text"
    assert config["routing"]["default_policy"] == "difficulty_based"
    assert config["router"]["risk_gate"]["static_only"] is True


def test_v3_2_static_gate_routes_numeric_without_model_call():
    config = {
        "router": {
            "enabled": True,
            "default_difficulty": "middle_ok",
            "risk_gate": {
                "enabled": True,
                "static_only": True,
                "default_label": "middle_ok",
                "answer_type_routes": {"numeric": "large_required"},
            },
        },
        "models": {"router_small": {"model_name": "router"}},
        "cache": {"enabled": False},
    }
    prompts = {"router": {"version": "v", "template": "{question}"}}

    class Client:
        def call_model(self, *args, **kwargs):
            raise AssertionError("static gate should not call router model")

    difficulty, response, error, route_info = RouterModule(config, prompts, Client()).run("1+1?", "", answer_type="numeric")

    assert difficulty == "large_required"
    assert response is None
    assert error is None
    assert route_info["static_gate"] is True
    assert route_info["route_reason"] == "answer_type:numeric->large_required"


def test_v3_2_dynamic_gate_limits_allowed_labels():
    config = {
        "router": {
            "enabled": True,
            "model": "router_small",
            "default_difficulty": "middle_ok",
            "risk_gate": {
                "enabled": True,
                "static_only": False,
                "default_label": "middle_ok",
                "dynamic_sources": {
                    "allenai/sciq": {
                        "default_label": "middle_ok",
                        "allowed_labels": ["small_ok", "middle_ok"],
                    }
                },
            },
        },
        "models": {"router_small": {"model_name": "router", "timeout": 1}},
        "cache": {"enabled": False},
    }
    prompts = {"router": {"version": "v", "template": "{question}\nAllowed: {candidate_labels}"}}

    class Client:
        prompt = ""

        def call_model(self, model_name, prompt, **kwargs):
            self.prompt = prompt
            return ModelResponse(True, model_name, "small_ok", 0.01)

    client = Client()
    difficulty, response, error, route_info = RouterModule(config, prompts, client).run(
        "Which option is correct?",
        "",
        source="allenai/sciq",
        answer_type="multiple_choice",
    )

    assert difficulty == "small_ok"
    assert response is not None
    assert error is None
    assert route_info["dynamic_gate"] is True
    assert "small_ok, middle_ok" in client.prompt


def test_v3_2_metrics_include_gate_and_switch_rates():
    summary = compute_metrics(
        [
            {
                "status": "success",
                "answer": "FINAL_ANSWER: A",
                "reference_answer": "A",
                "answer_type": "multiple_choice",
                "source": "allenai/sciq",
                "model_used": "phi3_medium",
                "route": {
                    "difficulty": "middle_ok",
                    "final_model": "phi3_medium",
                    "route_reason": "dynamic_default:allenai/sciq->middle_ok",
                    "static_gate": True,
                    "dynamic_gate": False,
                },
                "model_calls": [],
            },
            {
                "status": "success",
                "answer": "FINAL_ANSWER: 2",
                "reference_answer": "2",
                "answer_type": "numeric",
                "source": "meta-math/GSM8K_zh",
                "model_used": "gemma_large",
                "route": {
                    "difficulty": "large_required",
                    "final_model": "gemma_large",
                    "route_reason": "answer_type:numeric->large_required",
                    "static_gate": True,
                    "dynamic_gate": False,
                },
                "model_calls": [],
            },
        ]
    )

    assert summary["middle_usage_rate"] == 0.5
    assert summary["gemma_usage_rate"] == 0.5
    assert summary["static_gate_rate"] == 1.0
    assert summary["model_switch_rate"] == 1.0
    assert summary["text_answer"]["routing_by_source"]["allenai/sciq"]["model_usage_distribution"] == {"phi3_medium": 1}
