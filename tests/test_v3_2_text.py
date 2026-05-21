from adaroute.eval.metrics import compute_metrics
from adaroute.experiments.modes import resolve_mode_config, suite_modes
from adaroute.modules.router import RouterModule, parse_difficulty


def test_v3_2_suite_and_config_are_independent(tmp_path):
    base = {
        "routing": {
            "policies": {
                "always_small": {"default": "qwen_small"},
                "always_medium": {"default": "phi3_medium"},
                "always_gemma": {"default": "gemma_large"},
                "difficulty_based": {"simple": "qwen_small", "medium": "phi3_medium", "hard": "gemma_large"},
            }
        },
        "paths": {},
        "cache": {},
    }
    config = resolve_mode_config(base, "risk_aware_routing", output_dir=str(tmp_path / "run"), experiment_version="v3_2_text")

    assert suite_modes("text_fusion_v3_2_basic") == [
        "always_small",
        "always_middle",
        "always_gemma",
        "risk_aware_routing",
        "difficulty_routing",
        "random_routing",
        "adaroute_mm_full",
    ]
    assert suite_modes("text_fusion_v3_2_added_baselines") == [
        "random_routing",
        "adaroute_mm_full",
    ]
    assert suite_modes("text_fusion_v3_1_basic") == [
        "always_small",
        "always_gemma",
        "always_middle",
        "difficulty_routing",
    ]
    assert config["routing"]["default_policy"] == "difficulty_based"
    assert config["runtime"]["experiment_version"] == "v3_2_text"


def test_v3_2_full_baseline_keeps_text_settings_with_resource_fallback(tmp_path):
    base = {
        "routing": {
            "policies": {
                "latency_aware": {"simple": "qwen_small", "medium": "phi3_medium", "hard": "gemma_large"},
            }
        },
        "paths": {},
        "cache": {},
        "fallback": {},
    }
    config = resolve_mode_config(base, "adaroute_mm_full", output_dir=str(tmp_path / "run"), experiment_version="v3_2_text")

    assert config["vlm"]["enabled"] is False
    assert config["routing"]["default_policy"] == "latency_aware"
    assert config["fallback"]["enabled"] is True
    assert config["cache"]["enabled"] is True
    assert config["runtime"]["experiment_version"] == "v3_2_text"


def test_v3_2_risk_labels_parse_to_existing_difficulty_policy_labels():
    assert parse_difficulty("small_ok")[0] == "easy"
    assert parse_difficulty("middle_ok")[0] == "medium"
    assert parse_difficulty("need_gemma")[0] == "hard"


def test_v3_2_static_gate_routes_known_sources_without_model_call():
    config = {
        "router": {
            "enabled": True,
            "default_difficulty": "medium",
            "risk_aware_static_gate": {
                "enabled": True,
                "model_gate_enabled": False,
                "direct_answer_type_priors": {"numeric": "hard"},
                "direct_source_priors": {"ucinlp/drop": "hard", "meta-math/GSM8K_zh": "hard"},
                "hard_cues": [" not "],
                "source_rules": {
                    "allenai/sciq": {"simple_difficulty": "easy", "default_difficulty": "medium", "max_simple_chars": 600},
                    "mib-bench/arc_easy": {"simple_difficulty": "easy", "default_difficulty": "medium", "max_simple_chars": 600},
                    "mib-bench/arc_challenge": {"default_difficulty": "medium"},
                },
            },
        },
        "models": {"router_small": {"model_name": "router"}},
        "cache": {"enabled": False},
    }
    prompts = {"router": {"version": "v", "template": "{question}"}}

    class Client:
        def call_model(self, *args, **kwargs):
            raise AssertionError("v3_2 deterministic risk gate should not call router model")

    router = RouterModule(config, prompts, Client())

    assert router.run("What gas do plants need?", "", source="allenai/sciq", answer_type="multiple_choice")[0] == "easy"
    assert router.run("Which item below is NOT made from a natural material?", "", source="mib-bench/arc_easy")[0] == "medium"
    assert router.run("A two-step science question", "", source="mib-bench/arc_challenge")[0] == "medium"
    assert router.run("How many?", "", source="meta-math/GSM8K_zh", answer_type="numeric")[0] == "hard"


def test_metrics_include_route_distribution_breakdowns():
    summary = compute_metrics(
        [
            {
                "status": "success",
                "source": "allenai/sciq",
                "answer_type": "multiple_choice",
                "route": {"difficulty": "easy", "final_model": "qwen_small"},
                "model_calls": [],
            },
            {
                "status": "success",
                "source": "ucinlp/drop",
                "answer_type": "numeric",
                "route": {"difficulty": "hard", "final_model": "gemma_large"},
                "model_calls": [],
            },
        ]
    )

    assert summary["model_usage_by_source"]["allenai/sciq"] == {"qwen_small": 1}
    assert summary["difficulty_by_source"]["ucinlp/drop"] == {"hard": 1}
    assert summary["model_usage_by_answer_type"]["numeric"] == {"gemma_large": 1}
