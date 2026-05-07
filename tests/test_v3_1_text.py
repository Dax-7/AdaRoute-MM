import json

from adaroute.eval.metrics import compute_metrics
from adaroute.experiments.modes import resolve_mode_config, suite_modes
from adaroute.modules.router import RouterModule
from adaroute.v3.selection import select_verified_numeric


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_v3_1_suite_adds_middle_without_changing_v3_suite(tmp_path):
    base = {
        "routing": {"policies": {"always_medium": {"default": "phi3_medium"}}},
        "paths": {},
        "cache": {},
    }
    config = resolve_mode_config(base, "always_middle", output_dir=str(tmp_path / "run"), experiment_version="v3_1_text")

    assert suite_modes("text_fusion_v3_1_basic") == [
        "always_small",
        "always_gemma",
        "always_middle",
        "difficulty_routing",
    ]
    assert suite_modes("text_fusion_v3_basic") == ["always_small", "always_gemma", "difficulty_routing", "random_routing"]
    assert config["routing"]["default_policy"] == "always_medium"
    assert config["routing"]["policies"]["always_medium"]["default"] == "phi3_medium"
    assert config["runtime"]["experiment_version"] == "v3_1_text"


def test_router_numeric_prior_skips_model_call():
    config = {
        "router": {
            "enabled": True,
            "default_difficulty": "medium",
            "use_source_prior": True,
            "answer_type_priors": {"numeric": "hard"},
        },
        "models": {"router_small": {"model_name": "router"}},
        "cache": {"enabled": False},
    }
    prompts = {"router": {"version": "v", "template": "{question}"}}

    class Client:
        def call_model(self, *args, **kwargs):
            raise AssertionError("numeric prior should not call router model")

    difficulty, response, error = RouterModule(config, prompts, Client()).run("1+1?", "", answer_type="numeric")

    assert difficulty == "hard"
    assert response is None
    assert error is None


def test_metrics_default_latency_excludes_failed_samples():
    summary = compute_metrics(
        [
            {"status": "success", "latency": {"total": 1.0, "llm": 0.8}, "model_calls": []},
            {"status": "failed", "latency": {"total": 101.0, "llm": 100.0}, "model_calls": []},
        ]
    )

    assert summary["average_latency"] == 1.0
    assert summary["success_only_average_latency"] == 1.0
    assert summary["average_latency_all_samples"] == 51.0
    assert summary["failed_latency_excluded_count"] == 1
    assert summary["average_llm_latency"] == 0.8


def test_verified_numeric_selection_adds_gsm8k_filler(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    small = tmp_path / "small.jsonl"
    gemma = tmp_path / "gemma.jsonl"
    route = tmp_path / "route.jsonl"
    output = tmp_path / "verified.jsonl"

    rows = [
        {"id": "n1", "question": "q1", "answer": "1", "answer_type": "numeric", "source": "meta-math/GSM8K_zh"},
        {"id": "n2", "question": "q2", "answer": "2", "answer_type": "numeric", "source": "meta-math/GSM8K_zh"},
    ]
    _write_jsonl(dataset, rows)
    _write_jsonl(
        small,
        [
            {"sample_id": "n1", "answer": "FINAL_ANSWER: 0", "reference_answer": "1", "answer_type": "numeric", "status": "success"},
            {"sample_id": "n2", "answer": "FINAL_ANSWER: 2", "reference_answer": "2", "answer_type": "numeric", "status": "success"},
        ],
    )
    _write_jsonl(
        gemma,
        [
            {"sample_id": "n1", "answer": "FINAL_ANSWER: 1", "reference_answer": "1", "answer_type": "numeric", "status": "success"},
            {"sample_id": "n2", "answer": "FINAL_ANSWER: 2", "reference_answer": "2", "answer_type": "numeric", "status": "success"},
        ],
    )
    _write_jsonl(
        route,
        [
            {"sample_id": "n1", "answer": "FINAL_ANSWER: 0", "reference_answer": "1", "answer_type": "numeric", "status": "success"},
            {"sample_id": "n2", "answer": "FINAL_ANSWER: 2", "reference_answer": "2", "answer_type": "numeric", "status": "success"},
        ],
    )

    report = select_verified_numeric(
        dataset_path=dataset,
        small_results=small,
        gemma_results=gemma,
        route_results=route,
        output_path=output,
        target_size=2,
    )

    selected = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert report["selected_bucket_counts"] == {"verified_numeric": 1, "gsm8k_numeric_filler": 1}
    assert {row["id"] for row in selected} == {"n1", "n2"}
