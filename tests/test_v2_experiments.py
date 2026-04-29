from adaroute.core.types import ModelResponse
from adaroute.eval.yesno import compute_yesno_metrics, extract_yes_no
from adaroute.experiments.modes import resolve_mode_config
from adaroute.modules.vlm import VLMModule


def test_image_caption_cache_reuses_caption_for_multiple_questions(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")
    config = {
        "models": {"moondream_vlm": {"model_name": "moondream", "timeout": 1}},
        "vlm": {"enabled": True, "caption_mode": "image_caption"},
        "cache": {"enabled": True, "cache_vlm": True, "cache_dir": str(tmp_path / "cache")},
    }
    prompts = {
        "vlm": {"version": "v1", "template": "{question}"},
        "vlm_general": {"version": "v2", "template": "describe image"},
    }

    class Client:
        def __init__(self):
            self.calls = 0

        def call_model(self, model_name, prompt, images=None, timeout=None):
            self.calls += 1
            return ModelResponse(True, model_name, "a person at a table", 0.01)

    client = Client()
    module = VLMModule(config, prompts, client)
    first_caption, first_response = module.run("Is there a person?", str(image_path))
    second_caption, second_response = module.run("Is there a table?", str(image_path))

    assert first_caption == second_caption == "a person at a table"
    assert first_response is not None and first_response.ok
    assert second_response is not None and second_response.raw["cached"]
    assert client.calls == 1


def test_yesno_metrics_extracts_binary_answers():
    assert extract_yes_no("Yes, because the object is visible.") == "yes"
    assert extract_yes_no("No, it is not visible.") == "no"
    assert extract_yes_no("It is unclear.") == "invalid"

    metrics = compute_yesno_metrics(
        [
            {
                "answer": "Yes, because it is visible.",
                "multiple_choice_answer": "yes",
                "reference_answers": ["yes", "yes", "yes"],
                "answer_type": "yes/no",
            },
            {
                "answer": "No.",
                "multiple_choice_answer": "yes",
                "reference_answers": ["yes", "yes", "yes"],
                "answer_type": "yes/no",
            },
        ]
    )

    assert metrics["total_evaluated"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["invalid_rate"] == 0.0


def test_full_mode_enables_cache_and_fallback(tmp_path):
    config = resolve_mode_config(
        {
            "routing": {"policies": {"latency_aware": {}}},
            "paths": {},
            "cache": {},
        },
        "adaroute_mm_full",
        output_dir=str(tmp_path / "run"),
    )

    assert config["routing"]["default_policy"] == "latency_aware"
    assert config["vlm"]["caption_mode"] == "image_caption"
    assert config["cache"]["enabled"]
    assert config["cache"]["cache_vlm"]
    assert config["fallback"]["enabled"]

