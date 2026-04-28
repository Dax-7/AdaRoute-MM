from adaroute.core.types import ModelResponse
from adaroute.modules.vlm import VLMModule
from adaroute.utils.io import cache_key, write_cache


def test_vlm_ignores_empty_cached_caption(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")
    config = {
        "models": {"moondream_vlm": {"model_name": "moondream", "timeout": 1}},
        "vlm": {"enabled": True},
        "cache": {"enabled": True, "cache_vlm": True, "cache_dir": str(tmp_path / "cache")},
    }
    prompts = {"vlm": {"version": "v1", "template": "{question}"}}
    key = cache_key([str(image_path), "what is this?", "v1", "moondream"])
    write_cache(tmp_path / "cache" / "vlm", key, {"caption_text": ""})

    class Client:
        def call_model(self, model_name, prompt, images=None, timeout=None):
            return ModelResponse(True, model_name, "a mouse on fabric", 0.01)

    caption, response = VLMModule(config, prompts, Client()).run("what is this?", str(image_path))

    assert caption == "a mouse on fabric"
    assert response is not None
    assert response.ok
    assert response.raw is None


def test_vlm_treats_empty_model_response_as_failure(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")
    config = {
        "models": {"moondream_vlm": {"model_name": "moondream", "timeout": 1}},
        "vlm": {"enabled": True},
        "cache": {"enabled": True, "cache_vlm": True, "cache_dir": str(tmp_path / "cache")},
    }
    prompts = {"vlm": {"version": "v1", "template": "{question}"}}

    class Client:
        def call_model(self, model_name, prompt, images=None, timeout=None):
            return ModelResponse(True, model_name, "   ", 0.01)

    caption, response = VLMModule(config, prompts, Client()).run("what is this?", str(image_path))

    assert caption == ""
    assert response is not None
    assert not response.ok
    assert response.error == "EMPTY_VLM_RESPONSE"


def test_vlm_retries_empty_model_response_with_safe_prompt(tmp_path):
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"fake image")
    config = {
        "models": {"moondream_vlm": {"model_name": "moondream", "timeout": 1}},
        "vlm": {"enabled": True},
        "cache": {"enabled": False, "cache_vlm": True, "cache_dir": str(tmp_path / "cache")},
    }
    prompts = {"vlm": {"version": "v1", "template": "{question}"}}

    class Client:
        def __init__(self):
            self.prompts = []

        def call_model(self, model_name, prompt, images=None, timeout=None):
            self.prompts.append(prompt)
            if len(self.prompts) == 1:
                return ModelResponse(True, model_name, "", 0.01)
            return ModelResponse(True, model_name, "a mouse on fabric", 0.02)

    client = Client()
    caption, response = VLMModule(config, prompts, client).run("图中有什么？", str(image_path))

    assert caption == "a mouse on fabric"
    assert response is not None
    assert response.ok
    assert response.latency == 0.03
    assert response.raw is not None
    assert response.raw["retried"]
    assert client.prompts == ["图中有什么？", "Describe this image in detail."]
