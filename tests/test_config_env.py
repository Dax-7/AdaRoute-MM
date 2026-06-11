from adaroute.utils.io import load_config, ollama_base_url


def test_ollama_base_url_prefers_environment(monkeypatch):
    config = {"ollama": {"base_url": "http://from-config:11434"}}
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://from-env:11434")

    assert ollama_base_url(config) == "http://from-env:11434"


def test_ollama_base_url_falls_back_to_config_then_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert ollama_base_url({"ollama": {"base_url": "http://from-config:11434"}}) == "http://from-config:11434"
    assert ollama_base_url({}) == "http://localhost:11434"


def test_load_config_applies_model_environment_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
ollama:
  base_url: http://localhost:11434
models:
  qwen_small:
    model_name: qwen-old
  phi3_medium:
    model_name: phi3-old
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("ADAROUTE_SMALL_MODEL", "qwen2.5:1.5b")
    monkeypatch.setenv("ADAROUTE_MEDIUM_MODEL", "phi3:latest")

    config = load_config(config_path)

    assert config["models"]["qwen_small"]["model_name"] == "qwen2.5:1.5b"
    assert config["models"]["phi3_medium"]["model_name"] == "phi3:latest"
