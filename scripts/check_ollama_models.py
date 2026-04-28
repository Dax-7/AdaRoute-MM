from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adaroute.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Check required local Ollama models.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    base_url = config.get("ollama", {}).get("base_url", "http://localhost:11434").rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
        local = {model.get("name") for model in response.json().get("models", [])}
    except Exception as exc:
        print(f"Failed to query Ollama: {exc}")
        raise SystemExit(1)

    required = {cfg["model_name"] for cfg in config.get("models", {}).values()}
    missing = sorted(required - local)
    if missing:
        print("Missing models:")
        for model in missing:
            print(f"  ollama pull {model}")
        raise SystemExit(2)
    print("All configured Ollama models are available.")
