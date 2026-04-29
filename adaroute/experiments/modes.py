from __future__ import annotations

from copy import deepcopy
from typing import Any

from adaroute.utils.io import deep_merge


EXPERIMENT_SUITE = [
    "always_small",
    "always_gemma",
    "random_routing",
    "difficulty_routing",
    "difficulty_cache",
    "difficulty_fallback",
    "latency_aware_routing",
    "adaroute_mm_full",
]


_BASE_V2_OVERRIDES: dict[str, Any] = {
    "vlm": {
        "enabled": True,
        "caption_mode": "image_caption",
    },
    "router": {
        "enabled": True,
        "default_difficulty": "medium",
    },
    "fallback": {
        "enabled": False,
    },
    "cache": {
        "enabled": False,
        "cache_vlm": False,
        "cache_router": False,
        "cache_llm": False,
    },
    "runtime": {
        "experiment_version": "v2",
    },
}


_MODE_OVERRIDES: dict[str, dict[str, Any]] = {
    "always_small": {
        "routing": {"default_policy": "always_small"},
    },
    "always_gemma": {
        "routing": {
            "default_policy": "always_gemma",
            "policies": {"always_gemma": {"default": "gemma_large"}},
        },
    },
    "random_routing": {
        "routing": {"default_policy": "random"},
    },
    "difficulty_routing": {
        "routing": {"default_policy": "difficulty_based"},
    },
    "difficulty_cache": {
        "routing": {"default_policy": "difficulty_based"},
        "cache": {"enabled": True, "cache_vlm": True},
    },
    "difficulty_fallback": {
        "routing": {"default_policy": "difficulty_based"},
        "fallback": {"enabled": True},
    },
    "latency_aware_routing": {
        "routing": {"default_policy": "latency_aware"},
    },
    "adaroute_mm_full": {
        "routing": {"default_policy": "latency_aware"},
        "fallback": {"enabled": True},
        "cache": {"enabled": True, "cache_vlm": True},
    },
}


def available_modes() -> list[str]:
    return list(_MODE_OVERRIDES)


def resolve_mode_config(base_config: dict[str, Any], mode: str, output_dir: str | None = None) -> dict[str, Any]:
    if mode not in _MODE_OVERRIDES:
        raise ValueError(f"Unknown experiment mode: {mode}. Available modes: {', '.join(available_modes())}")

    config = deep_merge(deepcopy(base_config), deepcopy(_BASE_V2_OVERRIDES))
    config = deep_merge(config, deepcopy(_MODE_OVERRIDES[mode]))
    if output_dir:
        config = deep_merge(
            config,
            {
                "paths": {
                    "output_dir": f"{output_dir}/requests",
                    "log_dir": f"{output_dir}/logs",
                },
                "cache": {
                    "cache_dir": f"{output_dir}/cache",
                },
            },
        )
    config.setdefault("runtime", {})["experiment_mode"] = mode
    return config


def suite_modes(name: str) -> list[str]:
    if name != "vqav2_yesno_ablation":
        raise ValueError("Only suite 'vqav2_yesno_ablation' is currently defined")
    return list(EXPERIMENT_SUITE)
