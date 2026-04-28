from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any

import psutil


def _apply_overload(state: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    checks = [
        ("cpu_percent", "cpu_percent"),
        ("ram_percent", "ram_percent"),
        ("gpu_percent", "gpu_percent"),
        ("temperature", "temp_celsius"),
    ]
    overloaded = False
    for state_key, policy_key in checks:
        value = state.get(state_key)
        threshold = policy.get(policy_key)
        if value is not None and threshold is not None and value >= threshold:
            overloaded = True
    state["is_overloaded"] = overloaded
    return state


def get_psutil_state(policy: dict[str, Any]) -> dict[str, Any]:
    state = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_percent": psutil.virtual_memory().percent,
        "gpu_percent": None,
        "temperature": None,
        "backend": "psutil",
    }
    return _apply_overload(state, policy)


def parse_tegrastats_line(line: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "cpu_percent": None,
        "ram_percent": None,
        "gpu_percent": None,
        "temperature": None,
        "backend": "tegrastats",
    }
    ram_match = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if ram_match:
        used, total = int(ram_match.group(1)), int(ram_match.group(2))
        if total:
            state["ram_percent"] = round(used / total * 100, 2)

    cpu_values = [int(v) for v in re.findall(r"(\d+)%@\d+", line)]
    if cpu_values:
        state["cpu_percent"] = round(sum(cpu_values) / len(cpu_values), 2)

    gpu_match = re.search(r"GR3D_FREQ\s+(\d+)%", line)
    if gpu_match:
        state["gpu_percent"] = float(gpu_match.group(1))

    temps = [float(v) for v in re.findall(r"@([0-9.]+)C", line)]
    if temps:
        state["temperature"] = max(temps)
    return state


def get_tegrastats_state(policy: dict[str, Any]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["tegrastats", "--interval", "1000", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
        state = parse_tegrastats_line(line)
        return _apply_overload(state, policy)
    except Exception:
        return get_psutil_state(policy)


def get_system_state(config: dict[str, Any]) -> dict[str, Any]:
    system_cfg = config.get("system", {})
    backend = system_cfg.get("monitor_backend", "auto")
    policy = system_cfg.get("overload_policy", {})
    if backend == "tegrastats":
        return get_tegrastats_state(policy)
    if backend == "auto" and shutil.which("tegrastats"):
        return get_tegrastats_state(policy)
    return get_psutil_state(policy)
