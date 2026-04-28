from __future__ import annotations

from adaroute.core.types import RouteDecision
from adaroute.policies.difficulty_policy import DIFFICULTY_TO_CONFIG_KEY
from adaroute.policies.base import BaseRoutingPolicy


class LatencyAwarePolicy(BaseRoutingPolicy):
    name = "latency_aware"

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        policy_cfg = config["routing"]["policies"][self.name]
        selected = policy_cfg.get(DIFFICULTY_TO_CONFIG_KEY.get(difficulty, "medium"))
        overloaded = bool(system_state.get("is_overloaded"))
        if overloaded and policy_cfg.get("skip_large_when_overloaded", True):
            large_model = policy_cfg.get("hard")
            if selected == large_model:
                selected = policy_cfg.get("overload_model", selected)
            elif system_state.get("ram_percent") is not None:
                selected = policy_cfg.get("overload_model", selected)
        return RouteDecision(
            difficulty=difficulty,
            policy=self.name,
            selected_model=selected,
            reason="overload" if overloaded else "difficulty_and_latency",
            overloaded=overloaded,
        )
