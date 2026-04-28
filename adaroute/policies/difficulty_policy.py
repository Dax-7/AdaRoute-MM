from __future__ import annotations

from adaroute.core.types import RouteDecision
from adaroute.policies.base import BaseRoutingPolicy


DIFFICULTY_TO_CONFIG_KEY = {"简单": "simple", "中等": "medium", "困难": "hard"}


class DifficultyBasedPolicy(BaseRoutingPolicy):
    name = "difficulty_based"

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        policy_cfg = config["routing"]["policies"][self.name]
        selected = policy_cfg.get(DIFFICULTY_TO_CONFIG_KEY.get(difficulty, "medium"))
        return RouteDecision(difficulty=difficulty, policy=self.name, selected_model=selected, reason="difficulty_mapping")
