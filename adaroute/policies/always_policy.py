from __future__ import annotations

from adaroute.core.types import RouteDecision
from adaroute.policies.base import BaseRoutingPolicy


class AlwaysPolicy(BaseRoutingPolicy):
    def __init__(self, name: str):
        self.name = name

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        selected = config["routing"]["policies"][self.name]["default"]
        return RouteDecision(difficulty=difficulty, policy=self.name, selected_model=selected, reason="always")
