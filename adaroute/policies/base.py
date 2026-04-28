from __future__ import annotations

from adaroute.core.types import RouteDecision


class BaseRoutingPolicy:
    name = "base"

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        raise NotImplementedError
