from __future__ import annotations

import random

from adaroute.core.types import RouteDecision
from adaroute.policies.base import BaseRoutingPolicy

_RNGS: dict[int | None, random.Random] = {}


class RandomPolicy(BaseRoutingPolicy):
    name = "random"

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        policy_cfg = config["routing"]["policies"][self.name]
        seed = policy_cfg.get("seed")
        if seed not in _RNGS:
            _RNGS[seed] = random.Random(seed)
        rng = _RNGS[seed]
        selected = rng.choice(policy_cfg.get("candidates", []))
        return RouteDecision(difficulty=difficulty, policy=self.name, selected_model=selected, reason="random")
