from __future__ import annotations

from adaroute.core.types import RouteDecision
from adaroute.policies.base import BaseRoutingPolicy


DIFFICULTY_TO_CONFIG_KEY = {
    "easy": "simple",
    "medium": "medium",
    "hard": "hard",
    "简单": "simple",
    "中等": "medium",
    "困难": "hard",
    "绠€鍗?": "simple",
    "涓瓑": "medium",
    "鍥伴毦": "hard",
    "莽禄聽芒鈥毬┞嶁€?": "simple",
    "忙露鈥溍€β犆р€溾€?": "medium",
    "茅聧楼盲录麓忙炉娄": "hard",
}


class DifficultyBasedPolicy(BaseRoutingPolicy):
    name = "difficulty_based"

    def select_model(self, difficulty: str, system_state: dict, config: dict) -> RouteDecision:
        policy_cfg = config["routing"]["policies"][self.name]
        selected = policy_cfg.get(DIFFICULTY_TO_CONFIG_KEY.get(difficulty, "medium"))
        return RouteDecision(difficulty=difficulty, policy=self.name, selected_model=selected, reason="difficulty_mapping")
