from __future__ import annotations

from adaroute.policies.always_policy import AlwaysPolicy
from adaroute.policies.base import BaseRoutingPolicy
from adaroute.policies.difficulty_policy import DifficultyBasedPolicy
from adaroute.policies.latency_aware_policy import LatencyAwarePolicy
from adaroute.policies.random_policy import RandomPolicy


def get_policy(name: str) -> BaseRoutingPolicy:
    if name in {"always_small", "always_medium", "always_large"}:
        return AlwaysPolicy(name)
    policies = {
        "difficulty_based": DifficultyBasedPolicy,
        "random": RandomPolicy,
        "latency_aware": LatencyAwarePolicy,
    }
    if name not in policies:
        raise ValueError(f"Unknown routing policy: {name}")
    return policies[name]()
