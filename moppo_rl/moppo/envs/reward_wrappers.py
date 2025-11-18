"""Environment wrappers for multi-objective reward handling."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

from .reward_config import RewardConfig


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """Injects reward vectors into ``info`` while keeping scalar rewards compatible."""

    def __init__(self, env: gym.Env, reward_config: RewardConfig):
        super().__init__(env)
        self.reward_config = reward_config

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_vector = self.reward_config.evaluate(obs, action, reward, terminated, truncated, info)
        info = dict(info or {})
        info.setdefault("reward_vector", reward_vector)
        info.setdefault("reward_names", self.reward_config.names)
        return obs, float(np.sum(reward_vector)), terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: Dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info or {})
        # Provide zero reward vector on reset for convenience.
        info.setdefault("reward_vector", np.zeros(self.reward_config.reward_dim, dtype=np.float32))
        info.setdefault("reward_names", self.reward_config.names)
        return obs, info


__all__ = ["MultiObjectiveRewardWrapper"]
