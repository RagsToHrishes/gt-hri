"""Sample config that exposes the default scalar MiniGrid reward."""

from __future__ import annotations

from moppo.envs.reward_config import RewardConfig


def build_config(env_id: str) -> RewardConfig:
    return RewardConfig.scalar_default(env_id)


__all__ = ["build_config"]
