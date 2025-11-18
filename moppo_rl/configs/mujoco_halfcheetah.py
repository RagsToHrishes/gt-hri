"""Example reward vector for HalfCheetah-v4."""

from __future__ import annotations

from moppo.envs.reward_config import RewardComponent, RewardConfig


def build_config(env_id: str) -> RewardConfig:
    if env_id != "HalfCheetah-v4":
        raise ValueError("This config is tailored for HalfCheetah-v4.")

    def forward_reward(_obs, _action, _reward, _term, _trunc, info):
        return float(info.get("reward_run", _reward))

    def control_cost(_obs, _action, _reward, _term, _trunc, info):
        return float(info.get("reward_ctrl", 0.0))

    def survive_bonus(_obs, _action, _reward, _term, _trunc, info):
        return float(info.get("reward_survive", 0.0))

    components = [
        RewardComponent(name="forward", fn=forward_reward),
        RewardComponent(name="torque_cost", fn=control_cost),
        RewardComponent(name="healthy", fn=survive_bonus),
    ]
    return RewardConfig(env_id=env_id, components=components)
