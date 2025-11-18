"""Directional reward config for MiniGrid-Empty-5x5-v0."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from moppo.envs.reward_config import RewardComponent, RewardConfig


# Keep the directional shaping relatively small so the base MiniGrid reward
# remains the dominant learning signal.
CARDINAL_PREFERENCE_WEIGHT = 0.1


def _extract_agent_position(obs, info) -> Tuple[float, float]:
    def _coerce(value) -> Tuple[float, float] | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).flatten()
        if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
            return float(arr[0]), float(arr[1])
        return None

    pos = _coerce(info.get("agent_pos"))
    if pos is not None:
        return pos

    if isinstance(obs, dict):
        for key in ("agent_pos", "pos"):
            pos = _coerce(obs.get(key))
            if pos is not None:
                return pos

    return 0.0, 0.0


def _directional_reward(obs, info, axis: str, positive: bool) -> float:
    x, y = _extract_agent_position(obs, info)
    grid_size = float(info.get("grid_size", 5))
    denom = max(grid_size - 1.0, 1.0)
    coord = y if axis == "y" else x
    coord = np.clip(coord / denom, 0.0, 1.0)
    value = coord if positive else 1.0 - coord
    return float(value)


def _directional_preference(obs, info, axis: str, positive: bool) -> float:
    # Convert proximity (0..1) into a centered preference (-1..1) so the shaping
    # reward can nudge trajectories without overwhelming the task.
    return 2.0 * _directional_reward(obs, info, axis, positive) - 1.0


def _make_cardinal_component(name: str, axis: str, positive: bool, description: str) -> RewardComponent:
    def fn(_obs, _action, reward, _terminated, _truncated, info) -> float:
        bias = CARDINAL_PREFERENCE_WEIGHT * _directional_preference(_obs, info, axis, positive)
        return float(reward + bias)

    return RewardComponent(name=name, fn=fn, description=description)


def build_config(env_id: str) -> RewardConfig:
    components = [
        # RewardComponent(
        #     name="task",
        #     fn=lambda _obs, _act, reward, _term, _trunc, _info: float(reward),
        #     description="Baseline MiniGrid reward that only depends on completing the task.",
        # ),
        _make_cardinal_component(
            name="north",
            axis="y",
            positive=True,
            description="Shaping bonus that prefers positions closer to the north wall.",
        ),
        _make_cardinal_component(
            name="south",
            axis="y",
            positive=False,
            description="Shaping bonus that prefers positions closer to the south wall.",
        ),
        _make_cardinal_component(
            name="east",
            axis="x",
            positive=True,
            description="Shaping bonus that prefers positions closer to the east wall.",
        ),
        _make_cardinal_component(
            name="west",
            axis="x",
            positive=False,
            description="Shaping bonus that prefers positions closer to the west wall.",
        ),
    ]
    return RewardConfig(env_id=env_id, components=components)


__all__ = ["build_config"]
