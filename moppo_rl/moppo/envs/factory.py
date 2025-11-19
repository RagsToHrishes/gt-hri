"""Factory helpers for creating single or vectorized environments."""

from __future__ import annotations

import os
from typing import Callable, List, Optional

import gymnasium as gym
import minigrid
from gymnasium.wrappers import FlattenObservation

from .reward_config import RewardConfig
from .reward_wrappers import MultiObjectiveRewardWrapper


def _maybe_flatten_minigrid(env: gym.Env, env_id: str) -> gym.Env:
    """Use MiniGrid's native flatten wrapper when available."""
    if "minigrid" not in env_id.lower():
        return env
    try:
        from minigrid.wrappers import FlatObsWrapper  # type: ignore

        return FlatObsWrapper(env)
    except Exception:
        return env


def _ensure_flat_box_observations(env: gym.Env) -> gym.Env:
    """Flatten non-Box observation spaces so PPO always receives vectors."""
    if not isinstance(env.observation_space, gym.spaces.Box):
        return FlattenObservation(env)
    return env


def _disable_unhealthy_termination(env: gym.Env) -> None:
    """Force Mujoco-style envs to ignore unhealthy termination flags when available."""
    base_env = getattr(env, "unwrapped", env)
    for attr in ("terminate_when_unhealthy", "terminate_if_unhealthy"):
        if hasattr(base_env, attr):
            setattr(base_env, attr, False)


def make_env(
    env_id: str,
    reward_config: RewardConfig,
    seed: int,
    *,
    capture_video: bool = False,
    video_folder: str | os.PathLike[str] | None = None,
    video_name_prefix: Optional[str] = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a fully wrapped single environment instance."""

    env = gym.make(env_id, render_mode=render_mode)
    _disable_unhealthy_termination(env)
    env = _maybe_flatten_minigrid(env, env_id)
    if capture_video and video_folder is not None:
        env = gym.wrappers.RecordVideo(
            env,
            os.fspath(video_folder),
            name_prefix=video_name_prefix or "rl-video",
        )
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    env = MultiObjectiveRewardWrapper(env, reward_config)
    env = _ensure_flat_box_observations(env)
    return env

def _make_single_env(
    env_id: str,
    seed: int,
    reward_config: RewardConfig,
    capture_video: bool = False,
    video_folder: str | None = None,
) -> Callable[[], gym.Env]:
    def thunk() -> gym.Env:
        render_mode = "rgb_array" if capture_video else None
        return make_env(
            env_id=env_id,
            reward_config=reward_config,
            seed=seed,
            capture_video=capture_video,
            video_folder=video_folder,
            render_mode=render_mode,
        )

    return thunk


def make_vectorized_env(
    env_id: str,
    reward_config: RewardConfig,
    num_envs: int,
    seed: int,
    asynchronous: bool = True,
    capture_video: bool = False,
    video_folder: str | None = None,
    shared_memory: bool = True,
) -> gym.vector.VectorEnv:
    """Constructs a Sync/Async vectorized environment for training."""

    env_fns = [
        _make_single_env(
            env_id=env_id,
            seed=seed + idx,
            reward_config=reward_config,
            capture_video=capture_video if idx == 0 else False,
            video_folder=video_folder,
        )
        for idx in range(num_envs)
    ]
    if asynchronous:
        # MiniGrid tasks include custom MissionSpace observations which do not support Gym shared memory.
        if "minigrid" in env_id.lower():
            shared_memory = False
        return gym.vector.AsyncVectorEnv(env_fns, shared_memory=shared_memory)
    return gym.vector.SyncVectorEnv(env_fns)


__all__ = ["make_env", "make_vectorized_env"]
