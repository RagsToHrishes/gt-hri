"""Reward configuration utilities for defining multi-objective signals."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

RewardFn = Callable[
    [np.ndarray, np.ndarray, float, bool, bool, dict],
    float,
]


@dataclass
class RewardComponent:
    """Named reward component computed from a step transition."""

    name: str
    fn: RewardFn
    scale: float = 1.0
    description: str | None = None

    def evaluate(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> float:
        value = float(self.fn(obs, action, reward, terminated, truncated, info))
        return value * self.scale


@dataclass
class RewardConfig:
    """Defines how to map scalar rewards into a vector of components."""

    env_id: str
    components: Sequence[RewardComponent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.components:
            raise ValueError("RewardConfig must contain at least one component.")

    @property
    def names(self) -> List[str]:
        return [component.name for component in self.components]

    @property
    def reward_dim(self) -> int:
        return len(self.components)

    def evaluate(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> np.ndarray:
        values = [
            component.evaluate(obs, action, reward, terminated, truncated, info)
            for component in self.components
        ]
        return np.asarray(values, dtype=np.float32)

    @classmethod
    def scalar_default(cls, env_id: str) -> "RewardConfig":
        """Fallback configuration that only exposes the original scalar reward."""

        def identity_reward(
            _obs: np.ndarray,
            _action: np.ndarray,
            reward: float,
            _terminated: bool,
            _truncated: bool,
            _info: dict,
        ) -> float:
            return reward

        component = RewardComponent(name="environment", fn=identity_reward)
        return cls(env_id=env_id, components=[component])


def load_reward_config(path: Optional[str], env_id: str) -> RewardConfig:
    """Loads a reward config defined in a python module.

    The module can expose either a ``build_config`` callable returning a ``RewardConfig``
    or a top-level ``CONFIG`` variable.
    """

    if path is None:
        return RewardConfig.scalar_default(env_id)

    module_path = Path(path).expanduser().resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Reward config file not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import reward config module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]

    builder = None
    if hasattr(module, "build_config"):
        builder = getattr(module, "build_config")
    elif hasattr(module, "CONFIG"):
        builder = lambda *_args, **_kwargs: getattr(module, "CONFIG")

    if builder is None:
        raise AttributeError(
            f"Reward config module {module_path} must define build_config(...) or CONFIG."
        )

    config = builder(env_id)
    if not isinstance(config, RewardConfig):
        raise TypeError(
            f"Reward config builder in {module_path} must return RewardConfig, got {type(config)}"
        )
    return config


__all__ = ["RewardConfig", "RewardComponent", "load_reward_config"]
