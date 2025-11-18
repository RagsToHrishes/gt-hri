"""Neural network modules used by the MOPPO agent."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    layers = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(activation())
        last_dim = size
    return nn.Sequential(*layers)


class ConditionedGaussianActor(nn.Module):
    """Policy network that conditions on preference weights."""

    def __init__(self, obs_dim: int, action_dim: int, weight_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + weight_dim, hidden_sizes)
        last_dim = hidden_sizes[-1] if hidden_sizes else obs_dim + weight_dim
        self.mean_head = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, weights], dim=-1)
        features = self.net(x)
        mean = self.mean_head(features)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std


class ConditionedCategoricalActor(nn.Module):
    """Policy network producing logits for discrete actions."""

    def __init__(self, obs_dim: int, action_dim: int, weight_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + weight_dim, hidden_sizes)
        last_dim = hidden_sizes[-1] if hidden_sizes else obs_dim + weight_dim
        self.logits_head = nn.Linear(last_dim, action_dim)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, weights], dim=-1)
        features = self.net(x)
        return self.logits_head(features)


class MultiObjectiveValue(nn.Module):
    """Critic producing a value per objective, optionally conditioned on weights."""

    def __init__(
        self,
        obs_dim: int,
        reward_dim: int,
        hidden_sizes: Sequence[int],
        weight_dim: int,
        condition_on_weights: bool = True,
    ) -> None:
        super().__init__()
        input_dim = obs_dim + (weight_dim if condition_on_weights else 0)
        self.combine_weights = condition_on_weights
        self.net = build_mlp(input_dim, hidden_sizes)
        last_dim = hidden_sizes[-1] if hidden_sizes else input_dim
        self.value_head = nn.Linear(last_dim, reward_dim)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if self.combine_weights:
            obs = torch.cat([obs, weights], dim=-1)
        features = self.net(obs)
        return self.value_head(features)


__all__ = ["ConditionedGaussianActor", "ConditionedCategoricalActor", "MultiObjectiveValue"]
