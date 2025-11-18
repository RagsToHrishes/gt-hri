"""Rollout buffer storing trajectories for PPO-style updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import torch


@dataclass
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    advantages: torch.Tensor
    scalar_advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    weights: torch.Tensor


class RolloutBuffer:
    """A fixed-size buffer for on-policy rollouts."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        reward_dim: int,
        device: torch.device,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.reward_dim = reward_dim
        self.device = device

        obs_shape_full = (num_steps, num_envs, *obs_shape)
        action_shape_full = (num_steps, num_envs, *action_shape)

        self.observations = torch.zeros(obs_shape_full, dtype=torch.float32, device=device)
        self.actions = torch.zeros(action_shape_full, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, dtype=torch.float32, device=device)
        self.values = torch.zeros(num_steps, num_envs, reward_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, reward_dim, dtype=torch.float32, device=device)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.float32, device=device)
        self.weights = torch.zeros(num_steps, num_envs, reward_dim, dtype=torch.float32, device=device)

        self.advantages = torch.zeros_like(self.rewards)
        self.scalar_advantages = torch.zeros(num_steps, num_envs, dtype=torch.float32, device=device)
        self.returns = torch.zeros_like(self.rewards)

        self.step = 0

    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward_vector: torch.Tensor,
        done: torch.Tensor,
        weight: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise IndexError("RolloutBuffer is full. Call reset() before adding more data.")

        self.observations[self.step].copy_(obs)
        self.actions[self.step].copy_(action)
        self.log_probs[self.step].copy_(log_prob)
        self.values[self.step].copy_(value)
        self.rewards[self.step].copy_(reward_vector)
        self.dones[self.step].copy_(done)
        self.weights[self.step].copy_(weight)
        self.step += 1

    def reset(self) -> None:
        self.step = 0

    def compute_returns_and_advantages(
        self,
        next_values: torch.Tensor,
        next_dones: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        adv = torch.zeros(self.num_envs, self.reward_dim, dtype=torch.float32, device=self.device)
        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_value = next_values
                next_non_terminal = 1.0 - next_dones
            else:
                next_value = self.values[step + 1]
                next_non_terminal = 1.0 - self.dones[step + 1]
            delta = (
                self.rewards[step]
                + gamma * next_value * next_non_terminal.unsqueeze(-1)
                - self.values[step]
            )
            adv = delta + gamma * gae_lambda * next_non_terminal.unsqueeze(-1) * adv
            self.advantages[step] = adv
        self.returns = self.advantages + self.values
        self.scalar_advantages = (self.advantages * self.weights).sum(dim=-1)

    def _flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        tail_shape = tensor.shape[2:]
        if not tail_shape:
            return tensor.reshape(-1)
        return tensor.reshape(-1, *tail_shape)

    def get(self, batch_size: int) -> Iterator[Batch]:
        num_samples = self.num_steps * self.num_envs
        indices = torch.randperm(num_samples, device=self.device)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            batch = Batch(
                observations=self._flatten(self.observations).index_select(0, batch_idx),
                actions=self._flatten(self.actions).index_select(0, batch_idx),
                log_probs=self._flatten(self.log_probs).index_select(0, batch_idx),
                advantages=self._flatten(self.advantages).index_select(0, batch_idx),
                scalar_advantages=self._flatten(self.scalar_advantages).index_select(0, batch_idx),
                returns=self._flatten(self.returns).index_select(0, batch_idx),
                values=self._flatten(self.values).index_select(0, batch_idx),
                weights=self._flatten(self.weights).index_select(0, batch_idx),
            )
            yield batch


__all__ = ["RolloutBuffer", "Batch"]
