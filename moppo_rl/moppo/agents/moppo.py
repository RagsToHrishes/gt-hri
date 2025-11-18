"""Implementation of the Multi-Objective PPO algorithm."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from gymnasium.spaces import utils as space_utils

from ..models.networks import ConditionedCategoricalActor, ConditionedGaussianActor, MultiObjectiveValue
from ..storage.rollout_buffer import Batch, RolloutBuffer
from ..utils.weights import WeightSampler


@dataclass
class MOPPOConfig:
    learning_rate: float = 3e-4
    num_steps: int = 2048
    num_envs: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 512
    entropy_coef: float = 0.0
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    hidden_sizes: Tuple[int, int] = (256, 256)
    device: str = "cpu"
    weight_strategy: str = "dirichlet"
    dirichlet_alpha: float = 1.0


class MOPPOAgent(nn.Module):
    """Weight-conditioned PPO agent for multi-objective control."""

    def __init__(
        self,
        obs_space: gym.Space,
        action_space: gym.Space,
        reward_dim: int,
        config: MOPPOConfig,
    ) -> None:
        super().__init__()
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if not self.is_discrete and not isinstance(action_space, gym.spaces.Box):
            raise TypeError("MOPPOAgent currently supports Box or Discrete action spaces.")

        self.config = config
        self.device = torch.device(config.device)
        obs_dim = space_utils.flatdim(obs_space)
        self.obs_dim = obs_dim
        self.reward_dim = reward_dim
        hidden = config.hidden_sizes
        if self.is_discrete:
            self.action_dim = action_space.n
            self.actor = ConditionedCategoricalActor(obs_dim, self.action_dim, reward_dim, hidden).to(self.device)
            self.action_low = None
            self.action_high = None
        else:
            self.action_dim = action_space.shape[0]
            self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
            self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
            self.actor = ConditionedGaussianActor(obs_dim, self.action_dim, reward_dim, hidden).to(self.device)
        self.critic = MultiObjectiveValue(obs_dim, reward_dim, hidden, reward_dim, condition_on_weights=True).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)
        self.weight_sampler = WeightSampler(reward_dim=reward_dim, strategy=config.weight_strategy,
                                            dirichlet_alpha=config.dirichlet_alpha)

    def _dist(self, obs: torch.Tensor, weights: torch.Tensor):
        if self.is_discrete:
            logits = self.actor(obs, weights)
            return Categorical(logits=logits)
        mean, log_std = self.actor(obs, weights)
        std = torch.exp(log_std)
        return Normal(mean, std)

    def forward(self, obs: torch.Tensor, weights: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        dist = self._dist(obs, weights)
        values = self.critic(obs, weights)
        return dist, values

    def act(self, obs: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs, weights)
        if self.is_discrete:
            action = dist.sample()
            log_prob = dist.log_prob(action)
        else:
            action = dist.rsample()
            log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, values

    def evaluate_actions(
        self, obs: torch.Tensor, weights: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, values = self.forward(obs, weights)
        if self.is_discrete:
            log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy()
        else:
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        config = self.config

        updates = 0
        for epoch in range(config.update_epochs):
            for batch in buffer.get(config.minibatch_size):
                step_metrics = self._update_step(batch)
                for key, value in step_metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + value
                updates += 1

        if updates:
            for key in list(metrics.keys()):
                metrics[key] /= updates
        return metrics

    def _update_step(self, batch: Batch) -> Dict[str, float]:
        config = self.config
        adv = (batch.scalar_advantages - batch.scalar_advantages.mean()) / (
            batch.scalar_advantages.std(unbiased=False) + 1e-8
        )

        new_log_probs, entropy, values = self.evaluate_actions(batch.observations, batch.weights, batch.actions)
        ratio = (new_log_probs - batch.log_probs).exp()
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * adv
        policy_loss = -torch.min(unclipped, clipped).mean()

        value_loss = 0.5 * ((batch.returns - values) ** 2).sum(dim=-1).mean()
        entropy_loss = entropy.mean()

        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), config.max_grad_norm)
        self.optimizer.step()

        approx_kl = (batch.log_probs - new_log_probs).mean().abs().item()
        clip_frac = (torch.abs(ratio - 1.0) > config.clip_coef).float().mean().item()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
        }

    def act_deterministic(self, obs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.is_discrete:
                logits = self.actor(obs, weights)
                return torch.argmax(logits, dim=-1)
            mean, _ = self.actor(obs, weights)
            return self.clip_action(mean)

    def clip_action(self, action: torch.Tensor) -> torch.Tensor:
        """Clips continuous actions to the valid bounds."""
        if self.is_discrete:
            return action
        high = self.action_high.expand_as(action)
        low = self.action_low.expand_as(action)
        return torch.max(torch.min(action, high), low)

    def actions_to_env(self, actions: torch.Tensor) -> np.ndarray:
        if self.is_discrete:
            arr = actions.detach().cpu().numpy()
            return arr.astype(np.int64)
        clipped = self.clip_action(actions)
        return clipped.detach().cpu().numpy()

    def sample_weights(self, batch_size: int) -> torch.Tensor:
        weights = self.weight_sampler.sample(batch_size)
        return weights.to(self.device)

    def to_checkpoint(self) -> Dict[str, torch.Tensor]:
        return {
            "state_dict": self.state_dict(),
            "config": asdict(self.config),
            "reward_dim": self.reward_dim,
        }

    def load_checkpoint(self, checkpoint: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(checkpoint["state_dict"])


__all__ = ["MOPPOAgent", "MOPPOConfig"]
