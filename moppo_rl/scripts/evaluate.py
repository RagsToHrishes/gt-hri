"""Evaluation helper for trained MOPPO checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import gymnasium as gym
from gymnasium.spaces import utils as space_utils
import numpy as np
import torch
import tyro
from rich.console import Console

from moppo.agents.moppo import MOPPOAgent, MOPPOConfig
from moppo.envs.reward_config import load_reward_config, RewardConfig
from moppo.envs.reward_wrappers import MultiObjectiveRewardWrapper
from moppo.utils.device import resolve_device


console = Console()


@dataclass
class EvalArgs:
    checkpoint: str
    env_id: str
    reward_config: Optional[str] = None
    weights: Sequence[float] = (1.0,)
    num_episodes: int = 5
    device: str = "auto"
    seed: int = 42


def load_agent(checkpoint: dict, env: gym.Env, reward_config: RewardConfig, device: str) -> MOPPOAgent:
    config = MOPPOConfig(**checkpoint["config"])
    config.device = device
    agent = MOPPOAgent(env.observation_space, env.action_space, reward_config.reward_dim, config)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.eval()
    return agent


def evaluate(args: EvalArgs) -> None:
    resolved_device = resolve_device(args.device, console=console)
    args.device = resolved_device
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    reward_config_path = args.reward_config or checkpoint.get("reward_config_path")
    reward_config = load_reward_config(reward_config_path, args.env_id)

    base_env = gym.make(args.env_id, render_mode="rgb_array")
    env = MultiObjectiveRewardWrapper(base_env, reward_config)
    obs_space = env.observation_space
    obs, _ = env.reset(seed=args.seed)
    flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)

    agent = load_agent(checkpoint, env, reward_config, args.device)
    if len(args.weights) != reward_config.reward_dim:
        raise ValueError(
            f"Expected {reward_config.reward_dim} weights for this config, got {len(args.weights)}."
        )
    weights = torch.tensor(args.weights, dtype=torch.float32, device=agent.device).unsqueeze(0)
    weight_desc = ", ".join(
        f"{name}={float(value):.2f}"
        for name, value in zip(reward_config.names, weights.squeeze(0).cpu().numpy())
    )
    console.print(f"[magenta]Evaluation preference weights:[/magenta] {weight_desc}")
    episode = 0
    episode_return = np.zeros(reward_config.reward_dim, dtype=np.float32)
    returns = []

    while episode < args.num_episodes:
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=agent.device).unsqueeze(0).view(1, -1)
        action_tensor = agent.act_deterministic(obs_tensor, weights)
        env_action = agent.actions_to_env(action_tensor)[0]
        obs, reward, terminated, truncated, info = env.step(env_action)
        flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
        done = terminated or truncated
        reward_vector = info.get("reward_vector", np.array([reward], dtype=np.float32))
        episode_return += reward_vector
        if done:
            returns.append(episode_return.copy())
            episode_return[:] = 0.0
            episode += 1
            obs, _ = env.reset()
            flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)

    avg_return = np.mean(returns, axis=0)
    console.print(f"[bold green]Evaluated {args.num_episodes} episodes.[/bold green]")
    console.print(f"Average return per objective: {avg_return}")


if __name__ == "__main__":
    evaluate(tyro.cli(EvalArgs))
