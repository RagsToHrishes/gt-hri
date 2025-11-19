"""Pareto front plotting utility for trained MOPPO checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import utils as space_utils
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import torch
import tyro
from rich.console import Console

from moppo.agents.moppo import MOPPOAgent, MOPPOConfig
from moppo.envs.reward_config import RewardConfig, load_reward_config
from moppo.envs.factory import make_env as build_env


console = Console()


@dataclass
class ParetoArgs:
    """CLI arguments for Pareto plotting."""

    checkpoint: str
    env_id: str
    reward_config: Optional[str] = None
    output_dir: str = "outputs/pareto_plots"
    num_points: int = 11
    num_episodes: int = 3
    seed: int = 42
    device: str = "cpu"


def load_agent(checkpoint: dict, env: gym.Env, reward_config: RewardConfig, device: str) -> MOPPOAgent:
    """Instantiate an agent and load weights from a checkpoint."""
    config = MOPPOConfig(**checkpoint["config"])
    config.device = device
    agent = MOPPOAgent(env.observation_space, env.action_space, reward_config.reward_dim, config)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.eval()
    return agent


def rollout_returns(
    env: gym.Env,
    obs_space: gym.Space,
    agent: MOPPOAgent,
    weights: torch.Tensor,
    num_episodes: int,
    seed: int,
) -> np.ndarray:
    """Run deterministic evaluation for a single weight vector."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be >= 1 for evaluation.")

    obs, _ = env.reset(seed=seed)
    flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
    reward_dim = agent.reward_dim
    episode_return = np.zeros(reward_dim, dtype=np.float32)
    returns: list[np.ndarray] = []
    current_seed = seed

    while len(returns) < num_episodes:
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            action_tensor = agent.act_deterministic(obs_tensor, weights)
        env_action = agent.actions_to_env(action_tensor)[0]
        obs, reward, terminated, truncated, info = env.step(env_action)
        flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
        reward_vector = info.get("reward_vector")
        if reward_vector is None:
            reward_vector = np.array([reward], dtype=np.float32)
        episode_return += reward_vector
        if terminated or truncated:
            returns.append(episode_return.copy())
            episode_return.fill(0.0)
            current_seed += 1
            obs, _ = env.reset(seed=current_seed)
            flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)

    return np.mean(returns, axis=0)


def sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.lower())


def pareto_front_mask(points: np.ndarray) -> np.ndarray:
    """Boolean mask keeping only non-dominated points."""
    if points.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    mask = np.ones(points.shape[0], dtype=bool)
    for idx in range(points.shape[0]):
        if not mask[idx]:
            continue
        dominates = np.all(points >= points[idx], axis=1) & np.any(points > points[idx], axis=1)
        dominates[idx] = False
        mask[dominates] = False
    return mask


def plot_pair(
    name_i: str,
    name_j: str,
    returns: np.ndarray,
    weights: np.ndarray,
    equal_return: np.ndarray,
    output_path: Path,
) -> None:
    """Create and save a Pareto scatter plot for a specific objective pair."""
    x = returns[:, 0]
    y = returns[:, 1]
    if returns.shape[1] < 2:
        raise ValueError("Expected at least 2 return dimensions for plotting.")

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    color_values = weights[:, 0]
    scatter = ax.scatter(
        x,
        y,
        c=color_values,
        cmap="viridis",
        s=50,
        edgecolors="none",
        linewidths=0.0,
        alpha=0.35,
        label="Sampled returns",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(f"Weight on {name_i}", rotation=270, labelpad=14)

    mask = pareto_front_mask(returns)
    front = returns[mask]
    if front.size > 0:
        order = np.argsort(front[:, 0])
        front = front[order]
        ax.plot(
            front[:, 0],
            front[:, 1],
            color="crimson",
            linewidth=2.2,
            label="Pareto front",
        )
        ax.scatter(
            front[:, 0],
            front[:, 1],
            color="crimson",
            s=32,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.scatter(
        float(equal_return[0]),
        float(equal_return[1]),
        marker="x",
        color="red",
        s=120,
        linewidths=2.0,
        label="Equal weights",
    )
    ax.set_xlabel(f"Return: {name_i}")
    ax.set_ylabel(f"Return: {name_j}")
    ax.set_title(f"Pareto Front: {name_i} vs {name_j}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_pareto(args: ParetoArgs) -> None:
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    reward_config_path = args.reward_config or checkpoint.get("reward_config_path")
    reward_config = load_reward_config(reward_config_path, args.env_id)

    env = build_env(env_id=args.env_id, reward_config=reward_config, seed=args.seed)
    agent = load_agent(checkpoint, env, reward_config, args.device)
    obs_space = env.observation_space
    reward_names = reward_config.names
    reward_dim = reward_config.reward_dim
    if reward_dim < 2:
        raise ValueError("Pareto plotting requires at least two reward objectives.")
    if args.num_points < 1:
        raise ValueError("num_points must be >= 1 for Pareto sampling.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold green]Loaded checkpoint[/bold green] from {checkpoint_path}")
    console.print(f"Reward objectives: {', '.join(reward_names)}")

    rng = np.random.default_rng(args.seed)
    dirichlet_alpha = np.ones(reward_dim, dtype=np.float32)
    sampled_weights = rng.dirichlet(dirichlet_alpha, size=args.num_points).astype(np.float32)
    console.print(f"Sampled {args.num_points} weight vectors from a Dirichlet distribution.")

    def evaluate_weight(weight_vec: np.ndarray, eval_seed: int) -> np.ndarray:
        weight_tensor = torch.tensor(weight_vec, dtype=torch.float32, device=agent.device).unsqueeze(0)
        return rollout_returns(env, obs_space, agent, weight_tensor, args.num_episodes, eval_seed)

    all_returns = []
    for idx, weight_vec in enumerate(sampled_weights):
        avg_return = evaluate_weight(weight_vec, args.seed + idx)
        all_returns.append(avg_return)
    returns_matrix = np.stack(all_returns)

    equal_weights = np.full(reward_dim, 1.0 / reward_dim, dtype=np.float32)
    equal_return = evaluate_weight(equal_weights, args.seed + args.num_points)

    for i in range(reward_dim):
        for j in range(i + 1, reward_dim):
            console.print(f"[cyan]Evaluating pair:[/cyan] {reward_names[i]} vs {reward_names[j]}")
            returns_array = returns_matrix[:, [i, j]]
            weights_array = sampled_weights[:, [i, j]]
            plot_path = output_dir / f"pareto_{sanitize_name(reward_names[i])}_vs_{sanitize_name(reward_names[j])}.png"
            plot_pair(
                reward_names[i],
                reward_names[j],
                returns_array,
                weights_array,
                equal_return[[i, j]],
                plot_path,
            )
            console.print(f"[green]Saved plot to[/green] {plot_path}")

    env.close()
    console.print(f"[bold green]Finished.[/bold green] Pareto plots stored in {output_dir}")


if __name__ == "__main__":
    plot_pareto(tyro.cli(ParetoArgs))
