"""Training entrypoint for the MOPPO agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence
import time

import gymnasium as gym
from gymnasium.spaces import utils as space_utils

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

import hydra
from hydra.errors import HydraException
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from moppo.envs.factory import make_env, make_vectorized_env
from moppo.envs.reward_config import RewardConfig, load_reward_config
from moppo.agents.moppo import MOPPOAgent, MOPPOConfig
from moppo.storage.rollout_buffer import RolloutBuffer
from moppo.utils.weights import WeightSampler


console = Console()


METRIC_DESCRIPTIONS = {
    "policy_loss": "PPO objective for the actor; lower means updates are improving the policy.",
    "value_loss": "Mean-squared error between critic predictions and vector returns.",
    "entropy": "Average action-distribution entropy that encourages exploration.",
    "approx_kl": "Approximate KL divergence between the old and new policy distributions.",
    "clip_frac": "Fraction of samples where the PPO ratio hit the clipping threshold.",
    "avg_return": "Mean episodic reward vector over the last 10 completed training episodes.",
    "eval_avg_return": "Average reward vector from clean deterministic evaluation rollouts.",
}


@dataclass
class TrainConfig:
    env_id: str = "HalfCheetah-v4"
    reward_config: Optional[str] = "configs/mujoco_halfcheetah.py"
    total_steps: int = 50_000_000
    num_envs: int = 64
    num_steps: int = 1024
    asynchronous_envs: bool = True
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    update_epochs: int = 10
    minibatch_size: int = 4192
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    max_grad_norm: float = 0.5
    dirichlet_alpha: float = 1.0
    weight_strategy: str = "dirichlet"
    seed: int = 7
    device: str = "cpu"
    checkpoint_path: str = "checkpoints/latest-moppo.pt"
    eval_interval: int = 10
    eval_episodes: int = 2
    eval_weights: Optional[Sequence[float]] = None
    eval_render_dir: Optional[str] = None
    eval_render_mode: Literal["none", "video", "window"] = "none"
    eval_render_seconds: float = 5.0
    eval_weight_strategy: Literal["fixed", "dirichlet", "uniform"] = "fixed"
    eval_dirichlet_alpha: float = 1.0


def _get_workspace_dir() -> Path:
    """Return the directory where the training invocation was launched."""
    try:
        return Path(get_original_cwd())
    except HydraException:
        return Path.cwd()


def _resolve_path(path_str: Optional[str], workspace_dir: Path) -> Optional[Path]:
    """Resolve potentially relative config paths against the original workspace."""
    if path_str in (None, ""):
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = workspace_dir / path
    return path


def prepare_agent(env: gym.Env, reward_config: RewardConfig, args: TrainConfig) -> MOPPOAgent:
    config = MOPPOConfig(
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        dirichlet_alpha=args.dirichlet_alpha,
        weight_strategy=args.weight_strategy,
        device=args.device,
    )
    agent = MOPPOAgent(env.single_observation_space, env.single_action_space, reward_config.reward_dim, config)
    return agent


def resolve_eval_weights(
    args: TrainConfig, reward_dim: int, device: torch.device
) -> tuple[torch.Tensor | None, WeightSampler | None]:
    strategy = args.eval_weight_strategy
    if strategy == "fixed":
        if args.eval_weights is None:
            weights = np.full(reward_dim, 1.0 / reward_dim, dtype=np.float32)
        else:
            weights = np.asarray(list(args.eval_weights), dtype=np.float32)
            if weights.shape[0] != reward_dim:
                raise ValueError(f"Expected {reward_dim} eval weights, got {weights.shape[0]}.")
            total = float(weights.sum())
            if total > 0:
                weights = weights / total
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)
        return weight_tensor, None

    if strategy not in ("dirichlet", "uniform"):
        raise ValueError(
            "eval_weight_strategy must be one of 'fixed', 'dirichlet', or 'uniform', "
            f"got {strategy}."
        )

    sampler = WeightSampler(
        reward_dim=reward_dim,
        strategy=strategy,  # type: ignore[arg-type]
        dirichlet_alpha=args.eval_dirichlet_alpha,
    )
    return None, sampler


def run_evaluation(
    agent: MOPPOAgent,
    reward_config: RewardConfig,
    env_id: str,
    weights: torch.Tensor | None,
    weight_sampler: WeightSampler | None,
    num_episodes: int,
    seed: int,
    video_root: Path | None,
    step_label: str,
    render_mode: Literal["none", "video", "window"],
    max_display_seconds: float,
) -> tuple[np.ndarray, Optional[Path], int, list[str]]:
    capture_video = render_mode == "video"
    render_window = render_mode == "window"
    if capture_video and video_root is None:
        raise ValueError("eval_render_dir must be provided when eval_render_mode='video'.")

    env_render_mode: Optional[str] = None
    if capture_video:
        env_render_mode = "rgb_array"
    elif render_window:
        env_render_mode = "human"

    recording_dir: Optional[Path] = None
    if capture_video and video_root is not None:
        recording_dir = video_root / step_label
        recording_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(
        env_id=env_id,
        reward_config=reward_config,
        seed=seed,
        capture_video=capture_video,
        video_folder=recording_dir,
        video_name_prefix=step_label,
        render_mode=env_render_mode,
    )
    obs_space = env.observation_space

    def sample_eval_weights() -> torch.Tensor:
        if weight_sampler is not None:
            return weight_sampler.sample(1).to(agent.device)
        if weights is None:
            raise ValueError("Fixed evaluation weights required when eval_weight_strategy='fixed'.")
        return weights

    reward_names = reward_config.names

    def describe_weights(weight_tensor: torch.Tensor) -> str:
        flat = weight_tensor.detach().cpu().numpy().reshape(-1)
        return ", ".join(f"{name}={float(value):.2f}" for name, value in zip(reward_names, flat))

    weight_logs: list[str] = []

    current_weights = sample_eval_weights()
    weight_logs.append(f"episode 1: {describe_weights(current_weights)}")

    obs, _ = env.reset(seed=seed)
    flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
    episode_return = np.zeros(reward_config.reward_dim, dtype=np.float32)
    returns: list[np.ndarray] = []
    episode = 0
    display_deadline = (time.perf_counter() + max_display_seconds) if render_window else None

    while episode < num_episodes:
        obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=agent.device).unsqueeze(0).view(1, -1)
        with torch.no_grad():
            action_tensor = agent.act_deterministic(obs_tensor, current_weights)
        env_action = agent.actions_to_env(action_tensor)[0]
        obs, reward, terminated, truncated, info = env.step(env_action)
        flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
        if render_window:
            env.render()
        reward_vector = info.get("reward_vector")
        if reward_vector is None:
            reward_vector = np.array([reward], dtype=np.float32)
        episode_return += reward_vector
        if terminated or truncated:
            returns.append(episode_return.copy())
            episode_return.fill(0.0)
            episode += 1
            obs, _ = env.reset()
            flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
            if weight_sampler is not None and episode < num_episodes:
                current_weights = sample_eval_weights()
                weight_logs.append(f"episode {episode + 1}: {describe_weights(current_weights)}")
        if render_window and display_deadline is not None and time.perf_counter() >= display_deadline:
            break

    env.close()

    if returns:
        avg_return = np.mean(returns, axis=0)
    else:
        avg_return = np.zeros(reward_config.reward_dim, dtype=np.float32)

    recorded_video: Optional[Path] = None
    if recording_dir is not None:
        mp4_files = sorted(recording_dir.glob("*.mp4"))
        if mp4_files:
            recorded_video = mp4_files[-1]

    return avg_return, recorded_video, len(returns), weight_logs


def _save_checkpoint(
    agent: MOPPOAgent,
    reward_config: RewardConfig,
    args: TrainConfig,
    checkpoint_path: Path,
    reason: Optional[str] = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = agent.to_checkpoint()
    checkpoint.update(
        {
            "env_id": args.env_id,
            "reward_names": reward_config.names,
            "reward_config_path": args.reward_config,
        }
    )
    torch.save(checkpoint, checkpoint_path)
    reason_suffix = f" ({reason})" if reason else ""
    console.print(f"[bold blue]Saved checkpoint{reason_suffix} to {checkpoint_path}[/bold blue]")


def train(args: TrainConfig) -> None:
    workspace_dir = _get_workspace_dir()
    reward_config_path = _resolve_path(args.reward_config, workspace_dir)
    checkpoint_path = _resolve_path(args.checkpoint_path, workspace_dir)
    eval_render_dir = _resolve_path(args.eval_render_dir, workspace_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reward_config = load_reward_config(
        reward_config_path.as_posix() if reward_config_path is not None else None,
        args.env_id,
    )
    env = make_vectorized_env(
        env_id=args.env_id,
        reward_config=reward_config,
        num_envs=args.num_envs,
        seed=args.seed,
        asynchronous=args.asynchronous_envs,
    )
    agent = prepare_agent(env, reward_config, args)

    obs_shape = (space_utils.flatdim(env.single_observation_space),)
    action_shape = env.single_action_space.shape

    buffer = RolloutBuffer(
        num_steps=args.num_steps,
        num_envs=args.num_envs,
        obs_shape=obs_shape,
        action_shape=action_shape,
        reward_dim=reward_config.reward_dim,
        device=torch.device(args.device),
    )

    obs, info = env.reset(seed=args.seed)
    obs = obs.astype(np.float32)
    global_step = 0
    num_updates = args.total_steps // (args.num_envs * args.num_steps)
    episode_rewards = np.zeros((args.num_envs, reward_config.reward_dim), dtype=np.float32)
    completed_returns: list[np.ndarray] = []
    env_weights = agent.sample_weights(args.num_envs)

    eval_weights, eval_weight_sampler = resolve_eval_weights(
        args, reward_config.reward_dim, torch.device(args.device)
    )
    eval_video_root = eval_render_dir
    eval_render_mode = args.eval_render_mode
    if eval_render_mode == "none" and eval_video_root is not None:
        eval_render_mode = "video"
    eval_render_seconds = max(args.eval_render_seconds, 0.0)

    console.print(f"[bold green]Starting training for {num_updates} updates ({args.total_steps} steps).[/bold green]")

    for update in range(num_updates):
        buffer.reset()
        for step in range(args.num_steps):
            weights = env_weights
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(args.num_envs, -1)
            actions, log_probs, values = agent.act(obs_tensor, weights)
            env_actions = agent.actions_to_env(actions)

            next_obs, reward, terminated, truncated, info = env.step(env_actions)
            dones = np.logical_or(terminated, truncated)
            reward_vector = info.get("reward_vector")
            if reward_vector is None:
                reward_vector = np.expand_dims(reward, axis=-1)
            reward_vector = reward_vector.astype(np.float32)

            episode_rewards += reward_vector
            for env_idx, done_flag in enumerate(dones):
                if done_flag:
                    completed_returns.append(episode_rewards[env_idx].copy())
                    episode_rewards[env_idx] = 0.0

            buffer.add(
                obs_tensor,
                actions.detach().float(),
                log_probs.detach(),
                values.detach(),
                torch.as_tensor(reward_vector, dtype=torch.float32, device=agent.device),
                torch.as_tensor(dones, dtype=torch.float32, device=agent.device),
                weights.detach(),
            )

            obs = next_obs
            global_step += args.num_envs

            if np.any(dones):
                done_indices = np.nonzero(dones)[0].tolist()
                if done_indices:
                    env_weights[done_indices] = agent.sample_weights(len(done_indices))

        with torch.no_grad():
            next_obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(args.num_envs, -1)
            next_values = agent.critic(next_obs_tensor, env_weights)
            next_dones = torch.as_tensor(dones, dtype=torch.float32, device=agent.device)

        buffer.compute_returns_and_advantages(
            next_values=next_values,
            next_dones=next_dones,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )
        metrics = agent.update(buffer)
        avg_return = np.mean(completed_returns[-10:], axis=0) if completed_returns else np.zeros(
            reward_config.reward_dim
        )
        eval_avg_return: Optional[np.ndarray] = None
        eval_video_path: Optional[Path] = None
        eval_episodes_run = 0

        should_eval = args.eval_interval > 0 and (
            (update + 1) % args.eval_interval == 0 or update == num_updates - 1
        )
        if should_eval:
            (
                eval_avg_return,
                eval_video_path,
                eval_episodes_run,
                eval_weight_logs,
            ) = run_evaluation(
                agent=agent,
                reward_config=reward_config,
                env_id=args.env_id,
                weights=eval_weights,
                weight_sampler=eval_weight_sampler,
                num_episodes=args.eval_episodes,
                seed=args.seed + update,
                video_root=eval_video_root,
                step_label=f"update_{update + 1:05d}",
                render_mode=eval_render_mode,
                max_display_seconds=eval_render_seconds,
            )
            eval_str = ", ".join(f"{val:.2f}" for val in eval_avg_return)
            console.print(
                f"[bold cyan]Evaluation ({eval_episodes_run}/{args.eval_episodes} episodes) avg return:[/bold cyan] {eval_str}"
            )
            if eval_weight_logs:
                console.print("[magenta]Evaluation preference weights:[/magenta]")
                for log_entry in eval_weight_logs:
                    console.print(f"  {log_entry}")
            if eval_render_mode == "window" and eval_episodes_run < args.eval_episodes:
                console.print(
                    "[yellow]Stopped evaluation early after hitting the render time budget.[/yellow]"
                )
            if eval_video_path is not None:
                console.print(f"[cyan]Saved evaluation video to {eval_video_path}[/cyan]")
            if checkpoint_path is not None:
                _save_checkpoint(
                    agent=agent,
                    reward_config=reward_config,
                    args=args,
                    checkpoint_path=checkpoint_path,
                    reason=f"after evaluation update {update + 1}",
                )

        table = Table(title=f"Update {update + 1}/{num_updates}")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_column("Meaning")
        for key, value in metrics.items():
            description = METRIC_DESCRIPTIONS.get(key, "Gradient statistic averaged over updates.")
            table.add_row(key, f"{value:.4f}", description)
        avg_return_str = ", ".join(f"{val:.2f}" for val in avg_return)
        table.add_row("avg_return", avg_return_str, METRIC_DESCRIPTIONS["avg_return"])
        if eval_avg_return is not None:
            eval_avg_return_str = ", ".join(f"{val:.2f}" for val in eval_avg_return)
            table.add_row("eval_avg_return", eval_avg_return_str, METRIC_DESCRIPTIONS["eval_avg_return"])
        console.print(table)

    if checkpoint_path is not None:
        _save_checkpoint(
            agent=agent,
            reward_config=reward_config,
            args=args,
            checkpoint_path=checkpoint_path,
            reason="final",
        )


def _build_train_config(cfg: DictConfig) -> TrainConfig:
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("Expected dict-like config while building TrainConfig.")
    data.pop("hydra", None)
    return TrainConfig(**data)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    train(_build_train_config(cfg))


if __name__ == "__main__":
    main()
