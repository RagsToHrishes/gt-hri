"""Training entrypoint for the MOPPO agent."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Optional, Sequence
import subprocess
import threading
import time
import webbrowser

import gymnasium as gym
from gymnasium.spaces import utils as space_utils

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    "pareto_hypervolume": "Reference-dominated hypervolume computed from sampled evaluation weights.",
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
    eval_interval: int = 10
    eval_episodes: int = 2
    eval_weights: Optional[Sequence[float]] = None
    eval_render_dir: Optional[str] = None
    eval_render_mode: Literal["none", "video", "window"] = "none"
    eval_render_seconds: float = 5.0
    eval_weight_strategy: Literal["fixed", "dirichlet", "uniform"] = "fixed"
    eval_dirichlet_alpha: float = 1.0
    enable_tensorboard: bool = True
    tensorboard_port: int = 6006
    tensorboard_host: str = "127.0.0.1"
    tensorboard_auto_open: bool = True
    tensorboard_log_dir: Optional[str] = None
    tensorboard_log_videos: bool = True
    tensorboard_video_episodes: int = 1
    pareto_num_samples: int = 30
    pareto_weight_strategy: Literal["fixed", "dirichlet", "uniform"] = "dirichlet"
    pareto_dirichlet_alpha: float = 1.0
    pareto_weights: Optional[Sequence[Sequence[float]]] = None
    pareto_seed: int = 23
    pareto_eval_episodes: int = 1
    pareto_reference_point: Optional[Sequence[float]] = None
    pareto_plot_dir: Optional[str] = None
    pareto_log_json: bool = True


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


def _record_tensorboard_video(
    *,
    agent: MOPPOAgent,
    reward_config: RewardConfig,
    env_id: str,
    weights: torch.Tensor | None,
    weight_sampler: WeightSampler | None,
    num_episodes: int,
    seed: int,
    video_root: Path,
    update_index: int,
) -> Optional[Path]:
    if num_episodes <= 0:
        return None
    video_root.mkdir(parents=True, exist_ok=True)
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            cuda_state = torch.cuda.get_rng_state_all()
        except RuntimeError:
            cuda_state = None
    try:
        _, video_path, _, _ = run_evaluation(
            agent=agent,
            reward_config=reward_config,
            env_id=env_id,
            weights=weights,
            weight_sampler=weight_sampler,
            num_episodes=num_episodes,
            seed=seed,
            video_root=video_root,
            step_label=f"tensorboard_update_{update_index:05d}",
            render_mode="video",
            max_display_seconds=0.0,
        )
        return video_path
    except Exception as error:
        console.print(f"[yellow]Failed to record TensorBoard evaluation video: {error}[/yellow]")
        return None
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except RuntimeError:
                pass


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


def _sanitize_env_id(env_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in env_id)


def _build_run_artifact_dir(env_id: str, workspace_dir: Path) -> Path:
    sanitized_env = _sanitize_env_id(env_id)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    checkpoint_root = workspace_dir / "checkpoints"
    return checkpoint_root / f"{sanitized_env}_{timestamp}"


class MetricLogger:
    def __init__(self, run_dir: Path, env_id: str, reward_names: Sequence[str]) -> None:
        self.run_dir = run_dir
        self.log_path = self.run_dir / "training_metrics.jsonl"
        self._file = self.log_path.open("w", encoding="utf-8")
        metadata = {
            "type": "metadata",
            "env_id": env_id,
            "reward_names": list(reward_names),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._write(metadata)

    def _write(self, record: dict) -> None:
        json.dump(record, self._file)
        self._file.write("\n")
        self._file.flush()

    def log(
        self,
        *,
        update: int,
        global_step: int,
        avg_return: np.ndarray,
        eval_avg_return: Optional[np.ndarray],
        agent_metrics: dict[str, float],
        phase: Literal["train", "eval"] = "train",
    ) -> None:
        entry = {
            "type": "metrics",
            "update": update,
            "global_step": global_step,
            "avg_return": [float(val) for val in avg_return],
            "eval_avg_return": None
            if eval_avg_return is None
            else [float(val) for val in eval_avg_return],
            "agent_metrics": {key: float(value) for key, value in agent_metrics.items()},
            "phase": phase,
        }
        self._write(entry)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


@dataclass
class ParetoAnalysisResult:
    hypervolume: float
    radar_path: Optional[Path]
    front_returns: np.ndarray
    all_returns: np.ndarray
    weights: np.ndarray
    reference_point: np.ndarray


def _normalize_weight_matrix(weights: np.ndarray) -> np.ndarray:
    sums = np.sum(weights, axis=1, keepdims=True)
    sums[sums <= 0.0] = 1.0
    return weights / sums


def _sample_weight_matrix(
    *,
    strategy: Literal["dirichlet", "uniform", "fixed"],
    reward_dim: int,
    num_samples: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_samples <= 0:
        return np.empty((0, reward_dim), dtype=np.float32)
    if strategy == "dirichlet":
        alpha_vec = np.full(reward_dim, alpha, dtype=np.float32)
        samples = rng.dirichlet(alpha_vec, size=num_samples).astype(np.float32)
    elif strategy == "uniform":
        raw = rng.random((num_samples, reward_dim))
        raw = raw.astype(np.float32)
        samples = raw / np.clip(raw.sum(axis=1, keepdims=True), 1e-6, None)
    elif strategy == "fixed":
        raise ValueError("pareto_weight_strategy='fixed' requires pareto_weights to be provided.")
    else:
        raise ValueError(f"Unsupported pareto_weight_strategy: {strategy}")
    return samples


def _prepare_pareto_weights(args: TrainConfig, reward_dim: int, device: torch.device) -> torch.Tensor | None:
    if args.pareto_weights is not None:
        weight_array = np.asarray(args.pareto_weights, dtype=np.float32)
        if weight_array.ndim != 2 or weight_array.shape[1] != reward_dim:
            raise ValueError(
                f"pareto_weights must be a list of length-{reward_dim} vectors, "
                f"got shape {weight_array.shape}."
            )
    elif args.pareto_num_samples <= 0:
        return None
    else:
        rng = np.random.default_rng(args.seed + args.pareto_seed)
        samples = _sample_weight_matrix(
            strategy=args.pareto_weight_strategy,
            reward_dim=reward_dim,
            num_samples=args.pareto_num_samples,
            alpha=args.pareto_dirichlet_alpha,
            rng=rng,
        )
        weight_array = samples
    normalized = _normalize_weight_matrix(weight_array.astype(np.float32))
    tensor = torch.tensor(normalized, dtype=torch.float32, device=device)
    return tensor


def _pareto_front_mask(points: np.ndarray) -> np.ndarray:
    num_points = points.shape[0]
    mask = np.ones(num_points, dtype=bool)
    for i in range(num_points):
        if not mask[i]:
            continue
        dominates = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        dominates[i] = False
        mask[dominates] = False
    return mask


def _compute_reference_point(
    candidate: Optional[Sequence[float]],
    returns: np.ndarray,
) -> np.ndarray:
    reward_dim = returns.shape[1]
    if candidate is not None:
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.shape != (reward_dim,):
            raise ValueError(f"pareto_reference_point must have shape ({reward_dim},), got {arr.shape}.")
        return arr
    if returns.size == 0:
        return np.zeros(reward_dim, dtype=np.float32)
    min_vals = np.min(returns, axis=0)
    ref = np.minimum(min_vals, 0.0)
    return ref.astype(np.float32)


def _hypervolume_from_points(points: np.ndarray, reference: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    shifted = points - reference
    shifted = np.maximum(shifted, 0.0)
    valid_mask = np.any(shifted > 0.0, axis=1)
    valid = shifted[valid_mask]
    if valid.size == 0:
        return 0.0
    return _hypervolume_recursive(valid)


def _hypervolume_recursive(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    dims = points.shape[1]
    if dims == 1:
        return float(np.max(points[:, 0]))
    sorted_points = points[np.argsort(points[:, -1])]
    unique_levels = np.unique(sorted_points[:, -1])
    volume = 0.0
    prev = 0.0
    for level in unique_levels:
        width = float(level - prev)
        if width > 0:
            mask = sorted_points[:, -1] >= level
            slice_points = sorted_points[mask][:, : dims - 1]
            contribution = _hypervolume_recursive(slice_points)
            volume += width * contribution
        prev = float(level)
    return volume


def _plot_pareto_radar(front_points: np.ndarray, reward_names: Sequence[str], output_path: Path) -> Optional[Path]:
    if front_points.size == 0:
        return None
    num_objectives = len(reward_names)
    angles = np.linspace(0, 2 * np.pi, num_objectives, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    color_cycle = plt.cm.viridis(np.linspace(0, 1, front_points.shape[0]))
    for idx, values in enumerate(front_points):
        closed = np.concatenate([values, values[:1]])
        ax.plot(angles, closed, label=f"front #{idx + 1}", color=color_cycle[idx], linewidth=2.0)
        ax.fill(angles, closed, color=color_cycle[idx], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(reward_names, fontsize=10)
    ax.set_title("Pareto front radar plot")
    ax.grid(True, alpha=0.3)
    if front_points.shape[0] <= 10:
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _run_pareto_analysis(
    *,
    agent: MOPPOAgent,
    reward_config: RewardConfig,
    args: TrainConfig,
    weight_set: Optional[torch.Tensor],
    update_index: int,
    pareto_plot_dir: Path,
) -> Optional[ParetoAnalysisResult]:
    if weight_set is None or weight_set.shape[0] == 0 or args.pareto_eval_episodes <= 0:
        return None
    pareto_plot_dir.mkdir(parents=True, exist_ok=True)
    results: list[np.ndarray] = []
    for idx in range(weight_set.shape[0]):
        weight_tensor = weight_set[idx].unsqueeze(0)
        avg_return, _, _, _ = run_evaluation(
            agent=agent,
            reward_config=reward_config,
            env_id=args.env_id,
            weights=weight_tensor,
            weight_sampler=None,
            num_episodes=args.pareto_eval_episodes,
            seed=args.seed + 1000 * (update_index + 1) + idx,
            video_root=None,
            step_label=f"pareto_update_{update_index:05d}_weight_{idx:03d}",
            render_mode="none",
            max_display_seconds=0.0,
        )
        results.append(avg_return)
    return_matrix = np.stack(results, axis=0)
    weight_matrix = weight_set.detach().cpu().numpy()
    front_mask = _pareto_front_mask(return_matrix)
    front_points = return_matrix[front_mask]
    reference = _compute_reference_point(args.pareto_reference_point, return_matrix)
    hypervolume = _hypervolume_from_points(front_points, reference)
    plot_name = f"update_{update_index:05d}_radar.png"
    radar_path = _plot_pareto_radar(front_points, reward_config.names, pareto_plot_dir / plot_name)
    if args.pareto_log_json:
        json_path = pareto_plot_dir / f"update_{update_index:05d}_pareto.json"
        payload = {
            "update": update_index,
            "hypervolume": float(hypervolume),
            "reference_point": reference.tolist(),
            "weights": weight_matrix.tolist(),
            "average_returns": return_matrix.tolist(),
            "pareto_front_returns": front_points.tolist(),
        }
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    return ParetoAnalysisResult(
        hypervolume=float(hypervolume),
        radar_path=radar_path,
        front_returns=front_points,
        all_returns=return_matrix,
        weights=weight_matrix,
        reference_point=reference,
    )
class TensorBoardLogger:
    def __init__(
        self,
        *,
        log_dir: Path,
        reward_names: Sequence[str],
        enable: bool,
        host: str,
        port: int,
        auto_open: bool,
        log_videos: bool,
    ) -> None:
        self.enabled = enable
        self.log_dir = log_dir
        self.reward_names = list(reward_names)
        self.writer: SummaryWriter | None = None
        self._process: subprocess.Popen[str] | None = None
        self._log_handle: Optional[object] = None
        self._tb_url: Optional[str] = None
        self.log_videos = log_videos
        self.max_video_frames = 600
        if not enable:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.writer = SummaryWriter(self.log_dir.as_posix())
        except Exception as error:
            console.print(f"[yellow]Failed to initialize TensorBoard writer: {error}[/yellow]")
            self.writer = None
            self.enabled = False
            return
        console.print(f"[bold green]TensorBoard logs:[/bold green] {self.log_dir}")
        self._start_server(host=host, port=port, auto_open=auto_open)

    def _start_server(self, host: str, port: int, auto_open: bool) -> None:
        log_file_path = self.log_dir / "tensorboard.log"
        try:
            log_handle = log_file_path.open("w", encoding="utf-8")
        except OSError as error:
            console.print(f"[yellow]Could not open TensorBoard log file: {error}[/yellow]")
            log_handle = None
        cmd = [
            "tensorboard",
            "--logdir",
            str(self.log_dir),
            "--host",
            host,
            "--port",
            str(port),
        ]
        stdout_target = log_handle if log_handle is not None else subprocess.DEVNULL
        try:
            self._process = subprocess.Popen(cmd, stdout=stdout_target, stderr=subprocess.STDOUT, text=True)
            self._log_handle = log_handle
            self._tb_url = f"http://{host}:{port}/"
            console.print(f"[bold green]TensorBoard listening at {self._tb_url}[/bold green]")
            if auto_open:
                threading.Thread(target=self._open_browser, args=(self._tb_url,), daemon=True).start()
        except FileNotFoundError:
            console.print("[yellow]TensorBoard CLI not found; install 'tensorboard' to enable live metrics.[/yellow]")
            if log_handle is not None:
                log_handle.close()
            self._process = None
        except Exception as error:
            console.print(f"[yellow]Failed to launch TensorBoard: {error}[/yellow]")
            if log_handle is not None:
                log_handle.close()
            self._process = None

    @staticmethod
    def _open_browser(url: str) -> None:
        time.sleep(2.0)
        try:
            webbrowser.open(url, new=2, autoraise=True)
        except Exception:
            pass

    @property
    def should_log(self) -> bool:
        return self.writer is not None

    @property
    def wants_videos(self) -> bool:
        return self.writer is not None and self.log_videos

    def log_train_metrics(
        self,
        *,
        update: int,
        global_step: int,
        train_metrics: dict[str, float],
        avg_return: np.ndarray,
    ) -> None:
        if self.writer is None:
            return
        writer = self.writer
        for key, value in train_metrics.items():
            writer.add_scalar(f"train/{key}", float(value), global_step)
        for idx, name in enumerate(self.reward_names):
            writer.add_scalar(f"train/avg_return/{name}", float(avg_return[idx]), global_step)
        writer.add_scalar("train/update", update, global_step)
        writer.add_scalar("train/global_step", global_step, global_step)
        writer.flush()

    def log_eval_metrics(
        self,
        *,
        global_step: int,
        eval_avg_return: Optional[np.ndarray],
        eval_episodes_run: int,
        hypervolume: Optional[float],
        radar_path: Optional[Path],
    ) -> None:
        if self.writer is None:
            return
        writer = self.writer
        if eval_avg_return is not None:
            for idx, name in enumerate(self.reward_names):
                writer.add_scalar(f"eval/avg_return/{name}", float(eval_avg_return[idx]), global_step)
            writer.add_scalar("eval/episodes_run", eval_episodes_run, global_step)
        if hypervolume is not None:
            writer.add_scalar("eval/pareto_hypervolume", float(hypervolume), global_step)
        if radar_path is not None and radar_path.exists():
            try:
                image = imageio.imread(radar_path.as_posix())
            except Exception as error:
                console.print(f"[yellow]Failed to load radar plot for TensorBoard: {error}[/yellow]")
            else:
                tensor = torch.from_numpy(image)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(-1)
                if tensor.ndim != 3:
                    console.print("[yellow]Skipping radar image logging: unsupported image dimensions.[/yellow]")
                else:
                    channels = tensor.shape[2]
                    if channels == 4:
                        tensor = tensor[..., :3]
                    elif channels == 1:
                        tensor = tensor.repeat(1, 1, 3)
                    if tensor.shape[2] != 3:
                        console.print("[yellow]Skipping radar image logging: expected 1, 3, or 4 channels.[/yellow]")
                    else:
                        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float() / 255.0
                        writer.add_images("eval/pareto_radar", tensor, global_step)
        writer.flush()

    def log_video(self, video_path: Path, *, tag: str, global_step: int) -> None:
        if self.writer is None or not self.log_videos:
            return
        if not video_path.exists():
            return
        try:
            reader = imageio.get_reader(video_path.as_posix())
        except Exception as error:
            console.print(f"[yellow]Failed to read evaluation video for TensorBoard: {error}[/yellow]")
            return
        frames: list[np.ndarray] = []
        fps = 30
        try:
            meta = reader.get_meta_data()
            fps = int(meta.get("fps", fps))
        except Exception:
            pass
        try:
            for idx, frame in enumerate(reader):
                frames.append(frame)
                if idx + 1 >= self.max_video_frames:
                    break
        finally:
            reader.close()
        if not frames:
            return
        video_arr = np.stack(frames, axis=0)
        video_tensor = torch.from_numpy(video_arr).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
        self.writer.add_video(tag, video_tensor, global_step=global_step, fps=fps)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
def train(args: TrainConfig) -> None:
    workspace_dir = _get_workspace_dir()
    run_dir = Path.cwd()
    artifact_dir = _build_run_artifact_dir(args.env_id, workspace_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reward_config_path = _resolve_path(args.reward_config, workspace_dir)
    eval_render_dir = _resolve_path(args.eval_render_dir, workspace_dir)
    tensorboard_log_dir = _resolve_path(args.tensorboard_log_dir, workspace_dir)
    checkpoint_path = artifact_dir / "checkpoint.pt"
    if tensorboard_log_dir is None:
        tensorboard_log_dir = run_dir / "tensorboard"
    tensorboard_video_root = run_dir / "tensorboard_eval_videos"
    pareto_plot_dir = (
        _resolve_path(args.pareto_plot_dir, workspace_dir) if args.pareto_plot_dir else artifact_dir / "pareto_eval"
    )

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
    pareto_weight_set = _prepare_pareto_weights(args, reward_config.reward_dim, agent.device)

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

    metrics_logger = MetricLogger(run_dir=run_dir, env_id=args.env_id, reward_names=reward_config.names)
    tensorboard_logger = TensorBoardLogger(
        log_dir=tensorboard_log_dir,
        reward_names=reward_config.names,
        enable=args.enable_tensorboard,
        host=args.tensorboard_host,
        port=args.tensorboard_port,
        auto_open=args.tensorboard_auto_open,
        log_videos=args.tensorboard_log_videos,
    )
    console.print(f"[bold green]Run artifacts (checkpoints & Pareto logs) will be stored in {artifact_dir}[/bold green]")
    console.print(f"[bold green]Training metrics will be recorded in {metrics_logger.log_path}[/bold green]")

    console.print(f"[bold green]Starting training for {num_updates} updates ({args.total_steps} steps).[/bold green]")

    try:
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
            tensorboard_logger.log_train_metrics(
                update=update + 1,
                global_step=global_step,
                train_metrics=metrics,
                avg_return=avg_return,
            )
            metrics_logger.log(
                update=update + 1,
                global_step=global_step,
                avg_return=avg_return,
                eval_avg_return=None,
                agent_metrics=metrics,
                phase="train",
            )
            eval_avg_return: Optional[np.ndarray] = None
            eval_video_path: Optional[Path] = None
            tb_eval_video_path: Optional[Path] = None
            eval_episodes_run = 0
            pareto_result: Optional[ParetoAnalysisResult] = None
            extra_metrics: dict[str, float] = {}

            should_eval = (
                args.eval_interval > 0
                and args.eval_episodes > 0
                and ((update + 1) % args.eval_interval == 0 or update == num_updates - 1)
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
                _save_checkpoint(
                    agent=agent,
                    reward_config=reward_config,
                    args=args,
                    checkpoint_path=checkpoint_path,
                    reason=f"after evaluation update {update + 1}",
                )
                tb_eval_video_path = eval_video_path
                if tensorboard_logger.wants_videos:
                    video_source = tb_eval_video_path
                    if video_source is None and args.tensorboard_log_videos:
                        desired_episodes = args.tensorboard_video_episodes
                        if args.eval_episodes > 0:
                            desired_episodes = min(args.eval_episodes, desired_episodes)
                        if desired_episodes > 0:
                            video_source = _record_tensorboard_video(
                                agent=agent,
                                reward_config=reward_config,
                                env_id=args.env_id,
                                weights=eval_weights,
                                weight_sampler=eval_weight_sampler,
                                num_episodes=desired_episodes,
                                seed=args.seed + update,
                                video_root=tensorboard_video_root,
                                update_index=update + 1,
                            )
                    if video_source is not None:
                        tensorboard_logger.log_video(
                            video_source,
                            tag="eval/video",
                            global_step=global_step,
                        )
                pareto_result = _run_pareto_analysis(
                    agent=agent,
                    reward_config=reward_config,
                    args=args,
                    weight_set=pareto_weight_set,
                    update_index=update + 1,
                    pareto_plot_dir=pareto_plot_dir,
                )
                if pareto_result is not None:
                    console.print(
                        f"[magenta]Pareto hypervolume (update {update + 1}):[/magenta] {pareto_result.hypervolume:.4f}"
                    )
                    if pareto_result.radar_path is not None:
                        console.print(f"[magenta]Saved radar plot to {pareto_result.radar_path}[/magenta]")
                    extra_metrics["pareto_hypervolume"] = pareto_result.hypervolume
                tensorboard_logger.log_eval_metrics(
                    global_step=global_step,
                    eval_avg_return=eval_avg_return,
                    eval_episodes_run=eval_episodes_run,
                    hypervolume=pareto_result.hypervolume if pareto_result is not None else None,
                    radar_path=pareto_result.radar_path if pareto_result is not None else None,
                )
                metrics_logger.log(
                    update=update + 1,
                    global_step=global_step,
                    avg_return=avg_return,
                    eval_avg_return=eval_avg_return,
                    agent_metrics={**metrics, **extra_metrics},
                    phase="eval",
                )

            table = Table(title=f"Update {update + 1}/{num_updates}")
            table.add_column("Metric")
            table.add_column("Value")
            table.add_column("Meaning")
            table_metrics = list(metrics.items())
            for key, value in extra_metrics.items():
                table_metrics.append((key, value))
            for key, value in table_metrics:
                description = METRIC_DESCRIPTIONS.get(key, "Gradient statistic averaged over updates.")
                table.add_row(key, f"{value:.4f}", description)
            avg_return_str = ", ".join(f"{val:.2f}" for val in avg_return)
            table.add_row("avg_return", avg_return_str, METRIC_DESCRIPTIONS["avg_return"])
            if eval_avg_return is not None:
                eval_avg_return_str = ", ".join(f"{val:.2f}" for val in eval_avg_return)
                table.add_row("eval_avg_return", eval_avg_return_str, METRIC_DESCRIPTIONS["eval_avg_return"])
            console.print(table)

    except Exception as error:
        console.print(f"[bold red]Training interrupted due to error: {error}[/bold red]")
        
    finally:
        metrics_logger.close()
        tensorboard_logger.close()

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
