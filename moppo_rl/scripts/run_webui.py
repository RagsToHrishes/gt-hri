"""Launches a lightweight web UI for interactive reward-weight steering."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import gradio as gr
import gymnasium as gym
from gymnasium.spaces import utils as space_utils
import imageio
import numpy as np
import torch
import tyro

from moppo.agents.moppo import MOPPOAgent, MOPPOConfig
from moppo.envs.factory import make_env
from moppo.envs.reward_config import load_reward_config, RewardConfig


@dataclass
class WebUIArgs:
    checkpoint: str
    env_id: str
    reward_config: Optional[str] = None
    device: str = "cpu"
    seed: int = 0
    rollout_steps: int = 400
    fps: int = 30


def build_env(env_id: str, reward_config: RewardConfig, seed: int | None = None) -> gym.Env:
    # Reuse the full training wrapper stack so the agent sees identical observations.
    env_seed = 0 if seed is None else seed
    env = make_env(
        env_id=env_id,
        reward_config=reward_config,
        seed=env_seed,
        capture_video=False,
        video_folder=None,
        render_mode="rgb_array",
    )
    return env


def load_agent(checkpoint: dict, env: gym.Env, reward_config: RewardConfig, device: str) -> MOPPOAgent:
    config = MOPPOConfig(**checkpoint["config"])
    config.device = device
    agent = MOPPOAgent(env.observation_space, env.action_space, reward_config.reward_dim, config)
    agent.load_state_dict(checkpoint["state_dict"])
    agent.eval()
    return agent


def create_video(frames: List[np.ndarray], fps: int) -> str:
    if not frames:
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    temp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    with imageio.get_writer(temp.name, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return temp.name


def run_ui(args: WebUIArgs) -> None:
    checkpoint = torch.load(Path(args.checkpoint), map_location=args.device)
    reward_config_path = args.reward_config or checkpoint.get("reward_config_path")
    reward_config = load_reward_config(reward_config_path, args.env_id)

    env = build_env(args.env_id, reward_config, seed=args.seed)
    agent = load_agent(checkpoint, env, reward_config, args.device)
    env.close()

    reward_names = reward_config.names
    base_weight = 1.0 / len(reward_names)

    def rollout(*slider_weights: float) -> tuple[str, str]:
        weights = np.array(slider_weights, dtype=np.float32)
        if weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        env_local = build_env(args.env_id, reward_config, seed=args.seed)
        obs_space = env_local.observation_space
        obs, _ = env_local.reset(seed=args.seed)
        flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
        first_frame = env_local.render()
        frames: List[np.ndarray] = [first_frame] if first_frame is not None else []
        total_rewards = np.zeros(reward_config.reward_dim, dtype=np.float32)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=agent.device).unsqueeze(0)

        for _ in range(args.rollout_steps):
            obs_tensor = torch.as_tensor(flat_obs, dtype=torch.float32, device=agent.device).view(1, -1)
            action_tensor = agent.act_deterministic(obs_tensor, weight_tensor)
            env_action = agent.actions_to_env(action_tensor)[0]
            obs, reward, terminated, truncated, info = env_local.step(env_action)
            flat_obs = np.asarray(space_utils.flatten(obs_space, obs), dtype=np.float32)
            frame = env_local.render()
            if frame is not None:
                frames.append(frame)
            total_rewards += info.get("reward_vector", np.array([reward], dtype=np.float32))
            if terminated or truncated:
                break
        env_local.close()

        video_path = create_video(frames, fps=args.fps)
        returns_text = ", ".join(f"{name}: {val:.2f}" for name, val in zip(reward_names, total_rewards))
        return video_path, returns_text

    with gr.Blocks() as demo:
        gr.Markdown("## MOPPO Interactive Explorer")
        gr.Markdown("Adjust reward weights to see how the conditioned policy responds.")
        sliders: List[gr.Slider] = []
        for name in reward_names:
            sliders.append(
                gr.Slider(minimum=0.0, maximum=1.0, value=base_weight, step=0.01, label=f"Weight: {name}")
            )
        video = gr.Video(label="Rollout", autoplay=True)
        stats = gr.Textbox(label="Episode Returns", interactive=False)
        run_button = gr.Button("Generate Rollout")

        inputs = sliders
        run_button.click(fn=rollout, inputs=inputs, outputs=[video, stats])
        for slider in sliders:
            slider.release(fn=rollout, inputs=inputs, outputs=[video, stats])

    demo.launch()


if __name__ == "__main__":
    run_ui(tyro.cli(WebUIArgs))
