# MOPPO RL Playground

This repository provides a reference implementation of Multi-Objective Proximal Policy Optimization (MOPPO) inspired by the AMOR paper from Disney Research SIGGRAPH 2025. The goal is to make it easy to

- plug in Gymnasium / MuJoCo tasks with custom reward vector definitions,
- train a conditioned MOPPO agent that can smoothly adapt to preference weight changes,
- visualize trained checkpoints through a local web interface with live reward-weight dials.

## Key folders

```
configs/                # Reward vector definitions per environment
moppo/                  # Core library code
  agents/               # Algorithm implementations (MOPPO)
  envs/                 # Helper wrappers and reward adapters
  models/               # Actor / Critic networks
  storage/              # Rollout buffers
  utils/                # Shared utilities
  web/                  # Utilities for the live web UI
scripts/                # Entrypoints for training, evaluation, and the UI
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[web]
```

### Training

Training is now powered by Hydra. The defaults live in `moppo_rl/configs/train.yaml`, so you can either edit that
file, point `--config-path/--config-name` at a copy, or override individual fields inline with `key=value` style
arguments. Hydra stores outputs under `outputs/train/<timestamp>` by default; override `hydra.run.dir`
if you prefer a fixed log directory.

```bash
python scripts/train.py env_id=HalfCheetah-v4 total_steps=2_000_000 \
  reward_config=configs/mujoco_halfcheetah.py \
  eval_interval=10 eval_episodes=2 eval_render_mode=window \
  eval_render_seconds=6 eval_weights='[0.34,0.33,0.33]'
```

Every run automatically saves checkpoints under `checkpoints/<env_id>_<timestamp>.pt` (non-alphanumeric
characters in the environment name turn into `_`). Look for the checkpoint path printed after each evaluation
to point the UI or Pareto scripts at the most recent weights.

`eval_interval` controls how often training pauses for deterministic rollouts, `eval_episodes` sets the
number of episodes per evaluation, and `eval_weights` fixes the preference vector used for evaluation; omit the last
flag to default to equal weights. Use `eval_render_mode=window` to briefly show each evaluation in a native
viewer (capped by `eval_render_seconds`). To archive videos instead, pass `eval_render_mode=video` together with
`eval_render_dir=<folder>` and the run will save mp4s under per-update subdirectories.

Every run also streams summaries to TensorBoard under `<hydra run>/tensorboard`, launches a local server on
`http://127.0.0.1:6006`, and opens your browser automatically. PPO losses, gradient stats, and each reward component are
logged per update, while deterministic evaluation returns and videos appear under the `eval/` namespace. Disable or tweak
this behavior via Hydra overrides (use `tensorboard_log_dir=...` to redirect event files):

```bash
python scripts/train.py tensorboard_port=16006 tensorboard_auto_open=false \
  enable_tensorboard=false tensorboard_log_videos=false
```

When `tensorboard_log_videos=true`, the trainer either uploads the evaluation video you requested
(`eval_render_mode=video`) or quietly records a short rgb-array evaluation just for TensorBoard so you always get a clip;
set `tensorboard_video_episodes=<n>` to control how long those recordings are.

Each evaluation step also reuses a fixed batch of preference weights (`pareto_num_samples`, drawn with
`pareto_weight_strategy/pareto_dirichlet_alpha`) to sweep the learned policy and build a sampled Pareto front. The
resulting hypervolume is printed and logged to TensorBoard, while radar/spider plots and raw JSON data are saved under
`<hydra run>/pareto_eval/update_<step>_*`. Override `pareto_weights=[[...], ...]`, `pareto_eval_episodes`,
`pareto_reference_point`, or disable the extra sweep entirely with `pareto_num_samples=0`.

To randomize evaluation preferences at every episode, set `eval_weight_strategy=dirichlet`
or `eval_weight_strategy=uniform` and optionally adjust `eval_dirichlet_alpha`:

```bash
python scripts/train.py env_id=MiniGrid-Empty-5x5-v0 total_steps=500_000 \
  reward_config=configs/minigrid_empty.py \
  eval_interval=5 eval_episodes=4 eval_render_mode=window \
  eval_render_seconds=3 eval_weight_strategy=dirichlet eval_dirichlet_alpha=0.5
```

### Evaluation with fixed weights

```bash
python scripts/evaluate.py --checkpoint checkpoints/halfcheetah.pth \
  --env-id HalfCheetah-v4 --reward-config configs/mujoco_halfcheetah.py \
  --weights 0.4 0.4 0.2
```

### Maze-style directional rewards

For AntMaze or similar navigation tasks, point `--reward-config` at `configs/mujoco_antmaze.py`. The config exposes four reward components (`north`, `south`, `east`, `west`) that activate when the agent's planar velocity points toward the respective cardinal direction, making it easy to train policies that prioritize specific headings.

```bash
python scripts/train.py env_id=AntMaze-UMaze-v2 total_steps=2_000_000 \
  reward_config=configs/mujoco_antmaze.py \
  eval_interval=10 eval_episodes=2 eval_render_mode=window \
  eval_render_seconds=6 eval_weights='[0.25,0.25,0.25,0.25]'
```

Classic Minari point-maze datasets (e.g., `maze2d-medium-v1`) can use `configs/minari_point_maze.py` for the same directional reward vector while adapting to the point-mass observation format.

```bash
python scripts/train.py env_id=maze2d-medium-v1 total_steps=1_000_000 \
  reward_config=configs/minari_point_maze.py \
  eval_interval=5 eval_episodes=2 eval_render_mode=window \
  eval_render_seconds=4 eval_weights='[0.25,0.25,0.25,0.25]'
```

For lightweight grid worlds such as `MiniGrid-Empty-5x5-v0`, use `configs/minigrid_empty.py` (requires `gymnasium-minigrid`) to prefer different walls:

```bash
python scripts/train.py env_id=MiniGrid-Empty-5x5-v0 total_steps=500_000 \
  reward_config=configs/minigrid_empty.py \
  eval_interval=5 eval_episodes=4 eval_render_mode=window \
  eval_render_seconds=3 eval_weights='[0.25,0.25,0.25,0.25]'
```

### Interactive weight steering

```bash
python scripts/run_webui.py --checkpoint checkpoints/halfcheetah.pth \
  --env-id HalfCheetah-v4 --reward-config configs/mujoco_halfcheetah.py
```

The UI exposes sliders for each reward component and streams short rollouts generated with the latest weights.

For the lightweight MiniGrid setup that ships with this repo (checkpoint saved under `checkpoints/latest-moppo.pt`), launch the UI like so:

```bash
python scripts/run_webui.py --checkpoint checkpoints/latest-moppo.pt \
  --env-id MiniGrid-Empty-8x8-v0 --reward-config configs/minigrid_empty.py
```

### Pareto front visualization

After training, generate Pareto fronts across every objective pair with `scripts/plot_pareto.py`. For the bundled MiniGrid checkpoint:

```bash
python scripts/plot_pareto.py --checkpoint checkpoints/latest-moppo.pt \
  --env-id MiniGrid-Empty-8x8-v0 --reward-config configs/minigrid_empty.py \
  --output-dir outputs/pareto_minigrid --num-points 50 --num-episodes 5 --seed 13
```

This sweeps weights between each pair of objectives, averages returns over `num_episodes`, and marks the equal-weight point (red `X`) on every scatter plot saved under the chosen `output_dir`.

### Training reward curves

Every training run logs PPO statistics and the averaged reward vector per update to `training_metrics.jsonl` inside the
Hydra run directory (e.g., `outputs/train/<timestamp>/training_metrics.jsonl`). Plot the per-component reward curves
with:

```bash
python scripts/plot_training_rewards.py \
  --log-file outputs/train/2024-05-09_12-00-00/training_metrics.jsonl \
  --output outputs/train/2024-05-09_12-00-00/reward_curve.png \
  --include-eval
```

The output marks one line per reward component (optionally overlaying evaluation returns) so you can quickly inspect how
each objective evolves over time. The JSON log also keeps the underlying PPO metrics for any additional custom analysis.

## Notes

- Default reward vector definitions are provided for a few MuJoCo environments as a starting point. Extend `configs/` with your own components or programmatic generators.
- MOPPO trains a weight-conditioned policy; make sure to randomize preference weights during training to cover the regions you care about at inference time.
- Scripts expect Gymnasium environments. Install `gymnasium[mujoco]` to access MuJoCo control tasks.
- Evaluation video export depends on `moviepy`; installing the package (e.g., via `pip install -e .[web]`) pulls it in automatically.
- AntMaze task IDs come from `d4rl`/`minari` and require a local MuJoCo installation; install `gymnasium-robotics`, `d4rl`, and `minari`, configure MuJoCo per their READMEs, and run `python -c "import minari; minari.register_d4rl_datasets()"` once to expose the offline maze environments.
- MiniGrid examples depend on `gymnasium-minigrid`; install it separately if you want to run the empty-room command.
