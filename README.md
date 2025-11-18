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

```bash
python scripts/train.py --env-id HalfCheetah-v4 --total-steps 2_000_000 \
  --reward-config configs/mujoco_halfcheetah.py --checkpoint-path checkpoints/halfcheetah.pth \
  --eval-interval 10 --eval-episodes 2 --eval-render-mode window \
  --eval-render-seconds 6 --eval-weights 0.34 0.33 0.33
```


`--eval-interval` controls how often training pauses for deterministic rollouts, `--eval-episodes` sets the
number of episodes per evaluation, and `--eval-weights` fixes the preference vector used for evaluation; omit the
last flag to default to equal weights. Use `--eval-render-mode window` to briefly show each evaluation in a native
viewer (capped by `--eval-render-seconds`). To archive videos instead, pass `--eval-render-mode video` together with
`--eval-render-dir <folder>` and the run will save mp4s under per-update subdirectories.

To randomize evaluation preferences at every episode, set `--eval-weight-strategy` to `dirichlet`
or `uniform` and optionally adjust `--eval-dirichlet-alpha`:

```bash
python scripts/train.py --env-id MiniGrid-Empty-5x5-v0 --total-steps 500_000 \
  --reward-config configs/minigrid_empty.py --checkpoint-path checkpoints/minigrid-empty.pth \
  --eval-interval 5 --eval-episodes 4 --eval-render-mode window \
  --eval-render-seconds 3 --eval-weight-strategy dirichlet --eval-dirichlet-alpha 0.5
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
python scripts/train.py --env-id AntMaze-UMaze-v2 --total-steps 2_000_000 \
  --reward-config configs/mujoco_antmaze.py --checkpoint-path checkpoints/antmaze.pth \
  --eval-interval 10 --eval-episodes 2 --eval-render-mode window \
  --eval-render-seconds 6 --eval-weights 0.25 0.25 0.25 0.25
```

Classic Minari point-maze datasets (e.g., `maze2d-medium-v1`) can use `configs/minari_point_maze.py` for the same directional reward vector while adapting to the point-mass observation format.

```bash
python scripts/train.py --env-id maze2d-medium-v1 --total-steps 1_000_000 \
  --reward-config configs/minari_point_maze.py --checkpoint-path checkpoints/maze2d-medium.pth \
  --eval-interval 5 --eval-episodes 2 --eval-render-mode window \
  --eval-render-seconds 4 --eval-weights 0.25 0.25 0.25 0.25
```

For lightweight grid worlds such as `MiniGrid-Empty-5x5-v0`, use `configs/minigrid_empty.py` (requires `gymnasium-minigrid`) to prefer different walls:

```bash
python scripts/train.py --env-id MiniGrid-Empty-5x5-v0 --total-steps 500_000 \
  --reward-config configs/minigrid_empty.py --checkpoint-path checkpoints/minigrid-empty.pth \
  --eval-interval 5 --eval-episodes 4 --eval-render-mode window \
  --eval-render-seconds 3 --eval-weights 0.25 0.25 0.25 0.25
```

### Interactive weight steering

```bash
python scripts/run_webui.py --checkpoint checkpoints/halfcheetah.pth \
  --env-id HalfCheetah-v4 --reward-config configs/mujoco_halfcheetah.py
```

The UI exposes sliders for each reward component and streams short rollouts generated with the latest weights.

## Notes

- Default reward vector definitions are provided for a few MuJoCo environments as a starting point. Extend `configs/` with your own components or programmatic generators.
- MOPPO trains a weight-conditioned policy; make sure to randomize preference weights during training to cover the regions you care about at inference time.
- Scripts expect Gymnasium environments. Install `gymnasium[mujoco]` to access MuJoCo control tasks.
- Evaluation video export depends on `moviepy`; installing the package (e.g., via `pip install -e .[web]`) pulls it in automatically.
- AntMaze task IDs come from `d4rl`/`minari` and require a local MuJoCo installation; install `gymnasium-robotics`, `d4rl`, and `minari`, configure MuJoCo per their READMEs, and run `python -c "import minari; minari.register_d4rl_datasets()"` once to expose the offline maze environments.
- MiniGrid examples depend on `gymnasium-minigrid`; install it separately if you want to run the empty-room command.
