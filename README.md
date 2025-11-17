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
  --reward-config configs/mujoco_halfcheetah.py --checkpoint-path checkpoints/halfcheetah.pth
```

### Evaluation with fixed weights

```bash
python scripts/evaluate.py --checkpoint checkpoints/halfcheetah.pth \
  --env-id HalfCheetah-v4 --reward-config configs/mujoco_halfcheetah.py \
  --weights 0.4 0.4 0.2
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
