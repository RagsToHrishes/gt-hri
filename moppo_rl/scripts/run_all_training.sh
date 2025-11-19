#!/usr/bin/env bash

# Sequentially launch training runs for the standard Mujoco configs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIGS=(
  "train_cheetah"
  "train_hopper"
  "train_ant"
  "train_humanoid"
)

if [[ $# -gt 0 ]]; then
  echo "[info] Extra Hydra overrides will be forwarded to each run: $*"
fi

for config_name in "${CONFIGS[@]}"; do
  config_file="${PROJECT_ROOT}/configs/${config_name}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "[warn] Missing config ${config_file}; skipping ${config_name}" >&2
    continue
  fi
  echo "============================================================"
  echo "Starting training run for ${config_name} (config: ${config_file})"
  echo "============================================================"
  python3 "${SCRIPT_DIR}/train.py" --config-path "${PROJECT_ROOT}/configs" --config-name "${config_name}" "$@"
done
