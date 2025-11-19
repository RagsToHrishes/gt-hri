#!/usr/bin/env bash

# Launch training runs in parallel for the standard Mujoco configs.

set -euo pipefail

SCRIPT_DIR="/home/hrish/hri/moppo_rl/scripts"
PROJECT_ROOT="/home/hrish/hri/moppo_rl"

CONFIGS=(
  # "train_cheetah"
  # "train_hopper"
  # "train_ant"
  # "train_humanoid"
)

if [[ $# -gt 0 ]]; then
  echo "[info] Extra Hydra overrides will be forwarded to each run: $*"
fi

BASE_TENSORBOARD_PORT="${TENSORBOARD_PORT_BASE:-16006}"
user_tensorboard_override=false
for arg in "$@"; do
  case "$arg" in
    tensorboard_port=*|--tensorboard-port=*|--tensorboard_port=*)
      user_tensorboard_override=true
      break
      ;;
    --tensorboard-port|--tensorboard_port)
      user_tensorboard_override=true
      break
      ;;
  esac
done

pids=()
names=()
tb_ports=()
auto_port_offset=0

for idx in "${!CONFIGS[@]}"; do
  config_name="${CONFIGS[$idx]}"
  config_file="${PROJECT_ROOT}/configs/${config_name}.yaml"
  if [[ ! -f "${config_file}" ]]; then
    echo "[warn] Missing config ${config_file}; skipping ${config_name}" >&2
    continue
  fi
  tb_port=""
  tb_args=()
  if [[ "${user_tensorboard_override}" == false ]]; then
    tb_port=$((BASE_TENSORBOARD_PORT + auto_port_offset))
    tb_args=(--tensorboard-port "${tb_port}")
    auto_port_offset=$((auto_port_offset + 1))
  fi
  echo "============================================================"
  if [[ -n "${tb_port}" ]]; then
    echo "Starting training run for ${config_name} (config: ${config_file}, TensorBoard port: ${tb_port})"
  else
    echo "Starting training run for ${config_name} (config: ${config_file})"
  fi
  echo "============================================================"
  python3 "${SCRIPT_DIR}/train.py" --config-path "../configs" --config-name "${config_name}" "$@" "${tb_args[@]}" &
  pids+=($!)
  names+=("${config_name}")
  tb_ports+=("${tb_port}")
done

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "[warn] No training jobs were launched."
  exit 0
fi

status=0
for idx in "${!pids[@]}"; do
  pid=${pids[$idx]}
  name=${names[$idx]}
  port=${tb_ports[$idx]}
  if [[ -n "${port}" ]]; then
    port_note=" (TensorBoard port ${port})"
  else
    port_note=""
  fi
  if wait "${pid}"; then
    echo "[info] Training for ${name}${port_note} completed successfully."
  else
    echo "[error] Training for ${name}${port_note} failed." >&2
    status=1
  fi
done

exit ${status}
