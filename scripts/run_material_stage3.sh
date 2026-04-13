#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-GAG_material_stage3}"
export WANDB_NAME="${WANDB_NAME:-material_gag}"

CONFIG_PATH="${CONFIG_PATH:-config/gag_train/stageII_material_gag.yaml}"
EXTRA_ARGS=()

if [[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR_OVERRIDE}")
fi
if [[ -n "${TRAIN_FILE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--train_file "${TRAIN_FILE_OVERRIDE}")
fi
if [[ "${DISABLE_DEV_FILE:-false}" == "true" ]]; then
  EXTRA_ARGS+=(--dev_file "null")
elif [[ -n "${DEV_FILE_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--dev_file "${DEV_FILE_OVERRIDE}")
fi
if [[ -n "${MAX_TRAIN_STEPS_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS_OVERRIDE}")
fi

REPORT_TO_WANDB="False"
if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}"
  REPORT_TO_WANDB="True"
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.language_modeling.train \
  --config "${CONFIG_PATH}" \
  --report_to_wandb "${REPORT_TO_WANDB}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_NAME}" \
  "${EXTRA_ARGS[@]}"
