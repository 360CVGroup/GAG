#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-GAG_adjuvant_sft}"
export WANDB_NAME="${WANDB_NAME:-adjuvant_qwen3_1p7b_sft}"
export CONFIG_PATH="${CONFIG_PATH:-config/domain_adaptation/adjuvant_expert_sft_rebalanced_bertscore.yaml}"

EXTRA_ARGS=()
if [[ -n "${OUTPUT_DIR_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR_OVERRIDE}")
fi
if [[ -n "${TRAIN_PATH_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--train_path "${TRAIN_PATH_OVERRIDE}")
fi
if [[ -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--model_name_or_path "${MODEL_PATH_OVERRIDE}")
fi

REPORT_TO="none"
if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}"
  REPORT_TO="wandb"
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.domain_adaptation.train_domain_expert_sft \
  --config "${CONFIG_PATH}" \
  --report_to "${REPORT_TO}" \
  "${EXTRA_ARGS[@]}"
