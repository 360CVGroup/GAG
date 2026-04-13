#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-GAG_material_sft}"
export WANDB_NAME="${WANDB_NAME:-material_qwen3_1p7b_sft}"

REPORT_TO="none"
if [[ -n "${WANDB_API_KEY}" ]]; then
  wandb login --relogin "${WANDB_API_KEY}"
  REPORT_TO="wandb"
fi

python -m torch.distributed.run --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m src.domain_adaptation.train_domain_expert_sft \
  --config config/domain_adaptation/material_expert_sft.yaml \
  --report_to "${REPORT_TO}"
