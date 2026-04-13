#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/gag_eval/material_generation_metrics.yaml}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-64}"
STS_BATCH_SIZE="${STS_BATCH_SIZE:-128}"

python -m src.eval.compute_generation_metrics \
  --config "${CONFIG_PATH}" \
  --bertscore_batch_size "${BERTSCORE_BATCH_SIZE}" \
  --sts_batch_size "${STS_BATCH_SIZE}"
