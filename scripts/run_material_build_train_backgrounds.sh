#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-8}"
CLEAN_SHARDS="${CLEAN_SHARDS:-false}"
CONFIG_PATH="config/data_pipeline/build_material_train_backgrounds.yaml"
OUTPUT_PATH_OVERRIDE="${OUTPUT_PATH_OVERRIDE:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"

extra_args=()
if [[ -n "${OUTPUT_PATH_OVERRIDE}" ]]; then
  extra_args+=(--output_path "${OUTPUT_PATH_OVERRIDE}")
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  extra_args+=(--max_samples "${MAX_SAMPLES}")
fi
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  extra_args+=(--max_new_tokens "${MAX_NEW_TOKENS}")
fi

for ((rank=0; rank<NUM_GPUS; rank++)); do
  CUDA_VISIBLE_DEVICES="${rank}" \
    python -m src.data_pipeline.build_background_embeddings \
    --config "${CONFIG_PATH}" \
    --num_shards "${NUM_GPUS}" \
    --shard_rank "${rank}" \
    "${extra_args[@]}" &
done
wait

python -m src.data_pipeline.build_background_embeddings \
  --config "${CONFIG_PATH}" \
  --num_shards "${NUM_GPUS}" \
  --merge_shards true \
  --clean_merged_shards "${CLEAN_SHARDS}" \
  "${extra_args[@]}"
