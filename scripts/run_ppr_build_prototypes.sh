#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENCODER_NAME_OR_PATH="${ENCODER_NAME_OR_PATH:-models/Qwen3-1.7B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ppr/prototypes}"
NUM_PROTOTYPES="${NUM_PROTOTYPES:-32}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-256}"

mkdir -p "${OUTPUT_DIR}"

python -m src.ppr.build_domain_prototypes \
  --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
  --input_jsonl datasets/mixed_domain/offline/general.jsonl \
  --output_path "${OUTPUT_DIR}/general.pt" \
  --domain_name general \
  --num_prototypes "${NUM_PROTOTYPES}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}"

python -m src.ppr.build_domain_prototypes \
  --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
  --input_jsonl datasets/mixed_domain/offline/adjuvant.jsonl \
  --output_path "${OUTPUT_DIR}/adjuvant.pt" \
  --domain_name adjuvant \
  --num_prototypes "${NUM_PROTOTYPES}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}"

python -m src.ppr.build_domain_prototypes \
  --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
  --input_jsonl datasets/mixed_domain/offline/materials.jsonl \
  --output_path "${OUTPUT_DIR}/materials.pt" \
  --domain_name materials \
  --num_prototypes "${NUM_PROTOTYPES}" \
  --max_samples "${MAX_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --max_seq_length "${MAX_SEQ_LENGTH}"

