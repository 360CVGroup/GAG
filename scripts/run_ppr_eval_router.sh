#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENCODER_NAME_OR_PATH="${ENCODER_NAME_OR_PATH:-models/Qwen3-1.7B}"
EVAL_JSONL="${EVAL_JSONL:-datasets/mixed_domain/online/general_and_adjuvant_and_materials.jsonl}"
OUTPUT_REPORT="${OUTPUT_REPORT:-outputs/ppr/router_eval/general_materials_adjuvant_report.json}"
PROTOTYPE_DIR="${PROTOTYPE_DIR:-outputs/ppr/prototypes}"

mkdir -p "$(dirname "${OUTPUT_REPORT}")"

python -m src.eval.ppr.eval_ppr_accuracy \
  --encoder_name_or_path "${ENCODER_NAME_OR_PATH}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --general_proto "${PROTOTYPE_DIR}/general.pt" \
  --adjuvant_proto "${PROTOTYPE_DIR}/adjuvant.pt" \
  --materials_proto "${PROTOTYPE_DIR}/materials.pt" \
  --output_report "${OUTPUT_REPORT}"
