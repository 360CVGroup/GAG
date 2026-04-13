#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-config/ppr/mixed_domain_router.yaml}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/mixed_domain/ppr/routed_eval_general_materials_adjuvant.jsonl}"
METRICS_OUTPUT_FILE="${METRICS_OUTPUT_FILE:-outputs/mixed_domain/ppr/routed_eval_general_materials_adjuvant_with_metrics.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-outputs/mixed_domain/ppr/routed_eval_general_materials_adjuvant_summary.json}"
SUMMARY_TSV="${SUMMARY_TSV:-outputs/mixed_domain/ppr/routed_eval_general_materials_adjuvant_summary.tsv}"
BERTSCORE_BATCH_SIZE="${BERTSCORE_BATCH_SIZE:-64}"

python -m src.eval.ppr.run_routed_eval --config "${CONFIG_PATH}"

python -m src.eval.ppr.calculate_mixed_domain_metrics \
  --input_file "${OUTPUT_FILE}" \
  --output_file "${METRICS_OUTPUT_FILE}" \
  --summary_json "${SUMMARY_JSON}" \
  --summary_tsv "${SUMMARY_TSV}" \
  --bertscore_batch_size "${BERTSCORE_BATCH_SIZE}"
