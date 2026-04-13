#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/gag_eval/run_eval_adjuvant_rebalanced_bertscore.yaml}"

python -m src.eval.oracle_gag.run_eval --config "${CONFIG_PATH}"
