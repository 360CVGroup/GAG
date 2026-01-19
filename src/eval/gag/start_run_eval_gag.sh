#!/usr/bin/env bash
set -euo pipefail

# # 确保从 GAG 项目根目录运行（脚本放在哪都能跑）
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# GAG_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# cd "$GAG_ROOT"

# 确保能 import src.*
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Step 1/2: Run GAG eval (multi-domain generation)..."
python -m src.eval.gag.gag_run_eval \
  --config config/gag_eval/gag_run_eval_3domains.yaml

echo "[$(date '+%F %T')] Step 2/2: Calculate mixed-domain dataset metrics..."
python src/eval/gag/calculate_mixed_domain_dataset_metrics.py

echo "[$(date '+%F %T')] ✅ All steps completed successfully."
