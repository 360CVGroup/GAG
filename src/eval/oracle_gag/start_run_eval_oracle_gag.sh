#!/usr/bin/env bash
set -euo pipefail

# # 进入 GAG 项目根目录（脚本放在哪都能跑）
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# 确保能 import src.*
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "[$(date '+%F %T')] Step 1/3: Generate background embeddings on dev..."
python src/eval/oracle_gag/get_background_embedding_on_dev.py

echo "[$(date '+%F %T')] Step 2/3: Run oracle_gag eval (LLM generation)..."
python -m src.eval.oracle_gag.run_eval \
  --config config/oracle_gag_eval/run_eval_oracle.yaml

echo "[$(date '+%F %T')] Step 3/3: Calculate BERTScore metrics on dev..."
python src/eval/oracle_gag/calculate_metrics_on_dev.py

echo "[$(date '+%F %T')] ✅ All steps completed successfully."
