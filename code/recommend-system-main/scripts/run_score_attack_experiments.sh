#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results checkpoints ranklists

run_exp() {
  local tag="$1"
  shift
  echo "[$(date '+%F %T')] START ${tag}"
  python run.py "$@" \
    --tag "${tag}" \
    --result_file results/experiment_results.jsonl \
    --checkpoint_dir checkpoints \
    --show_progress false 2>&1 | tee "logs/${tag}.log"
  python scripts/summarize_results.py results/experiment_results.jsonl \
    --sort 'test_ndcg@10' \
    --write-score-attack results/final_score_attack_summary.md > "logs/${tag}.summary.log"
  echo "[$(date '+%F %T')] END ${tag}"
}

# Smoke tests: keep these cheap. Uncomment/run one at a time if debugging.
# run_exp mi-sasrec-smoke -m SASRec -d Musical_Instruments --epochs 1 --stopping_step 1
# run_exp mi-gru4rec-smoke -m GRU4Rec -d Musical_Instruments --epochs 1 --stopping_step 1
# run_exp mi-narm-smoke -m NARM -d Musical_Instruments --epochs 1 --stopping_step 1

# Main score-attack candidates.
# run_exp mi-sasrec-main -m SASRec -d Musical_Instruments --epochs 180 --stopping_step 15
# run_exp mi-gru4rec-main -m GRU4Rec -d Musical_Instruments --epochs 180 --stopping_step 15
# run_exp mi-narm-main -m NARM -d Musical_Instruments --epochs 180 --stopping_step 15
