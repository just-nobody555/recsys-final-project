#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs results

run_exp() {
  local tag="$1"
  shift
  echo "[$(date '+%F %T')] start ${tag}"
  python run.py "$@" --tag "${tag}" --result_file results/experiment_results.jsonl 2>&1 | tee "logs/${tag}.log"
  echo "[$(date '+%F %T')] done ${tag}"
}

run_exp smoke-mi -m UniSRec -d Musical_Instruments --epochs 3 --stopping_step 2 --show_progress false

# Uncomment one experiment at a time after the smoke run passes.
# run_exp mi-baseline -m UniSRec -d Musical_Instruments --show_progress false
# run_exp mi-transductive -m UniSRec -d Musical_Instruments --train_stage transductive_ft --show_progress false
# run_exp mi-temp003 -m UniSRec -d Musical_Instruments --temperature 0.03 --show_progress false
# run_exp mi-temp007 -m UniSRec -d Musical_Instruments --temperature 0.07 --show_progress false
# run_exp mi-lr5e5 -m UniSRec -d Musical_Instruments --learning_rate 0.00005 --show_progress false
# run_exp mi-drop02 -m UniSRec -d Musical_Instruments --hidden_dropout_prob 0.2 --attn_dropout_prob 0.2 --show_progress false
