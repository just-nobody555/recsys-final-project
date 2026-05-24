set -e
cd /root/recsys_project/recommend-system-main
run_one() {
  tag="$1"; shift
  echo "===== START $tag $(date '+%F %T') ====="
  /root/miniconda3/bin/python run.py "$@" --tag "$tag" --result_file results/experiment_results.jsonl --checkpoint_dir checkpoints --show_progress false > "logs/${tag}.log" 2>&1
  /root/miniconda3/bin/python scripts/summarize_results.py results/experiment_results.jsonl --sort 'test_ndcg@10' --write-score-attack results/final_score_attack_summary.md > "logs/${tag}.summary.log" 2>&1
  echo "===== END $tag $(date '+%F %T') ====="
}
run_one cds-gru4rec-lr5e4 -m GRU4Rec -d CDs_and_Vinyl --epochs 180 --stopping_step 15 --learning_rate 0.0005
run_one cds-narm-main -m NARM -d CDs_and_Vinyl --epochs 180 --stopping_step 15
