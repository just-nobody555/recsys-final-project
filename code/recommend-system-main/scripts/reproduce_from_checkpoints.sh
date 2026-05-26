#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-../../models/checkpoints}"
OUT_DIR="${OUT_DIR:-results/checkpoint_reproduce}"
mkdir -p "$OUT_DIR"

export_ranklist() {
  local tag="$1"
  local model="$2"
  local dataset="$3"
  local checkpoint="$4"
  shift 4

  python scripts/export_ranklist.py \
    -m "$model" \
    -d "$dataset" \
    -p "$checkpoint" \
    --split test \
    --topk 200 \
    --output "$OUT_DIR/${tag}.jsonl" \
    --summary "$OUT_DIR/${tag}.summary.json" \
    "$@"
}

echo "Export Musical_Instruments ranklists"
export_ranklist mi-unisrec UniSRec Musical_Instruments "$CKPT_DIR/mi-transductive-lr5e5-best.pth" --train_stage transductive_ft --show_progress false
export_ranklist mi-sasrec SASRec Musical_Instruments "$CKPT_DIR/mi-sasrec-main-best.pth" --show_progress false
export_ranklist mi-gru4rec GRU4Rec Musical_Instruments "$CKPT_DIR/mi-gru4rec-main-best.pth" --show_progress false
export_ranklist mi-narm NARM Musical_Instruments "$CKPT_DIR/mi-narm-main-best.pth" --show_progress false
export_ranklist mi-rich RichUniSRec Musical_Instruments_Rich "$CKPT_DIR/mi-richunisrec-main-best.pth" --show_progress false

python scripts/rrf_fusion.py \
  "$OUT_DIR/mi-unisrec.jsonl" \
  "$OUT_DIR/mi-sasrec.jsonl" \
  "$OUT_DIR/mi-gru4rec.jsonl" \
  "$OUT_DIR/mi-narm.jsonl" \
  "$OUT_DIR/mi-rich.jsonl" \
  --weights 8,1,1,1,8 \
  --rrf-k 60 \
  --topk 200 \
  --output-ranklist "$OUT_DIR/mi-final.jsonl" \
  --summary "$OUT_DIR/mi-final.summary.json" \
  --tag mi-final \
  --dataset Musical_Instruments

echo "Export Industrial_and_Scientific ranklists"
export_ranklist industrial-unisrec UniSRec Industrial_and_Scientific "$CKPT_DIR/industrial-inductive-baseline-best.pth" --show_progress false
export_ranklist industrial-sasrec SASRec Industrial_and_Scientific "$CKPT_DIR/industrial-sasrec-main-best.pth" --show_progress false
export_ranklist industrial-gru4rec GRU4Rec Industrial_and_Scientific "$CKPT_DIR/industrial-gru4rec-main-best.pth" --show_progress false
export_ranklist industrial-narm NARM Industrial_and_Scientific "$CKPT_DIR/industrial-narm-main-best.pth" --show_progress false
export_ranklist industrial-rich RichUniSRec Industrial_and_Scientific_Rich "$CKPT_DIR/industrial-richunisrec-main-best.pth" --show_progress false

python scripts/rrf_fusion.py \
  "$OUT_DIR/industrial-unisrec.jsonl" \
  "$OUT_DIR/industrial-sasrec.jsonl" \
  "$OUT_DIR/industrial-gru4rec.jsonl" \
  "$OUT_DIR/industrial-narm.jsonl" \
  "$OUT_DIR/industrial-rich.jsonl" \
  --weights 4,1,1,1,8 \
  --rrf-k 60 \
  --topk 200 \
  --output-ranklist "$OUT_DIR/industrial-final.jsonl" \
  --summary "$OUT_DIR/industrial-final.summary.json" \
  --tag industrial-final \
  --dataset Industrial_and_Scientific

echo "Export CDs_and_Vinyl ranklists"
export_ranklist cds-rich RichUniSRec CDs_and_Vinyl_Rich "$CKPT_DIR/cds-richunisrec-main-best.pth" --show_progress false
export_ranklist cds-sasrec SASRec CDs_and_Vinyl "$CKPT_DIR/cds-sasrec-main-best.pth" --show_progress false
export_ranklist cds-gru4rec GRU4Rec CDs_and_Vinyl "$CKPT_DIR/cds-gru4rec-main-best.pth" --show_progress false
export_ranklist cds-narm NARM CDs_and_Vinyl "$CKPT_DIR/cds-narm-main-best.pth" --show_progress false

python scripts/rrf_fusion.py \
  "$OUT_DIR/cds-rich.jsonl" \
  "$OUT_DIR/cds-sasrec.jsonl" \
  "$OUT_DIR/cds-gru4rec.jsonl" \
  "$OUT_DIR/cds-narm.jsonl" \
  --weights 1,0.04,0.04,0.04 \
  --rrf-k 60 \
  --topk 200 \
  --output-ranklist "$OUT_DIR/cds-final.jsonl" \
  --summary "$OUT_DIR/cds-final.summary.json" \
  --tag cds-final \
  --dataset CDs_and_Vinyl

python ../../evaluate_ndcg10.py \
  "$OUT_DIR/mi-final.jsonl" \
  "$OUT_DIR/industrial-final.jsonl" \
  "$OUT_DIR/cds-final.jsonl"
