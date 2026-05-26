#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"

DOMAINS=(
  "Musical_Instruments"
  "Industrial_and_Scientific"
  "CDs_and_Vinyl"
)

for domain in "${DOMAINS[@]}"; do
  python scripts/download_course_data.py --domain "$domain"

  python dataset/process_amazon.py \
    --domain "$domain" \
    --device "$DEVICE" \
    --output_dir dataset/processed

  python dataset/process_amazon.py \
    --domain "$domain" \
    --device "$DEVICE" \
    --output_dir dataset/processed \
    --output_domain "${domain}_Rich" \
    --rich_metadata \
    --with_user_features
done
