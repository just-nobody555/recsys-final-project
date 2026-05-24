#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f /etc/network_turbo ]; then
  # shellcheck disable=SC1091
  source /etc/network_turbo
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p logs results saved dataset/data dataset/processed

python -m pip install -U pip
python -m pip install -r requirements.txt

python scripts/check_environment.py

cat <<'EOF'

Next commands:

1. Put raw Amazon files under dataset/data/{domain}/:
   train.csv.gz, valid.csv.gz, test.csv.gz, item_metadata.jsonl.gz
   Or run:
   python scripts/download_course_data.py --domain Musical_Instruments

2. Process one cheap domain first:
   python dataset/process_amazon.py --domain Musical_Instruments --device cuda --output_dir dataset/processed

3. Verify processed files:
   python scripts/check_dataset.py --domain Musical_Instruments

4. Run a smoke experiment:
   python run.py -m UniSRec -d Musical_Instruments --epochs 3 --stopping_step 2 --show_progress false --tag smoke-mi
EOF
