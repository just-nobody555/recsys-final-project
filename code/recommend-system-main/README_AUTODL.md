# AutoDL Runbook

## 1. Recommended Image

Use an AutoDL official PyTorch image:

- First choice: `PyTorch 2.1.0 / Python 3.10 / CUDA 12.1`
- Fallback: `PyTorch 2.1.2 / Python 3.10 / CUDA 11.8`

Avoid Python 3.12 images for this project because RecBole dependencies are easier to break there.

## 2. Network Acceleration

Run this after SSH login:

```bash
source /etc/network_turbo
export HF_ENDPOINT=https://hf-mirror.com
```

Turn it off when it is no longer needed:

```bash
unset http_proxy && unset https_proxy
```

## 3. Setup

```bash
mkdir -p /root/recsys_project /root/autodl-tmp/recsys_data
cd /root/recsys_project
unzip recommend-system-main.zip
cd recommend-system-main
bash scripts/autodl_quickstart.sh
```

## 4. Data Processing

Put these files under `dataset/data/{domain}/`:

- `train.csv.gz`
- `valid.csv.gz`
- `test.csv.gz`
- `item_metadata.jsonl.gz`

Or download and rename the course files automatically:

```bash
python scripts/download_course_data.py --domain Musical_Instruments
```

Process the cheapest domain first:

```bash
python dataset/process_amazon.py --domain Musical_Instruments --device cuda --output_dir dataset/processed
python scripts/check_dataset.py --domain Musical_Instruments
```

## 5. Budgeted Experiments

Start with a smoke run:

```bash
python run.py -m UniSRec -d Musical_Instruments --epochs 3 --stopping_step 2 --show_progress false --tag smoke-mi
```

Then run one controlled experiment at a time:

```bash
python run.py -m UniSRec -d Musical_Instruments --train_stage transductive_ft --show_progress false --tag mi-transductive
python run.py -m UniSRec -d Musical_Instruments --temperature 0.03 --show_progress false --tag mi-temp003
python run.py -m UniSRec -d Musical_Instruments --temperature 0.07 --show_progress false --tag mi-temp007
python run.py -m UniSRec -d Musical_Instruments --learning_rate 0.00005 --show_progress false --tag mi-lr5e5
python run.py -m UniSRec -d Musical_Instruments --hidden_dropout_prob 0.2 --attn_dropout_prob 0.2 --show_progress false --tag mi-drop02
```

Summarize results:

```bash
python scripts/summarize_results.py
```
