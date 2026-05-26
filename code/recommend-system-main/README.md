# 推荐系统实验代码

## 环境安装

```bash
pip install -r requirements.txt
python scripts/check_environment.py
```

建议使用 Python 3.10、PyTorch 2.x 和 CUDA 环境运行。

## 数据准备

下载作业说明中给出的 Amazon Reviews 2023 5-Core 数据：

```bash
python scripts/download_course_data.py --domain Musical_Instruments
python scripts/download_course_data.py --domain Industrial_and_Scientific
python scripts/download_course_data.py --domain CDs_and_Vinyl
```

下载后的文件会放在 `dataset/data/{domain}/` 下，包括：

- `train.csv.gz`
- `valid.csv.gz`
- `test.csv.gz`
- `item_metadata.jsonl.gz`

其中 `valid.csv.gz` 对应作业 PDF 里的 `dev` 文件。

## 数据处理

普通数据：

```bash
python dataset/process_amazon.py --domain Musical_Instruments --device cuda
```

带 metadata 和历史侧字段的数据：

```bash
python dataset/process_amazon.py \
  --domain Musical_Instruments \
  --device cuda \
  --output_dir dataset/processed \
  --output_domain Musical_Instruments_Rich \
  --rich_metadata \
  --with_user_features
```

处理完成后可检查数据：

```bash
python scripts/check_dataset.py --domain Musical_Instruments
python scripts/check_rich_dataset.py --domain Musical_Instruments_Rich
```

## 模型训练

```bash
python run.py -m UniSRec -d Musical_Instruments --show_progress false
python run.py -m RichUniSRec -d Musical_Instruments_Rich --show_progress false
```

模型配置位于 `config/` 目录。

## 使用 checkpoint 复现最终结果

完整 checkpoint 位于仓库根目录的 `models/checkpoints/`。完成数据下载和处理后，在本目录运行：

```bash
bash scripts/reproduce_from_checkpoints.sh
```

脚本会加载 `models/checkpoints/*.pth`，重新导出三个数据集的 test ranklist，并调用仓库根目录的 `evaluate_ndcg10.py` 复算 Recall@10 和 NDCG@10。
