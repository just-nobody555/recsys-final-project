# 推荐系统实验代码

## 环境安装

```bash
pip install -r requirements.txt
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

普通 metadata：

```bash
python dataset/process_amazon.py --domain Musical_Instruments --device cuda
```

rich metadata：

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
python run.py -m SASRec -d Musical_Instruments --show_progress false
python run.py -m GRU4Rec -d Musical_Instruments --show_progress false
python run.py -m NARM -d Musical_Instruments --show_progress false
python run.py -m RichUniSRec -d Musical_Instruments_Rich --show_progress false
```

模型参数主要在 `config/` 目录中调整。

## 排序导出和融合

导出模型排序结果：

```bash
python scripts/export_ranklist.py --model UniSRec --dataset Musical_Instruments --checkpoint checkpoints/best.pth --output results/unisrec-ranklist.jsonl
```

RRF 融合：

```bash
python scripts/rrf_fusion.py results/model1.jsonl results/model2.jsonl --weights 1,1 --output-ranklist results/fused.jsonl
```
