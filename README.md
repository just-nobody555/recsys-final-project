# 推荐系统大作业

本仓库是推荐系统课程大作业的代码和结果文件。任务数据使用 Amazon Reviews 2023 的 5-Core 版本，实验类别包括：

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

训练、验证和测试集使用作业说明中给出的链接。代码里把 PDF 中的 `dev` 文件命名为 `valid`，内容对应同一份验证集。

## 目录说明

- `code/recommend-system-main/`：模型训练、数据处理和融合代码
- `report.docx`：实验报告，组员信息处提交前补充即可
- `final_results.csv`：三个数据集的最终测试结果
- `results_top10/`：三个数据集最终方法输出的 Top-10 推荐结果
- `evaluate_ndcg10.py`：根据 Top-10 结果复算 Recall@10 和 NDCG@10
- `models/checkpoints/`：主要实验的模型参数文件，使用 Git LFS 管理

仓库中不包含原始数据和处理后的特征文件，这些文件体积较大，按脚本重新下载和生成即可。模型参数文件保存在 `models/checkpoints/`，克隆仓库后需要执行 `git lfs pull` 下载完整 checkpoint。

## 最终结果

作业主要指标为测试集 `NDCG@10`。下面的基线是原始 `UniSRec` 结果。

| 数据集 | 最终方法 | Recall@10 | NDCG@10 | UniSRec 基线 | 绝对提升 | 相对提升 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Musical_Instruments | RRF(UniSRec+SASRec+GRU4Rec+NARM+RichUniSRec) | 0.0739 | 0.0383 | 0.0340 | +0.0043 | +12.6% |
| Industrial_and_Scientific | RRF(UniSRec+SASRec+GRU4Rec+NARM+RichUniSRec) | 0.0612 | 0.0311 | 0.0267 | +0.0044 | +16.6% |
| CDs_and_Vinyl | RRF(RichUniSRec+SASRec+GRU4Rec+NARM) | 0.1166 | 0.0553 | 0.0368 | +0.0185 | +50.1% |

三个数据集的 Macro Average `NDCG@10` 为 `0.0416`。

## 方法简述

在原始 UniSRec 的基础上，实验主要做了三点改进：

1. 增加 `SASRec`、`GRU4Rec`、`NARM` 等纯序列模型，补充不同结构的排序结果。
2. 扩展商品侧 metadata，使用 title、category、store、features、description、details、price、rating、rank 等信息。
3. 在 `RichUniSRec` 中加入商品 ID embedding、结构化 metadata embedding，以及用户历史评分、时间间隔和 recency 特征。

最后对多个模型的排序结果使用 RRF 做融合，得到最终 Top-10 推荐列表。

## 结果复算

```bash
python evaluate_ndcg10.py \
  results_top10/mi-rrf-rich-8-1-1-1-8-top10.jsonl \
  results_top10/industrial-rrf-rich-4-1-1-1-8-top10.jsonl \
  results_top10/cds-rrf-rich-fine-k60-1-0p04-0p04-0p04-top10.jsonl
```

复算结果应与 `final_results.csv` 中的 `NDCG@10` 一致。
