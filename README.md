# 推荐系统大作业

本仓库包含推荐系统大作业的源代码、结果文件和实验报告。数据使用作业说明中的 Amazon Reviews 2023 5-Core 版本，实验类别为：

- `Industrial_and_Scientific`
- `Musical_Instruments`
- `CDs_and_Vinyl`

训练集、验证集和测试集使用作业 PDF 中给出的链接。代码中将 PDF 里的 `dev` 文件命名为 `valid`，内容对应同一份验证集。

## 目录说明

- `code/recommend-system-main/`：数据处理、模型训练和结果导出代码
- `report.docx`：实验报告
- `final_results.csv`：三个数据集的最终测试结果汇总
- `results_top10/`：三个数据集最终输出的 Top-10 推荐结果
- `evaluate_ndcg10.py`：根据 Top-10 结果复算 Recall@10 和 NDCG@10
- `models/checkpoints/`：主要模型参数文件，使用 Git LFS 管理

仓库中不包含原始数据和处理后的特征文件，这些文件可按脚本重新下载和生成。模型参数保存在 `models/checkpoints/`，克隆仓库后需要执行 `git lfs pull` 下载完整 checkpoint。

## 最终结果

作业主要指标为测试集 `NDCG@10`。

| 数据集 | 测试样本数 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Musical_Instruments | 57439 | 0.0739 | 0.0383 |
| Industrial_and_Scientific | 50985 | 0.0612 | 0.0311 |
| CDs_and_Vinyl | 123876 | 0.1166 | 0.0553 |
| Macro Average | 232300 | 0.0839 | 0.0416 |
| Weighted Average | 232300 |  | 0.0458 |

## 结果复算

```bash
python evaluate_ndcg10.py \
  results_top10/Musical_Instruments_top10.jsonl \
  results_top10/Industrial_and_Scientific_top10.jsonl \
  results_top10/CDs_and_Vinyl_top10.jsonl
```

复算结果应与 `final_results.csv` 中的 `Recall@10` 和 `NDCG@10` 一致。

## 使用 checkpoint 复现

邮件提交包不包含 checkpoint。完整模型参数保存在 GitHub 仓库的 `models/checkpoints/` 中，使用 Git LFS 管理：

```bash
git clone https://github.com/just-nobody555/recsys-final-project.git
cd recsys-final-project
git lfs install
git lfs pull
```

之后进入 `code/recommend-system-main/`，安装依赖，下载并处理作业 PDF 中的三类数据：

```bash
cd code/recommend-system-main
pip install -r requirements.txt
python scripts/check_environment.py
python scripts/download_course_data.py --domain Musical_Instruments
python scripts/download_course_data.py --domain Industrial_and_Scientific
python scripts/download_course_data.py --domain CDs_and_Vinyl
```

数据处理完成后，加载已训练 checkpoint 重新导出最终结果：

```bash
bash scripts/reproduce_from_checkpoints.sh
```
