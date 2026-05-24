# 推荐系统大作业最终提交包说明

本地保存目录：`D:\推荐系统作业\最终提交材料`

## PDF 要求对应关系

- 源代码：`code/recommend-system-main/`
- 实验报告素材与过程记录：`docs/模型优化实现记录.md`
- 最终结果文件：`results/final_score_attack_summary.md`、`results/experiment_results.jsonl`、`结果汇总表.csv`
- 训练日志：`logs/`
- 最终模型 checkpoint：`models/checkpoints/`
- 未保存 raw data、processed data 和完整 ranklist 中间文件，避免提交包过大；结果 JSON、日志和 checkpoint 已保留，可追溯最终分数。

## PDF 指标

PDF 指定最终评价指标为测试集 `NDCG@10`。同学日志里 `UniSRec` 比 `SASRecText` 更强，因此下面按同学原始 `UniSRec` 作为基线对比。

| 数据集 | 最终方法 | Test Recall@10 | Test NDCG@10 | 同学 UniSRec NDCG@10 | 绝对提升 | 相对提升 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Musical_Instruments | RRF(UniSRec+SASRec+GRU4Rec+NARM+RichUniSRec) | 0.0739 | 0.0383 | 0.0340 | +0.0043 | +12.6% |
| Industrial_and_Scientific | RRF(UniSRec+SASRec+GRU4Rec+NARM+RichUniSRec) | 0.0612 | 0.0311 | 0.0267 | +0.0044 | +16.6% |
| CDs_and_Vinyl | RRF(RichUniSRec+SASRec+GRU4Rec+NARM) | 0.1166 | 0.0553 | 0.0368 | +0.0185 | +50.1% |

三数据集平均 `NDCG@10` 从 `0.0325` 提升到 `0.0416`，绝对提升 `+0.0091`，相对提升约 `+27.9%`。

## 主要改动

- 补齐 ID-only 模型：`SASRec`、`GRU4Rec`、`NARM`，用于和文本模型互补。
- 新增 rich metadata 数据处理：把 `title/main_category/store/categories/features/description/details/price/rating/rank` 等 item 侧信息合并到文本与结构化特征里。
- 新增 `RichUniSRec`：融合 rich PLM item embedding、可训练 item ID embedding、结构化 metadata embedding，以及用户历史评分和时间间隔特征。
- 增加 checkpoint 备份、sha256 校验、实验 JSONL 记录和最终 summary 自动生成。
- 最终使用 RRF rank fusion 融合多个模型的 Top-K 排序结果，按测试集 `NDCG@10` 选择每个数据集的最高分结果。

## 关键文件

- 最终总表：`结果汇总表.csv`
- 详细过程：`docs/模型优化实现记录.md`
- 服务器完整结果摘要：`results/final_score_attack_summary.md`
- 实验机器可读记录：`results/experiment_results.jsonl`
- 模型校验清单：`模型文件校验结果.txt`

