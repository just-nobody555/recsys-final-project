推荐系统大作业提交说明

文件说明：
1. code/recommend-system-main/
   源代码。

2. report.docx
   实验报告。

3. final_results.csv
   三个数据集的最终测试结果汇总。

4. results_top10/
   三个数据集最终方法输出的 Top-10 推荐结果。

5. evaluate_ndcg10.py
   根据 results_top10 中的结果复算 Recall@10 和 NDCG@10。

复算方式：
python evaluate_ndcg10.py results_top10/mi-rrf-rich-8-1-1-1-8-top10.jsonl results_top10/industrial-rrf-rich-4-1-1-1-8-top10.jsonl results_top10/cds-rrf-rich-fine-k60-1-0p04-0p04-0p04-top10.jsonl

最终 test NDCG@10：
Musical_Instruments: 0.0383
Industrial_and_Scientific: 0.0311
CDs_and_Vinyl: 0.0553
Macro Average: 0.0416

说明：
checkpoint 文件体积较大，未放入邮件附件。GitHub 仓库的 models/checkpoints/ 目录保留了主要模型参数，使用 Git LFS 管理，克隆后执行 git lfs pull 即可下载完整 checkpoint。

完整复现：
git clone https://github.com/just-nobody555/recsys-final-project.git
cd recsys-final-project
git lfs pull
然后进入 code/recommend-system-main/，安装 requirements.txt 中的依赖；使用 scripts/download_course_data.py 下载作业 PDF 中的三类数据，并用 dataset/process_amazon.py 处理数据。模型配置在 config/，checkpoint 在 models/checkpoints/。

使用 checkpoint 重新导出最终结果：
cd code/recommend-system-main
bash scripts/reproduce_from_checkpoints.sh
