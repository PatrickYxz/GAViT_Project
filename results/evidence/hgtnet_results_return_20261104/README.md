# HGTNet结果包离线核验（2026-11-04）

状态：**DOWNLOADED_RESULTS_METRICS_HISTORY_LOCALLY_RECOMPUTED**。本轮只分析已完成实验，没有新训练、GPU评估或模型/配方修改。

- 原上传文件为gzip tar，544296字节，SHA256为`4fc4d51cec3aac5900ac94cdde919818ba58cd0f76add625867623f54d052477`。[收据](archive_receipt.json)保存七个预期普通文件的身份；`reported_results.tar.gz`保留原包字节。显式限制名称、类型和大小，没有使用extractall或执行包内内容。
- [原始七文件](files/)包括test/history/summary/binding/initialization/checkpoint_reload/runtime。test为1364244字节，SHA256为`869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae`，与先前完成status一致。
- [主分析](analyze_results.py)、[742项检查](validation.json)和[运行摘要](analysis_stdout.json)：核对25200条目录真值/绑定标签/预测、全部45类指标与2025个混淆单元、195条history和100个验证混淆矩阵，以及来源/runtime/初值/重载metadata。OA94.11904761904762%、macro-F194.10529172930384%，23718正确、1482错误，两项test指标复算误差均为0。
- [独立预测审查](review_predictions/README.md)通过131/131检查，采用精确有理数及另一种F1公式。DF与HGT路径/类别/manifest引用一致；双方正确21303、HGT纠正2415、HGT退化503、双方错误979。净增加1912；没有从相对GAViT净4张推断逐图配对。
- [独立历史审查](review_history/README.md)通过498/498检查：100轮calibration、95轮fresh refit、68930次更新；唯一最佳95轮，完整195轮耗时及100个val混淆矩阵均保留。末轮OA92.46031746%，比最佳93.09523810%少正确8张。完整runtime和24个源码绑定项符合实际交付v1包；这不等于原模型/原图字节验证。
- [图表说明](figures/README.md)：四组各PNG/SVG/PDF，完整训练历史、45类及全混淆矩阵保留，双向混淆不删反方向。26/26图表检查通过，四份PDF均经渲染和视觉检查。逐类数值与F1标记的首稿重叠已修复，旧稿留在draft_01；最终12份图与SHA清单一致。
- [结果表更新收据](record_update.json)：只提升HGT行的证据等级，所有分数及其他四行原字节不变；主结果JSON/MD同步。`before_update/`保留修改前副本，历史服务器status没有重写。

[完整分析报告](../../../results/hgtnet_nwpu_seed42_analysis_20261104.md)、[机器分析记录](../../../results/hgtnet_nwpu_seed42_analysis_20261104.json)及[可复制纯文本](../../../output/hgtnet_seed42_verified_summary_20261104.txt)为主要入口。results目录另有八份完整CSV导出。

`verify_and_package.py`核对最终原件、图表、记录和文档后生成`final_validation.json`及图表数据ZIP；`bundle_manifest.json`记录包内有效载荷SHA，`delivery_receipt.json`记录ZIP身份和逐成员校验结果。脚本拒绝覆盖已有产物。ZIP保留results/output相对目录以便阅读报告，包含原始回传JSON、图表、数据和审计脚本；不包含模型、完整项目代码或依赖环境，审计脚本需要在原GAViT项目运行。ZIP不收录修复前草稿。

边界：未回收模型/optimizer张量、原图、完整原manifest和逐轮validation预测；相应SHA与初值一致属于metadata核验。单seed、不同完整方法配方和不同硬件的比较不建立模块因果、稳定优势、统计等价或公平速度结论。本轮没有Git发布、对外发信或自动租卡。
