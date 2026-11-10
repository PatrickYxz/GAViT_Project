# DFAGCN正式test.json逐图离线验收

复核日期2026-10-31，实验批次2026-10-30。状态DOWNLOADED_TEST_RECORD_METRICS_LOCALLY_RECOMPUTED。

- [原始test.json](test.json)，SHA256 `771dee2dcd1e676e4884c36654cc8fbcc4c067811362dbf9f3e997c903ccd4e9`。原文件/Users/patrick/Downloads/test.json保持不变。
- [可复现分析脚本](analyze.py)：标准库重建45类标签、逐图预测、混淆矩阵与全部指标，导出明细；脚本的输出路径独占，已有输出不会静默覆盖，复跑须使用独立新输出路径。
- [独立32项核验](review/audit.json)、[审阅说明](review/audit.md)。record seal匹配，manifest/cache引用及cached时段与归档summary吻合，25200路径唯一/有序/规范、45类各560、全部2025矩阵单元及45类指标通过。
- [作图版本与SHA](figures/plot_manifest.json)、[视觉QA](figures/visual_quality_check.json)。仅独立/private/tmp绘图依赖，未改模型环境。两图均PNG/PDF/SVG，完整矩阵为行归一化百分比且保留全部45类，矢量SVG没有内嵌位图；主代理已再次实际查看两张PNG。
- [更新前的结果与汇报](before_update/)，保留本次证据升级前版本。

从路径首目录按既有命名规则重建真实类别，与预测逐项复算，确认21806正确/3394错误；OA86.53174603174604%。macro-F1复算86.45890874660425%，与服务器报告差约2.84e-14，在1e-10绝对容差内，不称位级相同。每类precision另从列和计算，accuracy/recall以各行560为分母。

最弱五类palace/church/tennis_court/basketball_court/railway共910漏判，占26.8120%。前十有向混淆共498错误/14.6730%，前五双向类别对440错误/12.9641%；并列排名按类名排序。全部非零项和全部3394错误均在CSV，未挑例删除。

完整产物：[误差分析和可复制段落](../../dfagcn_nwpu_seed42_error_analysis_20261031.md)、[分析JSON](../../dfagcn_nwpu_seed42_analysis_20261031.json)、[45类指标](../../dfagcn_nwpu_seed42_per_class_20261031.csv)、[完整预测](../../dfagcn_nwpu_seed42_predictions_20261031.csv)、[错误清单](../../dfagcn_nwpu_seed42_errors_20261031.csv)。

边界：原manifest/图像、cache、refit记录与checkpoint原件未回收，本次仅确认返回record及引用自洽、指标可从其预测复算。旧三模型完整逐图预测不在本地，没有计算跨方法纠正/新增错误配对，也未从误差结构推断图模块或其他机制的因果贡献。没有新训练、GPU测试、Git提交推送或外发。
