# 章节事实与项目依据

2026-10-27重新读取现有源码与归档。以下路径相对于项目根目录；`S`代表 `.worktrees/sparse-hybrid-4n-top2`。代码事实与历史运行身份分别列示，当前HEAD不充当历史训练commit。

| 正文内容 | 核对依据 |
|---|---|
| Swin末级49×768 token与mean/LN/dropout/linear baseline | `models/swin_backbone.py:12`、`scene_classification/modeling.py:32`、`train_bigearth.py:166` |
| 固定分区公式、不等组大小与attention scorer | `models/region_grouping.py:149`、`:175`、`:193` |
| soft投影、每图原型、候选局部性、S/W两个归一化 | `models/soft_region_grouping.py:10`、`:38`、`:54`、`grouping_identity.py:4` |
| 48+32条非自环边、top2排除空间邻居、消息方向、tie break | `models/graph_construction.py:118`、`:150`、`:174` |
| 2层4头GAT、1024维、residual/LN/GELU/dropout、edge_weight未使用 | `models/graph_reasoning.py:23`、`:49`、`:64` |
| fixed/soft反馈、mean后classifier、无后续Transformer | `models/gavit.py:149`、`:216` |
| soft主版本无Gaussian prior/FFN，正式冻结ed480e6 | `results/evidence/bigearth_soft42_complete_20261017/summary.json:89`、`:95`；与当前soft模块SHA重新比对 |
| AID/NWPU数据与exact-duplicate分组 | `scene_classification/data.py:19`、`:58`、`:95`；`S/docs/thesis_experiments_discussion_20261006.md:13` |
| 当前NWPU三划分与预训练权重绑定 | `S/results/evidence/seed43_nwpu_final_20261011/reported_summary.json:11`、`docs/run_soft_assignment_nwpu.md:14` |
| 校准选轮数、重新初始化refit、T_max仍30 | `scene_classification/runner.py:282`、`:300`、`:342`、`:394` |
| AID/NWPU增强、精度、optimizer | `scene_classification/modeling.py:63`、`scene_classification/runner.py:50`、`:298` |
| BEN三个split/标签/数据身份 | `results/evidence/bigearth_soft42_complete_20261017/summary.json:123`；`baselines/bigearth/prepare_bigearth.py:220` |
| BEN RGB读取与uint8转换 | `models/bigearth_dataset.py:124` |
| BEN增强、训练与val选best | `train_bigearth.py:136`、`:209`、`:350`；`utils.py:225` |
| BEN精度差异和历史Swin42缺sidecar | `results/evidence/bigearth_soft42_complete_20261017/summary.json:155`；`results/evidence/bigearth_soft42_launch_20261016/preflight.json:5` |
| OA/macro-F1与混淆矩阵 | `scene_classification/metrics.py:5` |
| mAP/macro-/micro-F1与0.5阈值 | `test_bigearth.py:176` |
| AID同卡效率测量口径 | `S/results/aid_efficiency_summary.md:9`、`:28`；本地为回传摘要，不是本轮新测量 |
| HGTNet/DFAGCN明确要求与相关工作三主线 | `S/docs/supervisor_latest_email_received_20260924.md:17`、`:21` |
| 用户确认已请求作者材料/未回复 | `S/docs/external_methods_readiness_20260924.md:5`、`:7`；不代表独立核验邮件投递 |
| 现成中文稿可复用但实验部分已过期 | `S/docs/thesis_related_work_20261006.md`、`S/docs/thesis_experiments_discussion_20261006.md`；本轮未复制过期分数 |
| 新soft42及K9/K16回传在10-06稿之后完成 | `research_diary.md`顶部10-17至10-20条目；`docs/ben_gridpair42_visual_analysis_20261020.md` |

## 仍需完善而未填造的内容

1. 完整论文至少40篇真实相关引用；当前八条仅服务已写章节。正文和BibTeX中的原始来源均已打开，不能称为已读完所有论文全文。
2. 最终方法×数据集×seed结果与可比性，不引用早期空白/近似CSV作为终稿主表。
3. 统计检验的适用样本与实际结果；本轮未根据四舍五入或混合条件历史分数推导p值。
4. 全部模型的同条件效率；本轮仅描述已有AID benchmark的实际范围，不声称soft效率已测。
5. 具体预训练发行卡与版本的完整来源；现有权重文件和SHA可核对，未据此猜测发布方或其完整训练语料。

## 模板与文献来源

- 模板：`https://github.com/jackieyyang/ntu-dissertation-template`，commit `3094f44d8705919b7db3d08391678c4b4d709817`。
- Swin：`https://arxiv.org/abs/2103.14030`。
- GAT：`https://arxiv.org/abs/1710.10903`。
- NWPU：`https://arxiv.org/abs/1703.00121`。
- AID：`https://arxiv.org/abs/1608.05167`。
- BEN：`https://arxiv.org/abs/1902.06148`。
- BEN19 nomenclature：`https://arxiv.org/abs/2105.07921`。
- BEN v2 metadata来源：`https://zenodo.org/records/10891137`，与本地runbook所指同一记录。
- AdamW：`https://arxiv.org/abs/1711.05101`。

本表是作者工作记录，不编入论文正文。
