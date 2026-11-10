# 章节事实与项目依据

2026-10-27核对方法与实验设置，2026-10-28至29扩写绪论和文献综述并核对来源。以下路径相对于项目根目录；`S`代表 `.worktrees/sparse-hybrid-4n-top2`。代码事实与历史运行身份分别列示，当前HEAD不充当历史训练commit。

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

1. 当前四章实际引用50条独立文献，已满足至少40条的数量要求。逐篇原始来源、作者/题名/年份与机制核对范围见下方索引；有些来源只核到作者摘要或出版社索引，不能称已全文审读所有论文或完成穷尽综述。
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

## 绪论与文献综述的来源索引

| 范围 | 记录 |
|---|---|
| CNN遥感迁移、场景综述、遥感Transformer与预训练 | `provenance/literature-remote-20261028.json`，14篇候选的实际访问状态及逐篇用途/边界 |
| 图网络、分组、TokenLearner、两种STViT与GloRe | `provenance/literature-graph-grouping-20261028.json`，17篇候选的原始来源与高风险机制定位 |
| 绪论补充的ImageNet、遥感深度学习综述、Seasonal Contrast | `provenance/literature-intro-20261028.json` |
| 通用CNN/Transformer/MAE骨干 | `provenance/literature-foundations-20261028.json` |
| HGTNet、DFAGCN | `provenance/literature-direct-methods-20261028.json`；原作者上传论文文字层，未把原论文成绩抄入统一实测主表 |
| 参考文献合并去重、旧八条版本来源 | `provenance/bibliography-merge-20261028.json`；扩写前文件保存在 `provenance/before-intro-review-20261028/` |
| 本轮新增书目核对及复核收尾 | `provenance/final-review-20261029.json` |

写作采用作者自己的连接论述与设计比较，不复制摘要句子、论文图或算法图。GloRe的池化→图推理→广播、TokenFuser和super-token映回均按既有工作归属，不声明GAViT首创这一通用流程。两种STViT与TokenLearner正式会议题名分别核对。

2026-10-29最新日记和 `external_baselines/dfagcn/README.md` 已记录第一批独立组件与CPU检查；它们不构成完整DFAGCN、真实数据训练或正式外部比较。绪论与综述保留这一范围，不沿用“尚未开始任何实现”的过期说明。

## 篇幅与验证边界

用户确认50–60页包括前置页和参考文献，56页预算见README与本轮计划。绪论和综述按约4页/13页目标写；源码词数和引用检查不证明实际分页。原生编译不导出独立PDF，本轮不声称完成逐页视觉验收，也不声称整篇论文已到50–60页。
