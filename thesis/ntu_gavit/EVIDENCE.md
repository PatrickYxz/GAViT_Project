# 章节事实与项目依据

2026-10-27核对方法与实验设置，2026-10-28至29扩写绪论和文献综述并核对来源，2026-11-06接入外部方法补充比较。以下路径相对于项目根目录；`S`代表 `.worktrees/sparse-hybrid-4n-top2`。代码事实与历史运行身份分别列示，当前HEAD不充当历史训练commit。

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

1. 第1–4章实际引用50条独立文献，已满足至少40条的数量要求；本次补充章复用已有书目。逐篇原始来源、作者/题名/年份与机制核对范围见下方索引；有些来源只核到作者摘要或出版社索引，不能称已全文审读所有论文或完成穷尽综述。
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

历史说明：2026-10-29时仅完成DFAGCN组件与CPU检查。此后DFAGCN×NWPU、HGTNet×NWPU/AID均完成独立实现seed42正式实验和预测复算；当前正文状态以以下11月6日证据为准，不能继续将早期组件阶段当作现状。

## 2026-11-06外部方法补充章

| 正文内容 | 核对依据与边界 |
|---|---|
| 原定外部矩阵为HGTNet×NWPU/AID、DFAGCN×NWPU | `S/docs/supervisor_latest_email_received_20260924.md:17`；不因DFAGCN原文也有AID分数而新增本项目实测 |
| NWPU五模型seed42 OA、macro-F1、正确数、refit轮数、参数与硬件 | `results/external_baselines_nwpu_seed42_20261030.csv`；外部单seed不混入内部多seed统计 |
| HGTNet NWPU原预测、完整100/95轮、与DFAGCN逐图配对 | `results/hgtnet_nwpu_seed42_analysis_20261104.md`及其JSON/CSV；HGT↔GAViT净差4张不替代配对原件 |
| DFAGCN独立选择、输入224/448、VGG16冻结、GCN与LIBLINEAR | `configs/dfagcn_independent_v1.json`、`results/dfagcn_nwpu_seed42_20261030.json`；不称MatConvNet等价或官方复现 |
| AID内部seed42完整精度、正确数及真实逐图预测 | `results/evidence/external_results_integration_20261106/aid_core_raw/{swin,gavit}/`，10份原文件SHA核对通过；与HGTNet全部5000条路径/标签/类别顺序一致，本地独立重算三模型OA/F1与混淆矩阵 |
| AID三模型真实配对 | `results/aid_seed42_three_model_paired_analysis_20261106.json`及全部5000行配对CSV；HGTNet相对固定GAViT共同正确4771、纠正75、新增错76、共同错78，净-1；相对Swin纠正108、新增错79，净+29 |
| AID参数、internal refit轮数与同卡测量边界 | `S/results/aid_multiseed_summary.md`、`S/results/aid_seed42_comparison.md`；旧2080Ti测量不用于宣称相对新HGTNet的速度优势 |
| HGTNet AID OA96.92%、macro-F1 96.77286814931982%、4846/5000 | `results/evidence/hgtnet_aid_formal_complete_20261106/raw/test.json`、`root_audit.json`、`independent_audit.json`；本地原预测重算与服务器记录一致 |
| AID独立初始化、best96唯一972/1000、final100为971/1000、fresh refit96、55048接受步 | 同目录`history.json`、`summary.json`、`initialization.json`；保留100+96轮，不以refit训练100%代替test |
| HGTNet不同于internal 30epoch/AdamW/直接224resize | `configs/hgtnet_nwpu_formal_v1.json`、`configs/hgtnet_aid_formal_v1.json`及AID原`binding.json`；100epoch/Adam/batch16、256resize后224crop、独立部分Swin-B初值 |
| AID30类recall与双向混淆 | 原`test.json`的全部标签/预测及指标；Square143/165、Center113/130、Resort129/145、School135/150，Center/Square与Park/Resort各12错误；只作分布描述 |
| AID两幅原始离线图 | `results/figures/hgtnet_aid_seed42_20261106/hgtnet_aid_seed42_training_history.png`与`hgtnet_aid_seed42_confusion_matrix.png`；引用完整100/96历史和30×30混淆，不再执行模型 |
| AID恢复调用资源、checkpoint及来源身份 | `results/hgtnet_aid_seed42_20261106.json`、原`summary.json`、`runtime.json`、模型收据及两attempt日志；4986.588285893秒和CUDA峰值仅覆盖attempt2，全实验资源/费用未知，模型二进制未在本地重载 |

第五章是补充比较，不能据此宣称导师要求的统一训练/评估协议与多seed外部比较已经完成。AID固定GAViT比HGTNet净多对1图，现在已有三模型原始预测支持真实配对（HGTNet纠正75、新增76）；NWPU HGTNet比固定GAViT净多对4图仅为汇总差，不能推导该对逐图计数。两项都不构成显著性或稳定性证据。不同骨干、预训练、增强、预算与GPU均已披露。本文尚缺完整内部结果/消融、综合讨论及结论。

本次写作前五个原文件的字节与SHA256保存在 `provenance/before-external-results-20261106/`；未改动其他历史provenance、参考文献、模型、冻结划分或旧指标。11月6日子任务只做源码与事实检查。11月8日整合步骤已新鲜通过4项相关测试、35项独立数据/论述核验、50条引用/交叉引用/图片路径静态检查及原生LaTeX编译，4张分析PNG视觉复核通过。PDF导出及页面视觉检查最终范围另记在本次validation中；不沿用10月29日validation作为新增章节的验收结论。

## 篇幅与验证边界

用户确认50–60页包括前置页和参考文献，56页预算见README与本轮计划。绪论和综述按约4页/13页目标写；源码词数和引用检查不证明实际分页。原生编译只返回诊断，PDF导出与视觉检查以本次validation.json记录为准，不声称当前部分草稿已构成完整50–60页论文。

## 2026-11-08版式修正

实际导出首版PDF后发现目录自引用/负间距造成首行重叠，以及原图缩到论文宽度后字体太小。已保留首版PDF/日志和改前TeX，再移除目录自引用、改用正向标题间距；两幅论文图用 `provenance/render_external_aid_figures_20261108.py` 从相同完整数据按14.5cm宽重画，保存在 `images/external_aid_20261108/`。图轴/完整类名/legend为8pt，CM数值7pt；色值保持未取整百分比，数值取整，非零且小于1%的单元格写为<1。原始分析图、CSV及raw均保留，65项保护工件哈希未变。修正不改变正文12pt、页边距、行距或实验数值；修正后最终源码再次通过原生编译和缓存Tectonic编译，47页PDF完成19页直接视检与28页逐像素一致性映射；新版编译、来源及完整视觉复核记录见validation。
