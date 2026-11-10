



# Research Diary — GAViT Project

> 记录每次重要进展、实验结果和 debug 过程。每次完成功能模块、获得新结果或修复重要 bug 后必须更新。

---

## 2026-10-20 — 受训K9/K16同图文件审计与两页正式图稿完成

- 用户返回下载目录，[主归档](../../results/evidence/ben_gridpair42_visual_return_20261020/server_return)及[审计](../../results/evidence/ben_gridpair42_visual_return_20261020/validation.json)核对37文件SHA、两项exit0、8组真实记录/PT字节及CRC、488边、76项概率与原PNG/标签；未执行返回代码/pickle或本地神经网络。
- K16纠正3项标签、增加1项漏判，全部4例和负向结果保留；仍错9项不误称新退化。[分析与完整76项](../../docs/ben_gridpair42_visual_analysis_20261020.md)、[PDF/300dpi PNG/HTML](../../output/pdf/ben_gridpair42_20261020/README.md)完成，真实四例图及两页PDF均已渲染看图，像素/边/翻转/文本边界导出核对通过。主图画特征边，空间边及全部类别概率在HTML保留。
- K9训练summary补齐best10/final30/val best79.1052%及实际4090 UUID GPU-8734427f-061f-9fdb-97eb-fe9190a11225；原test和历史K16不改。当前CPU捕获21.68s。状态为可视化PAIRED_CAPTURE_AUDITED_FIGURES_COMPLETE，完整原训练history/checkpoint仍留服务器，审计范围注明；没有新增训练、性能CSV变更、外发或Git提交推送。导师汇报草稿已同步本项完成，Swin／固定／soft同图属于下一项。

---

## 2026-10-20 — K9/K16同图捕获报告8组通过，等待ZIP审计及真实图稿

- 用户回传安装/原训练身份门禁通过、CPU环境、4图×2模型全部8组CAPTURE_PASS与HTML/ZIP完成标记。[主原文](../../results/evidence/ben_gridpair42_visual_run_20261020/user_return.txt)、[解析](../../results/evidence/ben_gridpair42_visual_run_20261020/validation.json)归档；32项capture/restoration/pooling/feedback误差均0，K9每图42边、K16每图80边，共488边。此为观察/重建检查，不是分类错误率。
- [主状态](../../results/ben_grid3_seed42_status_20261018.json)改为SERVER_CAPTURE_REPORTED_COMPLETE_ARTIFACTS_PENDING。控制台未包含真实类别概率、图像、原始张量或实际run.exit/summary文件；本地文件审计/图稿仍未完成，原性能表不变。[下一步](../../docs/run_ben_gridpair42_visual_20261020.md)只下载现有ben_grid_pair_visual.zip并返回，无需重跑。无训练、代码/CSV修改、提交推送或外部发送。

---

## 2026-10-20 — 受训K9/K16同图可视化交接包已准备，真实捕获待执行

- 用户批准继续，新增tools/ben_grid_pair_visual.py、package_ben_grid_pair_visual.py及tests/test_ben_grid_pair_visual.py；[主运行说明](../../docs/run_ben_gridpair42_visual_20261020.md)、[CPU单行命令](../../results/evidence/ben_gridpair42_visual_prepare_20261020/install_and_capture_command.txt)与[核对](../../results/evidence/ben_gridpair42_visual_prepare_20261020/preparation_validation.json)保存。原模型/训练代码及历史指标不改；无训练或全量test。
- 使用全部4张历史原reader Resize224无损PNG，校验val划分/row/标签、两套seed42 best SHA、冻结13源码与原30轮/119825 summary/exit，CPU8线程共用同一FP32输入。真实观察hooks、pooling/feedback重建及K16旧图函数差需≤1e-5，实际42/80边、全部19概率、成败两方向、CPU张量与来源ZIP均保存。父summary会自动归档，返回ZIP后才能完成本地身份审计和真实案例图。
- 新增9项与旧K9回归11项共20项通过；完整unittest199项有21个已记录缺依赖错误（8torch/13pytest），非全套通过。本地安装fixture明确屏蔽dispatch，无神经网络推理；4张合成边/概率FIXTURE SVG已静态渲染看图，488边/12次原PNG对应通过，HTML浏览器file协议受限未交互验收。独立复核无Critical/Important项，minor计划状态已更新。
- [主状态](../../results/ben_grid3_seed42_status_20261018.json)为可视化HANDOFF_PREPARED_SERVER_CAPTURE_PENDING，服务器真实前向/图稿尚未完成；没有新正式分数、付费任务、提交推送或外部发送。

---

## 2026-10-20 — K9补跑的主要交付是受训3×3／4×4同图可视化

- 用户重申补跑用于“可视化33和44的对比”。已将[主运行说明](../../docs/run_ben_grid3_seed42_20261018.md#训练后的可视化交付)和[状态](../../results/ben_grid3_seed42_status_20261018.json)改为：先补summary身份，再用既有K9/42及K16/42权重捕获同图区域、实际空间/特征边与预测；mAP及逐类表仅作背景。此后继续Swin／固定4×4／软分配图，不把训练完成当作可视化完成。
- 已核对tools/diagnose_bigearth_feedback.py硬编码K16/80边，tools/ben_grid42_tensor_checks.py只记录训练首图边；尚不能代替K9/K16指定同图捕获。第一批沿用已归档4张历史验证输入并全部保留，旧seed44分数/图边不改标；固定边界不依赖训练，要补的是实际区域汇总权重、图边和预测。
- 当前只更新记录，无新训练/推理/图稿/分数，真实配对捕获待完成。

---

## 2026-10-20 — BEN固定K9 seed42回传队列exit0及119825张指标；summary元信息待归档

- [主仓库粘贴记录](../../results/evidence/ben_grid3_seed42_metrics_20261020/user_return.txt)及[核对](../../results/evidence/ben_grid3_seed42_metrics_20261020/validation.json)保存。总run.exit=0，test mAP71.07533208222321%、macro-F1 64.48830571218534%、micro-F1 76.68872316423352%，119825张/19类/阈值0.5；checkpoint SHA c4b77953d6dd2908d7a61ebe7a34f4004e8c219a0843de6ab05aaf207460feff。19类AP均值独立复算吻合，F1无原数组未复算；无直接服务器读取/新推理。
- [K9/K16独立对照](../../docs/ben_grid3_grid4_seed42_20261020.md)与[总体/全部类别JSON](../../results/ben_grid3_grid4_seed42_20261020.json)记录。K16−K9约mAP+0.92pp、macro-F1+1.51pp，micro基本持平；海岸湿地+13.12、沙滩+6.97、内陆湿地+1.14，天然草地−4.52、多年生作物−1.73。保留单seed、历史控制打印一位小数、GPU/软件批次限制，不称普遍/统计显著或建边机制证据。
- [状态](../../results/ben_grid3_seed42_status_20261018.json)标为TEST_METRICS_AND_QUEUE_EXIT0_REPORTED_SUMMARY_PENDING，已有真实指标不再null，但best/final、UUID/软件/精度、真实图/重载门禁和资源需[已有summary](../../results/evidence/ben_grid3_seed42_metrics_20261020/read_summary_command.txt)核对后归档。当前full_artifact_audit_complete=false；不重训/重测、不改原4seed表或性能CSV，未提交推送或外部发送。

---

## 2026-10-19 — BEN固定K9 seed42安装/分派已回传，正式训练启动尚未确认

- 用户回传安装/队列分派PID3599，目录/home/featurize/work/ben_grid3_seed42_20261018_01；[主仓库粘贴记录](../../results/evidence/ben_grid3_seed42_launch_20261019/user_return.txt)及[核对](../../results/evidence/ben_grid3_seed42_launch_20261019/validation.json)保存。root日志START preflight，末次打印train230000/237871、val110000/122342，无可见错误；未回传预检PASS、短测结果或正式首批/GPU进程与UUID，不把数据扫描或PID当训练成功启动。
- [状态](../../results/ben_grid3_seed42_status_20261018.json)更新为DISPATCHED_PREFLIGHT_RGB_SCAN_OBSERVED、formal started未知/null、verified=false、指标null。短测也有TRAIN_FIRST_BATCH_OK，需START train之后的首批及GPU进程/显存才能确认正式训练；下一步仅持续读取同一run.log，现有门禁队列自动继续，不重复安装/启动。本轮无直接远端执行、代码/CSV改动、提交推送或外部发送。

---

## 2026-10-18 — 用户授权补BEN固定K9 seed42；独立快照与队列已准备，服务器未执行

- [主仓库运行说明](../../docs/run_ben_grid3_seed42_20261018.md)、[状态](../../results/ben_grid3_seed42_status_20261018.json)及[安装/后台命令](../../results/evidence/ben_grid3_seed42_prepare_20261018/install_and_launch_command.txt)保存。用户授权只补一个K9 seed42正式30轮和best全test，复用历史固定K16/42为有限参考，不新跑4×4/Swin/soft。正式分数、实际UUID/耗时/费用保持null。
- 新增tools/ben_grid42_patch.py、run_ben_grid42.py、ben_grid42_tensor_checks.py、install_ben_grid42.py及tests/test_ben_grid42.py，不修改现有模型/训练入口或其未提交研究改动。从cd39be91a013bcf98099808c451300f87d3898b7导出独立13源码，K9扩展四邻接24+feature top2共18=42有向pre-GAT边，K16仍48+32，attention/GAT/token_feedback及优化配方保持；eval pin_memory=false IO修复、完整SHA和modified_frozen_snapshot身份明确记录，不假装Git clean。快照SHA f82d37a9cc1d18920a553fa1e4fd516e9d9ebda9d9abddb6595e12a69a944f58。
- 同一BEN划分237871/122342/119825、RGB/19类、原ImageNet SHA及增强，seed42/epochs30/batch32/workers4/AdamW3e-4/wd1e-4/Cosine30/BCE/阈值0.5，31366926参数。目标4090及UUID、FP32/noAMP/matmulTF32false/cuDNNtrue/8线程；旧4×4软件/精度及4090训练/3060测试批次限制保留，不称严格纯K因果。输出独立/home/featurize/work/ben_grid3_seed42_20261018_01，预检→真实张量/前向→96/64短测/保存/fresh重载→完整30→119825×19 best test，逐阶段日志/退出/资源/history/预测保存，短测通过先落输出目录experiment_diary.md再训练；失败即停，不覆盖/盲目重试。
- 新11项标准库检查通过；独立复核发现CUDA scatter应使用有限≤1e-5容差，已加回归修复，CPU K16图张量仍逐值比较。完整安装fixture已执行，GPU dispatch屏蔽，真实CUDA门禁未执行；本地全套因缺torch/pytest不能称通过，详情[准备核对](../../results/evidence/ben_grid3_seed42_prepare_20261018/preparation_validation.json)。旧价1.67元/h如仍适用，预留6～8h/10～14元，实际待测。未启动付费任务、改历史CSV、提交推送或外部发送。

---

## 2026-10-17 — 收到Featurize库存；未识别可比BEN 3×3性能或权重候选

- [主仓库原始回传](../../results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/server_return.txt)与[解析库存](../../results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/parsed_inventory.json)保存；两条命令与准备版本一致，510个不同路径（136个.pth/373个.pt/1个.safetensors）无K9/3x3文件名，第二条配置内容搜索无匹配且回到提示符。无可见错误，退出码未单独回传。本轮只读用户回传，没有直接访问服务器或加载模型。
- [审查](../../docs/ben_k9_k16_comparability_audit_20261017.md)及[状态](../../results/ben_k9_k16_comparability_audit_20261017.json)更新为当前本地/回传扫描均未识别可比BEN K9结果；K9 mAP和K16−K9差值仍null，现有4×4分数保持。大日志、未带配置的普通命名权重、剪枝目录及原NTU/其他存储仍有盲区，不据此宣称所有历史K9不存在。若继续区域数性能消融，需要先匹配配方并处理K16构图限制，本轮不自动启动新训练。
- 性能CSV、模型和旧图保持；无新训练/测试、提交推送或外部发送。

---

## 2026-10-17 — 检查既有3×3性能；本地无可配对BEN结果，服务器库存待查

- 用户要求先检查是否存在可比3×3性能，再进行对照。[主仓库审查说明](../../docs/ben_k9_k16_comparability_audit_20261017.md)及[结构化记录/来源SHA](../../results/ben_k9_k16_comparability_audit_20261017.json)保存；本轮未启动训练、模型评估或服务器访问，BEN K9 mAP与K16−K9差值保持null。
- 旧NWPU日记K9 test95.9%与K16 test95.8%同时改变池化/反馈，且旧K16为4725张test；不能当当前纯K消融。BEN表13条无K9；旧70.4%未修正kNN与70.6%修正K16都不能改标K9。初始BEN提交0b1e02c配方和作业均K16，针对性Git差异未找到K9仍不排除未入库覆盖参数。固定42的72.0%及cd39be9/K16/best13/final30维持，尚无有效3×3配对。
- 本地无模型权重，Featurize及原NTU库存待查；[两条单行只读命令](../../results/evidence/ben_k9_comparability_20261017/server_inventory_commands.txt)搜索权重和小于20MiB的文本配置线索，不读原图、不导入torch/加载checkpoint、不需GPU。命中只是候选，配置可配对的既有权重优先补测试，缺项不自动触发训练。26项[本地语法/合成fixture/来源核对](../../results/evidence/ben_k9_comparability_20261017/validation.json)通过，不当作服务器执行证明。CSV、模型、图稿保持，无提交推送或外部发送。

---

## 2026-10-17 — BEN固定3×3/4×4同图几何对照首批完成

- 用户选择BEN作为主展示数据集，并批准先完成分区本身。复用本地归档的全部4张历史验证输入（97389/14680/28450/4439），原来是seed44反馈诊断24例中的全部阈值翻转案例，选择偏差和历史来源明确保留，不冒充19类代表样本、42/43/46/47性能证据或新获益案例。
- [逐例分析及图注](../../docs/ben_grid3_grid4_analysis_20261017.md)、[单页PDF/300dpi PNG/四例SVG与PNG](../../output/pdf/ben_grid3_grid4_20261017/README.md)、[来源及49token成员](../../output/pdf/ben_grid3_grid4_20261017/figure_manifest.json)保存。按冻结ed480e6与当前分组源码相同的位置公式绘制K9/K16，4×4成员和数量与4份历史实际捕获逐一一致；源PNG/SHA及底图像素不变、未增强亮度或重新裁剪。没有模型推理、受训K9前向、图边、新预测或语义质量数值。
- 180×254mm PDF含12次原图放置和100个区域覆盖，源图仍224×224；300dpi导出2126×3000像素不增加遥感细节。四例及完整PDF已实际看图；160项[独立导出核对](../../output/pdf/ben_grid3_grid4_20261017/export_qa.json)通过，包含原始像素、K16捕获、整数参照分区、PDF矩形、SVG原图字节及字体/范围说明。构建命令为bundled Python运行主仓库output/pdf/ben_grid3_grid4_20261017/build_figure.py，之后Poppler渲染并运行同目录verify_figure.py；没有付费GPU成本。
- 观察：海岸4439中左列由96px缩为64px，包含岸带的空间单元覆盖水面范围变小，但岸水仍混合且连续岸线跨更多区域；14680下方浅绿斑块被更细分区进一步切开；另外两例的不规则纹理也未变成内容边界。支持“覆盖尺度与切分方式变化”，不支持普遍语义改善或4×4分类提升。四例原诊断分数未拿来充作本次网格结果。
- 中文汇报优先项、英文草稿可视化段和原NWPU审查的后续指针已同步，其他用户编辑保持。后续才核验BEN K9库存、三模型同图预测及实际边/扰动证据；当前缺项不自动触发新训练。性能CSV、模型/训练源码、历史图稿均未改，未外部发送或Git提交推送。

---

## 2026-10-17 — 优先审查3×3/4×4区域分析；现有内容部分满足要求

- 用户明确先做3×3/4×4区域分析，再处理其余内容。本轮按导师原文核对源代码、模型库存、旧图和新90例图册，形成[审查说明](../../docs/nwpu_grid3_grid4_visualization_audit_20261017.md)与[素材/SHA/分区复算](../../results/nwpu_grid3_grid4_visualization_audit_20261017.json)；中文汇报记录此优先顺序。未生成新对照图，未训练、推理或访问服务器。
- 本地90张224输入逐一SHA与清单吻合，45类各2张，可用于同图几何覆盖；旧12张诊断图实看教堂/机场均为K16 attention/token_feedback、kNN-5，不能替代当前稀疏混合图或3×3对照。旧K9权重/graph_vis仅有历史日记线索，本地没有模型权重；服务器库存尚未检查，不能据此断言已丢失。
- 分组位置公式与冻结ed480e6一致。7×7 token分配到3×3的单轴数量为3/2/2、区域4～9 token；4×4为2/2/2/1、区域1～4 token，显示边界分别96/160及64/128/192像素。4×4不是旧3×3的嵌套细分，更细不自动等于语义更连贯。此为源码公式复算，不是新模型实验或语义评分。
- 当前模型/稀疏混合构图拒绝K≠16，新捕获工具也固定49×16/80边；旧diagnosis手工重建attention且硬调用kNN，不能直接用于旧均值K9或当前sparse_hybrid。先用确定分区规则与既定90图做离线区域对照；随后核验旧K9权重/源码并捕获真实边。历史版本改变池化/反馈/构图，不把全部差异归因区域数；导师定性要求不自动意味着新增纯K训练消融。代码/CSV/性能结果保持，另将邮件seed文字不一致记入审查，未覆盖其余用户编辑。

---

## 2026-10-17 — 澄清NWPU图册的比较对象及共同错误含义

- 用户询问“3个优化5个退化”与固定GAViT高于Swin是否矛盾。重新读取捕获工具、图册清单和正式测试配对；从90条逐例记录独立复算为82双方正确、3 soft纠正fixed、0 soft新增错误、5双方错误，45类各2张。5张共同错误不等于新增退化。
- 图册用seed42校准best的fixed/soft权重、按val每类路径排序前两张，记录真实S/W与构造边并校验捕获前后logits/缓存；正式Swin对照来自独立refit后的25200张test。该test固定对Swin纠正586/新增543、净+43，OA93.93%→94.10%；不能由总体优势推出每张图改善，也不能以90例soft局部+3覆盖其正式test负面结果。
- [英文草稿](../../docs/supervisor_email_draft_20261012.md)及[中文汇报](../../docs/supervisor_progress_report_20261012.md)仅澄清可视化段落，其他用户编辑保留。已有纠正/共同错误实图已复看；本轮未新增训练、推理、CSV改动或外部发送。

---

## 2026-10-17 — BEN soft42正式完成；补齐指定四seed并保留负面结果

- 收尾独立复算与改动范围检查71项通过，两个工作区git diff --check通过；四seedCSV与追加条目预览已检查。[统计/记录核对](../../results/evidence/bigearth_soft42_complete_20261017/report_update_validation.json)保存。这些是数据/文档核对，不代表重新执行模型测试。
- 对用户回传的总队列/训练/test三项exit0、summary complete、final30、最佳验证checkpoint、119825张/19类测试和完成标记完成31项身份/数值核对；[原文](../../results/evidence/bigearth_soft42_complete_20261017/server_return.txt)、[summary](../../results/evidence/bigearth_soft42_complete_20261017/summary.json)、[核对边界](../../results/evidence/bigearth_soft42_complete_20261017/validation.json)保存。该核对基于回传，未独立下载服务器权重/预测数组或运行CUDA。
- 实验bigearth_soft42_20261013_01，seed42、模型ed480e60fea03ed786fdba70c144de5e815c34f5，已修复调度worker SHA09722f10542bed449d2e80df0041387e113cab58db403b3979e07f192194f767与两阶段一致。原调度失败历史保留。RTX4090 UUID GPU-44825cf0-bd98-bfdc-efbc-0100238a6789，FP32/noAMP、matmulTF32false/cuDNNtrue、8线程。
- 问题为补齐指定42/43/46/47的BEN soft缺项及逐类比较；沿用soft43配方，主要变量为训练seed42，物理GPU身份按实记录。RGB/19标签、原划分237871/122342/119825及SHA、原ImageNet权重；预处理/增强沿用冻结配方，未重建split。K16局部原型dim128/radius1/temp0.1/1次迭代、48空间+32特征边、GAT256/4头/2层、token_feedback/dropout0.1，31466766参数；AdamW lr3e-4/wd1e-4、Cosine30、batch32/workers4、BCE、F1阈值0.5。原命令和完整metadata见summary argv；由训练seed变化不能排除历史对照代码/硬件混杂。
- best10/val mAP78.75%，final30/val75.70723466069043%；test mAP70.1745705651681%、macro-F1 64.04828509185678%、micro-F1 76.7999586513447%。checkpoint SHA23314ea561f753b1aad73025148d7b9788bbdddc4f38f49afeca59a6aec91c35在train/meta/test/metrics一致，路径为/home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth；日志为同根run.log和gavit_train/run.log，曲线/探针留在服务器阶段目录。
- 训练20619.28s、完整test155.17s、队列20808.02s（5h46m48s）；有效训练处理量346.09图/s包含validation/IO/探针，不是纯前向吞吐。训练峰值allocated3.66GiB；按旧价1.67元/h计运行部分约9.65元，实际账单/准备空闲未知。
- 相比Swin42 mAP−0.73pp、固定42−1.83pp；soft对Swin草地+6.24、多年生作物+2.76，但海岸湿地−19.64、沙滩−4.43、工业区−2.32pp。与soft43方向反转，补足失败分析，不删除、不为了正向结果重跑。指定四seed Swin70.90±0.22、固定71.65±0.39、soft71.19±1.25%；soft对Swin2升2降/均值+0.29pp，对固定仅43正/均值−0.46pp。用户指定子集及历史44/45均保留，46/47原工件/F1/逐类仍缺。
- [四seed汇总](../../docs/ben_four_seed_three_model_20261017.md)、[CSV](../../results/ben_four_seed_three_model_20261017.csv)、[逐seed固定/soft分别对Swin的全部类别](../../docs/ben_seed42_seed43_vs_swin_20261013.md)、导师中文汇报/英文草稿及当前状态已同步。研究工作树bigearth_comparison.csv仅追加一个正式soft42条目，旧行字节不变；主表CSV由artifact-tool author/inspect/render，统计先按原精度计算再四舍五入两位。其余AID/NWPU/seed43记录与46/47 CSV保持。无新训练、外部发送、Git提交推送。

正式阶段命令（保存的argv；冻结调度负责入口加载、精度和资源记录）：

```bash
python -B -u /home/featurize/work/GAViT_SOFT_20260927/train_bigearth.py --model gavit --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits --epochs 30 --batch_size 32 --lr 0.0003 --seed 42 --num_regions 16 --grouping soft_prototype --edge_type sparse_hybrid --knn_k 5 --gat_hidden 256 --gat_heads 4 --gat_layers 2 --integration token_feedback --dropout 0.1 --run_stage formal --run_tag bigearth_soft42_20261013_01_gavit_train --pretrained_path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --checkpoint_path /home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth
```

```bash
python -B -u /home/featurize/work/GAViT_SOFT_20260927/test_bigearth.py --model gavit --data_dir /home/featurize/work/GAViT_Project/bigearth_files/splits --ckpt /home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth --batch_size 32 --seed 42
```

---

## 2026-10-17 — 用户报告实例已退还；等待持久化结果核对

- 用户称“实例被退还了，应该是跑完了”，未提供退出码/summary/test指标；只记录实例退还的用户声明，不能推断已训练或测试完成。此前最后技术回传仍是队列分派PID5807，首批运行也未回传。正式BEN soft42指标继续null。
- [状态](../../results/bigearth_soft42_status_20261013.json)与[查看说明](../../docs/run_ben_soft42_20261013.md)更新为读取持久化work下bigearth_soft42_20261013_01的总run.exit、gavit_train/run.exit、gavit_test/run.exit、summary.json和run.log；[只读命令及来源/完成条件](../../results/evidence/bigearth_soft42_result_check_20261017/user_report_and_read_plan.json)保存。仅检查文本/JSON不需要GPU、原始图像、解压、重新预检或训练；无法访问终端时取回同名持久化文件。
- 完成判据仍为总/训练/test三项exit0、summary complete/seed42、final_epoch30、完整119825×19test及完成标记；缺文件或非0先看原日志诊断，不按实例归还推测，也不重训。旧PID仅历史身份，不再据其做活跃进程判断。
- 两条只读命令Bash/zsh语法核验通过；本轮没有直接服务器访问、CUDA执行、代码或CSV改动、分数新增、外部发送、Git提交推送。

---

## 2026-10-16 — BEN soft42队列已分派PID5807；首批运行尚待核验

- 用户回传新BIGEARTH_SOFT42_PREFLIGHT_PASS与BIGEARTH_SOFT42_LAUNCH_DISPATCHED，PID5807、正式输出bigearth_soft42_20261013_01，training_start_verified=false。保存[原文](../../results/evidence/bigearth_soft42_launch_20261016/server_return.txt)、[预检](../../results/evidence/bigearth_soft42_launch_20261016/preflight.json)、[分派JSON](../../results/evidence/bigearth_soft42_launch_20261016/launch.json)及[核对](../../results/evidence/bigearth_soft42_launch_20261016/validation.json)。首个JSON缺左花括号仅解析时补入，原始字节/SHA保留。
- 与初次预检相比，除库存多出当前smoke的metadata/summary外，源码/划分/旧对照/hash/GPU UUID/软件与精度相同；formal_or_partial/unreadable仍空。旧37.0547与当前37.5025均为工程smoke，未当正式结果。此次粘贴未包含修复PASS标记/更新SHA，不据此宣称修复产物独立确认；当前能确认的是队列分派，不能确认首批、GPU进程/显存或训练完成。
- [状态](../../results/bigearth_soft42_status_20261013.json)记为launch_dispatched_first_batch_pending；server_training_started为null/未知、server_training_start_verified=false，正式指标仍null。下一步仅nvidia-smi和总run.log末尾，不重复任何安装、smoke、修复或启动。
- 对照SHA8ac0083f的实际冻结基座确认阶段日志/退出路径为gavit_train/run.log、gavit_train/run.exit及gavit_test/run.exit。原操作说明误写process.log，旧status helper误读stage.process.exit；本轮只更正[说明与读取命令](../../docs/run_ben_soft42_20261013.md)，结束后直接读真实三项退出文件与summary，不以旧helper的pending判失败，不改正在运行的服务器文件。
- 本轮只有回传/路径核对和记录，无GPU执行、代码修改、训练重启、正式指标新增、CSV修改或外部发送/Git提交推送。

---

## 2026-10-16 — BEN soft42工程产物完成；复现并修复两处调度错误，正式启动待回传

- 用户回传当前RTX4090的96train/64val、seed42、1轮训练/validation/保存/重载/图探针完成标记。完整[原文](../../results/evidence/bigearth_soft42_smoke4090_20261016/server_return.txt)、[相同的stdout与cat摘要](../../results/evidence/bigearth_soft42_smoke4090_20261016/summary.json)归档；参数31466766、CUDA重载差4.76837158203125e-7、实际80边/48+32、8图3快照和S/W归一化均与既定配置相符，GPU UUID/software/precision与本次预检一致。checkpoint回传SHA为4a6ae6e43001ffe5d729f7eb4e50ef9da4951e3b8422b2e5342b02f88a813398，summary与metadata一致；实际文件字节仍由服务器启动检查核验。本次本地未下载权重或运行CUDA。
- 工程耗时27.455166778秒/入口22.720187440秒、allocated3932788736/reserved4242538496字节。val mAP37.502483541378865%、macro-F1 1.5037593984962405%仅为工程子集（缺类），不进入正式BEN19结果、预算推断或候选选择。正式test指标保持null。
- 原外层短测命令抛FileNotFoundError：冻结训练包装用atexit写run.exit，控制器却在退出前读取；解释了Traceback后cat出现内部exit_code=0。原始外层退出码未回传，不能把该内部标记当作外层命令exit0。本地真实子进程fixture复现，移除过早读取，保留独立launch的完整检查。继续审计正式队列时发现ns['b']保存接口错误（42预检的b为函数局部），真实CLI fixture复现KeyError:b；改用队列自身的hash验证load_base().save，不改预检/模型配方。
- 新两项回归加既有检查共21项通过（42共10、43共11）；全量pytest仍因本地缺pytest未执行。原v1包/源码/失败记录保留，新v2首次安装包包含两处修复。当前服务器未修复，不能直接启动原v1队列；[带SHA校验/备份的修复并启动命令](../../results/evidence/bigearth_soft42_smoke4090_20261016/repair_and_launch_command.txt)只更新两个外部调度文件及manifest，先复用并核验本次短测，再进入已授权30轮→best全test。命令真实文件fixture证明备份准确、工件字节不变、拒绝重复修复/已有正式输出；最终GPU启动被拦截，不是服务器执行证明。详见[诊断与核对](../../results/evidence/bigearth_soft42_smoke4090_20261016/validation.json)。
- [当前状态](../../results/bigearth_soft42_status_20261013.json)和[操作说明](../../docs/run_ben_soft42_20261013.md)同步；不重训smoke、不删除旧输出、不换实验目录绕过异常。正式启动/精度待回传，历史CSV/其他seed/模型和导师邮件保持，无外部发送、Git提交或推送。

---

## 2026-10-16 — 收到BEN soft42服务器预检PASS；当前4090短测待执行

- 用户回传BIGEARTH_SOFT42_PREFLIGHT_PASS/seed42/new_training_started=false，完整[原文](../../results/evidence/bigearth_soft42_preflight_20261016/server_return.txt)、[规范化JSON](../../results/evidence/bigearth_soft42_preflight_20261016/server_return.json)与[核对记录](../../results/evidence/bigearth_soft42_preflight_20261016/validation.json)归档。原粘贴省略首个左花括号，仅为解析补入，原始字节和SHA保留；不将本地核对当作独立服务器读取。
- 三划分237871/122342/119825及SHA、冻结ed480e6、既有soft43的软件和FP32/TF32设置匹配。当前RTX4090 UUID为GPU-44825cf0-bd98-bfdc-efbc-0100238a6789，型号不变、物理卡变化正常记录。库存formal_or_partial/unreadable均为空，仅旧20260928的seed42工程smoke；未以该37.0547%小样本val分数替代正式结果。
- 旧Swin42 checkpoint/test log SHA首次随本次回传记录，70.9/64.2/76.6与既有打印相符，仍无sidecar，不新增其源代码/训练身份或权重与日志强绑定的证明。固定42 metadata与返回checkpoint SHA一致，cd39be9、best13、final30和72.0/66.0/76.7符合档案；历史代码/硬件混杂保留。
- [当前状态](../../results/bigearth_soft42_status_20261013.json)及[运行说明](../../docs/run_ben_soft42_20261013.md)更新为预检通过、当前卡96/64一轮smoke待回传；沿用已安装20261013调度和输出身份，不重新安装/换目录。正式训练/测试未回传，42正式指标继续null，未改CSV、模型或其他seed结果。本轮无GPU执行、外部发送或Git提交推送。

---

## 2026-10-13 — 用户授权补齐BEN软分配seed42；本地调度已准备

- 用户明确选择补跑BEN soft42，使指定42/43/46/47四seed的三模型主指标齐全；已有43/46/47保持，42结果仍为空。研究问题是补足缺项与观察多seed/逐类表现，不根据结果挑正向seed或重训旧Swin/固定42。先查服务器已有正式/部分checkpoint，能复用则不新增训练。
- [运行说明](../../docs/run_ben_soft42_20261013.md)、[当前状态](../../results/bigearth_soft42_status_20261013.json)已建立；独立脚本在既有研究工作树中，正式ed480e6模型源码不改。沿用soft43的K16局部原型/sparse-hybrid 48+32/token_feedback与237871/122342/119825、30轮、batch32/workers4、AdamW3e-4/wd1e-4、Cosine30、BCE/阈值0.5、原FP32/TF32及预训练SHA，仅计划改变训练seed42，记录当前4090 UUID。使用新工程及正式输出目录，从ImageNet初始化，验证选best后完整test，保存历史/探针/19类AP和预测/资源/退出码。
- 本地新增8项安全检查、原43调度11项回归、真实冻结CLI解析、AST、Bash/zsh、载荷SHA、安装/重复安装fixture通过。全量python3 -B -m pytest因本地无pytest未执行；未做本地CUDA/模型训练。服务器预检/短测/正式启动均尚未回传，未宣称已启动或填新指标。完整命令、SHA与限制见工作树results/evidence/bigearth_soft42_prepare_20261013/preparation_validation.json。
- 旧对照Swin42日志70.9/64.2/76.6和固定42 cd39be9/best13/30轮、72.0/66.0/76.7复用；前者来源记录有限，后者历史4090训练/3060测试，与新soft42代码/设备差异保留。上一soft43的4090队列约6h22m/按旧价1.67元每小时约10.64元仅用于预算；新任务预估6–8h/10–14元，准备/空闲另计。原CSV/正式分数/图稿/导师邮件保持；无外部发送或Git提交推送。

---


## 2026-10-13 — 主仓库逐seed类别展示补入BEN42

- 按用户要求新增主仓库[BEN42/43逐seed分别对Swin](../../docs/ben_seed42_seed43_vs_swin_20261013.md)/[结构化57条可得AP比较](../../results/ben_seed42_seed43_vs_swin_20261013.json)。BEN42固定版对Swin19类加入，43两组原38条保留；42软分配无正式test/AP归档，工程smoke不替代。主表显示较大数值变化，完整类别附录保留，不做统计显著性或四seed类别稳定性结论。原实验/分数/图稿不改，无新GPU工作。

---

## 2026-10-13 — 单seed逐类主展示统一以Swin为参照

- 主仓库新增[BEN43两个模型分别对Swin](../../docs/ben_seed43_vs_swin_20261013.md)/[结构化数值](../../results/ben_seed43_vs_swin_20261013.json)，按用户纠正重排主展示与导师汇报；原跨seed/跨比较素材保留为附录。主表展示较大AP变化，全部19类×两比較保留；数值改善/退化不写成统计显著。BEN43soft对Swin沙滩+12.47、内陆湿地+2.64，避免混用对fixed的+15.17/−0.26。原四seed主性能、模型、图稿不改，缺少46/47逐类状态仍保留，无新推理/训练。

---

## 2026-10-13 — 主仓库逐类别收益/退化分析补充

- 新增主仓库[逐类分析](../../docs/per_class_findings_20261013.md)/[结构化数值](../../results/per_class_findings_20261013.json)：NWPU42的三种比较各八条极值，BEN43完整19类三模型AP（soft对fixed 8升/8降/3舍入不确定），BEN固定对Swin42/43，以及AID42/43完整30类。图册3纠正/5共同错误与正式test类别退化分开，soft对fixed的火车站/教堂类整体退步不因个别图册纠正而抹去。
- 指定四seed42/43/46/47的主性能表不变；46/47逐类与NWPU43完整逐类未回传，不推测四seed类别稳定性。导师汇报已链接补充，原实验/图稿/CSV/模型保留，本轮仅离线复算与整理。

---

## 2026-10-13 — 解释可视化完成状态与语义一致性证据

- 重新核对90例NWPU图册/180SVG及3纠正/5共同错误，并查看教堂/河道实图；BEN图反馈四例PDF/300dpi PNG已归档。主仓库邮件与汇报已将泛泛的semantic coherence表述改为具体制作/审阅状态和语义质量未量化的范围。既有人工审阅没有一致地物完整性改善证据，但不能将跨多个区域直接视为分类失效；图边可见也不是GAT注意力或语义因果归因。性能主表仍为42/43/46/47，BEN图稿明确历史seed44反馈诊断身份。3×3/4×4受控比较未完成，原图/指标不改，无新GPU工作。

---

## 2026-10-13 — 按用户指定用42/43/46/47四seed作为导师汇报主表

- 用户明确指定本次汇报使用seed42、43、46、47。主表从46/47两seed补充汇总改为该四seed的Swin/固定版均值±样本SD，并统一两位小数。AID为96.37±0.48%→96.83±0.51%（配对均值+0.46pp），NWPU为93.81±0.11%→94.00±0.19%（+0.20pp），BEN为70.90±0.22%→71.65±0.39%（+0.75pp）。NWPU43仍负，不能把四seed解释为每次均优于baseline。
- 软分配按相同seed完整结果比较：NWPU四seed齐全；BEN soft42缺正式归档，仅43/46/47三seed的三模型配对，不用45替换42或与四seed均值直接相减。AID soft未运行。新46/47 F1及原始工件仍待补。汇报JSON新增指定子集与统计，原全量rows/summary、原CSV及历史44/45保留；邮件和中文汇报明确本次是已获得结果上的汇报子集，不作为全部预设seed的稳定性证明。
- 可视化和效率分析段落沿用，没有新deadline表述、外部发送或GPU工作。更改前两份草稿保留于证据目录，全部新统计均从原始可得精度计算后再显示。

---

## 2026-10-12 — 导师汇报删除deadline与倒排时间表

- 用户明确导师未给具体DDL；主仓库中文进度汇报与英文邮件草稿已删去截止日期、月底交付和倒排时间表。后续事项及外部方法复现咨询不附截止日期；两位小数实验表和可视化沿用。历史研究安排保留为历史记录，不作为导师给定期限。

---

## 2026-10-12 — 导师汇报统一两位小数；NWPU46/47回传精度确认

- 用户确认NWPU seed46/47为四舍五入后的OA，要求所有seed统一四舍五入至两位小数；中文导师汇报、英文邮件草稿及本批核对说明的单seed、均值/样本SD、配对差展示已统一。统计按原始可得精度先计算再展示，不从已舍入的三位小数二次舍入，不把回传精度补成额外观测。
- 当前CSV原值和全部已有数值JSON/统计数值保留；审计及汇报JSON新增展示规则和NWPU六个OA的两位小数回传来源，原始工件验收状态仍保持。BEN一位小数补尾零仅为排版。用户认为可视化部分可以，现有图册和分析段落沿用；该反馈不升级为语义改善或因果验证证据。
- 展示边界核对：全量NWPU固定均值94.004761…显示94.00；新46/47固定均值94.075显示94.08，soft均值93.545显示93.55，soft−Swin均值−0.175显示−0.18pp。当前全量种子范围和正负结果不变；无训练、推理或外部发送。

---

## 2026-10-12 — 主仓库AID确认更正与全量seed导师汇报

- 用户确认AID固定46为97.12%、Swin47为95.72，已更正主仓库[CSV](../../results/multiseed_test_results.csv)及[核对说明](../../results/seed46_seed47_audit_20261012.md)/[JSON](../../results/seed46_seed47_audit_20261012.json)。原表与更正前审计保留，非AID七行字节不变。AID本批Swin96.140±0.594%、固定96.600±0.735%，配对增益+0.56/+0.36pp，均值+0.460；不再以最近候选代替用户确认值。新批原始工件与辅助指标仍待核验，旧正式表不扩大证据口径。
- 主仓库新增[中文导师汇报](../../docs/supervisor_progress_report_20261012.md)、[英文邮件草稿](../../docs/supervisor_email_draft_20261012.md)、[完整精度及来源](../../results/supervisor_progress_data_20261012.json)。保留AID5、NWPU4、BEN6个全部已记录固定版配对及全部可得soft，不排除AID44、NWPU43、BEN44/45负面seed。混合历史均值仅为描述性进展，不能消除硬件/代码/最终轮及证据完整度差异。
- 导师要求对应状态同步到[邮件要求表](docs/supervisor_latest_email_received_20260924.md)：双数据集三模型多seed已记录，新46/47完整身份/F1待补；HGTNet/DFAGCN统一协议实测与3×3/4×4受控比较未完成。报告含图册/效率、写作进度与10-31拟议安排。独立核对24项CSV算术、15组主指标/全量统计、SHA与报告链接通过，见[验证JSON](../../results/evidence/seed46_seed47_reported_20261012/correction_and_report_validation.json)。本轮仅更正/汇报准备，无邮件发送、新GPU工作或Git提交/推送。

---

## 2026-10-12 — 主仓库新增seed46/47回传，AID两项抄录错误尚待原值

> 更正前历史状态：97.12/95.72已由用户明确确认并在上方条目归档，下文候选与待原值描述不再是当前状态。

- 新结果位于主仓库[数值表](../../results/multiseed_test_results.csv)，不是本worktree的正式结果目录。[核对报告](../../results/seed46_seed47_audit_20261012.md)与[JSON](../../results/seed46_seed47_audit_20261012.json)已保存。原表24项均值/样本SD/均值差算术符合打印精度；用户明确本批使用RTX4090、UUID不同于以前，测试数量和配置均不变。原始run/summary/退出状态/权重与split身份链仍缺，只归档用户回传与数值审计，不并入旧正式42/43汇总。
- AID固定46的97.13、Swin47的95.73被用户确认是记录错误。5000图OA步长0.02pp，最近合法候选97.12或97.14、95.72或95.74，无法唯一选末位2或4。候选和实际成绩严格区分，原CSV不改，AID实际均值/差值待correct或原test报告；本批NWPU固定−Swin均值+0.355pp、BEN+0.650pp，soft在NWPU两次负、BEN一升一降。BEN soft−Swin均值+0.05pp受一位小数舍入影响，不能认定整体提升或等效。保留所有历史正负结果，无新训练/推理或Git提交/推送。

---


## 2026-10-12 — 按用户范围汇总三模型seed42/43在AID、NWPU和BEN的正式性能

- 新增[三数据集汇总](results/three_models_seed42_seed43_overview_20261012.md)与[完整精度JSON](results/three_models_seed42_seed43_overview_20261012.json)，仅整合现有正式test、原始来源SHA与同seed差值。18个组合中15个有已归档结果；AID soft42/43和BEN soft42缺正式归档，不用seed45或smoke/val分数填空。固定模型使用当前稀疏混合4×4版本，不混入BEN历史未修正/纯kNN候选。AID与NWPU的OA/正确数、两模型/三模型macro-F1及BEN mAP/macro-F1/micro-F1从既有JSON/CSV核对。
- 42/43范围内：固定相对Swin的AID OA差+0.60/+0.30pp、NWPU +0.170635/−0.091270pp、BEN mAP约+1.1/+0.6pp；soft的NWPU差−0.063492/−0.230159pp，BEN43约+1.767571pp。BEN旧打印精度、fixed43保存21轮和历史代码/硬件差异、NWPU42/43跨GPU均保留。单标签micro-F1未单独归档，不推导补填；OA与mAP不跨任务平均。此表只是所请求42/43子集，不能替代AID既有三seed或BEN历史四seed总表。未修改历史分数、执行新训练/推理或提交/推送，新增seed的范围选择仍待用户回复。

---


## 2026-10-12 — BigEarth soft43最佳权重与资源核验收尾，seed43补实验完整归档

- 用户回传一次只读紧凑摘要；[原始字节](results/evidence/bigearth_soft43_complete_20261012/reported_summary_output.txt)、[结构化摘要](results/evidence/bigearth_soft43_complete_20261012/resource_summary.json)、[身份/历史/资源核验](results/evidence/bigearth_soft43_complete_20261012/resource_validation.json)已保存。complete/seed43、gavit_train与gavit_test各exit0、1–30轮编号、237871/122342训练/验证数量、31,466,766参数、代码ed480e60fea03ed786fdba70c144de5e815c34f5、预训练及三split SHA均与批准预检一致。实际train/test同一RTX4090 UUID GPU-cede40c0-f93d-7c8e-1006-cf6183e53592、软件/FP32/noAMP/TF32配置一致；两阶段checkpoint SHA与先前精确test返回均为c01403d67df77ec03f173877ca3466e1211d06015cd83dc89672629595e387a7。本轮仅审计用户回传，不直接连接GPU服务器或下载原模型文件。
- best为第8轮，完整精度val mAP78.99004691745016%、macro-F1 71.95576173181863%，metadata舍入78.9900%一致。末轮30的val mAP75.6996819828042%，较best低3.290364934646pp；test使用best，正式mAP72.86757085064232%、macro-F1 66.67389161697625%、micro-F1 76.90597117759339%未改变。摘要在服务器完整history上按只读命令断言选best；本地只有1–30轮编号与首/best/末三行，未独立复算全部30行曲线，不据末期回落证明具体机制。旧日志打印79.0%/原记录待摘要属于此前历史状态，本条补齐，不抹除历史。
- 正式输出仍为`/home/featurize/work/bigearth_soft43_20261011_01`，原完整启动命令见[冻结命令](results/evidence/seed43_bigearth_smoke4090_20261011/bigearth_launch_checked_command.txt)，只读摘要命令见[核验命令](results/evidence/bigearth_soft43_complete_20261012/read_summary_command.txt)。正式训练30轮阶段22679.135110秒=6小时17分59.14秒；best完整119825 test阶段197.353206秒=3分17.35秒；整批22941.731309秒=6小时22分21.73秒。整批与两阶段和的差65.242993秒不进一步归因，不用末轮754.068秒乘30或test循环02:53代替完整资源。
- CUDA allocated/reserved峰值：训练3.662695/3.943359GiB，test0.595415/1.027344GiB；主进程RSS分别2.047615/1.475349GiB，不含全部loader worker。有效训练吞吐314.656179图/秒=237871×30/完整训练阶段秒，分母含验证/IO/探针/工件检查，不写成纯模型速度或公平效率对比。固定8val初始/best/final三探针验收，S行/W列归一化最大误差2.98e-7，真实48空间+32特征=80边；仅支持工程路径正常，不证明语义改善。
- 按用户1.67元/h，BigEarth正式队列折算10.642414元（约10.64）；NWPU43三模型3617.327163秒折算1.678038元（约1.68）。两批合计26559.058473秒=7小时22分39.06秒、12.320452元（约12.32）。[费用记录](results/evidence/bigearth_soft43_complete_20261012/seed43_formal_cost.json)明确仅正式队列运行，解压/预检/短测/租赁空闲另计，不当实际账单。
- [正式三模型记录](results/bigearth_soft43_ablation.md)、test JSON、BigEarth主CSV新soft43行、[双seed摘要](results/bigearth_three_model_two_seed_summary.md)、[当前索引](results/soft_assignment_cross_dataset_update_20261011.json)、seed43状态/排期同步。只填已验证best与资源；旧CSV各行字节/指标、现有逐类AP、两seed统计及历史单seed快照均保留。本次授权的NWPU43三模型和BigEarth soft43训练/test/资源完整收尾，无需继续索取同批日志或重跑。最终三seed、公平效率/外部方法实测等论文缺口仍在，旧43轮数/版本/硬件混杂仍披露；本轮未新训练、改模型、提交或推送。

---


## 2026-10-12 — BigEarth soft43正式完成，三模型与双seed对比归档

- 用户回传总run.log尾40行与总/训练/test各exit0、精确metrics.json；[原始字节](results/evidence/bigearth_soft43_complete_20261012/reported_output.txt)、[解析与核验](results/evidence/bigearth_soft43_complete_20261012/validation.json)保存。epoch30、两STAGE COMPLETE、BIGEARTH SOFT43 COMPLETE、3745/3745与119825 test均符合；19类AP的均值精确等于mAP，阈值0.5。实际test设备/UUID/软件/精度与批准预检相同。支持正式训练/test已完成，不需要重训；完整资源与best epoch摘要仍待只读补齐。
- soft43正式test mAP72.86757085064232%、macro-F1 66.67389161697625%、micro-F1 76.90597117759339%；checkpoint SHA c01403d67df77ec03f173877ca3466e1211d06015cd83dc89672629595e387a7。相对旧Swin43打印71.1/64.1/76.0约+1.767571/+2.573892/+0.905971pp；相对fixed43打印71.7/65.7/76.8约+1.167571/+0.973892/+0.105971pp。旧对照仅一位小数，差为近似、参考舍入约±0.05pp；本seed三项方向均正，不把小数扩展冒充旧分数精度。[正式三模型表](results/bigearth_soft43_ablation.md)。
- [19类三模型CSV](results/bigearth_seed43_three_model_per_class.csv)按旧AP打印误差区间分类：对fixed43可确定8升8降3方向不确定；对Swin43可确定11升7降1不确定。soft−fixed最大为海滩/沙丘/沙地约+15.173898pp、海岸湿地+8.213406pp、多年生作物+1.397723pp；最大下降草地/稀疏植被−2.617726pp、工业商业区−0.986638pp、复杂耕作−0.619569pp。完整AP保留，不泛化为所有场景改进或因果归因。
- [43/45共同seed三模型表](results/bigearth_three_model_two_seed_summary.md)与CSV/JSON新增；soft−fixed mAP两次均升，中心值平均约+0.873127pp、样本SD0.416407pp；macro-F1两次均升，micro-F1一升一降且均差约−0.106566pp。soft−Swin mAP43升45降，平均约+0.463780pp、样本SD1.843839pp。相关统计含43旧打印中心值，仅描述n=2；不与Swin/fixed全四seed错配平均，也不声称稳定优于Swin。NWPU42/43 soft均低于Swin，仍无跨数据集一致优势。
- 限制继续保留：Swin43 full30/best6、fixed43 saved21/best8、旧bb9f1cf与历史4090 train/3060固定test；新soft43 full30/ed480e6/current4090。45三模型full30，但旧对照deb2b45、不同物理2080Ti，43/45的soft也2080Ti/4090不同。同seed不消除混杂，正面43和负面45均保留，不根据结果替换或追加seed。最终三seed/公平效率/外部方法实测仍未齐备。
- 第30轮loss0.0600898171325、val mAP75.6996819828%、macro-F1 71.3216787940%、nextLR0；best val只打印79.0%，精确值与best epoch未给出，CSV best值明确记打印精度、epoch留空。最后一轮754.068秒、test循环02:53不能当总资源/成本；总/阶段墙钟、峰值显存与有效吞吐待只读摘要。本地未下载原权重/predictions/probe/history；完成标记对应已核验冻结helper保存和验收路径，不推测额外机制。
- 主BigEarth CSV只追加soft43新行，所有旧行字节/分数保持；专门三模型/双seed/逐类表与日记/状态/排期/论文当前索引同步。原10-01单seedJSON和草稿保留为历史，新增当前指向，旧约96%NWPU表不混入新协议。交付[一次只读紧凑摘要命令](results/evidence/bigearth_soft43_complete_20261012/read_summary_command.txt)补精确best/轮数/资源，语法与字段路径核对通过；无GPU调用、新训练、Git提交或推送。

---


## 2026-10-11 — BigEarth soft43当前4090工程短测验收通过，交付正式训练→测试

- 用户回传`run.exit=exit_code=0`与完整summary；[结构化转录](results/evidence/seed43_bigearth_smoke4090_20261011/user_return.json)、[字段核验](results/evidence/seed43_bigearth_smoke4090_20261011/validation.json)保存。真实96train/64val、1轮、seed43、ed480e6 clean、31,466,766参数、19类BCE与局部原型配置、当前RTX4090 UUID/软件/精度均与批准预检一致。checkpoint SHA949504d4daf1d943be0dde59ff13eaea250212b0346126030d0f41146a580232，与回传metadata一致；原权重/探针没有下载到本地。
- 前后向/验证/保存/严格CUDA重建完成；logit最大差2.384185791015625e-7≤1e-5，实际48空间+32特征=80边。固定8张val、3份探针快照归一化S/W最大误差均2.384185791015625e-7；epoch1/best/final与训练状态验收由原包装器执行。训练loss0.5083718399、小样本val mAP31.6302547%、macro-F1 4.5297114%；只作为工程流程有限值，不进入完整19类test表、不判定模型效果。
- 训练入口11.863476秒，总含重建/检查14.093355秒；allocated峰值3.662695GiB、reserved3.951172GiB，主进程RSS1927744KiB不含所有worker。该时间/显存是小样本完整工程口径，不线性外推正式30轮费用或推理速度。[工程报告](results/bigearth_soft43_smoke_validation.md)。
- 为直接接已授权正式任务，交付[新的单行启动命令](results/evidence/seed43_bigearth_smoke4090_20261011/bigearth_launch_checked_command.txt)。在现有已验证parent之前增加stdlib只读核对：原工程exit/summary逐字段、metadata/真实权重SHA、批准预检、子集CSV/source SHA、必要探针与训练状态文件；重新执行原完整只读预检，并要求与本次已通过预检完全一致，之后才独占建立正式work/bigearth_soft43_20261011_01与后台进程，保存smoke_validation.json。原helper SHA7346c6aa25426998f44eeb55aa9aa33c1f43357d4cbabb3951413e5aa8d1b832、模型代码、30轮argv/优化/选best规则保持；旧交付文件保留，不能重复运行旧短测。
- Python AST、Bash/zsh、压缩载荷往返与原helper字节核对通过；9个stdlib分离fixture验证正常工件可接受，以及非零exit/改summary/权重/metadata/预检/CSV/source/缺探针全部拒绝、只读检查不修改原工件，exit0。[命令核验](results/evidence/seed43_bigearth_smoke4090_20261011/command_validation.json)、[fixture](results/evidence/seed43_bigearth_smoke4090_20261011/artifact_gate_validation.json)。本地没有CUDA或真实后台GPU任务。
- 下一步用户执行一次正式单soft43训练30轮→验证mAP best的119825test。出现PID后仍须GPU进程/显存、真实设备/237871+122342/首训练batch日志才记启动成功；首个完整epoch后按精确EPOCH RESOURCE更新时间预算。暂保留4090估计6–8h、按1.67元/h10.02–13.36元，非实测/实际账单。旧Swin43完整30/best6与fixed43保存21/best8、历史代码与GPU差异继续披露，不重训旧对照/NWPU，不据smoke改分组或阈值。
- 日记/当前状态/执行说明/排期同步。当前正式训练尚未回传启动或任何新test指标；无Git提交/推送。

---


## 2026-10-11 — NWPU43正式结果与双seed归档；BigEarth soft43预检通过

- 用户回传NWPU完整紧凑摘要，[原文](results/evidence/seed43_nwpu_final_20261011/server_return.txt)保留字节SHA，[解析与核验](results/evidence/seed43_nwpu_final_20261011/validation.json)确认complete/总exit0、43、ed480e6、冻结manifest/pretrained、各25200test、calibrate30轮、23/26/29自选预算与6300refit、各自refit/test权重SHA一致、有限指标/资源及重建差≤1e-5。preflight的training_started=false是训练前快照，不否定当前complete。本轮本地审计用户回传字段，没有直接连接GPU服务器或下载原权重/逐样本数组。
- 正式test OA：Swin93.85714285714286%、fixed93.76587301587301%、soft93.62698412698413%；macro-F1分别93.83452989448766/93.74351252940127/93.58782117143805%。fixed−Swin OA−0.091269841270pp（净少对23），soft−Swin−0.230158730159pp（净少对58），soft−fixed−0.138888888889pp（净少对35）。保留负面结果，不追加重试或选不同seed；43逐类/纠正与新增错误拆分尚未读取，不据净数推测场景。[正式表](results/nwpu_seed43_test_comparison.md)。
- 从seed42原始JSON精确合并：[六行CSV](results/nwpu_two_seed_comparison.csv)、[双seed统计](results/nwpu_two_seed_summary.md)。固定42正、43负，平均ΔOA+0.039682539683pp、样本SD0.185194633168pp；soft两次负、平均−0.146825396825pp、样本SD0.117851130198pp。n=2仅描述，不宣称稳定/显著优势、等效或方法整体无效。42为2080Ti、43为4090，seed与硬件一同变化，不能单独归因。最终三seed仍有缺口；新协议不混入旧约96%的表。跨数据集历史单seed快照保留并指向新统计。
- NWPU正式输出work/nwpu_three43_20261011_01；GPU-cede40c0-f93d-7c8e-1006-cf6183e53592/RTX4090/driver610.57.04/24564MiB。总3617.327163秒=1小时0分17.33秒，按1.67元/h折算1.678038元（约1.68），不含准备/短测/空闲或实际账单。参数Swin27555495/fixed31386920/soft31486760；六训练阶段峰值显存/耗时完整保存，口径含重建/IO，不当公平纯推理速度。固定/soft校准OA、完整43混淆/逐类以及公平推理效率待现有产物分析。
- 用户随后回传BIGEARTH_SOFT43_PREFLIGHT_PASS，[消息转录](results/evidence/seed43_bigearth_preflight_20261011/user_return.json)和[核对](results/evidence/seed43_bigearth_preflight_20261011/validation.json)保存。固定CSV237871/122342/119825、同当前4090 UUID、CUDA12.1/cuDNN8902/8线程/Python3.11.8及包版本，FP32/noAMP/matmulTF32false/cuDNNtrue符合预先门禁；原Swin43完整30/best6、fixed43保存21/best8和打印test71.1/71.7核验回传。旧2080Ti环境参照/4090训练/3060fixed测试、bb9f1cf与新ed480e6差异如实保留，预检不消除混杂。new_training_started=false，暂无新soft43分数。
- 按AGENTS目标设备工程检查交付[BigEarth96/64单轮命令](results/evidence/seed43_bigearth_preflight_20261011/bigearth_smoke_command.txt)。复用以前已验证工程包装器和Random42抽样，训练seed43、真实19类BCE/4worker/原精度，独立work/gpucheck_bigearth_soft43_4090_20261011_01；先核对GPU空闲/本次软件精度、代码/预训练，再解码原图并保存CSV/RGB子集/哈希，单轮训练→checkpoint→CUDA重建→metadata/有限loss/epoch1状态/探针归一化与48+32边验收。全部验收后才exit0/完成标记，工程分数不入论文表。Python AST/实际ed480e6 CLI/Bash/zsh/压缩载荷往返与成功标记顺序核对通过，本轮没有本地CUDA或新服务器训练。
- 下一步回传工程exit/summary，符合后进入已授权BigEarth单soft43完整30轮与best test；不重训旧Swin/fixed43，不重跑NWPU，不搜索架构或新增付费seed。日记/状态/执行步骤/排期已同步；无Git提交/推送。

---

## 2026-10-11 — NWPU43三模型九阶段全部验收完成，正式数值待回传

- 用户回传九次`STAGE_COMPLETE`（三calibrate→三refit→三test）及`NWPU43_THREE_MODELS_COMPLETE`，[原文](results/evidence/seed43_nwpu_queue_complete_20261011/server_return.txt)与[解析](results/evidence/seed43_nwpu_queue_complete_20261011/summary.json)保存。实际helper只在阶段process exit0和原入口run.exit0/身份/轮数/权重/history/25200test/45类混淆与预测文件检查通过后打印STAGE_COMPLETE；总完成标记在summary.json保存之后。支持本批训练、评估及调度验收已完成，不再将fixed/soft/refit/test记为待运行。
- 九阶段wall秒分别为calibrate506.992664/644.216864/644.594302，refit427.052295/592.482899/623.472711，test54.979153/61.762520/59.646505（Swin/fixed/soft顺序）。合计3615.199913秒=1小时0分15.20秒，约1.004222小时；按用户1.67元/h约1.677051元（显示1.68元）。这是阶段时间折算，含阶段验收，不包括队列阶段间预检、之前数据恢复/工程短测/租赁空闲，不能当实际账单或纯模型推理效率。
- 终态总run.exit、实际selected_epochs、三模型正式OA/macro-F1、参数/显存/完整运行环境和checkpoint SHA未在这段队列日志中展开。现仅保留之前Swin校准曲线93.095238%/首个最高epoch23，不能填最终test或断言fixed/soft优于baseline。
- [紧凑只读回传命令](results/evidence/seed43_nwpu_queue_complete_20261011/read_summary_command.txt)提取总exit0、seed43、选轮、三25200test OA/F1与权重/manifest身份、六训练阶段轮数/数量/参数/峰值显存/耗时/重建/权重SHA及preflight；不输出完整混淆矩阵。Python AST/Bash/zsh/正常字段和非零exit拒绝fixture通过，无GPU工作。[命令核验](results/evidence/seed43_nwpu_queue_complete_20261011/command_validation.json)。
- 下一步回传正式摘要，更新专门test表/双seed比较；同时可执行已准备的[BigEarth soft43只读预检](results/evidence/seed43_gpu4090_20261011/bigearth_preflight_command.txt)，以实际数据/软件/旧43对照核验决定下一项执行。BigEarth没有新训练或分数，软分配三seed缺口仍保留；未重训NWPU、修改模型或Git提交/推送。

---

## 2026-10-11 — NWPU43正式Swin校准30轮返回完成标记，等待队列阶段确认

- 用户贴附件启动与tail日志，原始字节已复制到[回传原文](results/evidence/seed43_nwpu_swin_calibrate_20261011/server_return.txt)并记录SHA；[解析摘要](results/evidence/seed43_nwpu_swin_calibrate_20261011/summary.json)及[30轮打印曲线](results/evidence/seed43_nwpu_swin_calibrate_20261011/history_from_printed_log.csv)保存。正式输出为`work/nwpu_three43_20261011_01`；实际Swin/RTX 4090/calibrate、5040train/1260val、首batch loss4.019081及30/30轮连续日志、`SCENE CALIBRATE COMPLETE`均返回。支持本阶段真实CUDA训练已执行并到入口完成，不只是PID。
- Swin最大val OA93.0952380952381%，首次epoch23，23/26/28/29/30打印同分；按固定first-tie选轮规则预期best23，实际metadata与阶段验收尚未回传。final epoch30 OA同值、train loss0.001668。macro-F1、checkpoint SHA、精确重建差、峰值显存、整个阶段wall和process exit未给出，不补造；这只是校准val，不作为25200test或三模型胜负证据。
- 图片校验完成31500/31500、41.4秒；30轮纯train打印合计335.83秒（5分35.83秒），平均11.1943秒/轮，范围10.98–11.69秒。该时间不含validation/checkpoint/IO，仍等待process.json总墙钟；不直接外推全九阶段或BigEarth费用。
- 当前查看的`tail -f swin_calibrate.process.log`不会自动切换阶段；下一步Ctrl+C退出查看（后台队列不受影响），只读总run.log确认`STAGE_COMPLETE swin_calibrate`和当前START。按预先队列，随后应为fixed_calibrate，再soft_calibrate，再各自refit/test；当前未回传总队列状态/后续阶段或nvidia-smi，不能声称已确认GPU占用、队列通过全部验收或全部九阶段完成。
- [进度表](results/nwpu_seed43_calibration_progress.md)、执行说明/seed43状态/排期同步；主test表、正式模型与命令身份保持。没有重复启动或本地GPU任务，没有Git提交/推送，BigEarth soft43尚待本批结束后预检。

---

## 2026-10-11 — 当前4090软分配工程短测返回完成，交付正式九阶段启动

- 用户回传31500/31500原图decode/file/pixel verify（34.2秒）、真实RTX 4090、smoke360train/90val、首batch loss3.830680、实际80边（48空间+32特征）、1/1轮loss3.334170、val OA46.666667%、train_s1.70与`SCENE SMOKE COMPLETE`。[原文](results/evidence/seed43_nwpu_smoke4090_20261011/server_return.txt)与[摘要](results/evidence/seed43_nwpu_smoke4090_20261011/reported_summary.json)保存，未下载原checkpoint或完整report。
- 核对实际ed480e6入口：完成标记在checkpoint保存、CUDA重建≤1e-5、run.json complete及run.exit0写入之后，支持本次工程流程已完成；具体权重SHA、重建差、完整环境与资源数值仍未回传，不据小样本46.67%写正式效果或把train1.70秒按大数据外推完整预算。
- 为避免另一次回传等待，在[正式单行启动命令](results/evidence/seed43_nwpu_smoke4090_20261011/nwpu_launch_checked_command.txt)前加入stdlib只读工程产物核对：exit/status/seed/360+90/1轮/软分配/原配方/源文件与输入SHA/当前软件/真实图边/重建有限差值/best字节metadata/history/必需probe；再次执行原预检并核对最新UUID。全部通过才创建原计划正式目录及后台进程，保存smoke_validation.json。不再训练工程smoke，不改9阶段argv、模型代码或现有helper字节SHA；旧命令与全部工程产物保留。
- 验证13个CPU fixture场景：正常产物、exit错误、seed/配方/图/源/权重/metadata/快照/loss/NaN重建拒绝，以及真实后台启动/拒绝不建目录/重复不改产物；Python AST/Bash/zsh/载荷往返通过，[记录](results/evidence/seed43_nwpu_smoke4090_20261011/validation.json)。没有本地或远程新GPU任务；正式实验尚未回传PID/GPU占用/首batch，不能记为已启动。
- 下一步用户执行已授权的NWPU seed43三模型calibrate→各自选轮refit→25200test九阶段，一次启动；首阶段GPU占用/进程/首batch验证后更新状态。BigEarth soft43仍待预检，已有主表分数、历史边界与后续范围不变；未Git提交/推送。

---

## 2026-10-11 — NWPU43第二次只读预检通过，物理4090发生变化

- 用户再次回传`NWPU43_PREFLIGHT_PASS`，[第二次JSON](results/evidence/seed43_nwpu_preflight_20261011/preflight_gpu2.json)与[核对](results/evidence/seed43_nwpu_preflight_20261011/validation_gpu2.json)保存。源commit、manifest/pretrained SHA、seed43、划分数量与九阶段一致；显卡仍RTX 4090、driver610.57.04、24564 MiB，但UUID由`GPU-b1911368-563d-916d-cb16-e7c2c9ead04a`变为`GPU-cede40c0-f93d-7c8e-1006-cf6183e53592`。确认换了物理卡，不据此猜测具体实例/服务器更换原因。
- 本次仍为只读预检，明确`training_started=false`，没有`SCENE SMOKE COMPLETE`或360/90工程运行回传；不能记为短测/正式启动。前次身份与证据保留，本次命令片段未含载荷SHA，不补造为再次独立核对了完整载荷字节。
- 下一步继续原已交付的一轮soft_prototype工程短测，目标为当前4090；独立工程输出不变，若已有该目录先读取日志/exit/身份，不删除或盲目新起retry。工程通过后按已有正式队列执行。正式输出身份、冻结模型/配方、主表分数不变，BigEarth预检仍待回传；无新GPU训练或Git提交/推送。

---

## 2026-10-11 — NWPU43新版只读预检通过，准备4090单轮工程检查

- 用户回传`NWPU43_PREFLIGHT_PASS`；[回传内容与JSON](results/evidence/seed43_nwpu_preflight_20261011/preflight.json)和[本地核对](results/evidence/seed43_nwpu_preflight_20261011/validation.json)保存。所示helper SHA `bb50d71a98aaa44b14c7f604c2236218c8b663a90f1e89254d6fd524c0340e5e`与本地修正后字节一致；ed480e6、manifest/pretrained SHA、seed43、5040/1260/25200、单RTX 4090（同UUID、driver610.57.04、24564 MiB）和计划九阶段符合。回传内容布局做本地规范化，未从服务器下载原文件。依据执行路径，31500文件路径、GPU空闲和既有正式43产物门禁通过；不把只读检查当像素解码/软件环境/真实训练完成。
- 用户随后要求继续。按项目AGENTS训练前检查，为新4090实例准备原发布入口的一轮soft_prototype短测：seed43、360train/90val、ImageNet/冻结manifest、batch32/workers2/8线程/原精度与训练配方；检查实际CUDA前后向、validation、checkpoint保存/重建，单轮工程结果不入主分数表、不选择新配置。独立输出`work/gpucheck_nwpu_soft43_4090_20261011_01`，保留已有2080 Ti短测与所有正式产物，不重划数据或改正式模型代码。
- [单行命令](results/evidence/seed43_nwpu_preflight_20261011/nwpu_smoke_command.txt)与回传步骤写入[执行说明](docs/run_seed43_three_models_20261011.md)。从本地Git对象读取实际ed480e6 CLI，验证参数完整解析与阶段固定1轮；Bash/zsh语法通过，仅参数fixture，无模型/CUDA调用。正式九阶段命令不变，短测通过后进入已有队列，BigEarth soft43预检仍待回传。
- 当前只记录预检与交付工程命令，尚未获得新4090短测或首个正式batch回传，没有新训练指标、Git提交/推送。排期和seed43状态同步，既有论文主表数值保持。

---

## 2026-10-11 — 修正seed43待执行命令的旧GPU限制，接受已授权4090

- 用户执行旧NWPU只读预检后回传显卡拒绝：[原始错误](results/evidence/seed43_gpu4090_20261011/server_return.txt)。实际nvidia-smi身份为RTX 4090、UUID `GPU-b1911368-563d-916d-cb16-e7c2c9ead04a`、driver610.57.04、24564 MiB；旧脚本仍要求2080 Ti，未同步用户4090决定。此断言位于训练前，回传支持GPU识别成功，不代表完整预检、像素解码或CUDA训练已完成；无需因本错误重复解压。只读路径没有创建正式输出或checkpoint。
- 用同一GPU行本地复现NWPU原错误，同时复现BigEarth预检把GPU纳入旧soft45环境严格等值导致换卡拒绝。更新NWPU单卡RTX 4090解析与实际UUID/driver/memory记录；BigEarth允许用户已授权的历史2080 Ti→当前4090，仍严格核对CUDA/cuDNN/Python/包版本/精度，报告历史显卡和物理身份差异。各批次内部GPU身份与工作进程硬件核验保持，不删除门禁、不改变正式模型、训练配方或划分。
- 两批预检/后台启动共四条单行命令重新打包并绑定新SHA，现链接见[执行说明](docs/run_seed43_three_models_20261011.md)。新交付与失败回传独立保存到`results/evidence/seed43_gpu4090_20261011`；旧`seed43_prepare_20261011`载荷/命令/验证原样保留。只读失败没有正式运行，维持原计划输出身份，无需新retry编号。
- 新4090与错误GPU/软件/精度漂移、BigEarth历史固定43保存21轮边界等回归及复用helper共23 tests通过、exit0；Python AST/Bash/zsh/载荷字节往返/真实后台fixture拒绝与重复产物保护通过，[验证记录](results/evidence/seed43_gpu4090_20261011/validation.json)。仅CPU调度fixture，没有真实GPU模型或正式训练。
- 排期与预算同步已沟通的4090估计：NWPU三模型约1–2h，BigEarth soft43约6–8h，总7–10h；按用户1.67元/h约11.69–16.70元，12h为20.04元，可预留25–30元覆盖数据恢复/预检等。不是4090 soft43实测，首轮再更新。已有分数与历史硬件/版本/21轮边界保留。
- 下一步为用户重新执行新版NWPU只读预检并回传JSON；完整服务器预检与目标CUDA工程验证待完成，未启动新训练、未修改主表分数、未Git提交或推送。

---

## 2026-10-11 — 后续补充 seed 统一使用 RTX 4090

- 用户在核对 BigEarthNet 四 seed 的训练硬件后明确决定：后续计划增加若干 seed，新增训练统一在 RTX 4090 上执行。新增 seed 的具体数量、列表和模型范围尚未确定；本条记录硬件安排，不代表新任务已经启动。
- 同一 seed 内需要比较的 Swin、固定分组 GAViT 和软分组 GAViT 对照均使用 RTX 4090，并固定软件版本、各数据集已有精度设置、数据划分、预训练、训练配方及选轮规则。尽量在同一实例内顺序执行配对模型，记录 GPU UUID；同型号不同物理卡或环境变化如实披露。
- 研究问题是统一硬件和执行条件后模型间差值能否在新增独立 seed 上复现；主要实验变量为训练 seed。结果进入对应数据集的配对多 seed 表和三模型消融汇总。启动前固定完整 seed 列表、模型范围、预算、代码与输出身份，保留所有约定运行，不按结果筛选 seed。已有 checkpoint 不能代替新增 seed 的独立训练，但满足身份的既有同 seed 产物先审计复用。
- 既有 BigEarthNet seed42/43/44/45 的结果、硬件和中断边界全部保留；新增统一 4090 的实验按批次报告，并提供包含历史结果的汇总，不因换卡或负面结果删除旧运行。统一硬件有助于减少后续比较的混杂，但不能据此认定历史波动已被证明来自硬件，也不预先保证 4090 上会获得提升。
- 尚按 RTX 2080 Ti 编写的待执行命令（包括 seed43 准备说明）须在实际执行前按本次决定更新 GPU 身份与资源检查；原数据集精度与训练配方保持。已完成运行和正在执行的任务不由本条自动中断或重启。当前只更新本地日记与排期，没有租用实例、启动 GPU、改变正式分数或提交/推送 Git。

---

## 2026-10-11 — 补NWPU43三模型与BigEarth soft43，命令准备完成

- 用户纠正范围：固定版NWPU42正式test确实高于Swin（94.103175%/93.932540%），要求补43，并明确软分配必须纳入导师要求的NWPU/BigEarth三模型比较；随后选择“同时补BigEarthNet seed43软分配”。本批范围为NWPU43三模型九阶段，加BigEarth soft43训练/测试；保留所有42/45与旧43产物，不新增架构搜索。
- [冻结配置、单行预检与启动说明](docs/run_seed43_three_models_20261011.md)已准备：NWPU沿用ed480e6、同manifest/pretrained、5040/1260校准30轮，再按本次各模型自己的最高val OA首轮6300pool refit，最后25200test；43只改变训练seed，不重新划分或套用42的29/29/28。新根目录work/nwpu_three43_20261011_01，九个calibrate/refit/test子目录；CE/AdamW/cosine30/batch32/workers2/FP32双TF32-off与旧协议保持。
- BigEarth soft43沿用已完成soft45的ed480e6及局部原型配方、冻结CSV/pretrained、237871/122342/119825、30轮、BCE/AdamW/cosine30/阈值0.5/batch32/workers4/FP32与原TF32设置，新根目录work/bigearth_soft43_20261011_01；从ImageNet开始，仅评估val mAP最佳权重。独立调度器沿用soft45的记录/探针/权重与19类测试验收，正式模型代码不变。
- 预检核对干净发布代码/数据/预训练/GPU和已存seed43结果；NWPU同时搜索失败run.json对应的checkpoint metadata，发现已有产物停止审计复用，不能清目录重跑。BigEarth核对旧43两权重与metadata、保存轮数、已回传test日志及完整RGB文件，环境以已验收soft45参照；实例身份尚未回传，当前未完成服务器预检或启动训练。后台文件和输出独占，任何失败保留，阶段exit0但产物不完整也停止。
- 历史边界保留：Swin43完成30/best6、固定43停止21/best8，旧4090训练/3060固定测试、bb9f1cf；新soft43计划完整30、ed480e6/current2080Ti。同seed不消除这些差异，不写成三模型完全同条件或稳定净贡献。补完后NWPU各42/43两seed、BigEarth soft43/45两seed；三模型配对统计使用共同seed，Swin/fixed全四seed另报，完整三seed仍有缺口。
- 成本参照旧NWPU九阶段约2小时18分、BigEarth soft45总14小时44分；暂排NWPU2–3h→BigEarth15–18h串行，数据恢复/预检另计，当前显卡、首轮吞吐与实际租价待确认。不按小样本外推费用、不保证按旧时长完成。
- 本地测试先观察缺实现、缺历史和缺部分checkpoint发现检查的失败，再实现；最终新9项与复用helper12项共21 passed、exit0。两批载荷Python/Bash/zsh/实际后台fixture启动与拒绝/重复不改产物通过，[验证记录](results/evidence/seed43_prepare_20261011/validation.json)区分调度fixture与GPU结果。默认Python缺torch的旧helper初测已保留说明，改用已有QA环境复测通过，没有修改服务器依赖。
- 已更新排期/范围/待运行状态；没有任何新精度、GPU训练、Git提交/推送，不把命令交付记为实验完成。下一项为用户执行只读预检回传身份，随后按已授权范围运行；新正式结果返回后立即补日记与专门结果表。

---

## 2026-10-10 — 用户明确月底交付，收敛研究窗口

- 用户明确本月月底交付，并希望项目具有明确贡献；按2026-10-31作为当前截止日。导师是否要求模型必须超过Swin，用户表示尚不确定，不能记录为已确认的验收硬条件。
- 再读归档导师原邮件：统一协议、多seed、相关方法比较和双数据集三模型消融是明确要求；原文没有“所有数据集/seed必须超过Swin”的措辞。争取稳定收益的研究目标保留，当前固定/soft及已关闭候选证据不改写，论文剩余缺口不取消。
- [月底路线建议](docs/month_end_delivery_20261010.md)提出10-12前确定关键对照/预算、10-18结束新架构探索、10-19—23评估与必需对照、10-24起集中整合修改；实验窗口为建议，不是已批准或已执行的训练。推荐先在现有统一NWPU协议下明确跨区域消息的独立训练对照，90例冻结权重干预不能替代它。
- 本轮只完成期限确认、原邮件复核与路线整理；未实现候选、启动服务器任务或Git提交/推送。最低有效提升、具体候选、成本上限和剩余方法比较可行性仍须在训练前明确，不承诺一定得到正向结果或全部导师要求按时完成。

---

## 2026-10-10 — BigEarthNet seed43逐类AP补齐，修正四seed类别方向

- 用户回传两份现有测试日志的 `grep -A 30 'Test mAP'` 输出：[原始回传](results/evidence/bigearth_seed43_per_class_20261010/server_return.txt)已保留终端命令与提示符；[解析与来源SHA](results/evidence/bigearth_seed43_per_class_20261010/summary.json)记录本次原文和既有42/44、45类别CSV的身份。没有重新训练、测试、下载权重或改变既有正式分数。
- seed43两模型均含19个唯一且顺序一致的类别，Swin/GAViT的mAP71.1/71.7%、macro-F1 64.1/65.7%、micro-F1 76.0/76.8%与既有主表一致。逐类10升、8降、1打印持平；最大提升为荒野/石南灌丛/硬叶植被+4.2pp、多年生作物+3.3pp、内陆湿地+2.9pp，最大下降为海滩/沙丘/沙地−2.7pp。[seed43逐类表](results/bigearth_seed43_per_class_comparison.csv)与[四seed逐类表](results/bigearth_four_seed_per_class_comparison.csv)已生成。
- 四seed42/43/44/45没有任何类别四次全部提升或四次全部下降。只看42/44/45时耕地与内陆水体三次正向、自然草地三次负向，在43分别变为−0.2/−0.1/+1.3pp，因此此前三次方向不能作为四seed一致结论。农业与自然植被混合区、内陆湿地为3升1打印持平；自然草地、海滩沙丘沙地为1升3降；海岸湿地2升2降，平均差−7.42pp、配对差样本SD12.85pp，波动尤大。完整19类解释见[四seed分析](results/bigearth_four_seed_class_analysis.md)。
- 42/43/44类别AP只有一位小数，45保留完整返回精度；方向计数为显示精度下的描述，持平不证明真实相等。43打印AP均值71.110526/71.736842%、差+0.626316pp与日志71.1/71.7%相容，聚合主表继续使用日志值。既有四seed整体mAP平均差约−0.23pp的结论不变；不做显著性或图模块因果声明。
- Swin43此次再次回传测试wall234.20秒、峰值allocated0.58GiB，与已记录时长一致；GAViT43片段没有计时/显存，仍待确认。原日志节选未含退出状态、权重SHA或逐图预测，不据此新增这些验收声明；沿用原身份和epoch8 best、训练epoch21停止边界，不写成30/30。不同硬件/历史执行条件仍使跨seed波动不能唯一解释为随机seed。
- 主结果表只补记seed43逐类证据状态，原聚合分数、训练轮数和资源边界保留；当前四seed摘要与旧42/44后续分析已链接新表。既有数据算术校验和输出重读通过，未修改模型或发起服务器任务、未Git提交/推送。

---

## 2026-10-10 — 下一机制问题：区分GAT邻居消息与分支自身变换

- 用户要求继续研究，先只读复核当前代码、已闭环的幅度/FFN/pooling/空间先验结果及缓存保存schema。目标仍是稳定且值得额外复杂度的收益，没有保证任何新方式一定提升，也未重新打开已关闭候选。候选问题为：原soft模型的分类反馈是否真正依赖跨区域GAT消息，而不仅是逐区域的投影/残差/LayerNorm/GELU/反馈变换？
- 现有alpha=0整支移除不能回答这一问题；原GAT同样包含逐节点可学习变换。提出原边与仅自身环两条件的冻结权重对照，保持regions/S/参数/readout，改变跨区消息。它是已训练模型的条件性干预，不是重新训练的无图baseline；不能把删除邻居后表现等同图模块净贡献，亦不声称逐区域输入没有来自Swin/soft pooling的跨区信息。
- 已核实可行的低成本输入组合：pooling42的selected_cases.pt中90例normal regions/S/edges/pooled/logits，feedback42 soft_cache.pt按path对齐的独立mean_tokens，加原soft best.pth的GAT/feedback/classifier参数。先CPU重放原边核对pooled/logits，再考虑自身环；不能用pooled减本次重算反馈作为唯一原token来源，避免基准部分循环自证。全1260缓存没有regions/edges，因此只能承诺原固定90例，不能声称已具备全val图干预数据。本地未读服务器真实PT，来源缺失/不一致则停，不自动恢复原图或重跑主干。
- 独立只读代理核对相同缓存限制及非重复性。旧mean pooling的区域质量加权等价在此前已知，池化前FFN也已失败，不包装成新发现。温度/半径/距离强度搜索没有新增依据；更早层插图属于较大架构变化，当前不直接付费启动。
- 核对原始ViG/STA论文摘要与GATConv官方文档：前者区分图交互与节点变换，后者采用超token采样/交互/回映，均不能证明当前GAViT照搬必有收益。PyG最新文档是2.9.0，另读本地2.8.0实际GATConv的自环加入逻辑，没有用最新版冒充服务器版本。[来源、缓存schema及SHA记录](results/evidence/next_mechanism_review_20261010/review.json)。
- 当前完成的是机制缺口与缓存可行性审查，探针设计限一次CPU90例原边/自身环对照，无自动边搜索、全val扩展或训练。未实现/执行新工具、未改模型、未发起服务器任务、未Git提交/推送。不能把源码可行性当作真实缓存验收或实验发现。

---

## 2026-10-10 — 空间先验缓存收尾完成，关闭该候选

- 用户返回IDENTITY、METRICS、PAIRED和完整30轮压缩曲线；原文逐字保存在[closeout_server_return.txt](results/evidence/nwpu_spatial_calibration_20261010/closeout_server_return.txt)，[最终解析/算术记录](results/evidence/nwpu_spatial_calibration_20261010/closeout_summary.json)已验证样本总数、配对边际/净差、OA、首个最高轮和曲线有限性。原文没贴最后SPATIAL_CACHED_CLOSEOUT_PASS，但全部实质输出完整，前置检查在这些打印之前，不为补标记重跑命令。此前“待回传”为历史状态。
- 本次身份明确为67fe84d且dirty=false，RTX2080Ti/CUDA12.1/PyTorch2.2.2、8线程，实际图48空间+32特征边，冻结manifest/pretrained和实际best文件SHA一致；相同配方/路径/标签、45类各28样本、缓存OA/F1与history/文件SHA的验证由服务器只读命令执行。本地仍只核对返回摘要，未取得原预测/权重/全部图缓存或独立前向。
- 完整val1260：Swin OA94.206349207%、macro-F1 94.171552170%；旧soft93.730158730%、93.765721402%；空间先验92.380952381%、92.342474473%。相对soft差−1.349206pp/−1.423247pp，配对纠正26/新增错误43/双方对1138/双方错53，净−17；相对Swin差−1.825397pp/−1.829078pp，纠正21/新增44/双方对1143/双方错52，净−23。候选不是仅某个指标的权衡，两个指标均下降；单seedval不能做跨seed/正式test结论。
- epoch24同时最高OA/F1，train loss0.003957；最终epoch30 OA92.063492%、F192.001548%、loss0.003366，较best少对4张。后六轮没有恢复/超越，曲线支持按原规则停止，不支持追加轮数；较低训练loss与验证平台不能唯一归因于过拟合、空间混合或优化问题。
- 两位小数曲线计时累计train1646.28秒、val221.64秒、训练总图/训练时间约91.84图/秒；train每轮32.11–104秒，中位48.585秒，val3.52–17.7秒。总wall2997.438秒的剩余1129.52秒未分项，可能包含入口校验/存盘/重载等已知阶段，不能据此断言网络盘故障或新模块复杂度翻倍；实际费用未知。此次不是公平效率测量。
- [候选结果表](results/nwpu_spatial_prior_trial.md)、运行说明和排期已同步收尾状态。关闭anchor_gaussian_v1，不追加seed、强度搜索、refit/test或BigEarthNet；无需进一步服务器补件。完整训练S/W几何和逐类原表未本地取回，不影响执行预先停止条件，也不用于编造机制解释。本轮未改模型代码、新训练、GPU诊断、Git提交/推送；已有闭环负面证据保留。

---

## 2026-10-10 — 空间先验校准负结果：触发冻结停止条件

- 用户回传 `work/nwpu_spatial42_calibrate_20261009_01` 的run.exit0/status complete：NWPU seed42、5040train/1260val、final epoch30、best epoch24、best val OA92.38095238095238%。回传JSON已[原样保存](results/evidence/nwpu_spatial_calibration_20261010/server_return.json)，[算术摘要](results/evidence/nwpu_spatial_calibration_20261010/summary.json)区分观察值与待核验项。
- 由OA反算1164/1260，较旧soft1181少17（−1.349206pp），较Swin1187少23（−1.825397pp），较固定分组1172少8（−0.634921pp）。逐图预测尚未下载/重算，净少对不等于新增错误数；macro-F1、逐类/混淆矩阵仍保存在服务器，未回传就不填造。
- 实测总2997.438秒=49.96分钟，峰值allocated3932652032字节=3.66257GiB，重载最大logit差9.536743164e-7。probe为8例val、last epoch30、7.659秒；完成摘要不等于逐元素探针验收。最佳checkpoint报告SHA为a1c7f3b173e369ec3d40e2122fec91fa9ebc70d39ac6a96f5db90017c76fb579，实际文件SHA待只读核对。当前回传未展开硬件/commit/配方/每轮吞吐，不能把短测配置直接当本次已核验身份或据总耗时定位性能根因；实际费用未知。
- 原冻结命令为67fe84d发布版、单候选anchor_gaussian_v1、S加入固定0.5空间距离项、attention/GAT/feedback配方保持、FFN关闭、原ImageNet重新初始化、batch32/workers2/8线程、30轮AdamW3e-4/wd1e-4+cosine30、CE、FP32/双TF32false，完整命令在[运行说明](docs/run_nwpu_spatial_calibrate_20261009.md)。该段是预注册身份，本次实跑身份将从run.json确认。
- OA低于旧soft，执行预先停止条件：不追加seeds、空间强度搜索、refit/test或BigEarthNet训练。[候选表](results/nwpu_spatial_prior_trial.md)记录负结果。结论只限这个候选/单seed验证协议没有提供收益；不能证明空间约束或所有软分配普遍无效，也不能仅凭摘要归因于过拟合、语义混合或图反馈。后续研究不由这次回传自动触发新训练。
- [一次只读收尾命令](docs/read_nwpu_spatial_closeout_20261010.txt)读取现有Swin/soft/spatial三组CSV/JSON和新best文件，核对相同路径/标签/配方、45类支持数、OA/macro-F1重算、配对计数、曲线首次最高选轮和新权重SHA；只输出精简身份/指标/30行压缩曲线，不展示混淆矩阵。无需原图、Torch或GPU，不修改服务器文件。真正运行及缺失指标仍待回传。
- 本地合成CSV/JSON正常流程与坏checkpoint、错路径、首次最佳轮错误四场景通过，Python载荷AST及bash/zsh语法一致，见[交付验证](results/evidence/nwpu_spatial_calibration_20261010/readback_validation.json)。独立审阅代理因使用额度限制未完成，没有额外审阅通过声明；主代理已自行核对算术/源码schema及本地检查。未修改模型或提交/推送。

---

## 2026-10-09 — 空间先验真实短测回传，准备一次 NWPU 校准

- 用户回传 `work/nwpu_spatial42_smoke_20261008_01`：exit0/status complete，smoke seed42、360train/90val、final epoch1；源代码67fe84d干净，固定manifest/pretrained SHA与设计一致，RTX2080Ti、CUDA12.1/PyTorch2.2.2、8线程。最佳checkpoint SHA为89803222fd393f3cfecc0c16a2bbb80922932aaec0df81c25413259bbad513c0；训练状态观察SHA为b67c5455e9af4c2e53026b713435ece0fe1f45caf4ed732b1fe6fd18fd5c908f。完整命令/架构/优化配方沿用[已发布短测说明](docs/run_nwpu_spatial_prior_20261008.md)，不把观察SHA当作已有参考比对。
- 图实际前向48空间+32特征边，重载最大logit差3.5762786865e-7；probe为val8例，epoch1、1.734秒。总98.576秒、train8.261秒/43.578图每秒、val0.572秒，训练loss3.598312；峰值allocated3.66156GiB、reserved4.03906GiB，统计包含重载。未回传短测分类分数，不推测或写入主比较CSV。
- 保存[回传JSON](results/evidence/nwpu_spatial_smoke_20261009/server_return.json)与[摘要算术](results/evidence/nwpu_spatial_smoke_20261009/summary.json)。用户将终端内容插入旧运行说明尾句，先逐字保存整个文档到source_document_with_pasted_return.txt，再仅修复截断句子并链接下一步；不丢弃原始粘贴。原权重/PT/逐图预测仍在服务器，本地未下载或独立复算模型输出。
- 下一步仅一组固定seed42、5040train/1260val、30epoch校准，仍为S加入固定空间先验一个主要变量，从原ImageNet初始化，不续接smoke。旧soft1181/1260、Swin1187/1260及原停止规则不变；候选结果将进入[状态表](results/nwpu_spatial_prior_trial.md)。已有checkpoint无法回答新空间先验从训练开始是否改善分类，故先通过工程短测再做此一组；不启动test/refit/其他seed/BigEarthNet。
- [复制命令](docs/run_nwpu_spatial_calibrate_20261009.md)将八例CPU快照归一化/掩码/别名/历史、原图变换输入SHA、源码及冻结输入身份合并预检，成功才发起唯一后台校准；不增加GPU诊断或重复短测。要求日志首batch、进程与GPU占用齐全才视为已启动。输出固定work/nwpu_spatial42_calibrate_20261009_01，当前尚未在服务器启动。
- 本地只用合成缓存/八张小图和发布CPU类验收交付：Python AST/两shell语法与8个正常/拒绝场景通过，Git/GPU/启动进程被mock，没有真实训练；证据见[handoff_validation.json](results/evidence/nwpu_spatial_smoke_20261009/handoff_validation.json)。独立只读审查无阻塞问题。未修改模型代码、未提交/推送。
- 成本估算：旧同型号完整soft墙钟24.73分钟；新smoke train/val机械外推61.83分钟，含首轮启动开销不能当稳态。先预留约一小时、参考25–70分钟，完整首轮再更新；不得把98.58秒整段乘420。实际费用未知。

---

## 2026-10-09 — 空间先验独立分支已发布，等待 NWPU 一轮短测

- 用户明确批准提交/推送后，在058f1d8干净隔离工作区创建 `codex/soft-spatial-prior-nwpu-20261008`；仅复制昨日7个代码/测试文件，SHA全部与验收快照一致。其他17个共享运行文件与基底一致，FFN保持关闭。原研究工作区与旧分支保留。
- 实际发布目录完整tests：162 passed / 44 warnings / exit0 / 56.35秒；代码没有因发布改变，旧92项开发验证保留。发布包含5运行文件、2测试文件、聚焦日记/运行说明及验证证据，共19文件。原始pytest失败日志/补丁context空格作为原始数据保留，仅这些字节校验通过的证据不做行尾空格检查，代码和文档diff check通过。
- 新提交 `67fe84d431685cf2c7788edfae21fa4e9eb040a4` 已推送；GitHub分支SHA与本地一致，ahead/behind为0/0，发布工作区干净。未合并main，未创建PR。[发布回执](results/evidence/nwpu_spatial_prior_20261008/publication_receipt.json)保存文件清单、远端核对和固定commit同步命令。
- [服务器运行说明](docs/run_nwpu_spatial_prior_20261008.md)已补源码/空闲单2080Ti/8GiB/全图路径/输出不存在检查、8线程和实际checkpoint SHA回传。命令fixture覆盖正常、源码脏、GPU忙/型号变化/低显存、缺图/旧输出及损坏checkpoint。候选没有FFN专属artifact_sha256字段，回传改为检查best/last sidecar并独立计算训练状态SHA；不宣称有单参数梯度实机审计。
- 仍只交付360train/90val的一轮smoke：发布不等于服务器已运行，当前GPU/原图状态及新候选CUDA/预训练/效果待实测，不把162项本地验证当模型收益。完整校准、refit/test和多seed未启动。

---

## 2026-10-08 — 空间先验候选已实现：仅修改 S，先做 NWPU 工程短测

- 用户在 pooling 诊断后要求继续，落实单候选 `anchor_gaussian_v1`；假设为保持区域锚点的一定空间约束可能改善完整 val 分类。公式是 `cos/0.1 - 0.5 * ||(token - anchor_centroid)/1.75||²` 后再做原局部掩码和区域 softmax。中心来自实际硬分区，单轴 `[0.5,2.5,4.5,6]`；不由 attention/S 更新。固定强度、无搜索、无新损失、无新可学习参数。
- 原型 attention、最终 W=softmax_token(logS+attention)、图构造公式/GAT/feedback、seed/初始化/配方均保留；S 改变会连带改变 W 数值、区域特征和特征边。原冻结 pooling 负结果只支持停止直接替换 W，不能证明空间先验一定有效。
- CLI 显式 `--spatial-prior anchor_gaussian_v1`，新完整配置版本 `local_prototype_cosine_spatial_v1` 写入 architecture/probe。旧默认和 checkpoint keys 不变；候选独有49×16固定buffer使严格加载拒绝新旧身份误配。场景入口贯通构建/保存/重载/val评估，并限定 seed42、NWPU smoke/calibrate、禁止叠 FFN/refit/test；本地 synthetic smoke 是工程例外。
- 测试先观察缺接口失败再实现：分组7项 red，入口11项 red；实现后64项相关检查通过。覆盖独立几何赔率、S/W归一化和梯度、初始化/RNG、真实图模块、合成图片生命周期与probe、旧scene/FFN回归。额外BigEarth入口回归遇到临时环境缺依赖及OpenMP沙箱问题，补齐临时环境并在沙箱外离线CPU复测后28项通过（2807.53秒），共92项不同测试通过。失败原因及可用日志保留，最终状态见[验证记录](results/evidence/nwpu_spatial_prior_20261008/local_validation.json)。本地测试不等于真实预训练/CUDA或模型效果验收。
- [冻结设计与短测参数](docs/run_nwpu_spatial_prior_20261008.md)、[结果状态表](results/nwpu_spatial_prior_trial.md)、[7文件补丁及SHA](results/evidence/nwpu_spatial_prior_20261008/release_identity.json)已整理。补丁以已发布058f1d8为基底，仅5运行文件和2测试文件；另外17个运行文件字节不变，FFN关闭。独立只读审查未发现阻塞问题。原未提交研究材料保留，尚无新commit/push。
- 下一步先发布独立源码，再进行360train/90val的一轮短测；真实GPU/数据/耗时待核验。工程通过后才考虑固定seed42完整校准，与旧soft1181/1260及Swin1187/1260配对。若无超越旧soft，结束该候选；小幅超过仍不代表稳定。当前没有启动新服务器训练，主实验表不填新指标，不重跑BigEarthNet。

---

## 2026-10-08 — 最终汇总干预回传完成：保留attention，关闭纯推理替换分支

- 用户回传1260完整val末批PASS、`NWPU POOLING PROBE COMPLETE`和exit0，输出`work/nwpu_pooling42_20261008_01`。保存[原文](results/evidence/nwpu_pooling_probe_20261008/server_return.txt)、[解析与算术](results/evidence/nwpu_pooling_probe_20261008/server_return_summary.json)和[对照结果表](results/nwpu_pooling_probe_seed42.md)。本地没有下载原PT/逐样本预测或再跑GPU；本次硬件/耗时/缓存SHA/完整逐类指标未在短回传展开，不拿旧信息补造。
- 原W复现1181/1260，OA93.730158730%、macro-F1 93.765721402%、CE0.241419598；替换为U后1174/1260、93.174603175%、93.192050080%、0.242198229。差值−0.555556pp/−0.573671pp/CE+0.000778630。配对1169双方正确、5纠正、12新增错误、74双方错，净−7；总数、边际与OA差本地复核一致。
- S最大差0；恢复logit差2.8610e-6、U列误差2.3842e-7、原缓存概率差2.9802e-7、CPU分类头重放差7.6294e-6。回传完成及这些检查支持本次实机诊断完成，不把本地摘要复核说成原始缓存独立验收。
- GAT前区域余弦均值由0.751306069降至0.708353294；每图32条特征边平均保留18.496032（57.80%），平均替换13.503968。均值不能外推每张图同向，保留率不是Jaccard。更低的区域相似度没有在此次干预中带来分类收益，不能将其等同语义改善、消除塌缩或稳定增益。
- 决定：保留现有attention，停止此次直接替换汇总的推理方案，不扩大混合系数/重复前向。不据冻结权重干预断言新结构重训后一定差，也不把attention或图边判为唯一根因。空间先验仍只是候选；下一步建议在保留attention条件下仅改变S中的固定空间距离项，从训练开始做单变量NWPU对照，配置与停止条件须预先冻结。当前没有新模型修改、训练、test/BigEarthNet/seed扩展或Git提交推送。
- 独立只读复核确认算术与上述结论边界；模型主比较表/消融分数保持，新增结果进入推理敏感性记录。运行说明和前一批缓存结果已标明该后续完成，避免重复付费诊断。

---

## 2026-10-08 — NWPU 最终汇总推理干预准备完成，待服务器执行

- 用户在S/W诊断回传后要求继续，完成 `tools/diagnose_nwpu_pooling.py` 与[单行启动/回传说明](docs/run_nwpu_pooling_probe_20261008.md)。固定旧soft seed42 calibration best、全部1260val，只将最终W替换为S列归一化U，原型attention/S/模型参数保持，图边从新区域表示实际重算。没有修改正式模型/训练入口、重新训练或使用test；空间先验仍是候选，不当作已证根因。
- 正常→干预→恢复三次真实前向，检查S逐元素不变、48+32边与节点输入、恢复预测、旧缓存概率及原完整val OA93.730159%；分类头输入/logits/头参数保存后CPU重放再验收。全量指标/逐图特征统计落盘，屏幕只打印摘要。另保存原固定90例GAT前区域特征、S/W和实际边，补上前一批缓存缺特征的限制；不按本次成败挑样本。
- 复用两个已发布诊断helper且导入前核验完整SHA；它们校验冻结源码/清单/权重/预测/原图或ZIP字节像素身份。新输出 `work/nwpu_pooling42_20261008_01` 及启动日志独占创建，缺失输入/已有输出拒绝，不自动重试。服务器命令读取持久化ZIP，无需恢复大型原始数据；当前仅交付，真实模型/CUDA/完整数据尚未运行。
- 先写7项测试并观察缺少实现的失败，实现后7项通过；加入交付脚本一致性、两shell语法、缺输入/旧输出拒绝，以及两个真实JPEG/ZIP复用检查，最终 **12 passed、exit0、4.06秒，39条既有Torch/Python3.12弃用提示**。日志与身份见[验证记录](results/evidence/nwpu_pooling_probe_20261008/validation.json)。本地1260合成数据流程用小型token骨干配真实soft grouping/PyG，不当作预训练Swin或CUDA性能验证。未扩大到训练全套测试。
- 独立只读静态审阅未发现阻断项，覆盖干预隔离、钩子恢复、schema、冻结helper来源和防覆盖；未由审阅代理重复跑测试。最终源码/命令语法与补丁空白检查通过。没有Git提交/推送，不覆盖任何旧实验；返回状态仍待服务器完成标记与exit0。
- 解释冻结：U质心更接近S是公式必然，不当作收益；干预后升降均仅描述现有联合训练权重的敏感性，不代表新结构训练后优劣或稳定提升。结果将进入推理敏感性记录，不修改正式训练对比表，不自动追加空间先验/BigEarthNet/多seed试验。

---

## 2026-10-07 — 90例 S/W 真实缓存摘要回传：检查通过，下一步定位最终汇总重加权

- 用户回传90例全部PASS、`ASSIGNMENT CACHE COMPLETE`及exit0，目录为`work/nwpu_assignment_cache42_20261007_01`。保存[原文](results/evidence/nwpu_assignment_cache_20261007/server_return.txt)、[解析摘要](results/evidence/nwpu_assignment_cache_20261007/server_return_summary.json)和[结果表/解释](results/nwpu_assignment_cache42_20261007.md)。本地只复核回传摘要及当前公式，没有获取完整PT、逐图/分位数、当前源文件SHA清单或服务器耗时。
- soft的S熵0.741116、W邻居重叠0.478292、有效token数7.015423，均低于均匀邻域的1/0.640693/18.777516；不支持简单称为均匀扩散或语义塌缩。S锚点距离0.102003小于参照0.120128，而W距离0.181653更大、自身RMS0.184898更小；更适合描述为有选择且相对集中的跨锚点汇总，是否有害尚未确定。
- 正确单位为模型输入宽高比例：soft S/W锚点距离约0.714/1.272个token间距。两均距之差不是逐区域S→W位移。原82双方正确/3纠正/0新增错误/5双方错保持；90固定验证案例不能支持总体、稳定或因果收益，原缓存缺特征使特征冗余仍未知。
- 用户进一步“继续”后，将下一步收窄为现有soft checkpoint的1260val最终汇总干预：只比较原W与S列归一化U，保留用于原型的attention和S；跟随区域表示变化重算特征边，记录真实边变化。直接读持久化ZIP，无需解压/重训；该对照目前处于本地准备阶段，未发布新模型或启动服务器。结果将进入推理敏感性记录，不写入正式训练对比表；空间距离先验保留候选，尚非已证根因或最终选择。

---

## 2026-10-07 — 重回软分配机制研究：90例 S/W 缓存检查准备完成，真实结果待回传

- 用户要求先研究适合 GAViT 的软分配，暂缓继续写作，并同意先复用已有产物核验机制。当前空间距离先验是待验证候选，未声称最优或能保证稳定提升；本轮没有修改模型、发起训练、追加 seed、调参或 Git 提交/推送。
- 新增独立 CPU 工具 `tools/analyze_nwpu_assignment_cache.py` 与[执行说明](docs/check_nwpu_assignment_cache_20261007.md)、[单行命令](docs/launch_nwpu_assignment_cache_20261007.txt)。只读服务器 `work/nwpu_regions42_20260929_01` 中原90例 fixed/soft 的 S/W、图边、logits与身份记录；无需恢复原图、GPU前向或上传工具包。新输出为 `work/nwpu_assignment_cache42_20261007_01`，已有输出拒绝覆盖。
- 研究问题：当前 soft 是否表现出超出候选几何参照的重复权重汇总/空间偏移，为下一步空间先验是否值得试验提供线索。指标包括按候选数归一化的 S 熵、区域质量、W 重叠、有效 token 数、分别基于 S/W 的质心漂移与 RMS 扩散；保留逐图逐区域/24对无向邻居值，并按图汇总 mean/median/p90。固定/soft 支持上限不同，不把归一化支持比例之差当绝对数量差。
- 独立公式审阅得出均匀邻域参考本身就有 W 邻居重叠 0.6406926407、有效 token 均值18.77751634、归一化输入坐标漂移0.1201276687、RMS0.2641628789；这些是数学参照，**不是90例实测模型结果**。权重重叠不等于区域语义塌缩；旧缓存没有原 tokens/最终 region features，无法据此定位唯一失败原因。90例是固定校准验证案例，原有3纠正/0新增错误不能外推总体或因果收益。
- 先写测试并观察缺少实现的导入失败；实现后18项定向CPU检查通过，最终加入嵌入单行命令的实际执行/两shell语法/缺失源检查后 **20 passed，exit0，3.16秒**，见[日志](results/evidence/nwpu_assignment_cache_20261007/tests.log)及[验证记录](results/evidence/nwpu_assignment_cache_20261007/validation.json)。均使用合成PT，未读取服务器真实90例。旧临时环境Python包文件不完整，重建隔离QA环境；macOS沙箱OpenMP共享内存阻断一次子进程检查，获准在沙箱外运行同一CPU测试后通过。
- 独立代理完成数学、实际缓存schema、文件防覆盖与结论范围静态审阅，无待修主问题；其额外随机对照因自身环境缺torch未执行，不当作通过证据。实际90例、服务器耗时、下一步候选选择仍待回传。没有新准确率实验，正式模型比较表/消融分数表保持原值；本条记录诊断准备，不视为软分配性能问题已解决。

---

## 2026-10-06 — 相关工作三主线初稿与12篇原始来源整理完成

- 用户继续已确定的论文写作阶段。新增[相关工作初稿](docs/thesis_related_work_20261006.md)，串联视觉/遥感Transformer、图像图推理与固定/k-means/软分组三条主线，具体比较GAViT与HGTNet、DFAGCN、SSN、Slot Attention、GroupViT及STViT；不引入新训练、不试图以文献成绩替代E02实测。
- 已核对12篇定向原始来源，生成[BibTeX](docs/thesis_related_work_20261006.bib)和[来源索引](results/evidence/related_work_20261006/sources.json)。区分ViT预印本/会议年、DFAGCN在线/卷期年，补齐STViT五位作者，SSN页码统一采用Springer章节363–380。SLIC DOI因一手读取未确认而省略；不凭搜索片段补猜完整未知字段。
- 当前代码核对明确：固定区域内attention不改变成员；历史余弦硬聚类分配detach、汇总仍可反传；soft原型来自当前图像，S归属与W汇总归一化轴不同；空间边仍用锚点先验，top-2离散；余弦仅选拓扑、不作为GAT输入边权；FFN候选不等于ViG复现。汇总—交互—回映已有STViT先例，不能无限定宣称首创、全可微图拓扑或语义分割质量。
- 两代理分别完成遥感方法与区域分组原始来源核验，主代理核对当前源码并整合；后续独立稿件复审请求因额度限制未完成，没有独立最终审阅签字。MDPI直接读取429时使用出版社索引正文；Swin CVF403/ViT OpenReview验证页以作者arXiv和官方仓库替代。STViT no-grad沿用已标明commit的09-26静态审计，本轮锁定源码远程读取失败，不冒称重新确认最新版。
- [本地引用一致性验证](results/evidence/related_work_20261006/validation.json)通过：12个唯一引用/书目键及字段对应、标题一致、字段括号平衡、10个本地链接目标存在，保存初稿/书目/来源及8项本地机制证据SHA。该验证不替代逐主张科学评审，也未做文献管理器导入或TeX编译；当前交付为Markdown与UTF-8 BibTeX初稿。
- 导师对照、排期、实验矩阵及E02接入记录同步“相关工作初稿完成、待审阅/格式修订”。外部方法实测、多seed软分组、公平新结构效率和3×3/4×4受控比较继续保留为未完成项；历史数据、模型代码与实验结果保持，未Git提交/推送。

---

## 2026-10-06 — 三数据集实验讨论成稿，外部比较继续保留缺口

- 用户在FFN候选关闭后继续推进，并说明尚未收到HGTNet/DFAGCN作者回复。本轮仅使用已有本地证据与有界公开材料检索，没有访问邮箱、发送消息、启动GPU或追加seed；FFN停止决定及全部原分数保持。
- 新增[实验与讨论初稿](docs/thesis_experiments_discussion_20261006.md)：整合AID三seed、BigEarth四seed历史统计、NWPU42/BigEarth45正式软分配三模型表、90例结构与正反例图、完整val反馈幅度诊断、FFN四格校准及AID效率范围。明确测试/验证/案例层级、固定split与训练seed、历史中断/舍入、无稳定/因果结论；没有拿旧NWPU近似值补新协议表。
- [表格核验](results/evidence/thesis_synthesis_20261006/validation.json)通过：26条数值行逐项按摘要重建，重算AID/BigEarth均值及样本SD、配对边际/净差、六行soft差值与19类贡献、FFN停止条件与新增参数、效率相对变化；21个去重本地链接目标和两张既有插图存在。保留可复核脚本、来源SHA和初稿SHA。仅检查已有摘要和算术，不冒称新GPU前向、逐样本预测重算或全部原始数据已本地备份。
- [外部材料复查](docs/external_methods_refresh_20261006.md)执行6条网页查询并核对合作者主页/两篇全文页面；未取得新的作者完整实现、权重或代码许可，结论限于本次范围。排除交通预测同名DFAGCN与HGTNet页面推荐论文中的ECA-MSDWNet链接；依赖库链接不作为完整实现。澄清DFAGCN已有C与部分融合系数网格，实际仍缺完整配置与验证选择规则，不再笼统称范围未公开。
- 更新导师要求对照、E02接入状态、矩阵和排期，纠正软分配“尚未设计/实现”及NWPU划分“仍计划”的过时现状。保留导师英文原文和按日期的历史；外部方法实测、多seed软分配、新结构公平效率、3×3/4×4受控比较及三主线相关工作仍未完成。先推进材料/论文工作，不把初稿当作全部论文要求完成。
- 两项并行只读审计因额度限制未完成，没有独立审阅签字；主代理完成上述来源核对与数值验证。不提交/推送Git，不改训练代码或历史实验产物。

---

## 2026-10-05 — FFN缓存收尾回传完成，停止候选扩训

- 用户完整回传COMPACT_CLOSEOUT_PASS。原文8150字节逐字节归档于results/evidence/nwpu_ffn_calibration42_20261005/cached_closeout_server_return.txt，附SHA；粘贴命令AST与已交付只读脚本一致。四组1260唯一同路径同标签、45类各28张、共同manifest/pretrained/training、30行history/首次最高OA选轮、重算OA/F1与所选缓存一致，以及两个新best实际文件SHA，均由服务器完成检查。本地没有下载原预测/权重或GPU复跑。
- [最终四格表](results/nwpu_ffn_calibration42_comparison.md)已补齐：新Swin+FFN macro-F1为94.044703%，比原版−0.126849pp；新soft+FFN93.145677%，比原soft−0.620044pp。OA差仍−0.158730/−0.555556pp，三项冻结继续条件均失败；沿用停止候选、不追加训练/test的决定。
- 配对结果：soft+FFN对旧soft纠正27/新增34/双方正确1147/双方错误52（净−7）；对新Swin28/39/1146/47（−11）；对旧Swin24/37/1150/49（−13）；新Swin对旧Swin28/30/1157/45（−2）。四格总数/边际/净差/全部非零逐类净和及跨模型逐类差可加性，本地复核与独立只读审阅均一致。soft对原版类别12升/15降/18平；不挑改善类别外推机制。
- 新soft best epoch22，末轮OA92.936508%、比best少3张；旧soft末轮也比best少3张。当前best比较已正确，不能以最佳轮提前单独证明过拟合。新末5轮OA均值Swin93.809524%、soft92.936508%；末轮loss约0.001694/0.002503。
- 新参数27852903/31784168，均比对应原版多297408；训练秒数836.361268/1018.461295，验证85.025076/116.455914。新两组报告hardware一致：RTX2080Ti/CUDA12.1/torch2.2.2/8线程、code_git 058f1d8 clean；两个actualBestSHA与上次报告完全一致。未含UUID/驱动，不声称与旧物理卡相同或公平推理提速。
- 本次有界训练与缓存收尾完成，无需再补跑或回传同批摘要。PT探针逐元素、last/state字节审计、本地完整曲线镜像在约定scope之外；不把这些写成已验收。保留原始服务器成果与负面结果，原论文多seed/外部方法/公平效率仍是独立未完项。

---

## 2026-10-05 — FFN两组30轮校准完成回传，候选未过冻结OA门槛

- 用户返回swin_ffn与soft_ffn具名摘要：均exit0/status complete、phase calibrate/seed42/train5040/val1260/final30。best val OA为94.047619%（epoch29）与93.174603%（epoch22），best文件报告SHA分别bec792b9…38fb9bf1和a926e6bd…e2e87ed；原始字段与完整SHA保存于results/evidence/nwpu_ffn_calibration42_20261005/server_return.json。本地仅核对回传，未访问服务器或独立下载权重。
- [独立四格校准表](results/nwpu_ffn_calibration42_comparison.md)及结构化JSON已更新：旧Swin/soft为94.206349/93.730159%，FFN变化−0.158730/−0.555556pp（−2/−7张）；新soft比新Swin低0.873016pp（−11），比旧Swin低1.031746pp（−13）。新正确数1185/1174由OA严格换算，尚未逐条预测复算。独立审阅与本地算术核对一致，三项严格超过条件均失败。
- 已交付[一次缓存收尾单行命令](docs/read_nwpu_ffn_closeout_20261005.md)，标准库只读，核对四模型1260同路径标签并重算OA/macro-F1、首次最高OA选轮、新best实际SHA、四组配对及全部非零逐类净变化；输出9条记录，无矩阵。10项本地合成文件/错误拒绝/语法检查通过（样例输出2118字节），服务器缓存收尾尚待回传，不称已完成真实预测复算。
- 按原协议将FFN候选作为负面结果收尾，不追加训练/调参/seed/refit/test。结论限本次seed42与验证选择，不把微小差距解释为稳定退步，不以best提前推断过拟合，也不修改主要指标。新macro-F1/曲线/配对及完整身份仍待一次缓存只读补件，不标为论文归档全部完成。
- 两段总耗时1160.389391/1341.166717秒，合计41分41.556秒；allocated峰值3.564785/3.666964GiB，重载差0/2.3842e-7。该合计不含外层启动预检/间隙，不作推理提速比较；新GPU/软件完整快照未在本次摘要中，当前物理卡不能仅沿用旧记录。

---

## 2026-10-04 — FFN双模型GPU短测回传通过，双校准命令交付

- 用户补充swin_ffn_smoke、soft_ffn_smoke的具名精简回传：两项exit0/status complete、NWPU seed42/360训练90验证/1轮、干净058f1d8、RTX2080Ti。总耗时108.947310/116.192555秒，allocated峰值3.565/3.667GiB（回传取整），重载logit差0/2.3842e-7；[短测表与证据边界](results/nwpu_ffn_smoke_validation.md)及结构化原字段/本地核对保存。未本地访问服务器或独立下载权重。
- 两组297408新增参数/初始FFN张量SHA完全一致，初始残差0；首步fc2非零、fc1/norm为0，第二步全部非零，selected/final均已偏离initial；soft实际48+32=80边通过。soft首次捕获8张来自训练前验证探针，不改变训练batch32。该结果是工程短测成功，不代表FFN指标收益；旧负面结果保留。
- 已准备[两组30轮校准单行命令](docs/run_nwpu_ffn_calibration_20261004.md)，复用已发布058f1d8与旧Swin/soft无FFN校准。目标既有nwpu_ffn42_20261003_01下新swin_ffn/soft_ffn；独占calibration日志/claim、完整发布与输入身份/GPU检查、已有短测best实际SHA核对、首失败停止，不自动test/refit/扩seed或重试。11项本地调度/拒绝/防覆盖/语法检查通过，独立审阅无可行动问题；本地没有启动服务器任务。当前两组正式校准仍待用户启动回传。
- 冻结问题不变：soft+FFN能否超过旧Swin94.206349%、旧soft93.730159%及新Swin+FFN；校准5040/1260、seed42、各30轮，val最高OA首达选best。粗预算1–1.5小时参考旧同卡两组47分58秒加余量，非新版实测；需首轮后修正。原始短测未回传纯训练吞吐，不直接以109/116秒乘30。正式比较表不填smoke准确率；实验结果仍待测试。

---

## 2026-10-04 — FFN短测回传截断，收窄输出等待关键字段

- 用户回传从45类混淆矩阵中部开始的文本；尾部含一条run_exit=exit_code=0和两组smoke范围声明。模型名称、两组完成状态、FFN梯度/初始化与checkpoint身份在该片段不可见，暂不将两组GPU短测标为完整验收。原文保存在results/evidence/nwpu_ffn_smoke_20261004/user_readback_truncated.txt。
- 原因定位到本地交付readback最后一条print：展开整个run.json及history.validation，导致45×45矩阵与逐类统计大量占屏。只修改输出字段投影，前置验收逻辑保持；另提供每模型一行的只读补充摘要。此短命令不替代完整权重哈希检查，不需要重新训练或恢复原图。此次不产生新模型指标、不更新正式对比表，等待精简回传。

---

## 2026-10-04 — FFN独立分支已发布，服务器双模型短测待执行

- 用户明确同意将9文件候选提交并推送。使用原soft发布`ed480e60fea03ed786fdba70c144de5e815c34f5`建立受管干净发布目录`/Users/patrick/.codex/worktrees/token-ffn-nwpu-release/GAViT_Project`；9文件逐字节匹配已验收候选，原开发目录的其他修改/实验产物保留。
- 新提交`058f1d8b79d0be538ec358eb742d2cd3992921d4`，独立分支`codex/token-ffn-nwpu-20261003`已推送。推送成功后独立ls-remote核对完整SHA一致，发布目录干净；未合并主分支或旧soft分支。首次远端复核遇到连接重置，重试只读查询成功，不将本地tracking ref当远端证据。
- 完整发布checkout的tests目录144项通过/exit0、51.17秒、44项既有提示；开发目录先前234项含其他未发布工具，两者范围区分。独立代码审阅本轮完成，9文件范围未发现可行动问题；旧验收中的额度中断记录保留为历史。来源/测试/远端证据见`results/evidence/nwpu_ffn_release_20261003`。
- 已交付[服务器同步、顺序双模型GPU短测与只读回传命令](docs/run_nwpu_ffn_smoke_20261003.md)，无需上传工具包。保留冻结20261003命名，目标新代码目录GAViT_FFN_20261003，独占输出nwpu_ffn42_20261003_01；先检查代码/清单/权重SHA、31500路径、2080Ti空闲及至少8GiB显存，然后Swin+FFN/soft+FFN各360/90一轮，第一组失败不运行第二组，不自动30轮/refit/test。
- 启动/回传单行Python/Bash/zsh语法与受控调度验证通过，真实bash首阶段失败停止、重复输出保持/缺图/脏源码/GPU忙拒绝、未完成产物不冒报成功；修正了把519MiB桌面占用误判为GPU忙的过严128MiB条件。该修改只在本地交付包装器，不改变已发布9文件或模型协议。全量逐图内容哈希仍由训练入口完成；没有访问Featurize或真实GPU结果，当前状态是待用户执行。

---

## 2026-10-03 — NWPU轻量FFN候选本地实现与验证完成，尚未发布/训练

- 用户同意按有界候选推进并继续实现。研究问题：反馈后的token经过共享非线性变换再池化，能否改善soft GAViT，并超过同样加FFN的Swin。只增加两组NWPU seed42校准；旧Swin/soft对照复用。最高val OA仍为选轮与筛选主指标，macro-F1取同checkpoint；须同时严格超过旧Swin94.206349%、新Swin+FFN和旧soft93.730159%才考虑后续，不将单seed过线称稳定。详见[冻结协议](docs/run_nwpu_ffn_trial_20261003.md)。
- 实现可选residual_ffn_v1：LN→Linear768/192→精确GELU→Linear192/768加残差，位于池化前，GAViT在token反馈后。fc2 weight/bias为0；完整构造在独立CPU seed42003 RNG作用域内。两模型同初始化、默认旧模型参数/随机数/架构字段/state键保持。新增297408参数；真实CPU模型总参数Swin27852903/soft31784168，初始logits与哈希匹配ed480e6原类最大差均0。第一步W1/LN任务梯度为0不代表AdamW不做权重衰减，第二步非零梯度已验证。
- 新scene开关与严格metadata重建已实现；拒绝非42seed、refit/test、不支持模型/数据集和改变的NWPU清单/预训练身份。本地synthetic smoke例外；新版不会自动扩BigEarth、seed或调整超参数。保存首次前向形状/零残差、两步梯度、初始/selected/final FFN张量SHA及best/last/训练状态文件SHA；其余训练/验证/图边/soft探针沿用。
- 测试先失败再实现；真实Swin/soft CPU合成图训练、反传、保存重建及旧路径回归通过。完整tests目录234项通过/exit0，119.99秒，44项既有依赖/合成TIFF提示。裸pytest因既有CLI文件顶层argparse中断；首次沙箱内9项子进程因OMP共享内存限制失败，获准沙箱外运行后全部通过。设计独立复核完成；代码审阅代理额度中断，没有独立代码签字，主代理完成限定diff核查。[本地验收与所有失败/成功日志](results/nwpu_ffn_local_validation.md)已归档，不称真实CUDA通过或已有模型收益。
- [9文件发布候选](deliverables/nwpu_ffn_trial_20261003.patch)仅含4个代码修改、2个新代码、2个测试与1份协议；改动前文件与ed480e6发布记录共30文件SHA核对通过，补丁在该文件树的检查/应用/字节一致性通过。拟独立分支codex/token-ffn-nwpu-20261003；目前未建分支、提交/推送或修改服务器。当前目录其他既有修改与历史结果保持。
- 后续发布完整checkout验收和Git同步后，先两组各360/90、一轮GPU短测；通过后仅两组30轮校准。历史旧两次校准47分58秒仅作参考，当前GPU/费用/FFN真实耗时尚未知。新val/test指标均待测试；不自动refit/test、加轮或转BigEarth，不改变此前软分配与反馈幅度负面结果。

---

## 2026-10-03 — NWPU反馈幅度诊断收尾：所有减弱设置均退步

- 用户回传nwpu_feedback42_20261002_01的run.exit=0与完整summary；1260张校准val、seed42、fixed/soft原best、预设α=0/0.25/0.5/0.75/1相符。原文按字节归档，结构化摘要与[本地算术核验](results/evidence/nwpu_feedback42_20261003/validation.json)保存；没有下载远程identity/日志/PT或逐类文件，不声称本地独立前向/F1/CE复算。
- α=1固定/软OA93.015873%/93.730159%与此前校准best完全一致；全部10行配对四格、正确数/OA/差值一致。两模型四种较小α的OA和macro-F1全部下降、CE全部上升。关闭反馈分别纠正8/12、新增17/21个错误，均净少对9/1260、OA下降0.714286 pp。α=0是联合训练模型的扰动，不是独立Swin或无图训练。
- 原缓存概率差≤3.8743e-7、捕获重建logit差2.8610e-6；CPU复算差fixed8.5831e-6/soft1.0490e-5。脚本采用atol/rtol各1e-5并要求argmax一致；不把稍超绝对atol误读成失败，也不称所有绝对误差小于1e-5。远端张量未回传，组合容差通过依据交付脚本完成语义。
- 全部原图直接来自ZIP，总46.331209秒；两缓存各约9.31MB，SHA与资源记录已归档。本次摘要未含GPU型号，不以历史配置代替实测。阶段时间含读盘/构建/缓存，不作公平模型速度结论。[完整结果表与论文表述](results/nwpu_feedback_sensitivity_seed42.md)已保存。
- 在该权重/验证集/预设范围内，“反馈过强，简单调小可改善”的假设不获支持，本诊断收尾；不扩α网格、不因结果重训或直接实现门控，不更改正式test主表。当前结果不定位软分配唯一失败根因、不外推BigEarth；多seed、外部方法与公平效率继续保持独立未完项。排期/矩阵/执行说明/综合讨论已补链接，未提交/推送Git。

---

## 2026-10-02 — NWPU反馈强度诊断本地准备完成，服务器待执行

- 用户在讨论软分配混合结果及下一步后明确继续；本轮实现有界、无训练的冻结权重诊断。研究问题是降低图反馈幅度如何改变NWPU完整val预测；固定seed42、ed480e6、原1260张val及fixed/soft校准best，仅预设α=0/0.25/0.5/0.75/1。α=0不是独立训练的Swin或无图模型，结果不能直接解释BigEarth或证明因果根因。
- 新增tools/diagnose_nwpu_feedback.py，复用哈希锁定的原图册来源验证；一次原前向捕获mean token/feedback、复算五档分类头，核对α=1缓存概率/预测/原val OA。直接读取work ZIP、逐图核对文件/像素身份，不需解压。保存两份可离线复算PT、配对预测、OA/macro-F1/CE/逐类/混淆矩阵及显式退出状态；旧输出不覆盖，失败不自动重试。
- 本地25项针对性/原来源检查通过；真实Swin与两种GAViT的随机权重CPU合成输入前向比较误差≤5.96e-7，缓存重载差0。单行命令解码/语法/假进程交付、重复保护和缺文件拒绝通过；[本地验收](results/nwpu_feedback_sensitivity_local_validation.md)区分真实CUDA尚未执行。
- [单行启动与回传说明](docs/run_nwpu_feedback_sensitivity_20261002.md)已准备，独立输出nwpu_feedback42_20261002_01；服务器尚未启动，指标与耗时待回传。只比较预设幅度趋势及类别损益，不自动选α、改模型、看test调参或扩seed；既有正式结果与多seed/外部方法/公平效率未完范围保持。未提交/推送Git。

---

## 2026-10-01 — 双数据集软分配消融整合与论文讨论草稿完成

- 用户同意继续整理既有实验。新增[双数据集正式表、原因边界与中英文正文](results/soft_assignment_cross_dataset_ablation.md)，以及6行精确JSON主表；分别报告NWPU seed42和BigEarth seed45，不混合OA/mAP、不将两个数据集视为同任务的两个seed、不改旧协议表。每组n=1，不填SD或显著性结论。
- 从已归档摘要重新核对三模型test、NWPU正确数/全部配对边际、BigEarth全部19类AP均值及贡献分解。soft−fixed分别为NWPU OA−0.234127 pp、BigEarth mAP+0.578683 pp；后者macro-F1+0.569346、micro-F1−0.319103 pp。NWPU纠正538/新增597；BigEarth海岸湿地贡献+0.826882 pp，其余18类合计−0.248200 pp。全部正负变化保持。
- 冻结ed480e6的gavit/soft grouping/identity源码与当前文件逐字节一致。反馈后立即mean pool意味着给定token与区域反馈时，分类输入只保留区域总质量加权反馈；合成数值核对误差2.78e-17，仅验证代数，不是新模型前向。S仍改变上游W、区域特征/图边/F，不能据此判软分配无效或宣称已确定因果根因。
- 将已观测抵消、类别集中、可视化变化与待检验的局部限制/原型趋同/反馈接口/训练随机性分开。8图探针与90图册不证明全局塌缩或稳定语义改进；相对fixed新增99,840参数不等于运行开销可忽略。生成脚本、来源哈希和算术核验保存在results/evidence/soft_assignment_synthesis_20261001。
- 本轮论文分析材料收尾，未改模型、启动GPU、按test调参或自动扩seed。排期及实验矩阵已链接新整合稿；既有NWPU报告顶部标明后续BigEarth/图册已完成，正文旧日期判断保留。多seed、外部方法统一比较与新结构公平效率仍未完成；不把本次整合写成导师全部要求完成，未Git提交/推送。

---

## 2026-10-01 — BigEarth soft45完整资源摘要回传，本批归档收尾

- 完整终端摘要已回传；原始粘贴缺外层左花括号、含末尾提示符，原始字节保持，仅结构化副本修补。核对冻结脚本SHA、完整入口参数、模型配方、split/预训练、批准GPU/精度和对照身份；训练metadata/test metrics与前批逐字段相同。[核验记录](results/evidence/bigearth_soft45_complete_20261001/summary_validation.json)通过。权重、预测和探针张量未下载，未声称本地重新前向或逐样本复算。
- 队列53058.733037秒（14小时44分18.733秒）；训练52678.069086秒（14小时37分58.069秒）；测试322.636569秒（5分22.637秒）。训练allocated/reserved峰值分别3929639936/4326424576字节；测试639322112/1103101952字节；主进程RSS分别2067432/1546488 KiB，不含loader子进程。有效训练吞吐135.466810张/秒包含验证和IO等全阶段开销，不能视为模型前向速度；费用未知，不同物理GPU的旧fixed时长不作严格效率比较。
- 固定8张val、初始/best/final三快照检查通过，S/W归一化最大误差2.384185791e-7/2.980232239e-7，实际48+32边；探针至epoch30、累计5.398102462秒。工程检查不证明语义合理性、全局无塌缩或多seed稳定性。
- 结果不变：soft test mAP70.836919%、macro-F1 64.864823%、micro-F1 76.383354%；较fixed +0.578683/+0.569346/−0.319103 pp，三项仍低于Swin。本轮训练/测试/资源归档完整收尾；报告、主CSV、排期和实验矩阵已补齐，历史CSV行保留原字节。无需新GPU运行或继续索取同批摘要；双数据集多seed、公平效率和外部方法比较仍是独立未完项，未提交/推送Git。

---

## 2026-10-01 — BigEarth soft45已完整完成，mAP较固定分组改善但仍低于Swin

- 用户归还实例后回传持久化产物：总队列/训练/测试exit_code均0，history到epoch30，test3745/3745和BIGEARTH SOFT45 COMPLETE标记齐全；测试119825张。训练已在实例归还前完成，无需恢复数据、续训或重跑。原始附件字节归档，文件首行缺O和history前段截断保持，不伪造完整原日志。
- soft45 test mAP70.83691876%、macro-F1 64.86482284%、micro-F1 76.38335431%；相对固定45分别+0.57868251/+0.56934585/−0.31910255 pp，相对Swin45分别−0.84001078/−0.21938919/−0.34891006 pp。属于单seed混合结果，不能声称三指标全面改善或稳定超过Swin。完整[三模型消融与逐类表](results/bigearth_soft45_ablation.md)和主BigEarth CSV已登记，历史行逐字节保留。
- 19类AP均值重算与macro-mAP一致，12升7降。海岸湿地AP+15.710764 pp、对mAP贡献+0.826882 pp，其他18类合计−0.248200 pp；草地/稀疏植被−7.489263 pp，沙滩/沙丘/沙地+3.147280 pp。保留全部类别；这是分解而非删类/因果/显著性结论，不按结果改配置、重跑或挑seed。
- metadata clean ed480e6、seed45/30轮/soft_prototype及冻结配方相符；best epoch11、val mAP78.7037%。metadata/test报告checkpoint SHA同为4969f401…bd35e2；本地未下载权重字节，不称独立服务器重哈希。最终loss0.07173317、val mAP76.22925382%，较best低2.474446 pp；完整可解析曲线只含17–30轮，最佳选取/全30轮由冻结队列完成检查支持。
- best权重125955898字节、last state377875857字节、预测NPZ9307603字节、summary13618字节均在work中有报告。后14轮墙钟均值1744.81秒，test循环显示05:04；全程墙钟/峰值显存/具体probe摘要尚未返回，不能将片段外推为实测资源或费用。下一步只读取已有summary补齐归档，训练/测试已完成；双数据集多seed和外部方法等导师范围仍未完成。

---

## 2026-10-01 — 用户报告实例已归还，soft45完成情况待持久化产物核验

- 用户报告昨晚睡着后实例已归还，询问如何检查输出。上一条已知执行证据仍为epoch1的1201/7434；不能按隔夜时间或原预算推定已完成，也不能直接认定从头白跑。原实例进程不会跨归还继续运行，是否此前已完成尚未知。
- 先读取持久化work中总/训练/测试退出码、history、best metadata、best/last state及测试产物存在性和有界日志尾部。只需标准库，不加载权重、不用GPU/原图，不先恢复BigEarth或再次执行独占启动命令。只读单行命令语法及缺目录/部分产物情形通过，文件保持不变；服务器输出待用户回传。
- 缺run.exit不等于已有权重丢失；磁盘run.json的running也不能证明归还后的进程仍活跃。若未完成，后续先确认last state完整性、实际完成轮次与探针一致性，再确定续训/仅补测试方案；本轮没有改脚本、清除claim、删除产物或重新启动训练。正式指标仍待测试/确认。

---

## 2026-09-30 — BigEarth soft45真实CUDA首轮已开始

- 用户回传START gavit_train、RTX2080Ti/CUDA12.1/8线程及原批准GPU UUID/软件/精度；train237871、val122342、31,466,766参数、Device cuda、计划30轮均相符。Epoch1已到1201/7434（显示16%），确认超过仅有PID的阶段，真实训练batch已执行；未收到独立nvidia-smi显存快照，峰值显存与最终资源仍待产物。
- 当前训练进度03:53、剩余21:01、显示4.94batch/s；仅按该快照推算首轮训练循环约24分54秒，不含其后完整验证/存盘/探针。总预算暂保持16–18小时规划，首个完整epoch记录回传后再修正，不根据局部吞吐声称完整训练耗时已测得。
- [启动日志结构化转录](results/evidence/bigearth_soft45_prepare_20260930/startup_user_return.json)已归档；无val/test新指标、最佳权重或退出码，不标实验完成。继续既有队列，30轮后自动评估best；不重新启动、恢复数据、改配置或另开GPU任务。排期、矩阵和运行说明同步为训练中，历史完成结果保持。

---

## 2026-09-30 — BigEarth soft45服务器预检通过，交付单模型正式训练/测试队列

- 用户回传BIGEARTH_SOFT45_PREFLIGHT_PASS：全split237871/122342/119825及冻结SHA一致，旧Swin/固定GAViT最佳权重实际SHA和30/30/正式test记录通过。RTX2080Ti/CUDA12.1/cuDNN8902/Python3.11.8/包版本与精度相符；当前UUID为GPU-f57c9bd1…54ffab，旧对照GPU-0fa5e2ad…1de00b，驱动同610.57.04。记录不同物理卡，不宣称完全同设备。用户摘要已转录归档，未直接SSH或下载原文件。
- 新增独立tools/run_bigearth_soft45.py调度冻结ed480e6入口：先独立子进程重核预检，随后只做soft45的30轮训练→best完整test；复用原已核验seed45 helper，不创建假的smoke报告或修改正式模型代码。输出bigearth_soft45_20260930_01独占创建，失败停止、无自动重试/覆盖/扩seed。实际服务器启动仍待用户粘贴执行。
- 完整记录入口参数、脚本SHA、精确历史与best/last身份、0–30轮分配探针及实际48+32边、test精确三指标/19类AP/逐样本NPZ和阶段资源范围。主进程不初始化CUDA，避免预检上下文阻止自身GPU空闲检查；独占claim先于退出码写入，重复调用不破坏活动结果。
- 本地先建立失败检查再实现；12项新旧针对性测试通过（含真实CPU分配/图边快照、损坏拒绝、记录hook的返回/RNG保持），独立审阅无新增问题。单行载荷与源码一致、Python/Bash/zsh语法、后台实际封装的预期失败停止、两种重复入口保留原字节检查通过。均不替代正式GPU执行证据；原双数据集GPU smoke保留，无需重复。
- [完整启动/进度/完成命令](docs/run_bigearth_soft45_20260930.md)已交付，无需上传文件；[本地验证](results/evidence/bigearth_soft45_prepare_20260930/launch_validation.json)与tests.log归档。预算仍按16–18小时规划，首轮后修正。新正式指标全为待测试，单seed不构成稳定性；未Git提交/推送、未修改模型或旧结果。

---

## 2026-09-30 — 用户重新确认BigEarth软分配消融，先复用完整seed45对照

- 用户看到导师要求后明确同意继续BigEarth正式软分配，并回复当前RTX2080 Ti、原数据需要恢复。此指令取代下方同日较早的撤下安排；尚未启动GPU训练，也不将当前准备写成完成实验。
- 核对现有seed45 Swin/固定GAViT均30/30、test119825、完整退出和资源记录，优先重新核验服务器权重身份后复用，仅补已发布soft_prototype seed45。选择依据是完成与可追溯性，不是测试分数；不重跑旧对照、不自动扩多seed、不改现有负面结果。
- [正式准备说明](docs/run_bigearth_soft45_20260930.md)写明假设、主要模块变量、三模型消融表及单seed边界，冻结原split/预训练/30轮/batch32/优化和精度。soft代码ed480e6、旧对照deb2b45如实区分；当前只增加新模块、反馈和探针支持，沿用已有真实GPU短测。best依val mAP选取，最终须完整test的mAP/macro-F1/micro-F1/19类AP、预测及资源记录。
- 已准备无需上传文件的单行只读预检，检查原工具哈希、旧对照完整身份/权重、软分配发布代码、GPU与软件环境和全split原RGB文件；现有formal soft45产物则停止供检查。Python/Bash/zsh语法、载荷往返通过，复用文件检查用例通过；服务器真实预检仍待数据恢复后回传，没有真实CUDA新执行证据。
- 原固定GAViT同型号GPU训练阶段14.89小时；soft暂按16–18小时规划训练/测试，另加数据恢复，首轮实际耗时后修正。当前小时租价未知，未捏造费用。排期/E05状态已同步；本轮无模型修改或Git提交/推送，完整训练包装器待预检结果后交付。

---

## 2026-09-30 — 用户取消当前BigEarth软分配训练安排，先收敛研究问题与成本

- 用户明确表示不想在BigEarthNet跑软分配，原因是规模大、耗时长，并质疑新增实验能证明什么。尊重当前决定：撤下此前建议的下一步BigEarth正式训练，不准备/启动该付费任务；此前并无已启动的BigEarth软分配正式作业需要停止。
- 澄清实验问题：完整同协议三模型对比只能进一步测量该候选在BigEarthNet上的性能/成本表现，补另一数据集证据；单seed不能证明稳定性，分类指标不能单独证明分组语义更合理或解释因果。NWPU seed42目前未显示整体优势且只是单seed，不足以宣布所有软分配方案无效；同样没有依据仅因模块已实现就自动扩大付费验证。
- 复核导师原邮件确实要求NWPU和BigEarthNet上的Swin/固定网格/可学习软分配消融。该要求与当前执行决定分开记录：BigEarth条目是未执行、论文范围调整待沟通，不标完成、不改导师原文、不擅自用AID或小样本替代，未对外发邮件。
- 下一步优先明确已有证据支持的论文主张和真正缺少的对照、整理既有结果/写作与外部方法材料；不立即新增另一轮数据集、seed或参数搜索。排期和E05主体同步，所有正负结果保持，未修改模型或Git历史。

---

## 2026-09-30 — 真实90图册审阅完成：局部分组改变，尚无一致语义区域增益证据

- 用户传回cases_compact.html，43,162,303字节/41.16MiB，比原HTML减84.43%，SHA d92ad8c7…4027已按实际文件计算、逐字节归档。解析90唯一路径/45类各2张/180SVG，预测复算82双方正确、3纠正、0新增、5共同错误；实际查看全部90总览及8例完整S图。[审阅报告及正反例图](results/nwpu_regions_visual_review_20260930.md)。
- SVG精确区域ID/边属性统计：1724/4410 token（39.09%）改变主归属、每图8–30个；两模型特征边共有992/2880（34.44%），空间边一致。它们是显示层结构比较，不是S连续量、边质量或因果；原90例PT/W未下载，前向数值门禁仍引用服务器摘要。
- 3纠正为church_063、medium_residential_005、railway_station_016；5共同错误为forest_006、freeway_189、palace_016、river_043、runway_025。混合水林、厂房/道路竞争线索、宫殿/教堂、机场/跑道共存及狭长河道均已结合实际图片描述，未认定标签错误或单一根因。软分组并未显示出持续沿地物边界形成连贯区域的充分证据；成功案例同样不能解释为某条边导致纠正。
- 全90图总览与8例细图、全部3纠正/5共同错误合成图及中英文图注已归档。强调7×7 token显示、锚点固定、S不同于实际W聚合、边不代表注意力；无argmax赢家不等于区域塌缩。正式test soft对fixed净−59保持，不能用90例+3替代。此批可视化审阅收尾，其余E05/外部方法/效率未完；无模型修改、新训练或Git提交/推送。

---

## 2026-09-29 — NWPU 90图真实捕获完成，轻量图册待取回审图

- 用户回传run.exit=0/status complete，45类各2张val、calibration best fixed/soft、90张全来自ZIP，1173.4355秒（19分33.44秒）。已转录原字段及来源声明、核对配对整数与误差范围；证据是用户回传摘要，未本地下载核验原ZIP或实际语义图稿。详见[本批完成表](results/nwpu_regions_server_completion.md)。
- fixed/soft正确82/85；双方正确82、soft纠正3/新增0、双方错误5。捕获/重建/恢复最大logit差2.861e-6、缓存概率1.520e-6；S/W误差2.384e-7/3.576e-7，实际48+32边、all_capture_checks_passed=true。工程诊断完成；固定案例计数不能代替正式test或稳定性结论。正式soft−fixed仍净−59，负面结果保留。
- 原HTML277287895字节、证据ZIP159449863字节/报告SHA03e899fd…afd75，前者重复嵌入4+16份底图；独立[轻量导出](docs/compact_nwpu_regions_20260929.md)只去重相同PNG引用，保留全部90例/16区域细图与原证据。90合成案例180SVG、原文件不变/防覆盖/错数量拒绝、两种SVG离线渲染像素相同和命令语法/载荷往返通过；服务器导出大小/真实审图仍待回传。
- 静默阶段后已完成，无需继续PID诊断或重跑；总耗时不代表纯推理时间，早期慢阶段根因未作无证据定论。下一步接收图册做语义观察，无新GPU任务、模型修改、Git提交/推送或多seed扩展；E05其余双数据集多seed/效率及外部方法比较保留。

---

## 2026-09-29 — 可视化静默阶段定位到首模型权重加载

- 用户回传PID2723仍在，Rsl，elapsed08:28、CPU累计04:53（57.7%）；日志已出现SOURCE PASS及RTX2080Ti/CUDA12.1。输出有cases、selection.json、identity.json；源码顺序表明已通过原90图读取、哈希和预处理阶段。fd44打开fixed/best.pth，尚无CAPTURE/退出摘要。
- 当前证据指向首个模型的torch.load权重读取/反序列化阶段，不能仅凭进程活跃断言正常吞吐，亦不能把08:28整体归为该步骤耗时。work网络盘可能影响读取，但尚无两次IO/文件位置对比或栈证据，根因未确认。
- 已归档[服务器状态摘要](results/evidence/nwpu_regions_20260929/startup_server_status.json)，准备15秒间隔双快照的只读进度检查，比较CPU累计、IO、fd44位置、产物数量及日志。Python/Bash/zsh语法和双快照本地模拟通过；不kill、不重复启动，不修改活动脚本或重新解压。

---

## 2026-09-29 — 90图可视化后台已发起，首条日志尚未回传

- 用户回传PID2723、输出nwpu_regions42_20260929_01，数次tail均无可见内容；仅确认父启动器发起了子进程，尚无SOURCE PASS/DEVICE/首例CAPTURE，不标记CUDA诊断启动成功或完成。
- 源码确认首条print位于manifest/run.json/checkpoint/模型源码/缓存预测校验、Torch/模型依赖导入、CUDA检查之后；静默阶段缺乏进度可见性。当前不知道实际停留在哪步，不能将空日志直接归因为数据缺失、GPU故障、输出缓冲或死锁。
- 已准备仅查看PID2723的ps、proc I/O/wchan/打开文件、输出列表及已有日志/summary/exit的单行命令，Python/Bash/zsh语法通过；等待服务器证据。没有kill、重复启动、修改活动脚本或重新解压；Ctrl+C结束tail不停止该独立会话后台进程。

---

## 2026-09-29 — 用户确认90图可视化方案，本地实现与验收完成

- 新增独立`tools/visualize_nwpu_regions.py`；两份已有校准best、原val每类路径前2张共90张、两模型真实S/W及图推理输入，保存全部正负案例。原图/像素/权重/run.json/代码哈希及原缓存概率核对；原数据丢失可直接读持久化ZIP，不整包解压。未修改模型和训练入口。
- 产物为自包含HTML、每例4列SVG与16区域完整S图、原像素PNG、原始张量/身份和证据ZIP。区分主归属/真实软分布、锚点先验/真实地物边界、构造边/GAT自环，不能把线宽视为注意力或把熵视为预测不确定性。
- 本地15项新增+13项相关回归共28 passed；90合成图完整流程、真实分组/PyG及542成员ZIP通过。另用完整Swin-T/PyG随机权重CPU前向，捕获/重建/恢复差均0；这些不是受训权重/CUDA/真实图片验收。图稿经离线SVG渲染实际查看；浏览器file打开被策略阻止，未绕过。详见[验收记录](results/nwpu_regions_local_validation.md)。
- 独立审阅1项P2已用实际runner复现并修复：现在推理前保存原PNG/选图身份，缓存不符也保留失败PT和error；exit1且原输出不覆盖。最终28项通过；脚本解码/后台启动参数/Bash/zsh语法/防重复启动的本地检查通过。
- [直接粘贴执行说明](docs/run_nwpu_regions_20260929.md)已生成，无需上传工具包或改训练仓库。服务器目标nwpu_regions42_20260929_01，尚未启动；真实运行时长、GPU结果及语义分析待用户回传。本轮无Git提交/推送或PR合并、没有新增训练；E05其他未完范围保持。

---

## 2026-09-29 — 收尾后可视化入口只读核对

- 用户要求继续。先核对最新排期、E05矩阵、既有结果表及Featurize runbook；修正矩阵主体仍写“NWPU校准待执行”的过期状态，seed42已完成范围以校准/refit/test/配对收尾为准。
- 下一项建议补跨类别固定/软分配真实图边案例，使用现有校准best与原val图，避免将refit训练池内的val图称为未见样本。建议每类按路径固定取2张，共90张，不根据正确性、置信度或test退步类别选图，全部案例保留。它是结构观察，不能代替正式总体指标或机制因果验证；此采样/绘图扩展方案尚待确认，未实现或运行。
- 现有`visualize_region_diagnosis.py`的`forward_diagnosis`手工重建固定分组且硬编码`build_knn_graph`，不能用于当前sparse_hybrid/soft模型。`tools/diagnose_bigearth_feedback.py`虽捕获真实前向，但限定19类固定分组，不能直接套NWPU。`SoftAssignmentProbe`已保存S/W和真实edge_index，但scene入口只取排序前8图，不能作为跨类别覆盖证据；本地也未取得这批真实张量和原图。
- 拟扩展既有捕获/绘图流程，逐图真实forward，区分S归属与W汇总、48空间+32特征构造边和GAT内部自环；argmax图只能显示主归属，附完整软分配及分配熵，不称像素分割真值或因果解释。旧命令不能直接交付；本轮无训练、GPU执行、模型代码改动或Git提交/推送。

---

## 2026-09-29 — NWPU seed42缓存配对回传，本批完整收尾

- 用户附件返回NWPU_TEST_PAIRS_COMPLETE；OA/macro-F1从25200条同路径同标签唯一预测复算，与正式队列完全一致。原文缺最外层左花括号，解析副本只补该字符，原始字节保持；本地核对三张2×2表的总数、模型正确数/净差、逐类重复计数和混淆上界，全部相符。[收尾分析](results/nwpu_seed42_closeout.md)。
- fixed对Swin纠正586/新增543、双方正确23128/双方错误943，净+43；soft对Swin纠正570/新增586、23085/959，净−16；soft对fixed纠正538/新增597、23117/948，净−59。一对一错分别1129/1156/1135张（约4.48/4.59/4.50%），总分接近不等于逐图行为相同；双方错误也可能预测不同类。
- soft相对fixed主要退步为密集住宅−18、火车站−17、教堂−13、沙漠−13；改善为工业区+14、稀疏住宅+13、网球场+11、跑道+11，每类560张，榜单各仅前四。宫殿↔教堂双向混淆Swin/fixed/soft为135/134/130，局部混淆下降未形成整体增益。没有原图/因果干预，不将计数解释为区域塌缩或图反馈原因。
- 新协议正式结果表补齐配对收尾链接，历史主CSV和其他数据集结果保持。seed42校准→refit→test→缓存诊断链已收尾，无需再跑本批命令；不改参数、不补跑seed42、不按test选择赢家或新种子。本轮没有模型代码修改、Git提交/推送或PR合并。
- E05双数据集多seed、跨类别区域/图边可视化与独立效率仍未完成；本批完成不代表论文全实验完成。没有新增GPU任务，后续付费范围不自动扩大。测试集正式结果保持Swin93.932540/fixed94.103175/soft93.869048%，当前不支持soft整体或稳定提升。

---

## 2026-09-29 — NWPU seed42六阶段完成：fixed小幅领先，soft验证优势未延续到test

- 用户回传queue.exit=0、六阶段process_exit=0/checks_passed=true及六个run.exit=0，队列3965.2774秒（66分05.28秒）。原始附件逐字节归档并解析，三模型6300/0、refit29/29/28、架构/校准best来源/最终test权重SHA链及OA整数计数本地核对通过。[正式测试表](results/nwpu_seed42_test_comparison.md)。
- 25200test：Swin OA93.93253968%、macro-F1 93.90914547%；fixed94.10317460%/94.08993346%；soft93.86904762%/93.85498722%。正确数23671/23714/23655；fixed−Swin净+43（+0.170635 pp），soft−Swin净−16（−0.063492 pp），soft−fixed净−59（−0.234127 pp）。单seed小差距不作稳定/显著结论，也不把净差误当纠正/破坏数。
- soft校准val相对fixed +0.714286 pp没有延续到正式test。校准与refit的训练池、学习轨迹和评价样本不同，不能据排序逆转断言错误、过拟合或区域塌缩；全部正负结果保留，不按test修改配置或补跑本seed。两个GAViT实际48+32边、重载差9.54e-7通过，Swin重载差0。
- 单项refit墙钟1108.58/1257.25/1240.20秒；连同校准累计约138.54分钟，不含解压/空闲，租价未知。当前只获取摘要字段，完整training/code/hardware一致性依据队列checks，原始checkpoint/预测CSV未下载，不冒充独立服务器文件复核。
- 本seed付费流程完成，下一步仅用缓存test CSV配对与逐类统计收尾；不再启动GPU或新训练。本地模型代码、Git/PR状态及历史协议表保持；对应新协议测试表、排期和矩阵更新。E05完整双数据集多seed与多类别可视化仍未完成，不把单seed完成扩大为整个课题完成。

---

## 2026-09-29 — 用户确认继续，交付NWPU seed42三模型refit/test队列

- 用户在1–1.5小时预算估计与29/29/28选轮方案后回复“继续”，本轮完成六阶段命令与本地调度验证，服务器尚未启动。[执行说明](docs/run_nwpu_refit_test42_20260929.md)固定代码ed480e6、原manifest/预训练/seed42/配方、三个校准checkpoint SHA和新输出nwpu_refit_test42_20260929_01；不重复校准或smoke，不修改模型代码。
- 研究问题是soft相对fixed的机制收益是否延续到独立test，同时如实保留与Swin差距；三种架构为主要变量。先从原ImageNet权重分别在6300图训练池拟合29/29/28轮，Cosine T_max仍30，后统一对25200test评估三个last.pth。全程三模型串行，不用test选择配置或轮数。新指标用于同协议NWPU消融表，当前指标均待测试。
- 准备命令只读核对Git/输入/三个校准来源、31500路径及GPU空闲，搜索work中nwpu目录及下一层的已完成同清单seed42 refit以避免重跑；有缺图或旧结果时在新目录创建前停止。队列再核验冻结身份，独占启动标记防重入，每项须进程、run.exit、来源/数量/权重/硬件等检查通过才继续；失败保留已完成模型和日志，不自动重跑。
- Python/Bash/zsh语法及模拟六阶段成功、首项/中项/后期测试失败、返回exit0但身份错误、重复启动保护和缺图提前拒绝均通过，详见results/evidence/nwpu_refit_test42_20260929/local_validation.json。zsh语法检查返回0但沙箱出现nice(5)非致命提示；不涉及真实GPU或训练。没有重新运行神经网络回归，因为发布代码和既有入口均未改变。
- 原GPU实测缩放仅训练约56.87分钟，含检查/测试规划1–1.5小时（不含解压/初始化）；实际耗时、显存、训练曲线、测试OA/macro-F1/逐类/混淆矩阵与成本待回传。已有队列与全部正负结果保持，无Git提交/推送或PR合并，多seed及BigEarth完整软分配消融仍未完成。

---

## 2026-09-29 — NWPU缓存诊断完成：软分配存在纠正与破坏，进入正式拟合前的决策

- 用户附件完整至CACHED_VALIDATION_READ_COMPLETE，逐字节归档并解析1260张配对统计、45类支持检查、三模型30轮舍入曲线与0/28/30探针；本地整数/曲线/身份一致性复算通过。共同training/code/hardware均true，Git ed480e6干净、RTX2080Ti、FP32双TF32-off，原图与GPU无需再次读取。[完整诊断](results/nwpu_seed42_cached_diagnosis.md)。
- Swin/fixed/soft macro-F1为94.171552/92.976564/93.765721%，同OA排序。soft对fixed纠正40/破坏31，净+9；对Swin纠正30/破坏36，净−6。末5轮OA均值94.0000/92.8889/93.5873%；soft最佳到最终净少对3张，无明显后期崩落证据，不因结果偏低盲目加轮。
- 软探针投影原型cosine均值0.33662→0.78264、熵0.67858→1.48263，提示更相似/更分散；0/28/30无低范数、最小质量约1，feedback比降到0.515。它只覆盖排序前8张val而非类别分层，不能据此证明全局塌缩或造成精度差距。源码推断偏向airplane，具体sample_ids此次未返回；不更改模型或事后挑样本取代原探针。
- 新协议表补macro-F1、参数/资源和诊断链接，负面结果保留。建议冻结v1，下一批按已定选轮规则进行三模型refit29/29/28（train+val6300，Cosine周期30，从原预训练重启）及25200test；单seed研究价值是检验相对fixed的机制增益，不保证超过Swin。按当前吞吐估纯训练约56.87分钟，含检查/测试等规划1–1.5小时、租价待定。尚未启动付费任务或扩多seed/BigEarth正式训练，E05完整消融仍待完成；本轮无Git提交/推送、模型改动或PR合并。

---

## 2026-09-29 — NWPU三模型seed42校准完成，软分配改善固定网格但仍低于Swin

- 用户回传nwpu_ablation42_20260928_01队列及三阶段exit0，三模型status complete、30/30、train5040/val1260、seed42；共同manifest与预训练SHA符合冻结值。返回字段逐项转录及本地数值核对已存档，未直接下载服务器run.json/权重。当前新协议[消融结果表](results/nwpu_seed42_calibration_comparison.md)独立于历史约96%的70/15/15记录。
- Swin最佳val OA94.206349%/epoch29，fixed93.015873%/epoch29，soft93.730159%/epoch28；换算1187/1172/1181张正确。soft−fixed +0.714286 pp（净+9张），soft−Swin −0.476190 pp（净−6张）；仍未达到超过骨干的假设，不能声称稳定提升或解释因果。
- 实际48+32边均通过，最大重载差0/3.5763e-7/3.5763e-7。全过程1394.24/1455.64/1483.65秒，队列4346.87秒（72分26.87秒）；allocated约3.5584/3.6559/3.6626GiB，属于完整运行峰值和墙钟，不是推理效率。完整training/code/hardware/params、macro-F1、逐类预测及曲线待只读回传。
- 研究问题、预定commit/命令/配方沿用[启动记录](docs/run_nwpu_ablation42_20260928.md)，完整architecture、checkpoint/清单/预训练哈希见结构化证据。下一步仅读取已有验证预测、history及soft探针；不追加训练，不自动refit/test、多seed或调温度。E05正式双数据集消融与稳定性仍未完成。本次仅记录与诊断命令，无模型修改、Git提交/推送或PR合并。

---

## 2026-09-28 — 双数据集工程验收收尾，交付NWPU三模型seed42校准队列

- 用户回传BigEarth最终只读检查JSON（所有断言之后打印，开头PASS行未粘贴）。S/W归一化误差均2.3841858e-7，快照身份/mask/别名/日志一致性和480波段逐文件ZIP哈希通过，备份9620902 bytes；[完整结果](results/bigearth_soft_smoke_validation.md)及机器可读核对记录更新，BigEarth/NWPU预定工程门槛均收尾。
- 精确短测loss0.5011217395、val mAP37.0546909675%、macro-F1 2.6737323859%，含train/val/存盘但不含probe的epoch wall7.815954543秒。train无类16正例，val无类14/16正例，所以本次mAP只对17个有正例类求平均；不进正式表，也不据低F1改阈值或换样本。探针熵0.77295→1.11003，S变化0.05066，低范数均0，反馈/token均值1.36484→1.49532；不扩大为语义或收益证据。
- NWPU哨兵图false，先用持久化ZIP（固定SHA beeecd0b...1f53）校验并恢复；不重新下载/prepare。正式train/val前仍全量图像哈希验证，哨兵不能替代全量验收。
- [三模型队列](docs/run_nwpu_ablation42_20260928.md)对应已批准E05研究问题：Swin/固定4×4/soft_prototype同ed480e6、manifest5040/1260/25200、预训练、seed42和scene配方，各30轮calibrate，以val OA为主，不启动refit/test或多seed。准备时检查可复用同协议结果、GPU空闲及新目录，串行失败即停，保留旧产物。原历史70/15/15结果不复用为新baseline。各命令和身份在启动前固定，用户复制执行；尚未有队列实际启动/训练结果。
- 本地仅做命令语法及队列调度验证（模拟子训练成功/首项失败/中项失败）；三种路径的执行顺序/停止/退出记录通过。首次测试mock误拦截Git内部subprocess调用，调整测试边界后通过，未因此改交付队列。该测试不冒充GPU正式训练；源码不改、无Git提交/推送/PR合并。后续根据首轮实测调整时间估计，不用小样本26秒乘30报价。

---

## 2026-09-28 — BigEarth软分配96/64真实GPU训练与重建回传通过

- 用户返回exit_code=0（首行粘贴为xit_code=0）与training_and_reload_complete。RTX2080 Ti、96train/64val、1轮，实际48+32边、两张原val探针重建logit差4.76837158203125e-7；完整[工程验收记录](results/bigearth_soft_smoke_validation.md)、回传摘要和本地数值核对已存档，不冒充直接获取服务器原文件。
- best epoch1、val mAP37.0547%仅用于短测；19类中无正例类会被入口跳过，完整逐类支持和精确loss/val F1尚待日志返回，不能与正式完整验证/测试比较。报告权重SHAfbde01de...944daef。
- 入口墙钟24.338881547秒、包含重建26.110388833秒，峰值allocated3.6598GiB/reserved4.0371GiB，主进程RSS1.7790GiB；固定8val探针epoch1、累计1.713090071秒。FP32/无AMP/matmul TF32 false/cuDNN TF32 true，保留BigEarth原配方；不把跨数据集/不同TF32设定的资源数当公平效率结论。
- 至此双数据集真实训练与重载路径已有证据；BigEarth缓存S/W身份/归一化/别名、有限loss及支持数、480原波段备份逐文件哈希仍需一次CPU只读收尾。检查命令已在同一合成CPU产物上验证通过，不追加训练；也报告NWPU哨兵图存在性以衔接下一阶段。
- 下一阶段仍为同协议NWPU Swin/固定4×4/soft_prototype的seed42三模型校准，先val比较，再按冻结规则refit/test；正式训练和多seed尚未启动。本轮未改正式代码、历史准确率表、提交/推送或合并PR。

---

## 2026-09-28 — 用户报告BigEarth数据恢复，交付96/64单轮短测命令

- 用户回复“恢复好了”，作为人工恢复报告记录，尚无恢复后480波段逐项/解码输出或CUDA训练结果。提供[准备、单轮训练及结果回传三步](docs/run_bigearth_soft_smoke_20260928.md)，保留固定Random(42)抽样和原CSV哈希，代码仍为ed480e6。
- 准备命令检查所选160图原B02/3/4及真实reader解码，拒绝merged图像覆盖，成功才创建独立`bigearth_soft_smoke42_20260928_01`，保存小CSV、正例支持数、完整哈希及CRC校验过的480原波段ZIP到work，方便以后短测恢复；这些服务器产物尚未产生。不会用于替代完整正式训练集。
- 训练包装器使用原train_bigearth入口，19类BCE、seed42、1epoch、batch32、workers4、train pin=True/val=False，保留原TF32并记录；Cosine T_max随smoke轮数为1。记录退出码/设备/耗时/峰值，结束后同配置strict重建、同2张原val图max logit diff≤1e-5，hook核对48+32边，确认last state及四探针文件。不会只凭PID宣布启动成功，完整诊断仍以回传验收。
- 本地命令测试使用已发布隔离源码、真实Swin/PyG、合成uint16TIF，随机权重/CPU/2线程/workers0；相同96/64训练原入口exit0、重载差0、48+32实际边通过。prepare/CRC/清单通过，已有输出与缺波段拒绝；Bash/zsh/Python语法通过，详见[本地证据](results/evidence/bigearth_soft_preflight_20260928/launch_local_validation.json)。这些不代表真实GPU或预训练验收，未把合成指标写入正式表。
- 本轮仅生成命令/文档/验证记录；未修改或推送正式模型代码、未连接服务器代跑、未启动完整训练或合并PR。下一步用户按顺序复制命令，返回run.exit和summary。

---

## 2026-09-28 — BigEarth原包定位返回：data目录不存在

- 用户回传ls/find均报告`/home/featurize/data`不存在；work最大深度2仅列AID（2628474154 bytes）、NWPU（427389445 bytes）与两个小诊断ZIP，未找到BigEarthNet原始包。[定位证据](results/evidence/bigearth_soft_preflight_20260928/archive_locator_user_return.json)已存档，不据有限深度搜索断言服务器全局没有原包。
- 当前缺乏定向提取480波段所需的源归档。核对既有Featurize runbook，下一步重新添加此前使用的BigEarthNet-S2数据；若界面支持仅下载，先取原包并回传路径，检查成员后再定向恢复，否则按平台提供流程恢复。未现场核验平台当前选项，不承诺可绕过自动解压。
- 计划恢复后将固定160张原RGB另存小归档到work，避免今后短测重复恢复全量；尚未创建，也不替代正式训练的完整数据。原CSV/模型保持，BigEarth短测尚未启动，NWPU已完成的工程证据有效。

---

## 2026-09-28 — BigEarth软分配短测预检返回160图原波段缺失

- 用户回传固定96train/64val需要480波段、missing480、BIGEARTH_FILES_MISSING，首例为T34VER_07_76的B02/B03/B04。原train/val CSV哈希、19列二元标签、唯一绝对路径等断言已执行到最终输出。结构化摘录保存在[预检证据](results/evidence/bigearth_soft_preflight_20260928/data_presence_user_return.json)。
- 只确认抽样文件在原路径不存在，不据此断言全部数据或ZIP已删除；结合实例本地data生命周期，下一步先查已有压缩包位置/格式，再定向恢复相同160图的480个RGB波段，避免未经确认的整包解压或重新抽样。
- [定位命令](docs/check_bigearth_soft_data.md)仅列data和data/work两层压缩文件，Bash/zsh语法检查通过；服务器执行结果仍待返回。未启动BigEarth GPU短测/正式训练、改模型或正式结果表，NWPU工程验收仍有效。

---

## 2026-09-28 — NWPU快照与CUDA分配梯度回传通过，工程检查收尾

- 用户回传两条完整命令与PASS结果，归档到[原始结果摘录](results/evidence/nwpu_soft_smoke42_20260927_02/final_checks_server_return.txt)，本地解析并核对阈值、摘要一致性及有限非零梯度；未直接下载服务器checkpoint/tensor。
- 快照S行和误差2.384185791015625e-7、W列和误差2.980232238769531e-7，S平均变化0.03925492987036705与既有摘要一致；最低区域质量1.000672459602356。命令的身份、有限非负、mask外零、epoch1/best/final别名及JSON核对全部走到PASS。
- best权重与原8图预处理SHA256匹配后，仅前2图在CUDA做S随机线性目标反传；投影/注意力/骨干梯度绝对值和6396.894643783569/9804.477229118347/596998.582154274，均有限非零，GAT/feedback/classifier均无梯度；未更新或保存参数。不能把梯度量级或小样本区域质量当成分类收益或语义无塌缩证据。
- 合并先前360/90训练、验证、48+32图边、保存重载与探针证据，NWPU预定工程检查收尾；[验收记录](results/nwpu_soft_smoke_validation.md)、核对JSON及排期已更新，正式比较/消融精度表不填smoke数值。
- 下一步按既定计划验证BigEarthNet多标签BCE路径（96train/64val、seed42、1轮）；先用[只读数据检查](docs/check_bigearth_soft_data.md)核对冻结CSV及固定抽样480个RGB波段存在，避免无条件整包解压。当前数据状态未知，尚无BigEarth新GPU结果，也未启动正式训练、改发布源码、提交/推送或合并PR。

---

## 2026-09-28 — NWPU探针摘要回传，准备两项剩余工程检查

- 用户回传_02的history/probe小摘要：train3.235914936秒、360/秒数=111.2513793图/秒，val0.525494186秒，loss3.579246097。固定8val的熵0.692803→1.014223、平均绝对S变化0→0.03925493，低范数原型数均0，最小原型范数0.556113→0.661625；反馈/token比均值1.038349→0.759035。epoch0/1/best/final文件存在均true。[记录及解释](results/nwpu_soft_smoke_validation.md)已更新，提供的截断JSON仅补外层结构，不冒充服务器原文件。
- 数值和吞吐复算一致；分配变化不证明新投影专属路径可微或分类提升，熵升高不直接判定好坏，低范数为0不等同于排除语义塌缩。无需改超参或重训此smoke；正式比较表仍不填入40%的小样本分数。
- [两条剩余检查](docs/check_nwpu_soft_final.md)核对S/W归一化、mask、快照身份/别名与JSON变化一致性，以及原checkpoint/原8图输入哈希匹配后的2图CUDA S专属反传。投影、注意力、骨干应收到有限非零梯度，GAT/分类头不应收到；不更新或保存权重。已用本地真实Swin/PyG、合成图/随机权重CPU检查命令可运行，并拒绝损坏归一化和人为断开梯度两个反例；该结果不冒充服务器验收。
- 服务器两项检查及BigEarth96/64原图短测仍待执行；本轮没有修改发布代码、推送、转PR就绪或启动正式训练。PR与已完成_02产物保持。

---

## 2026-09-28 — NWPU软分配_02真实GPU训练和重载完成

- 用户回传`nwpu_soft_smoke42_20260927_02`的exit0、status complete、1轮、360train/90val、seed42；原始附件逐字节归档，JSON提取后核对20个运行源码哈希与发布ed480e6及本地一致，dirty=false。完整soft配置、冻结清单/预训练SHA、batch32/workers2/FP32/TF32-off一致。[工程验收表](results/nwpu_soft_smoke_validation.md)与核对JSON已保存，未直接取服务器checkpoint文件。
- RTX2080 Ti、Python3.11.8/torch2.2.2/CUDA12.1；实际48空间+32特征边通过，31,486,760参数。best/final epoch均1，小样本val OA40%；保存重载差3.5762786865234375e-7（≤1e-5），checkpoint报告SHA41c595b7...939982c。该OA仅工程小样本数据，不能作为正式效果或与历史NWPU~96%比较。
- 总墙钟131.205秒（含预检和重建），CUDA峰值allocated3.6616 GiB/reserved4.0391 GiB，主进程RSS1.7564 GiB。固定8张val探针完成epoch1、累计1.590秒。纯train秒/吞吐/损失与epoch0/1诊断详情未返回，不能用总时间直接外推正式成本。
- 本次确认训练/验证/图边/保存重载链路通过；快照张量与分配专属CUDA梯度仍待核验，BigEarth96/64真实短测未执行。下一步只读已有history/probe摘要，不重训或修改配置。保留_01失败，正式结果表不填写这次smoke分数，PR保持草稿，未提交/推送新代码。

---

## 2026-09-28 — NWPU短测缺图失败后，复用输出目录被保护逻辑拒绝

- 用户回传：ZIP SHA256检查OK，`unzip -n -q`返回；随后再次启动`nwpu_soft_smoke42_20260927_01`，PID2703 exit1。追加launch日志同时包含`airplane/airplane_001.jpg`缺失和最新`FileExistsError`，不能把两段当成同一次调用。
- 发布源码`_run_output`先创建独占输出目录，再进入`_train/load_manifest`。最新调用在mkdir处失败，未重新读图或训练；此前缺图失败会保留目录、failed run.json和run.exit。解压命令成功返回不替代恢复后的文件存在检查，目前不能确认所有原图已恢复。
- 提供[路径核对与_02重试命令](docs/retry_nwpu_soft_smoke_02.md)：先只读核对固定manifest的31500个路径及新目录未占用，成功再使用同代码/清单/seed/配方跑相同1轮，输出和launch日志改为_02；保留_01，无需重新prepare或改模型。重试尚待用户执行，没有GPU通过或指标结果，未改代码、PR或历史成绩。

---

## 2026-09-28 — 软分配独立分支发布，草稿PR创建

- 用户选择选项2（推送独立分支并创建PR），随后确认继续。从`deb2b45`准备的隔离副本提交并推送`ed480e60fea03ed786fdba70c144de5e815c34f5`，分支`codex/soft-assignment-20260927`。[草稿PR#1](https://github.com/PatrickYxz/GAViT_Project/pull/1)已创建并附到当前任务，base为`codex/aid-entry-20260920`，未合并。
- 发布23文件：18个实现/测试文件加5个文档/证据文件；30个源码/测试哈希与133项通过的候选完全一致，未重复修改算法。提交前Git whitespace和文档Bash/zsh语法通过，GitHub返回的PR head与文件清单匹配。原研究工作树未提交材料和历史成绩未随本次发布混入。
- 下一步服务器从现有AID代码目录fetch新分支，再按上述完整SHA创建独立`GAViT_SOFT_20260927`目录，执行`nwpu_soft_smoke42_20260927_01`的360train/90val、1轮seed42验证。GPU、原图多worker、重载/探针及耗时显存须以返回证据为准；尚未启动或完成该短测，更未启动正式30轮。BigEarth96/64及S专属CUDA梯度检查仍待后续。固定版本命令见[短测说明](docs/run_soft_assignment_smoke.md)。

---

## 2026-09-27 — NWPU服务器版本、GPU和冻结文件身份确认

- 用户回传`codex/aid-entry-20260920`与origin对齐、`deb2b45`且无本地修改；RTX2080 Ti显存total22528/free22009 MiB为查询时值，不代表已测新模型峰值需求。
- 固定manifest `/home/featurize/work/nwpu_protocol_20260927_02.json`的SHA256为`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`；预训练文件SHA256为`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`，与已有ImageNet文件一致。[终端回传](results/evidence/nwpu_prepare_20260927/server_identity_return.txt)已归档，未直接连接服务器。
- 本地30个实现/测试文件与09-27最终验收SHA256完全相符。从服务器同版本`deb2b45`整理只含18个必要代码/测试差异的发布候选，避开原工作区其他未提交研究材料；不重新prepare、不改变训练协议。发布前CPU复测与独立版本交付仍在进行；尚未提交、推送或启动GPU。
- 本轮候选验收完成：[发布记录与补丁](results/soft_assignment_release_validation.md)。真正本地克隆的完整tests目录133项通过、44warnings、47.52秒、exit0；与原190项差别是未带入57项无关工具测试，无skip。首次源码导出缺Git身份及沙箱OpenMP的失败保留，改用本地克隆并获准沙箱外复测通过，未改生产代码。18文件补丁及30文件哈希已固定，diff/补丁检查通过；独立分支发布方式待确认，GPU短测尚未启动。

---

## 2026-09-27 — NWPU分组清单prepare成功，固定后续三模型划分

- 用户返回全量31500图扫描47.6秒及`MANIFEST COMPLETE`，train5040/val1260/test25200符合20%训练池、池内验证协议；按前条命令输出为`/home/featurize/work/nwpu_protocol_20260927_02.json`。保留seed42/4242和group策略，后续Swin/固定4×4/soft_prototype共用这一清单，不再重新抽划分。
- 8组同类像素重复、8个额外副本、31492个唯一像素图像，全部31500文件保留。机场6组、宫殿2组；组归属train2/val1/test5，均未跨划分。完整逐组回传摘录保存在[准备证据](results/evidence/nwpu_prepare_20260927/group_prepare_return.txt)。这是用户终端证据，不冒充本地读取原始manifest；manifest SHA256、服务器当前commit/dirty、GPU及预训练文件身份仍待只读返回。
- 数据准备阶段完成，尚无新GPU训练或准确率结果。下一步核对上述身份、将已通过本地验证的soft代码发布为独立固定版本，再做NWPU360train/90val的1轮短测。已更新执行说明与排期；未改模型代码、历史成绩、提交或推送Git，也未启动GPU任务。

---

## 2026-09-27 — NWPU完整扫描通过，首次prepare因同类重复终止

- 用户返回`IMAGE SCAN prepare: 31500/31500; elapsed=46.4s`，随后reject策略报告`airport/airport_050.jpg`与`airport/airport_420.jpg`像素重复。当前只显示首组，不能断言仅有这一组。错误发生在划分与manifest写入前，尚无成功`nwpu_protocol_20260927_01.json`，没有本轮训练或测试成绩。
- 核对现有数据代码：group按像素哈希把同类重复绑定为同组，保留所有文件，禁止跨split；不同标签的重复仍拒绝，无法维持精确份额时失败，不自动放宽数量。既有AID同类问题已有此通用实现，无需修改服务器代码或重新下载/解压。
- 下一次准备采用`--duplicate-policy group`、原seed42/4242、新manifest路径`nwpu_protocol_20260927_02.json`。协议明确记录为group-v2，三模型统一使用成功后的同一manifest，预期5040/1260/25200不变。真实重复组数/归属与prepare成功仍待回传；不混用旧历史NWPU协议或挑split求收益。
- 本次仅核对既有分组行为、运行相关本地回归并更新执行说明，未改模型/数据算法、提交/推送或启动GPU。回归证据在`results/evidence/nwpu_prepare_20260927/grouped_tests.log`。

---

## 2026-09-27 — NWPU原图下载、校验和数量检查回传成功

- 用户服务器wget完成`NWPU-RESISC45_a826b44.zip`：427389445/427389445字节，报告传输75秒；预定SHA256 `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`校验OK。已解压至`/home/featurize/data/NWPU-RESISC45`，断言45类、每类700张JPG、总31500张通过。持久化ZIP无需重下；证据为用户完整终端回传，尚非逐图解码完成。
- 下一步逐图RGB/256×256校验、文件/像素哈希及重复检查，按seed42/4242生成独立`nwpu_protocol_20260927_01.json`，预期5040/1260/25200；默认reject重复，出现异常先保留并分析，不删除图像或改seed。此时manifest尚未产生，不标记划分完成。
- 本地核对已发布`deb2b45`的prepare支持NWPU，其数据模块、包初始化和experiment_identity与当前版本逐字节一致；可复用服务器`GAViT_AID_20260920`的数据准备入口，目录名不代表只支持AID。仍需回传服务器commit和GPU型号，新soft训练代码未发布，未启动GPU、提交或推送。后续命令见[短测说明](docs/run_soft_assignment_smoke.md)。

---

## 2026-09-27 — NWPU下载来源核对与服务器准备

- 用户在平台仅找到NWPU-VHR-10/RSOD检测包，进一步搜索RESISC45无结果；不能替代45类场景分类数据。已准备[服务器直下载命令](docs/download_nwpu.md)，避免电脑中转和新工具包。
- 从TorchGeo官方源码确定原图ZIP固定revision `a826b44d938a883185f11ebe3d512d38b464312f`；原仓库地址已重定向至`isaaccorley/resisc45`。本地读取HF API确认大小427389445字节、SHA256 `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`；镜像HEAD返回200/ZIP/同大小。只验证来源与响应头，没有下载完整数据或确认用户服务器已成功。
- 命令将ZIP存入work，哈希一致才解压到data，并要求45类/31500图/每类700的输出。只取图片，不采用TorchGeo自带划分；新manifest仍按项目协议prepare。未启动GPU、提交或推送代码，真实数据验收仍待回传。

---

## 2026-09-26 — E05局部原型软分配实现与本地验收完成

- 用户批准按计划继续实施，在现有`codex/sparse-hybrid-4n-top2`隔离工作树完成，HEAD仍为`8f2ed07671ed50387998a46a2d2167a7a4dfc2ed`，未提交/推送；旧未提交材料及正式checkpoint/分数保留。源码前后SHA256、环境、测试完整输出及退出码见[本地验收记录](results/soft_assignment_local_validation.md)。
- 实现`soft_prototype`：复用旧区域注意力生成图像内原型，共享LN+768→128无bias投影、局部cosine/温度0.1、单次可微软分配；S行归一化，W列归一化，区域汇总和图反馈一致使用软归属。新增参数99,840；Swin、K16、48+32边构造和GAT保持。空间边仍为锚点先验，不声称语义纯净、真实接壤或像素级分割。
- 版本配置、旧权重路径、新配置缺失/冲突拒绝、scene/NWPU与BigEarth入口、校准/refit身份传递和固定探针已完成。探针每轮记录熵、区域质量、原型范数/相似度、相对初始S变化与反馈量级，只取train/val/pool且恢复RNG/模式，不触碰test；保留epoch0/best/final张量供后续图稿。
- 新建macOS/Python3.12.13 CPU环境；修改前81项基线通过。功能分阶段RED/GREEN后首次完整187项通过；独立审阅揭示中断发生在probe记录与命名快照选择之间会让续训exit0但best/final缺失。两种原入口中断均复现失败后补恢复校验，16项相关测试通过，最终完整**190 passed、44 warnings、exit0、47.20秒**。警告为Torch旧AST和合成TIFF无地理参考；无跳过项。测试使用真实Swin/PyG及合成图像，预训练下载与loader workers在测试工厂内替换，不能冒充真实数据或CUDA验证。
- 本地已验证公共参数/RNG不变、one-hot旧模型等价、仅S损失经分配路径反传、CE/BCE前后向、48+32图边、严格重载、诊断无副作用和受控续训；尚未验证服务器workers4/CUDA/完整真实校准→refit/资源或准确率。BigEarth旧续训不保存完整RNG和全split/源码身份的限制保留并披露，八图探针不替代完整实验身份。
- 下一步按[短测说明](docs/run_soft_assignment_smoke.md)确认NWPU目录、发布独立版本，NWPU每类8train/2val、BigEarth96train/64val各1轮seed42验证；通过后才推进NWPU三模型同协议校准与后续BigEarth正式消融。没有启动付费训练、重分旧数据或填入未测指标；HGTNet/DFAGCN仍等待作者材料。

---

## 2026-09-26 — 软分配设计确认，实现计划与当前工程基线

- 用户回复“ok”确认09-26单路线设计。已完成[实现计划](docs/superpowers/plans/2026-09-26-local-prototype-soft-assignment.md)，覆盖独立测试环境、版本化配置、可微分组、软反馈、无副作用探针、scene/BigEarth训练重载及服务器短测交付。推荐助手在当前任务顺序实现；当前计划待实施前审阅，不重复要求用户选择算法。
- 已用git-dir/common-dir确认继续使用现有codex/sparse-hybrid-4n-top2独立工作树，未另建worktree；保留既有未提交研究材料。已核对当前attentive_spatial、metadata解析、scene校准/refit和BigEarth入口，旧NWPU入口须明确拒绝新架构，避免只在CLI添加选项而重载丢配置。
- 新配置计划完整保存在architecture.soft_assignment中，旧架构不新增必填字段；新增投影用独立CPU RNG作用域，不改变公共层初值。分配学习通过仅依赖S的梯度测试验证，诊断须保存/恢复RNG与模型模式，refit不取val/test探针。以上是待实现要求，尚非通过证据。
- 本轮重跑19项experiment_identity与3项sparse_hybrid_static_contract，共22项通过、各exit0；[原始输出及环境探测](results/evidence/soft_assignment_plan_20260926/baseline.json)已保存。旧临时Python3.12.13环境目录仍在，但torch/torchvision/timm/PyG/numpy/pytest均缺失；没有沿用历史CPU测试通过记录，新Tensor测试需要先恢复独立环境。
- 当前只新增计划/证据并更新设计及排期；没有改模型/训练业务代码、安装依赖、启动GPU训练、Git提交或推送。旧正式指标保持不变；本地实现完成和双数据集GPU短测仍待后续验收。

---

## 2026-09-26 — 用户要求明确决策：收敛为局部区域原型软分配

- 用户表示对选型没有头绪，要求助手明确推荐路线。已形成[单路线设计](docs/superpowers/specs/2026-09-26-local-prototype-soft-assignment-design.md)，不再要求用户选择分组算法；旧草案标注被替代，保留历史。
- 推荐共享LN/无bias的768→128投影、单位化特征、旧区域注意力生成当前图片原型、局部候选内cosine/固定温度0.1、单次可微软分配。保留原重要性加权区域汇总及S软反馈；取消原90%/10%先验与编号MLP。新增参数静态99840，尚非运行测量或收益证据。
- 首批仅按导师三核心模型安排：Swin、固定4×4、软分配。先本地工程检查、真实小样本，再NWPU seed42校准；BigEarthNet长训及多seed后续按工程/研究结果和预算安排。取消首批额外冻结分配训练，不声称三模型足以隔离每种机制；导师双数据集多seed要求仍保留。
- 设计明确归一化、梯度、共同初始化、旧权重兼容、metadata与成本顺序；用户继续复制服务器命令。当前只有设计记录，未实现模型、启动训练、提交或推送；下一阶段为据此编排实现与测试。

---

## 2026-09-26 — E05 GitHub源码核对：借鉴局部关联，暂不冻结初稿分配器

- 按用户要求查询对应实现，跨09-25/26核对STViT、SSN、Slot Attention、GroupViT及PyG DiffPool共5仓库11文件，固定commit与路径存入[来源索引](results/evidence/soft_assignment_github_20260925/source_index.json)。[比较记录](docs/soft_assignment_github_review_20260926.md)区分作者实现、参考实现和库实现；未执行外部代码或安装依赖。
- STViT最接近汇总→区域交互→映射回token的数据流，但关联在no_grad中计算；值汇总仍可反传，不能说整个网络不学习，也不能直接称分配头可微。SSN距离/汇总CUDA有backward；Slot Attention支持按slot竞争后按输入归一化；GroupViT默认可用Gumbel和直通硬分配；DiffPool同时粗化图，不能保持现有稀疏边消融不变。
- 初稿0.9/0.1和输出区域编号的MLP属于本项目候选，并非已发表验证的配置。草案已补充状态：保留局部约束/S与W区分/软反馈，优先进一步细化当前图片区域原型相似度候选；尚不批准实现或启动双方案训练。若冻结投影但骨干仍更新，分配仍可变，不能冒称固定分配对照。
- SPFormer检索中的同名方法未混用；本轮未确认Mei等目标论文的作者代码，不声称不存在。已有模型、实验数值、训练协议保持，E02仍等待作者材料；本轮只更新设计依据和研究记录，未提交或推送Git。

---

## 2026-09-25 — E05软分配设计草案：局部空间先验、双向软映射与冻结对照

- 用户同意先进行下一步设计。[设计草案](docs/superpowers/specs/2026-09-25-soft-assignment-design.md)已形成，供审阅，尚未批准实现。比较局部空间约束、全局自由分配和迭代slots三条路线；推荐局部软分配，保留Swin/16区域/GAT/图预算，避免同时改变多个主要变量。
- 提议新增768→128→16分配头及LayerNorm，从原区域0.9、邻近区域合计0.1的固定先验开始学习；保留原标量注意力。S按区域归一化、W按token归一化，分别服务软反馈与区域汇总；候选外严格为0。48条固定边明确解释为空间锚点先验，不能冒充学习区域的真实接壤。
- 已核对反馈后立即mean pool，推导最终反馈等于区域质量加权和；不以分配热图声称存在后续token细化。新增分配头静态参数102032，batch32单个分配矩阵100352字节；标准库核算初始行和、候选覆盖和16区域正质量通过，均非运行时/GPU测量。
- 除导师要求的Swin/固定GAViT/可学习GAViT外，建议NWPU seed42增加独立训练的冻结软分配机制对照，区分学习与平滑作用，不默认扩成全多seed。候选只看train/val，配置冻结后才看test；旧协议、历史缺失与多seed范围明确披露，不据负面结果删seed。
- 设计列出归一化/梯度、one-hot兼容、共同层初始化、真实边、metadata重建、数值与失活诊断及双数据集短测要求。SSN/Slot Attention作为有出处的相关方法，未声称复现或软分配首创。E02继续等待材料；本轮仅设计和文档，不改模型、不启动训练、不提交或推送。

---

## 2026-09-25 — 用户已发送作者请求邮件；等待期间核对软分配接口

- 用户确认已发送邮件。E02状态更新为等待作者回复与材料；未独立核验收件人、发送正文或投递回执，尚无新源码/权重/配置回传。09-24草稿及检索证据保留为历史，不把当时的未联系状态追溯改写。
- 等待外部材料期间推进E05本地设计准备，E02仍是论文必做项。当前代码的attentive_spatial只学习固定区域内的汇总权重；区域归属仍为7×7 token到4×4区域的硬分配，图反馈按该归属gather。
- 软分配设计须同时明确区域汇总与反馈分配。现有sparse_hybrid空间边依赖区域编号的4×4位置，若区域归属自由学习，不能未经定义继续把编号相邻等同于实际区域相邻。这些是接口约束，不是当前收益不稳定的已证实原因，也不保证新模块提升。
- 已同步作者请求记录、接入记录、排期和矩阵。本轮仅只读代码核对及文档更新；未实现新模块、修改实验数值、启动GPU、发送消息或提交/推送Git。下一步形成可审阅设计，再安排与风险相称的本地验证和服务器短测。

---

## 2026-09-24 — 继续E02：排除同名仓库，落实作者材料请求与接入缺口

- 按用户“继续”及导师优先级先推进外部方法。读取当前日记、结果表、旧方法审计和Featurize规则；不重复已完成的AID三seed与BigEarthNet45，也未修改模型或启动GPU。
- GitHub连接器执行8条具体仓库查询并检查返回。HGTNet缩写返回灌溉制图、生物信息及无关文档仓库，均不符目标论文；DFAGCN缩写0结果，DOI尾号出现两个无关URL碰撞，README检索只得论文列表。作者范围查询422留作失败，不当作0结果。查询边界与候选README哈希保存至[证据](results/evidence/external_methods_20260924/search_record.json)，不声称代码不存在。
- 作者主页及两篇作者全文核对了公开通讯方式。HGTNet图聚合描述与预训练映射、DFAGCN完整特征/GCN/LIBLINEAR流程仍有关键配置缺口；不能为赶进度接入同名近似模型。IEEE页面直接读取失败，技术核对来源明确为作者全文。
- 完成[接入记录](docs/external_methods_readiness_20260924.md)与[两封可复制的材料请求](docs/external_methods_requests_20260924.md)。邮件仍为本地草稿，未联系任何作者。现有AID清单可复用，外部方法公共训练配方未冻结，NWPU正式清单与多标签适配范围仍待落实；没有把共享split等同于同训练协议。
- E02本轮检索与材料请求准备完成，正式实现/比较尚待材料；E05必做项保留，其设计不依赖作者回复。排期/矩阵同步，已有实验数值保持不变，未提交或推送Git。

---

## 2026-09-24 — 最新导师邮件原文归档，确认软分配为必做项

- 用户在当前任务提供最新英文邮件全文；已原文归档于[邮件与要求对照](docs/supervisor_latest_email_received_20260924.md)。邮件发送日期未提供，2026-09-24仅为归档日。此前基于早期邮件的“未回复/未找到最新原文”不再作为现行依据。
- 明确要求实现并评估learnable soft assignment，在NWPU与BigEarthNet至少比较Swin、固定4×4 GAViT、可学习分组GAViT；该模块尚未设计实现，不能用未来工作替代。当前收尾只针对固定网格实验批次，不代表论文整体完成。
- AID已完成两个核心模型三seed；HGTNet/DFAGCN外部方法比较仍未完成。最新邮件要求各数据集内统一训练/评估协议与多seed均值/SD，已修正旧矩阵将不同原方法配方作为主比较的表述；完整方法×数据集矩阵仍须落实，不静默删减。
- 邮件对前两seed一致提升的评价保留为历史反馈；后续四seed包含正负结果，主要mAP平均差约−0.23 pp，不能沿用早期稳定提升表述或据此更换主要指标。
- 排期按邮件顺序保留剩余必做项：外部方法比较、软分配双数据集消融、三主线相关工作；图稿已完成范围与3×3/4×4等尚需核验范围分开。此次只归档和校正项目文档，不实施新模块、不启动服务器训练、不改实验数值、不提交或推送Git。

---

## 2026-09-24 — 按用户要求收尾：记录收益不稳定的证据与解释边界

- 用户明确要求“先收尾”，本轮停止追加训练与诊断，整理[收尾说明](results/bigearth_closeout_interpretation.md)。已有主表、四seed统计和原始回传不变；未进行Git提交、推送或服务器操作。
- 区分总分来源与机制根因：seed45湿地AP差约−23.19 pp，经19类等权平均贡献约−1.22 pp，占净mAP差约86%；这只解释分数如何下降，不能据此判定图边或反馈导致湿地退步。
- 代码确认固定4×4区域、仅区域内注意力汇总、同区域共享直接残差反馈。语义混合、上下文干扰及与Swin已有关系建模的互补性不足作为假设记录；原图24例反馈纠正2/破坏2仅支持案例中利弊并存，不是总体指标或训练消融。
- val/test湿地与内陆水体共现比例8.5%/46.2%为泛化线索，同时保留seed44仅内陆组GAViT AP更高的反证，不写成“GAViT更依赖海洋”的确定原因。GAViT45测试已用best epoch10，不能将末轮验证下降直接当作测试退步原因。根因尚未由受控消融确立。
- 本轮结果与解释整理结束；E02/E05和E06完整数据链等独立事项仍未完成，本次不启动。没有为负面结果增加seed、改阈值或重选checkpoint。

---

## 2026-09-24 — BigEarthNet45完整验收，四seed mAP平均差−0.23 pp，核心配对收尾

- 用户回传总队列exit0、Swin/GAViT各smoke/train/test六阶段exit0及complete。双方正式训练均30轮，test119825张。原始粘贴首行缺Q、尾有shell提示符；按原字节归档，提取JSON另存，不冒充服务器summary.json原文件。[证据](results/evidence/bigearth_seed45_20260922_01/summary.json)、来源说明、可重跑核验脚本和summary_validation.json已保存。前次GAViT舍入日志仍保留，已核对与精确值相容。
- Swin45 Test mAP71.67692953551497%、macro-F1 65.08421202330264%、micro-F1 76.73226436237192%；GAViT45为70.25823624286052%/64.29547698363885%/76.70245685151849%。配对差−1.4186932926544529/−0.7887350396637913/−0.029807510853430585 pp。best val Swin78.6509%于epoch6、GAViT78.2951%于epoch10；双方final30。Swin末轮val与完整曲线未返回留空；GAViT末轮76.61513448708149%沿用此前同目录摘录，不补造历史。
- seed45 recipe、19类结构与80边metadata、预训练/三个split SHA均匹配冻结协议。clean deb2b45；launcher实际本地SHA8ac0083f...与回传一致；本地再次确认BigEarthNet入口/模型/utils/identity与a23a566无源码差异。参数27535501/31366926；smoke重载差0/4.76837e-7。RTX2080Ti UUID/软件、FP32无AMP、matmul TF32=false、cuDNN TF32=true/8线程按共同身份与匹配launcher的服务器检查记录；未把它视为AID效率TF32-off协议。
- Swin最佳权重SHA78cf048b6e2690e40789d13c78030c2350b238c899024a632347af7f0997ed8c，GAViT4268de448d7fabee4fde6db042241efbf38459953f6108337b95a3025313940c。各自metadata/训练摘要/test指标三处一致、与smoke权重不同；未本地收到checkpoint字节，哈希验证范围为报告链。原始各阶段run/history/predictions及独立图边捕获未本地收到，不声称重算F1或完成逐样本显著性检验。
- 19类AP均值各自精确重现mAP，8类升、11类降。海岸湿地33.854200825581785→10.66842766288083%，差−23.185773162700954 pp，对总mAP贡献−1.2203038506684713 pp，约占净下降86.02%；其余18类合计贡献约−0.1984 pp。是事后算术分解，不排除任何类，不据此确定图反馈因果或按test调模型。[逐类表](results/bigearth_seed45_per_class_comparison.csv)。
- Swin/GAViT训练测量阶段45857.68239937/53606.540994166秒（12.7382/14.8907小时）；测试270.560692114/407.965277058秒；含两个smoke共27.826336658464164小时。按此前参考价¥0.69/h条件换算约¥19.20，非实际账单；不含计时前导入/身份检查、全量预检、解压、队列间隔和闲置。训练allocated峰值3.556476/3.653889GiB，范围含保存状态核验；RSS不含worker，不能当全作业内存或标准推理性能。
- 全部42/43/44/45保留，主要mAP两正两负。四seed描述性均值±样本SD：Swin71.269232±0.340729%、GAViT71.039559±0.944096%，配对差−0.229673±1.266451 pp。macro-F1均值64.771053/65.298869%，micro-F1 76.508066/76.825614%；次要指标均值正向不替代主要指标结论。历史42—44仅一位小数、45精确；跨seed硬件不同，GAViT43/44历史停止21/26、Swin44最终退出未确认，不能当纯随机seed方差或显著性证据。
- [seed45完整表](results/bigearth_seed45_comparison.md)、[四seed主表](results/bigearth_four_seed_summary.md)/JSON、bigearth_comparison.csv、bigearth_multiseed_summary.csv、排期与矩阵已同步。CSV旧实验行原字节保留；旧三seed汇总另存快照。AID与NWPU结果未修改。E04配对测试与汇总完成，未将历史不完整训练改写为30轮，也不为负向45追加重跑。此次仅本地归档和统计，无新GPU训练、发布代码修改、Git提交或推送。

---

## 2026-09-23 — BigEarthNet45队列完成标志回传；GAViT mAP70.3%，Swin及退出摘要待补

- 用户回传同一目录bigearth_seed45_20260922_01的尾日志，部分重复粘贴，按一次运行归档。实际设备仍RTX2080Ti，未切4090。GAViT第30/30轮训练7434/7434、STAGE COMPLETE gavit_train，测试119825张/3745批到100%、STAGE COMPLETE gavit_test及BIGEARTH SEED45 COMPLETE均可见。按本地队列源码，末尾标志须六阶段退出及共同环境检查通过后打印；当前未直接取得服务器run.exit及脚本SHA，不将间接标志写成已收到退出文件。
- GAViT45 Test mAP70.3%、macro-F1 64.3%、micro-F1 76.7%，最佳val mAP78.3%均为终端舍入值；最佳epoch及精确分数待run.json/metrics.json。最后一轮loss0.08172658044316794、val mAP76.61513448708149%、macro-F1 71.71295447541755%、next_epoch_lr0。19类AP已录入证据；海岸湿地10.7%、沙滩9.7%、自然草地47.6%，不据此单独断定图反馈因果或按test调参。
- 日志明确torch2.2.2/torchvision0.17.2/timm1.0.27/PyG2.8.0/numpy1.26.4/sklearn1.9.0/Pillow10.3.0/Python3.11.8/CUDA12.1/cuDNN8902/8线程、RTX2080Ti及GPU UUID，FP32、无AMP、matmul TF32 false、cuDNN TF32 true，与BigEarthNet既定默认行为一致，不与AID效率FP32/TF32-off混同。
- 第30轮训练24:21，整轮含验证/保存1824.7612908370065秒（30.41分钟）；test循环05:11，不是完整命令墙钟。总训练/各阶段耗时、峰值资源与实际费用尚待报告，不能将30倍单轮时间当实测总时长。
- [阶段结果表](results/bigearth_seed45_partial.md)、[规范化摘录](results/evidence/bigearth_seed45_20260922_01/reported_excerpt.txt)及结构化回传指标已落地。尚无Swin45分数、双方精确best/权重身份和退出文件；不能给seed45配对差值或更新四seed均值。已有三seed统计及AID保持原数值，不因这次70.3%重跑。下一步仅以一行只读命令汇总持久化JSON与退出记录，无GPU计算、数据解压或新训练。

---

## 2026-09-22 — 用户优先选择BigEarthNet seed45；配对队列准备完成，服务器结果待回传

- 研究问题：执行既定E04第四个训练seed，检验冻结模型的配对收益波动，结果进入BigEarthNet多seed表。用户明确选择先seed45，并确认本实例BigEarthNet-S2已完整解压；本地尚未验证服务器文件。保留42/43/44全部正负结果及历史中断/硬件边界，未调整AID结果或模型。现有checkpoint分析不能替代新的独立训练随机性，因此安排Swin/GAViT各30轮。
- 代码复用服务器干净deb2b45发布目录GAViT_AID_20260920。本地git diff证明其BigEarthNet train/test、models、utils、experiment_identity与seed44 a23a566字节一致；无需切分支、更新仓库或上传工具包。独立外部包装保存精确epoch记录/资源/逐样本预测，不改原训练计算、保存返回值或默认精度。单行启动及完整命令、数据/架构/优化协议见[seed45执行说明](docs/run_bigearth_seed45.md)，源码见tools/run_bigearth_seed45.py。
- 唯一主要变量为seed45；固定原237871/122342/119825及回传证据中的三个CSV SHA、预训练SHA f7e9953f...、RGB TIF/224/原增强、batch32/AdamW lr3e-4 wd1e-4/Cosine30/BCE/阈值0.5。GAViT沿用16区、attentive_spatial、4n+top2 sparse_hybrid、GAT256/4头/2层、token_feedback；train pin_memory=True、val/test False、workers4。保留BigEarthNet FP32无AMP默认，记录配对TF32/线程状态，不擅自套AID配置。
- 新持久化输出/home/featurize/work/bigearth_seed45_20260922_01。预检全split RGB存在、冻结SHA、标签/路径无重复/无merged覆盖，先两个各1轮96train/64val smoke，覆盖原图解码、训练/验证/保存及重载；随后Swin30→best test→GAViT30→best test，GPU空闲检查、独立进程串行、失败停止、拒绝覆盖、锁阻止重复队列。不会把文件存在检查当全量解码验证。所有阶段与总队列必须有exit0及完成证据。
- 逐轮精确loss/val mAP/F1和epoch墙钟持续落盘，首轮含初始化；最佳轮对照history核验，末轮state必须epoch30；测试保存119825×19概率/标签/路径及精确指标。阶段记录参数量、GPU UUID/软件/源码commit、CUDA峰值、主进程RSS及墙钟；RSS不含loader子进程，费用和目标卡真实速度待首轮回传，不用AID分钟级时间估算此任务。
- 本地先见新增测试因实现缺失失败，后4项通过：冻结split/缺图/重复/merged拦截、第二阶段exit7立即停止与目录拒覆、正式recipe及best测试路径、CPU小模型下保存回调返回值/精确指标/重载一致。该CPU替身验证包装，不代表真实GAViT或CUDA验收。bash/zsh语法及实际后台bootstrap模拟缺源exit1、无正式启动、二次启动拒覆通过。[验证证据](results/evidence/bigearth_seed45_launch_20260922/validation.json)。未改已发布训练源码、未提交或推送，未连接服务器或启动付费训练；GPU smoke/正式结果待用户回传。

---

## 2026-09-22 — E06模型前向结果回传：单张延迟+34.41%，batch32吞吐−15.31%

- 用户回传`aid_efficiency_20260922_01/run.exit`和`summary.json`的终端输出：exit0、status complete，Swin/GAViT各batch1/32四阶段退出码均0，总wall time88.260855秒。按摘要验收完成，不把早先目录缺失推定为历史运行中断，也不重复当前测量。原文和提取JSON已存[证据目录](results/evidence/aid_efficiency_20260922_01/summary.json)。
- 研究问题和协议保持预先固定：已有seed42最终refit权重，同一RTX2080Ti、FP32、8线程、无AMP/TF32、224×224固定合成RGB归一化输入预置GPU，独立进程串行。每项预热20次后5×50次同步主机计时，含完整分组/图构建/GAT/反馈；排除原图读取/处理/传输、模型加载和指标计算。训练/选轮/准确率评估均未重跑。
- Swin/GAViT batch1平均延迟14.891226/20.015683 ms，增加5.124457 ms（+34.4126%），吞吐67.153638/49.960823图/秒（−25.6022%）。batch32平均批次耗时56.648042/66.884989 ms（+18.0711%），吞吐564.891547/478.433210图/秒（−15.3053%）。全部预定组保留；Swin b1第一组较慢，未据结果删组或改变统计口径。
- batch1峰值allocated133.267578/147.755371 MiB（+10.8712%），reserved158/180 MiB；batch32 allocated594.482422/609.353027 MiB（+2.5014%，+14.870605 MiB），reserved942/962 MiB。范围为预热后的模型/驻留输入/当前输出与临时张量/分配缓存，不是整卡显存，不与训练含checkpoint重建峰值直接对比。参数27543960/31375385，增加3831425（+13.9102%）。
- 两个checkpoint SHA与seed42原test一致，manifest/预训练及probe SHA匹配冻结值，报告Git deb2b45 clean；17源码哈希逐一匹配本地发布工作树。报告硬件/软件与原seed42同型号同版本，驱动610.57.04及GPU UUID按本次回传记录，不推断训练与测量为同一物理卡。GAViT报告真实48空间+32特征边检查通过。
- 本地重算4×5组均值、样本SD、min/max、吞吐、摊销耗时及相对开销均一致，总前向计时约39.610秒不超过88.261秒wall time。原始250次计时、逐阶段input hash/独立身份/遥测、完整run.log与checkpoint字节未本地收到，因此中位数/P95按回传引用、不声称独立重算或已排除温漂；脚本成功汇总前的配对/身份/退出检查与摘要来源分开说明。
- 结合固定split三训练seed的OA均值差+0.0467 pp、两正一负，当前固定4×4区域图实现未显示足以支撑额外计算开销的稳定整体收益；这不证明所有图方法无效或某个子模块的耗时因果。[效率表与论文表述](results/aid_efficiency_summary.md)、[三seed主表](results/aid_multiseed_summary.md)、JSON、排期/矩阵及执行说明已同步。原准确率/配对/逐类统计保持不变。
- 状态：E06模型前向测量的回传和摘要核验完成；完整数据读取/处理/传输链仍待测，E02/E05等独立研究项仍待办。此次本地仅归档和统计，无正式代码改动、新训练、Git提交或推送。

---

## 2026-09-22 — E06已有AID权重的模型效率入口准备，本地验证通过，CUDA结果待回传

- 恢复检查回传：`OUTPUT_EXISTS: False`、空文件列表、`NO_RUN_LOG`，`pgrep`未列出benchmark进程。因此当前指定输出目录未建立，也无匹配运行进程；不能反推此前一定启动过或被重启中断。直接提供既有、重新核对解码SHA及bash/zsh语法的完整启动单行命令，创建尚不存在的`_01`目录；不改源码、身份或测量协议，不解压数据。是否首个CUDA前向正常及最终完成继续等日志回传。
- 后续用户说明已重启实例，读取`aid_efficiency_20260922_01/run.exit`与`summary.json`均提示不存在。目前只能确认预期完成记录不可用；尚未取得目录、启动日志或进程证据，不能判断启动命令未执行、运行被重启中断或其他失败，也不认定持久化文件丢失。下一步只读目录、run.log及相关进程；此次model-only测量不需解压原图，不自动重复已有任务或覆盖结果。
- 用户在三seed与限定曲线/混淆诊断收尾后确认继续E06。研究问题：当前OA均值收益+0.0467 pp、参数增加3831425（+13.91%）时，实际模型延迟/吞吐/显存的代价是多少？预先固定第一个seed42配对refit权重，不依据分数或速度挑seed。复用已有checkpoint即可回答，无新训练、原图解压或准确率评估。
- 新增独立 `tools/benchmark_aid.py`，复制文本方式写入服务器新的 `/home/featurize/work/aid_efficiency_20260922_01`。正式训练仓库仍要求干净deb2b45；先核对源码、manifest、预训练来源、refit身份、run.exit及checkpoint实际SHA。输入是同batch逐字节一致的固定合成RGB归一化张量，预先驻留GPU；计入完整Swin/GAViT含图构建与反馈，排除数据加载/处理/传输与权重加载。完整数据处理链耗时仍待测，不把本次model-only结果当数据集吞吐。
- 固定FP32、无AMP/TF32、eval/inference_mode、8 CPU线程；Swin b1/GAViT b1/GAViT b32/Swin b32四项串行独立进程。每项真实前向/图输入预检后移除hook，清除加载缓存，预热20次，测5组×50次，每次CUDA同步后停表；保存原始耗时、均值/中位数/P95、重复均值SD、GPU身份/温度/时钟/功耗，以及预热后allocated/reserved峰值。竞争GPU进程快照检查/同任务文件锁、失败即停及输出不可覆盖。
- 本地先见5项新测试失败，再实现并通过；CPU随机权重两模型真实前向/重载logit差均0，参数正确，GAViT捕获48+32边；合成身份通过而source失败、权重损坏、manifest变更被拒。单行启动在bash/zsh通过语法和实际隔离运行检查，GPU占用模拟触发exit1/failure，无成功摘要，重复执行不覆盖。详情与证据见[本地验证](results/aid_efficiency_local_validation.md)。这些不是正式权重GPU验证或性能测量。
- [执行说明及一行启动/回传命令](docs/run_aid_efficiency.md)已准备，不要求上传工具包、Git同步或修改正式训练代码。发布工作树仍干净，未提交/推送。服务器由用户复制命令；当前没有访问服务器或新效率数值，必须看到4阶段完整结果、AID EFFICIENCY COMPLETE和exit0后验收。三seed准确率表保持不变；E06完整处理链、E02/E05等仍待办。

---

## 2026-09-22 — AID曲线与重点混淆核验收尾，未发现选轮或LR协议错误

- 用户回传seed44四阶段共106条epoch/loss/lr/val OA四元组，以及三seed两模型各Commercial/RailwayStation两条真实标签行，共12条混淆行。原文与去尾部shell提示符的JSON、来源范围、核验和派生分析已保存在[证据目录](results/evidence/aid_seed44_curve_confusion_20260922/summary.json)。不是完整history或完整混淆矩阵，没有原始逐样本预测或新的前向推理。
- 30/30/25/21轮均连续完整，与前次run摘要一致；loss有限非负，LR有限为正。全部LR符合lr0.0003、T_max30、epoch−1的余弦公式，最大绝对误差8.13e-20。Swin校准第25/26轮并列最高97.0%，选最早25；GAViT第21/22/26轮并列最高97.0%，选21。refit25/21与各自预算一致，GAViT21轮结束不是训练中断，未因为较短预算更改余弦周期。
- 校准末轮val OA为Swin96.6%、GAViT96.7%，较各自最佳下降0.4/0.3 pp；末轮loss约0.000822/0.000875。refit末轮loss约0.001007/0.002999，GAViT第9→10轮0.116852→0.197366、第13→14轮0.028954→0.064490存在反弹但随后下降。没有发现曲线中的明显不收敛或协议错误；没有逐batch/梯度信息，不能单凭epoch平均loss排除所有问题或定性为图反馈故障、欠拟合或过拟合。
- 每条混淆行的支持数175/130、对角线正确数与对应seed逐类准确率一致。seed44 Commercial正确171→150，主要误分School1→8（+7）、StorageTanks0→6、Church/Resort各0→3；RailwayStation正确126→116，主要误分Industrial1→10（+9）。两类净多31次错误，其余28类净多7次，重现全局−38正确。这里只是边际计数变化，不能断言对应相同图片的具体预测转移。
- Commercial→School三个seed的Swin/GAViT计数分别22/4、2/2、1/8；RailwayStation→Industrial分别3/0、1/0、1/10，纠错方向随seed变化。这提供事后错误讨论，不能证明图边、区域划分或背景依赖的因果。三seed总体指标保持OA平均差+0.0467 pp/F1+0.0151 pp，当前固定区域图版本未显示稳定整体收益；保留负面seed44，不按test追加轮数或挑seed。
- [完整限定范围分析](results/aid_seed44_curve_confusion_analysis.md)、seed44阶段表、三seed主表、排期/矩阵和已完成的只读说明同步更新。本轮没有正式代码修改、重训、重新解压、GPU推理或Git提交。四列曲线和两类行已取得，完整history字段/矩阵仍未本地取得，区分已收尾检查与资料完整度。
- 下一步E06复用已有匹配checkpoint，在相同GPU/输入/batch/精度、预热和同步条件测推理延迟、吞吐和峰值显存，并明确数据读取/传输计时边界。已知参数增加3831425（+13.91%），实际推理代价待测；当前没有编写/运行计时任务，不将训练wall time当推理效率。E02/E05等独立研究项保持待办，不直接用本次测试类别来选择候选。

---

## 2026-09-22 — AID三seed完成：seed44负向，整体均值近零且未显示稳定收益

- 用户回传seed44六阶段摘要，运行目录标签20260921，均complete/exit0。Swin/GAViT均校准30轮、最佳val OA97.0%，各自best25/21轮用于5000pool refit；test均5000。当前回传日期不作为服务器实际开始/结束时间。来源best哈希、最终last哈希与测试报告一致。
- seed44 test Swin OA96.84%、macro-F1 96.66111935641514%；GAViT OA96.08%、macro-F1 95.87462595609628%。差−0.76 pp/−0.78649340031886 pp，正确4842→4804。配对4744两者都对、60纠正、98新增错误、98两者都错，净−38；本地核对四格总和、双方正确数、30类整数正确数加权OA及平均F1均一致。
- 类内准确率8升/8平/14降，F1为8/1/21。Commercial净−21、RailwayStation−10，合计−31，其余28类净−7；Park−5，DenseResidential/Resort各+5。三seed Commercial为+23/+3/−21，RailwayStation+2/+3/−10，Park+7/−1/−5。不能从类表推断具体误分方向或图模块因果。
- 与42/43共同代码deb2b45干净、17源码哈希、RTX2080Ti及软件身份相同，训练配方除seed外完全一致。显式architecture与seed42和本次calibrate/refit相同，manifest/预训练一致。GAViT实际48空间+32特征边；Swin重载差0，GAViT2.384e-7/1.907e-6均小于1e-5。当前摘要没有显示协议错配或重载失败，但尚未读取原始训练曲线，不把低分直接归因于代码bug或过拟合。
- 四训练阶段1372.88/1344.11/1143.45/1035.15秒，合计81.59分钟；参数27543960/31375385、精确内存/权重路径/哈希记录于[seed44表](results/aid_seed44_comparison.md)。不含测试/解压/等待，内存范围含重建；逐轮吞吐、推理效率与费用未取得，不推算。
- 已保留[原始粘贴](results/evidence/aid_seed44_batch_20260922/reported_output.txt)，规范化仅提取JSON并去掉开头eaturize/work残片和末尾shell提示符；摘要、来源边界及核验JSON同目录。未下载原始run/config/history、完整预测/混淆矩阵或权重，未把规范化摘要充当原文件字节副本。checkpoint SHA核对的是报告链，未本地重哈希权重。
- 预定42/43/44全部保留并汇总：Swin OA96.6800±0.2946%、GAViT96.7267±0.5707%；macro-F1 96.5221±0.2611%和96.5372±0.5804%，均为n3样本SD（ddof1）。配对OA差+0.046667±0.714516 pp，F1差+0.015139±0.709070 pp。两正一负，当前固定4×4图版本未显示稳定整体增益；不宣称已证明等效或统计显著，不排除seed44、不因负面结果重跑或新增挑选seed。+30/+15/−38跨运行合计+7次判断，不当成15000独立图片或7张唯一图片。
- [三seed主表/JSON](results/aid_multiseed_summary.md)、[跨seed类别表](results/aid_multiseed_class_stability.md)、排期与矩阵已更新，E03预定重复指标汇总完成。GAViT额外3831425参数（+13.91%），近零均值增益未形成清晰收益；标准化推理速度和内存尚未测，不把训练wall time冒充效率。
- 下一步优先只读已有history与混淆矩阵：核对44四训练阶段最早最佳轮/曲线，并读取三seed两模型Commercial/RailwayStation真实标签行，补具体错误讨论。单行命令见[只读检查](docs/analyze_aid_three_seeds.md)，不需要原图、解压、GPU或上传工具包，不用于按test改模型。命令已在合成数据上验证语法、四曲线/六测试输出、矩阵方向与最早同分规则及错误选轮拒绝；真实曲线/混淆仍待回传。之后E06复用已有checkpoint；E02/E05等独立事项仍保留，未启动新训练、修改发布代码或Git。

---

## 2026-09-21 — AID seed43六阶段完成：OA+0.30 pp，准备既定seed44重复

- 用户回传统一摘要：Swin43 `_02`校准及其余五阶段均complete/exit0。Swin校准best25、val OA97.0%，GAViT best29、val OA97.3%；均最终30轮。两次refit在5000训练池分别执行25/29轮，来源best checkpoint哈希匹配，最终test均5000张并匹配自身last权重。校准末轮val与逐轮history未回传，不补填。
- Swin test OA96.86%、macro-F1 96.68420610630325%；GAViT OA97.16%、macro-F1 96.95587595593263%。差值+0.30 pp/+0.27166984962938 pp，正确4843→4858。配对四格4796两者都对、62纠正、47新增错误、95两者都错，净+15；服务器汇总已按5000唯一图片路径和同标签匹配，本地重算总和与双方正确数一致。
- 两份30类support合计5000，与seed42类映射/支持数一致；按逐类整数正确数和F1平均重现OA/macro-F1。类内准确率12升/10平/8降，F1为16/2/12。Commercial净+23→+3、Park+7→−1；Bridge/Center本次各+4，Square−4、Stadium−3。两次类别贡献有波动，不据test改配置，也不推断混淆去向或图反馈因果。
- 训练配方除seed42→43外完全一致；共同硬件RTX2080Ti、软件版本和代码身份相同，干净deb2b45及17个源码哈希与本地发布工作树逐一相符。manifest、预训练和参数量匹配冻结协议；GAViT两训练阶段真实前向48+32边，重载最大差7.1526e-7/3.5763e-7，Swin均0。
- 四训练阶段1259.43/1328.66/1025.11/1229.32秒，合计80.71分钟，不含测试/等待/解压。校准与refit allocated约Swin3.5582GiB、GAViT3.6549/3.6556GiB；报告范围含重建，不能作为标准推理显存/速度。精确资源、权重路径/哈希见[seed43阶段表](results/aid_seed43_comparison.md)，完整命令沿用[既定队列](docs/run_aid_seed43_batch.md)。逐轮吞吐与费用待补。
- [原始粘贴及摘要](results/evidence/aid_seed43_batch_20260921/summary.json)、来源说明和核验结果已归档；保留尾部shell提示符的原文副本，提取JSON仅去该尾部。未取得原始run.json/config/history、完整预测/混淆矩阵或checkpoint文件，未把阶段摘要伪装成服务器原文件；checkpoint SHA仅核对报告链，未本地重算权重。
- [多seed阶段表](results/aid_multiseed_summary.md)及JSON已建立：42/43均正向，固定划分n=2阶段OA均值Swin96.60%、GAViT97.05%，均值差+0.45 pp；样本SD约0.3677/0.1556 pp。不能合并成10000独立test或声称稳定/显著结论。排期、矩阵同步为42/43完成、44待运行。
- 用户随后要求继续。准备[seed44六步顺序命令](docs/run_aid_seed44_batch.md)，范围为既定两模型各校准/refit/test，唯一主要变量训练seed44；数据、模型、配方、选轮规则冻结，结果进入三seed均值±样本SD。更便宜的现有权重分析不能回答第三次独立训练随机性；不重复smoke或42/43。输出`aid_{model}_{phase}44_20260921_01`，以各自44校准最佳轮refit，失败停止，最后统一回传。
- 本地已验证新命令bash/zsh语法、模型/seed/来源路径、真实printf→nohup bash包装下六个模拟阶段串行、第三阶段失败停止、缺图拦截及5000行乱序配对汇总；失败报告被拒绝。无真实GPU运行、服务器连接、代码修改或Git提交/发布。seed44尚待用户执行和回传。

---

## 2026-09-21 — 按用户要求准备seed43剩余五步连续队列

- 用户明确要求后续命令一起提供，连续运行后统一回传。保持已冻结研究问题：只改变训练seed检验固定划分下的重复性，不依据seed42 test调参。替代此前逐阶段人工回传的执行安排，不增加seed44或其他候选。
- [完整队列与汇总命令](docs/run_aid_seed43_batch.md)先等待Swin校准`aid_swin_calibrate43_20260921_02/run.exit`（最多120分钟），要求exit0，再串行执行GAViT43校准、Swin43 refit、GAViT43 refit、Swin43 test、GAViT43 test。后续输出均为`aid_{model}_{phase}43_20260921_01`，总日志`/home/featurize/work/aid43_followup_20260921_01.launch.log`。不重启当前Swin、不覆盖失败目录，bash -e遇失败停止；每个Python进程的stdin为/dev/null。
- 沿用干净deb2b45、冻结manifest/预训练和配方。refit自动读取各自seed43校准最佳轮，Cosine T_max仍30，最终测试各自last.pth。现有入口继续校验数据、来源、权重与保存重载；完成后以AID43_COMPLETE和六阶段exit0/complete验收。用户复制命令执行，未改发布代码或Git身份。
- 只读汇总输出共享训练/代码/硬件身份、六阶段最佳轮/最终轮/权重/资源、两份5000test OA/F1/逐类指标及路径配对计数。核查5000唯一同路径同标签、报告OA与逐样本正确数一致，避免再要求上传工具包或原图。
- 本地验证：四条shell命令均通过bash/zsh语法检查；汇总Python编译通过；模拟五任务严格串行、第二任务失败停止、源Swin失败/超时均不启动任务；合成5000行逆序配对覆盖四种正确性结果，失败阶段报告被拒绝。这些是命令验证，不是真实GPU结果。当前Swin `_02`及后续队列启动/完成均待回传，未新增正式指标。

---

## 2026-09-21 — Swin43首次预检缺图退出；用户恢复ZIP，准备新目录重试

- 用户启动`aid_swin_calibrate43_20260921_01`，shell PID2916明确exit1；traceback在load_manifest→_image_identity报告`/home/featurize/data/AID/AID/Airport/airport_1.jpg`不存在。源码确认发生在数据预检，早于模型构建和训练，不计入已完成的seed43实验。原始run.exit/run.json未回传，不编造服务器报告；[回传错误](results/evidence/aid_swin_calibrate43_20260921_01/reported_error.txt)与[诊断摘要](results/evidence/aid_swin_calibrate43_20260921_01/failure_summary.json)已保存。
- 用户询问是否因重启实例；实例本地data缺图与该解释一致，但未独立确认发生了何种重启/实例重建，也未从一个缺失文件推定全目录已清空。后续用户执行`unzip -n -q /home/featurize/work/AID_visionflan_20260920.zip -d /home/featurize/data`并返回提示符，无显示错误。恢复后显式退出码和全量哈希尚未回传，仍由训练入口验证10000张原图与冻结manifest。
- 保留失败`_01`目录，当前重试改为`aid_swin_calibrate43_20260921_02`；同seed43、manifest、预训练、30轮校准和全部配置，不重新prepare或改变划分，也不重新运行seed42。入口拒绝已有输出目录，launch日志仍追加。三条启动/跟踪/验收命令已更新于[执行说明](docs/run_aid_seed43.md)，通过bash/zsh语法检查。
- 当前只准备重试命令，未代为启动服务器任务或确认原图已全部恢复。须看到完整图片校验、实际GPU、TRAIN FIRST BATCH OK和EPOCH1/30才能记录成功启动；新实例实际GPU/运行环境仍待回传。未修改训练代码或发布Git身份，保持deb2b45干净。

---

## 2026-09-21 — AID seed42逐类分析收尾：纠正93/新增错误63，准备固定协议seed43

- 用户回传按路径匹配5000张的统计：两者都对4754、Swin错/GAViT对93、Swin对/GAViT错63、两者都错90，净多正确30。匹配命令已检查唯一完整路径、相同标签和正确总数；本地核对四项和5000及4817/4847正确数一致。原始预测未直接下载，不能把156个正确性变化样本当作全部预测不一致样本。
- 两份30类表的类名/索引/support一致，support合计5000；从类内准确率×support还原整数正确数、按样本加权重算OA96.34%/96.94%、平均类F1重现96.22088514124282%/96.7811271561119%。准确率17类提高/7类下降/6类持平；类F1为21/7/2。
- Commercial正确145→168（+23，准确率82.8571%→96.0%），Park156→163（+7，89.1429%→93.1429%），其余28类净变化合计0；并非只有这两类有变化。主要正确数下降Industrial−6、Resort−5、Square−3。Industrial与School准确率下降但F1上升，完整混淆方向未知，不由类表推断错误去向或图模块因果。
- [回传表与计数](results/evidence/aid_seed42_offline_20260921/reported_per_class.txt)、规范化JSON与来源说明已归档；[30类分析报告](results/aid_seed42_per_class_analysis.md)、[分析JSON](results/aid_seed42_per_class_analysis.json)、阶段表、汇总、排期与矩阵同步。E01的单seed流程及所需逐类/配对分析收尾；完整逐样本CSV与混淆矩阵仍由服务器保存，不声称已本地备份。
- 下一步E03按既定42/43/44路线测训练重复性，模型为E01前已冻结的Swin与固定4×4 GAViT，不根据本次test换配置。仅主要变量train_seed42→43，外层split_seed42/val_seed4242及manifest、预训练、架构、优化和选轮规则保持。已有smoke/完整流程/离线统计无法回答训练seed方差，因此新的预定重复有独立研究价值；不重复seed42或新增调参。
- seed43仍先各自30轮4000/1000校准，再按各自本次最佳轮5000pool重训/5000test，不复用seed42的21/20轮。正式身份Git deb2b45干净，命令、研究问题、完整配置和成本参照见[seed43执行说明](docs/run_aid_seed43.md)。当前仅准备Swin43新输出`aid_swin_calibrate43_20260921_01`供用户复制执行，未代为启动或自动排队其他运行；保留43/44的所有结果，不挑seed。多seed稳定性尚未建立。

---

## 2026-09-21 — AID seed42配对test完成：OA+0.60 pp、macro-F1+0.560242 pp

- 用户回传GAViT完整数据扫描10000/10000（113.2秒）、SCENE EVALUATION COMPLETE、run.json和指标摘要，并明确双方exit_code0。GAViT 5000test OA96.94%、macro-F1 96.7811271561119%；Swin此前OA96.34%、macro-F1 96.22088514124282%的显式退出码现已补齐。两模型报告complete，refit/train_plus_val。
- GAViT checkpoint SHA `a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`与refit匹配；与Swin评估的manifest、17个源码哈希、Git deb2b45/dirty=false、RTX2080Ti软硬件环境、batch32/workers2/float32/TF32关闭/pin_memory=false一致。argv仅权重来源及输出不同。扫描113.2秒不作为模型推理耗时。
- test差值OA+0.60 pp、macro-F1+0.56024201486908 pp；由精确OA和5000样本推得正确4817→4847、错误183→153，净多正确30张。尚未读取逐样本预测，不把净差当成仅纠正30例，也不声称统计显著或跨seed稳定优于baseline。校准val的+0.8 pp与最终test分开保留。
- [GAViT报告与摘要](results/evidence/aid_gavit_test42_20260921_01/run.json)、退出码、回传末段及来源说明已归档；Swin run.exit和provenance同步补齐。所有聊天JSON为规范化副本，未声称服务器字节副本。已更新[正式阶段表](results/aid_seed42_comparison.md)、[数值汇总](results/aid_seed42_test_summary.json)、排期及矩阵；完整metrics/per_class/predictions仍留服务器、尚未本地取得。
- 下一步复用持久化predictions.csv/per_class.csv，按图片路径配对并检查5000唯一条目、相同路径/标签和4817/4847正确总数，再取得纠正/新增错误数、逐类准确率/F1和支持数。目的是补齐类别分析与失败讨论，不改变模型、不重复推理、不使用test重新选择。用户复制单行只读命令，见[离线分析说明](docs/analyze_aid_seed42.md)；无需原图、GPU或工具包上传。
- 单行统计在本地合成CSV检查四种结果及乱序匹配、缺行/重复路径/标签不一致拒绝；这只验证统计命令，不充当真实服务器分析结果。真实计数仍待回传。当前不自动排队新seed或完整训练；E02外部方法、E03多seed、E05可学习分组及E06标准效率测量仍按矩阵保留。

---

## 2026-09-21 — Swin AID test OA96.34%、macro-F1 96.220885%，继续GAViT配对测试

- 用户回传Swin test完整run.json和指标摘要：5000样本、split=test、dataset=AID、status complete，OA96.34%、macro-F1 96.22088514124282%，training_phase=refit、fit_scope=train_plus_val。对应21轮5000训练池的last.pth，不是校准best。
- checkpoint SHA `fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018`与已验收refit一致，manifest SHA一致；17个源码哈希、Git deb2b45/dirty=false、RTX2080Ti软硬件环境与refit一致。评估batch32/workers2、float32、TF32关闭、pin_memory=false，command_argv明确最终权重与test清单。
- 本次没有回传run.exit和SCENE EVALUATION COMPLETE日志，不补造显式exit0；现有报告/指标支持继续既定独立GAViT测试，同时只读补取Swin退出码，无需重跑Swin。完整metrics.json、逐类指标/混淆矩阵/逐样本预测尚未回传，当前只保存摘要，未宣称完整材料已本地归档。
- [规范化run.json](results/evidence/aid_swin_test42_20260921_01/run.json)、[metrics摘要](results/evidence/aid_swin_test42_20260921_01/metrics_summary.json)及provenance已归档，恢复聊天转义下划线与__init__.py，不是服务器原文件字节副本。更新阶段结果、排期和实验矩阵；最终配对比较仍待GAViT与退出证据齐全。
- 下一步GAViT评估沿用冻结last.pth SHA `a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`，同5000test和评估配方，不依据Swin分数改配置；输出`aid_gavit_test42_20260921_01`。启动、查看日志及双方退出码/GAViT摘要回传命令见[执行记录](docs/run_aid_formal_seed42.md)。未代为执行服务器命令或改动发布代码/Git。

---

## 2026-09-21 — GAViT AID最终拟合完成：5000张/20轮/exit0，进入双方最终测试

- 用户回传epoch16—20/20、SCENE REFIT COMPLETE、完整run.json及exit0：GAViT / refit / complete，seed42、train5000/val0、final_epoch20，末轮loss0.006303。val_OA=None、best_epoch/best_val=null符合固定预算流程。实际前向48空间边+32特征边，保存重载logit最大差4.76837158203125e-07，低于1e-5。
- 自身校准的架构、图捕获、参数量和所有冻结身份匹配；与Swin refit的代码、硬件环境、manifest、预训练、seed、training、counts及阶段逐项一致。17个源码哈希匹配干净发布deb2b45，来源best checkpoint SHA匹配GAViT校准；不使用本地格式化报告与服务器来源run.json做字节哈希比较。
- 最终权重`/home/featurize/work/aid_gavit_refit42_20260920_01/last.pth`，SHA256 `a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a`。RTX2080Ti，全程819.7612688440004秒（约13分40秒），可见末五轮train_s30.66—31.86秒。CUDA峰值allocated3.6556GiB/reserved3.8770GiB（含重建），主进程RSS1.6547GiB。总轮数不同、wall time含扫描/I/O，不能据此声称GAViT推理更快；完整history及费用未回传。
- 已归档[原始回传](results/evidence/aid_gavit_refit42_20260920_01/server_output.txt)、提取run.json、run.exit和provenance；更新[阶段表](results/aid_seed42_comparison.md)、排期与矩阵。服务器checkpoint未直接下载，两模型最终test均待运行。
- 下一步复用双方last.pth评估相同5000张test：目的为获得单seed正式泛化比较，模型与轮数均冻结，不增加训练或按test选择。先Swin后GAViT，评估CUDA/batch32/workers2/float32/TF32关闭/无随机增强，manifest及代码沿用已有身份。新输出`aid_swin_test42_20260921_01`与`aid_gavit_test42_20260921_01`；完整命令与验收见[执行记录](docs/run_aid_formal_seed42.md)。入口会核对原始数据、权重/metadata、来源完成和源码；只有完成标志、exit0、5000样本及正确身份齐全才记录正式结果。
- 当前evaluate保存OA、macro-F1、逐类指标、混淆矩阵及逐样本预测；不打印逐batch进度、不记录标准化推理耗时/显存，因此不新增代码或要求不存在的日志。完整扫描10000张与测试5000张分开说明；仅提供摘要回传命令，完整产物保留。未代为运行GPU测试或修改发布代码/Git状态。

---

## 2026-09-21 — Swin AID最终拟合完成：5000张/21轮/exit0，准备GAViT refit20轮

- 用户回传epoch16—21/21、SCENE REFIT COMPLETE、完整run.json及exit_code0：Swin / refit / complete，seed42、train5000/val0、final_epoch21。末轮loss0.003510，val_OA=None、best_epoch/best_val=null符合固定预算拟合，不是验证失败。
- 与Swin校准逐项核对architecture、code、hardware、manifest、预训练、seed、全部training、参数量与数据集一致；来源best checkpoint SHA与校准报告一致。17个源码哈希与干净发布deb2b45一致。refit_source.run_json_sha256按服务器报告归档，不与格式化后的本地校准报告计算哈希比较。
- 最终权重`/home/featurize/work/aid_swin_refit42_20260920_01/last.pth`，SHA256 `fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018`，重载logit最大差0。RTX2080Ti全程892.877999542秒（约14分53秒）；可见末六轮train_s30.61—31.26秒。峰值CUDA allocated3.5582GiB/reserved3.8301GiB（含重建），主进程RSS1.4012GiB；完整history及费用未提供。
- 已归档[原始回传文本](results/evidence/aid_swin_refit42_20260920_01/server_output.txt)、提取的run.json、run.exit和来源说明，更新[分阶段表](results/aid_seed42_comparison.md)、排期及实验矩阵。checkpoint未直接下载；两模型test均待运行，不补填测试指标。
- 下一步GAViT refit20轮，用同一5000训练池和ImageNet来源重新初始化，Cosine T_max仍30，其余冻结身份沿用校准。目的为完成既定50%训练池配对，预算事先由各自val OA选定；已有便宜smoke、两次完整校准及Swin refit，不重复校准或新增超参。新输出`aid_gavit_refit42_20260920_01`，完整单行命令和验收条件见[执行记录](docs/run_aid_formal_seed42.md)。先看实际首batch/图80边/首轮，再按真实每轮耗时估总时长；Swin14分53秒仅作参照，GAViT实测待回传。
- 本轮仅整理证据与运行文档，未修改发布训练代码、Git身份或代为启动GPU任务。GAViT refit完成并验收后评估双方last.pth的5000test，保存OA、macro-F1、逐类/混淆矩阵/预测。

---

## 2026-09-20 — GAViT AID校准完整验收：best20 / val OA97.7%，两模型进入最终拟合

- 用户回传完整run.json与run.exit=0：GAViT / calibrate / complete，seed42、train4000/val1000、final_epoch30，best_epoch20、best_val_oa97.7%；此前末轮val OA97.1%、loss0.000893分开保留。实际RTX2080Ti，48空间边+32特征边，actual_forward_checked=true；保存重载logit最大差4.76837158203125e-07，低于1e-5。
- 与Swin完整报告逐项核对：manifest、预训练、seed、全部training、hardware、code、counts及阶段一致；17个源码哈希与干净发布deb2b45一致。架构与GAViT smoke一致。最佳val相对Swin96.9%为+0.8个百分点，仅为单seed校准验证差，不能写作test或稳定性增益。
- GAViT全程1366.7442648819997秒，约22分47秒；CUDA峰值allocated3.6549GiB/reserved3.8770GiB（含重建），主进程RSS1.6646GiB；参数31375385全部可训练。完整history、macro-F1、逐类/混淆矩阵及费用未回传；不把全程wall time差当推理速度比较。
- 最佳checkpoint `/home/featurize/work/aid_gavit_calibrate42_20260920_01/best.pth` / SHA256 `6a1a99e337b7f810b60bce86308ec23ca52c7a327ec7a8e9a9c4e39ebbfff7eb`。已归档[完整报告](results/evidence/aid_gavit_calibrate42_20260920_01/run.json)、退出码和来源说明；聊天JSON仅规范化，不是服务器原文件逐字节副本，未直接下载checkpoint。
- 下一实验目的为完成既定50%训练池/50%test协议：两模型从同一ImageNet权重重新初始化，分别用自己的验证选轮预算Swin21/GAViT20，在5000训练池拟合，Cosine T_max仍30，最后使用last.pth测试。主要变量仍为架构，各自选轮策略事先已规定，不根据test改变。已有两次smoke和完整校准已回答工程与预算选择问题，不重复校准。
- 本地使用发布入口validate_calibration只读核对refit配置与回传报告，两模型分别返回21/20轮；未加载服务器数据/checkpoint，完整来源校验仍由服务器入口完成。完整冻结身份、命令、输出路径、成本估计和验收见[执行记录](docs/run_aid_formal_seed42.md)。先提供Swin新目录`aid_swin_refit42_20260920_01`命令，用户执行且验收后再安排GAViT；未代为启动训练/test，无训练代码或Git变更。
- [阶段结果](results/aid_seed42_comparison.md)、排期与实验矩阵同步更新；正式test比较表保持待测试。发布工作树维持干净deb2b45，避免refit来源身份变化。

---

## 2026-09-20 — GAViT AID校准日志到30轮，完整最佳轮与退出报告待核对

- 用户回传epoch27—30及SCENE CALIBRATE COMPLETE，输出目录`/home/featurize/work/aid_gavit_calibrate42_20260920_01`，后台job显示done。日志支持30轮校准完成，但本轮尚未收到显式exit0或run.json，不将done当作退出码。
- 末轮val OA97.1%、loss0.000893、train_s25.12秒；可见片段最高为epoch27的97.2%，epoch1—26未提供，全程best仍未知。已归档[末段日志](results/evidence/aid_gavit_calibrate42_20260920_01/reported_tail.log)及边界摘要，并更新[seed42阶段表](results/aid_seed42_comparison.md)。
- 下一步读取GAViT run.exit/run.json核对身份、全程best、实际图前向、保存重载与资源；确定自身refit预算。Swin预算已确认为21轮；两模型随后各从同一预训练在5000训练池重训，并使用固定预算末轮权重评估5000test。当前不启动refit/test，不将校准val差当作正式测试收益。

---

## 2026-09-20 — Swin AID seed42校准完整验收：best21 / val OA96.9%，准备GAViT同配方校准

- 用户回传run.exit=0和完整run.json：model swin / phase calibrate / status complete，train4000、val1000、seed42、final_epoch30，best_epoch21、best_val_oa96.9%，checkpoint重载logit最大差0。此前末轮日志val OA96.3%、loss0.000756，与best分开保留。
- 重启后实际硬件确认仍RTX2080Ti，Python3.11.8、Torch2.2.2/CUDA12.1、timm1.0.27等环境及全部training配置与既有Swin smoke逐项一致。code Git deb2b45 / dirty=false，17个源码哈希与干净发布工作树全部一致；manifest与预训练SHA也一致。源码/指标检查为只读，未重跑训练。
- 总耗时1281.2088429679998秒，约21分21秒；峰值CUDA allocated3.5582GiB/reserved3.8301GiB（含重建）、主进程RSS1.4433GiB；参数27543960全部可训练。完整每轮history、macro-F1/per-class/confusion与费用未提供，不根据末十轮补造。
- 最佳checkpoint `/home/featurize/work/aid_swin_calibrate42_20260920_01/best.pth` / SHA256 `3542b3b5cd40fbf1ea44881f284ab0d997322e9e2ed01687341fc2300bb8e939`。用户JSON因聊天转义被规范化后归档[报告](results/evidence/aid_swin_calibrate42_20260920_01/run.json)及provenance，不当作服务器原文件字节副本；[正式分阶段表](results/aid_seed42_comparison.md)已更新。
- E01下一步为GAViT seed42校准，研究问题和冻结协议保持原计划；主要变量仅架构，4000train/1000val、30轮、AdamW3e-4/wd1e-4/CE/Cosine30、batch32/workers2，同预训练和manifest。E00便宜前置已通过。新输出`aid_gavit_calibrate42_20260920_01`，完整命令见[执行记录](docs/run_aid_formal_seed42.md)，bash/zsh语法通过；待用户在服务器执行，未代为启动。
- Swin后续5000训练池refit预算已选21轮，T_max仍30，重新从预训练初始化；目前暂不启动refit/test。GAViT轮数由自身校准选择；单seed验证结果尚不能支持稳定优于baseline。GPU成本以GAViT完整首轮/全程实测补充，Swin21分21秒仅作资源参照，不承诺相同耗时。

---

## 2026-09-20 — Swin AID正式校准日志到30轮；完整报告待核对

- 用户回传`aid_swin_calibrate42_20260920_01`的epoch21—30和SCENE CALIBRATE COMPLETE，后台job显示done。确认按日志已完成30轮校准；尚未回传run.exit/run.json，不将shell done冒充显式exit0。
- 第30轮loss0.000756、val OA96.3%、train_s24.88秒；可见片段最高val OA96.9%在第21轮，epoch1—20未提供，完整best epoch/value仍待报告，不能按片段选轮数。可见train_s为24.55—31.20秒，非含验证/存盘的全轮耗时。
- [日志片段](results/evidence/aid_swin_calibrate42_20260920_01/reported_tail.log)、[有边界的摘要](results/evidence/aid_swin_calibrate42_20260920_01/tail_summary.json)及[seed42阶段表](results/aid_seed42_comparison.md)已更新。当前仅为校准val指标；5000训练池refit和5000test尚未运行，没有将其写成最终测试精度。
- 下一步用户读取run.exit与run.json，核对全程best、实际GPU（实例已重启过）、代码/manifest/预训练身份、保存重载差与资源统计，然后启动同配置GAViT正式校准。不重跑Swin、不修改数据或代码。

---

## 2026-09-20 — 实例重启后AID原图恢复；正式训练状态待核查

- 后续用户回传指定正式输出目录的run.log与run.exit均不存在；确认没有这两份运行证据，不推断其他目录或进程状态。继续使用原计划输出身份首次启动命令，入口会拒绝覆盖已有目录；重启后的nvidia-smi及首个正式batch/epoch仍待回传。
- 用户确认重启实例后AID图片计数为0，随后使用work中保留的ZIP执行unzip -n并报告已全部解压。恢复路径沿用`/home/featurize/data/AID/AID`；本次尚未回传恢复后全量哈希检查，后续入口会基于已冻结的`aid_protocol_20260920_03.json`复核，不重新prepare或改变划分。
- 既有两模型GPU短测证据保留；重启后GPU型号/可用性需重新查看。此前已提供Swin正式校准命令，但尚未收到启动/运行回传，不能假设重启前未训练、已完成或需要从头重跑。
- 下一步先只读检查原`aid_swin_calibrate42_20260920_01`的run.log/run.exit与nvidia-smi；若输出不存在说明没有该目录下的运行证据，若中断则保留checkpoint再确定处理方式。当前入口没有自动resume，不覆盖目录或直接换编号重训来跳过状态核查。此轮没有新增GPU任务或代码改动。

---

## 2026-09-20 — AID GAViT GPU短测通过，E00完成；准备E01 seed42正式校准

- 用户回传GAViT在RTX2080Ti上的train240/val60、seed42、1 epoch完整日志与run.json/run.exit，status complete、SCENE SMOKE COMPLETE及exit_code=0齐全。实际前向确认48空间边+32特征边=80，actual_forward_checked=true；checkpoint重载logit最大差4.76837158203125e-07，低于1e-5。
- 训练loss3.010718、smoke val OA60.0%，best/final epoch均1；数据扫描140.5秒、训练3.19秒、全程157.37608910299969秒。CUDA峰值allocated3.6549GiB/reserved3.8770GiB（含重建），主进程RSS1845030912字节，参数31375385全部可训练。短测精度差和计时差不用于正式效果/效率结论。
- checkpoint `/home/featurize/work/aid_gavit_smoke_20260920_01/best.pth` / SHA256 `c3ca53bb504196cff398fa76854e33d3592a48a7733bc5385cf72040fba82dae`。17个源码哈希与本地发布deb2b45一致；两模型manifest、预训练、seed、training、hardware、code、counts逐项相同。证据由用户文本提取归档于[run.json](results/evidence/aid_gavit_smoke_20260920_01/run.json)，并更新[验收表](results/aid_smoke_validation.md)，未直接下载服务器checkpoint。
- E00已完成；正式E01问题为固定区域图在AID上的配对收益与成本，主要变量为模型结构，已用两次便宜smoke排除工程问题。下一步只给用户Swin seed42校准命令：4000train/1000val、30轮、val OA选epoch。其后GAViT同流程，再各自5000pool重训并最终5000test；不利用test选轮数。完整冻结身份、超参、产物、命令和成本估算方式写入[正式执行记录](docs/run_aid_formal_seed42.md)。
- 正式计划Git deb2b45，输出`/home/featurize/work/aid_swin_calibrate42_20260920_01`；启动前要求干净代码，入口会验证全量数据/权重，首个batch/完整首轮后核查真实速度和GPU占用。30轮总耗时待首轮全量训练/验证测量，费用未提供。命令bash/zsh语法通过；没有代替用户启动训练、改代码或新增Git同步。正式结果表[seed42](results/aid_seed42_comparison.md)目前全部待运行/待测试。

---

## 2026-09-20 — AID Swin真实GPU短测通过，GAViT短测待执行

- 用户回传Swin在NVIDIA GeForce RTX 2080 Ti上的完整结束输出及run.json/run.exit。train240、val60、seed42、1 epoch，TRAIN FIRST BATCH OK、EPOCH 1/1、SCENE SMOKE COMPLETE、status complete、exit_code=0全部齐全；checkpoint重载logit最大差0。本地核对报告中17个源码SHA256与干净发布工作树deb2b45全部一致，未重新运行服务器任务。
- ImageNet预训练指定文件成功加载，CUDA forward/backward、双worker数据加载、validation及保存重建流程通过。训练loss3.098898、smoke val OA41.666666666666664%，best/final epoch均1。数据完整性扫描101.5秒，训练3.89秒，全程114.53324206199977秒；CUDA峰值allocated3.5582GiB/reserved3.8301GiB（含重建），主进程RSS1504694272字节，参数27543960。这些是小样本工程验收指标，不写入正式性能表或当作效率benchmark。
- manifest SHA256 `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；预训练SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`；checkpoint `/home/featurize/work/aid_swin_smoke_20260920_01/best.pth` / SHA256 `73efc23a14992873b2f3ab2cd4dac608d68708be1fe7fad5962a1966657cfc0e`。完整训练配方、环境和源码身份保存在[回传证据](results/evidence/aid_swin_smoke_20260920_01/run.json)，[工程验收表](results/aid_smoke_validation.md)已更新。源于用户文本，未直接下载服务器checkpoint；未提供指标不补填。
- E00的真实数据和Swin部分完成；下一步同清单/权重/seed/batch32/workers2执行GAViT的1 epoch短测，额外检查48空间+32特征边。没有启动正式calibrate/refit/test，没有改代码或要求再次Git同步；GAViT尚待用户运行。

下一步命令（待执行，bash/zsh语法通过）：

```bash
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase smoke --seed 42 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_gavit_smoke_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2
```

---

## 2026-09-20 — AID全量数据预检通过，准备Swin GPU短测

- 用户回传完整扫描10000/10000，扫描耗时109.0秒，出现MANIFEST COMPLETE；train4000/val1000/test5000。一组重复、1份多余副本、9999张唯一图像；Industrial/industrial_203.jpg与industrial_205.jpg均在train，未跨划分。源输出归档于[服务器输出](results/evidence/aid_data_preflight_20260920/server_output.log)及[摘要](results/evidence/aid_data_preflight_20260920/summary.json)。
- 结合上一次ZIP SHA256完全匹配，本轮数据准备门槛已通过。当前仅有用户回传文本，未直接读取服务器manifest；清单路径沿用上一条命令的`/home/featurize/work/aid_protocol_20260920_03.json`，清单SHA及真实运行Git身份将由短测自动记录。用户未单独打印prepare退出码；以写清单后打印的完成标志确认此步成功，不冒填exit0。
- 下一步为E00的Swin工程验收：seed42、240张train/60张val、1 epoch、batch32、workers2、AdamW3e-4/wd1e-4、CE、Cosine周期30，复用指定ImageNet预训练文件。目的为核对真实AID和CUDA上的权重加载、forward/backward、validation、保存和重载；不用于论文性能结论，不启动30轮正式训练。Swin成功后同清单测试GAViT。
- 计划Git版本为deb2b45，实际commit/脏状态由入口记录；输出新目录`/home/featurize/work/aid_swin_smoke_20260920_01`。首次真实GPU短测采用前台命令，错误和进度直接可见，入口另存run.log/run.json/run.exit及checkpoint；尚未实际启动。前置load_manifest会重新扫描原图，并显示进度。
- 核对现有入口参数及日志写入流程；提供的命令通过bash/zsh语法检查。没有改动训练代码或正式指标表，服务器无需再次Git同步。

计划命令（待用户执行）：

```bash
python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase smoke --seed 42 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_swin_smoke_20260920_01 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2
```

---

## 2026-09-20 — AID压缩包哈希确认；prepare被手动中断，补充扫描进度

- 进度提示已提交并推送`deb2b45`到既有AID分支；服务器仍待用户拉取重跑，全量预检尚未完成。
- 用户回传Git fast-forward 2ba112a→06fddd0及ZIP完整SHA256 `cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7`，与固定来源预期一致；据服务器用户输出确认下载完整性，未在本地重新下载/哈希整个压缩包。
- group prepare在`_image_identity`像素SHA256计算处收到Ctrl+C，抛KeyboardInterrupt。没有新的数据错误证据；未达到清单写入步骤，未完成全量预检，不能当作数据或GPU验收通过。日志没有开始时间，不能估算本次已运行多久或归因为性能故障。
- 原入口只在全量扫描后打印结果，容易误以为卡住；新增开始、首张、每250张和最终张的已完成数量/总数、已用秒数并立即flush。prepare及训练/评估的load_manifest共用提示；不改变哈希、分组、seed、划分或输出清单内容。
- 18项现有数据测试和5个subtest通过；实际CLI使用600张合成图含1组重复，prepare/load退出0，进度0/1/250/500/600均可见，清单与06fddd0逐字节一致。见[CLI证据](results/evidence/aid_progress_20260920/cli.log)。未新增仅镜像日志实现的单元测试。
- 下一步同步代码后用新编号`aid_protocol_20260920_03.json`前台运行，等待MANIFEST COMPLETE及DUPLICATES；通过后再进行Swin GPU短测。没有要求重下/重解压，没有启动正式训练或改正式结果表。

---

## 2026-09-20 — AID重复图片预检修复：保留原图并按像素组隔离划分

- 修复已提交并成功推送：`06fddd0602e0e35d831f5515f8d469edf0a2e5fb`，分支`codex/aid-entry-20260920`；push确认2ba112a→06fddd0，本地origin跟踪SHA一致，发布工作树干净。服务器仍由用户复制单行pull和prepare group命令执行。
- 用户服务器2ba112a运行prepare，在`Industrial/industrial_205.jpg`与`industrial_203.jpg`解码像素重复处退出。原入口对所有重复一律报错，未生成清单、未开始训练；此输出不证明ZIP整体哈希或所有图片均已验收。已有压缩包和图片保留，不要求重新下载/解压。
- 新增显式`--duplicate-policy group`，采用schema 2 / scene-pool-group-v2。同类完全相同像素不可拆组，固定seed按组分配，保持每类精确50%训练池及池内20%val；跨类标签冲突仍拒绝。确定性子集和先选pool，再选val；选定pool无法满足精确val时拒绝而不改比例/换seed。默认reject及旧schema 1兼容，无重复类别沿用旧划分。
- 清单记录重复组路径、像素哈希、类别、split、唯一图像/多余副本数。加载时重算并校验数据哈希、报告、无跨划分重复与seed划分。全部原图保留，指标仍按全部图片计数；这是自定义防完全重复泄漏协议，不称AID官方固定划分，也未处理近似重复或地理重叠。两模型必须共用同一清单。
- 新增7项回归：同类跨编码重复、原图/数量保留、跨类拒绝、无法精确划分拒绝、跨split泄漏拒绝、报告一致性、旧划分兼容，并覆盖新清单真实CPU训练/重载/评估流程。发布工作树完整测试96 passed、34 subtests、exit0、10.52秒；39条现有Torch/Python3.12弃用警告。合成CLI prepare及指南bash/zsh语法通过；代码审查无阻断项。见[验证记录](results/aid_duplicate_fix_validation.md)。
- 下一步服务器Git同步后以原data-root和新清单`/home/featurize/work/aid_protocol_20260920_02.json`执行prepare group，回传MANIFEST COMPLETE及完整DUPLICATES报告；实际重复总数和完整数据预检仍待确认。ZIP源预期SHA256 cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7，尚无用户核验回传。未启动GPU短测/正式训练、未改正式结果表。

---

## 2026-09-20 — AID备用镜像服务器下载完成；SHA256与数据预检待执行

- 用户回传服务器07:51:00启动hf-mirror.net直连，HTTP200、application/zip，最终100%、saved [2628474154/2628474154]，文件`/home/featurize/work/AID_visionflan_20260920.zip`。日志显示耗时3m50s、显示速度12.3MB/s；该速度是进度显示值，不当作全程平均吞吐。
- 完整保存提示先于后续Ctrl+C，支持文件传输完成；不能替代完整SHA256核验，尚无下载进程退出码记录。预期源SHA256仍为cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7。
- 下一步由用户执行sha256sum，一致后解压到data并用2ba112a现有入口prepare，实际data-root为`/home/featurize/data/AID/AID`。预计train4000/val1000/test5000，需MANIFEST COMPLETE回传确认；尚未解压验收或启动GPU短测。
- 下载网络问题已有成功传输证据，停止继续更换来源。保留work压缩包供以后恢复，数据协议、代码及正式结果表不变。

---

## 2026-09-20 — AID直连失败，备用镜像小段下载检查通过

- 用户回传直连HF解析出108.160.172.208及2a03:2880:f10c:283:face:b00c:0:25de，随后IPv4超时/IPv6不可达。原代理429与直连失败均已确认；不继续改全局DNS/代理或关闭TLS校验。ls片段显示预训练文件约109M存在，只算存在性证据。
- 对第三方hf-mirror.net的同revision AID.zip执行无认证256字节Range GET，直接HTTP206、总长2628474154、未重定向；与原HF同范围字节完全一致。[记录](results/evidence/aid_source_probe_20260920/mirror_net_probe.json)。完整镜像SHA及服务器网络仍未验证，后续必须核对源发布方cdb1af…703bf7，再进行prepare。
- 向用户提供同路径--no-proxy、-c、tries=1续传命令；保留原文件、配置和数据协议。[指南](docs/download_aid.md)更新，真实服务器下载仍待回传，未启动任何训练。
- AI Studio 52025公开页有AID_dataset.zip但未迁移，下载会要求百度登录；未进行登录、凭据传递或绕过。hf-mirror.com的本地GET与HEAD均跳原HF，不与本次.net入口混淆。仅更新研究工作树文档与证据，发布代码2ba112a不变。

---

## 2026-09-20 — AID下载返回429；暂缓解压，收集限流节点证据

- 用户只回传“429了”，尚无原始下载日志/错误URL/响应头。确认当前下载未验收，不推断为具体账号/IP/CDN原因，不开始解压或训练；已请用户提供失败最后10行并去掉凭据。
- 随后用户提供原始日志：wget连接172.16.0.13:5848，Proxy request sent后429，URL为原固定revision HF resolve入口。确认经代理请求，不足以区分代理生成/转发429；下一步给出仅增加--no-proxy、tries=1的同文件续传命令做直连测试，全局代理配置不变。直连结果尚待回传。
- 核对HF官方限流文档；另实测同revision的HF-Mirror路径只会308跳回原HF resolve入口，本地最终200不能代替服务器连通性，也不能作为独立备用源。未要求用户反复盲试此域名或向其提供token。
- [下载说明](docs/download_aid.md)补充故障边界。现有ZIP保留便于恢复后续传，SHA256和全量prepare仍必需。没有改代码、Git发布版本、数据协议或正式结果。

---

## 2026-09-20 — 服务器确认AID代码版本，平台未找到数据；直链镜像完成目录抽查

- 用户回传服务器HEAD为2ba112a，与发布分支一致；停在Git pager的(END)不影响版本确认。服务器预训练文件检查仍未回传，随后用户确认Featurize未找到AID。
- 核对Vision-Flan固定revision的AID.zip公开镜像：HTTP可达，2628474154字节；只读取中央目录及三张图，30类10000个图片条目，三张均600×600 RGB；实际层级AID/AID/类别。发布方LFS SHA256 cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7。没有下载全包或声称与AID作者版本逐字节一致；[证据](results/evidence/aid_source_probe_20260920/probe.json)保留边界。
- [下载说明](docs/download_aid.md)提供服务器单行wget断点续传、sha256sum、unzip及现有prepare命令。压缩包放work持久化目录，原图放data，所有data-root应采用实际嵌套路径；全量解码、重复检查和固定划分仍由服务器prepare完成。
- 另查SkyScript公开AID包，目录含23886个图片条目，分在其train/val/test，未采用也未通过去重/换划分适配。官方OneDrive/百度网盘入口仍可在项目页找到，但没有验证服务器直下。镜像本地可达不能代替Featurize网络验证；真实下载/数据预检及GPU运行均待用户输出。
- 只更新研究工作树的来源记录和本地指南；发布的2ba112a代码及正式实验表不变，没有新增服务器任务或付费训练。

---

## 2026-09-20 — AID代码已发布独立分支；下一步由用户在服务器拉取并准备数据

- 用户选择继续自行复制单行命令执行。使用独立工作树`/Users/patrick/.codex/worktrees/aid-entry/GAViT_Project`整理发布，基于原8f2ed07，仅纳入8份经过审查的Python代码/测试、运行指南、验证记录/日志与一条分支日记，合计12个文件；原研究工作树及其他未提交实验材料保留，没有合并main或旧实验分支。
- 分支`codex/aid-entry-20260920`已提交并推送至origin，commit `2ba112a3d034a6c723bb6e9c099855d12b21fb35`。推送后用`git ls-remote`确认远端完整SHA一致；独立工作树干净。分支历史日记未包含研究工作树其间未提交内容，整体进展仍以本日记为准。
- 在隔离分支运行完整tests/：89项测试、34个subtest通过，exit0、8.85秒。Python文件与此前审查版本逐字节一致，服务器命令块均通过bash/zsh语法检查；[发布验证](results/aid_entry_release_validation.md)和[日志](results/evidence/aid_entry_release_20260920/tests.log)留档。没有将合成数据指标写入正式表。
- [指南](docs/run_aid.md)改为服务器全新目录`/home/featurize/work/GAViT_AID_20260920`，使用git clone直接取得代码，继续复用旧目录的预训练文件；不再要求上传诊断工具包。尚无用户拉取/下载成功证据，实际AID、CUDA和workers=2验收仍待完成，不启动正式训练。
- 已核对AID官方页面仍提供OneDrive/百度网盘入口，但本次未取得可验证服务器直链；Featurize公开数据集页面未能确认AID条目。先由用户在平台搜索AID，数据来源/实际目录确认后执行prepare和两个独立smoke。无需恢复BigEarthNet原始数据。

---

## 2026-09-20 — AID统一入口本地完成：74项相关测试通过，真实数据/GPU待验收

- 用户两次要求继续AID接入，并确认服务器尚未准备AID。新增`run_scene.py`和`scene_classification/`，复用现有模型，保留旧NWPU及BigEarthNet入口；已实现分层固定清单、30类身份及原图预检、共享训练配置、完整单标签指标、smoke/calibrate/refit与独立评估。见[指南](docs/run_aid.md)、[工程验证](results/aid_entry_local_validation.md)。
- 数据清单记录相对路径、文件/像素哈希、类别映射、外层seed42/内部val seed4242，检查重复/损坏/更改/路径逃逸/划分漂移。新Swin/GAViT共用增强、AdamW3e-4/wd1e-4、Cosine30、CE；GAViT显式固定K16 sparse-hybrid/token_feedback。checkpoint绑定清单、阶段、配置、真实源码/预训练身份；每次新输出，入口保存日志与退出状态。
- 18项新增测试及相关旧模型/图/身份/训练状态测试共74项通过、31个subtest通过，exit0；另从仓库外以30类合成图和随机权重完成两模型CLI训练/独立val评估，共5个命令exit0。保存前到重建模型最大logit差均0，真实图输入48空间+32特征边通过。独立验证同一份本地生成权重在两模型backbone中逐张量一致，不声称验证实际ImageNet文件。合成分数不写入正式结果表。
- 审查发现并修复：未完成训练可被评估、refit报告和权重身份绑定不足、绝对路径启动记录错误Git目录、评估用当前代码却不校验源码变化。失败回归及通过结果保留。新增配置记录实际参数组、dropout位置、软件与设备，论文资源结论仍待同硬件测量。
- 本地Mac双worker沙箱检查因OpenMP共享内存错误退出；获准在沙箱外重试后180秒超时，后者根因未定。CPU命令行成功使用workers=0，不能代替服务器workers=2；两次失败证据保留，GPU验证仍须检查实际加载过程。
- 在独立临时环境安装测试依赖，未改项目环境；没有恢复BigEarthNet数据、启动服务器/GPU正式训练或修改已有正式指标。代码尚未提交、推送、同步，真实AID/ImageNet/CUDA及calibration/refit正式运行均待完成。E00本地部分完成，整体待数据/服务器验收；seed45及可学习分组等排期保留。

---

## 2026-09-19 — W1-09文献与实现可行性审计完成，下一步AID接入

- 用户同意按排期继续核对AID、HGTNet、DFAGCN并形成实验矩阵。本次完成公开原文/官方数据页/作者论文列表检索及现有单标签训练评估代码审查；[方法审计](docs/external_methods_audit_20260919.md)和[实验矩阵](docs/experiment_matrix_20260919.md)记录来源、缺口和具体验收。
- 两篇方法论文共同设置为AID50/50、NWPU20/80，但重复次数、backbone和训练方法不同；不能和当前NWPU70/15/15近似分数直接排名。DFAGCN是VGG16特征、GCN聚合与LIBLINEAR分类的分阶段流程，不是现有端到端模型的替换选项。官方方法代码、权重及代码许可证尚未核实；未找到不等于断言没有。
- 当前Swin/GAViT单标签脚本均写死45类/NWPU身份；增强不一致，Swin的AdamW未显式传weight_decay；现有test主体只输出Accuracy。先做AID共享数据/训练/指标入口及小样本smoke，再安排配对正式实验；不只改数据目录就启动训练。
- 矩阵提出AID外层50/50，池内验证选预算后在完整训练池重新拟合；这是一项待实现、待冻结的本项目方案，不冒充原文选择规则，也不把40/10/50校准阶段称为实际50%训练。E00完成标准、seed42优先、条件多seed、成本估算与外部方法缺口均已列明。
- W1-09审计与矩阵完成，AID接入和外部复现仍待完成。seed45、seed43边界披露及NWPU/BigEarthNet可学习分组待办保留，第二周接入进度偏差已写入排期。没有新增正式指标、服务器任务、训练代码改动、第三方代码导入或对外联系；未提交或推送Git。

---

## 2026-09-19 — 四例论文图稿完成：PDF/PNG导出核验通过，W1-08收尾

- 用户要求继续论文图整理。复用既有四例PNG/JSON/捕获SVG，完成 `output/pdf/seed44_feedback_figure/` 中的单页PDF（180×229.18 mm）和300 dpi PNG（2126×2707），附中英文图注及LaTeX插图片段。按a–d保留全部2个误报纠正/2个漏报案例，无新选样、模型前向或训练。
- 图上12个面板为四例各自的RGB/空间边/特征边；4张输入像素未做亮度、裁剪或重编码修改。320条捕获有向边和原曲线路径、节点中心、token矩形保留；为印刷可读性调整字体、线宽、箭头及节点字号。源区域填色透明度0.25，未把颜色或边宽解释为注意力。
- 使用ReportLab生成科学图稿，Poppler渲染最终PDF并逐行检查。初次渲染遇到bundled字体配置引用构建机路径，设置项目临时fontconfig后解决；同时将可见字体嵌入PDF。预览发现初版填色设置顺序使透明度被颜色重置，已修复并重新渲染；最终没有遮挡、文字越界或布局重叠。没有绕过原HTML的file://浏览器限制。
- 导出核验：PDF恰1页，4张嵌入图片RGB像素与归档逐一一致，12次图像放置；PDF中的320条实际边曲线与源路径换算最大误差约4.99e-5 pt；392个token填色矩形均alpha0.25。全部关注概率/row匹配、实际绘制字体嵌入、文字边界位于页面内。最终Poppler渲染exit0且stderr为空。
- PDF SHA256 `c680785164c4d9f1ac72e610ee08d86f8570a8ba0ba5afd402a856f588ade3fa`；PNG SHA256 `f7e4960d8d0d12a6f78cce40a674a5e7680de7b7adcd66435858c0c9ab5577d2`。[交付说明](output/pdf/seed44_feedback_figure/README.md)、逐边manifest和export_qa.json保留来源及验证。当前按180 mm宽验收，放入其他投稿模板后仍需复查实际字号；源RGB只有224像素，不把300 dpi导出当更高遥感分辨率。
- W1-08真实sparse-hybrid图示完成。研究结论保持定性边界：纠正与损害并存，背景干扰仍待验证，不从四例推导总体AP或特定边因果作用；正式test表与三seed统计不变。没有Git提交或推送。

---

## 2026-09-19 — 真实案例HTML返回：四例内容检查完成，固定区域与局部目标提供研究线索

- 用户返回 `feedback_cases.html`，1,166,876字节，SHA256 `bca16d0a7eab8deca452a0c1890367727eb1848b4e69084b6c875c4c83ba5c63`。原件按字节归档到 `results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919/`，离线提取未改动PNG、原SVG、两种分层SVG和完整四例JSON，没有执行HTML脚本或绕过浏览器file://限制。
- 四个row恰为已固定24例的全部翻转。PNG/原SVG哈希及JSON按生产格式还原后的哈希均匹配来源清单；每例48空间+32特征边、每节点2条非相邻特征入边、节点坐标/49 token矩形、分层SVG字节与原捕获一致。19类概率与缓存差核对通过，四例最大差4.76837158203125e-7，关注分数与终端四位小数一致；不替代24例的约1e-6。没有原始TIF/PT及完整run.json/loaded_model.json，未声称独立重算前向。
- 实际查看4张224×224 RGB：97389深绿斑块与开阔斑块混合；14680暗绿纹理与下部浅绿开阔带共存；28450深绿、褐色规则地块及浅色窄带交错；4439为大面积水面与左侧狭窄浅色岸带。标签取自返回记录，不凭RGB纠正标签或给出像素真值。
- 四例assignment相同，为固定空间分组；区域内attention可学习不等于区域边界自适应。row4439的实际特征边包括1→4、5→8、13→8、5→12，可见水面区域到沿岸混合区域的连接。可提出局部目标/背景干扰假设；整支反馈干预不能归因某条边或确认水面信息导致漏判。
- 数值结论保留2例抑制误报/2例增加漏报，全部四例一起进入定性分析。已写[视觉分析](results/seed44_feedback_visual_review.md)，更新[中英文图注](results/seed44_feedback_figure_caption.md)、原图报告及排期。没有新GPU推理/训练、正式指标修改、Git提交或推送。
- 原图对齐、数值干预、四例内容与图结构检查阶段收尾；下一步仅整理论文版面并验收导出，本地材料已足够。HTML浏览器排版仍未验收，W1-08不整体勾选；不再要求用户重复解压、推理或回传同一日志。

---

## 2026-09-19 — 4个原图翻转案例的独立展示页已准备；服务器生成和真实图像检查待完成

- 用户同意继续案例图整理。检查本地证据与Downloads后，尚未发现原图运行的row PNG/SVG/JSON，只有终端及解析结果。因此不能直接进行真实遥感内容判断或声称论文图已完成。
- 按已确认的4例（草地97389/14680、沙滩28450/4439）新增 `tools/render_feedback_case_sheet.py`，只读服务器原图诊断产物。每例展示原图、区域+空间边、区域+特征边，并标注标签及zero/normal分数；边从已有SVG筛选，与JSON捕获边核对，不重建kNN、不调用Torch或GPU。
- 输出一份内嵌PNG/SVG/案例JSON/来源文件哈希的 `feedback_cases.html`，另留evidence.json；接受complete+original及既定checkpoint，检查24图清单和4例身份、缓存误差、PNG224尺寸、SVG边对应，拒绝缺失数据及覆盖输出。用户可整段复制[命令](docs/paste_feedback_case_sheet.sh)到服务器，无工具包上传；后续只需单个HTML供查看和视觉分析。
- 6项新测试先因功能缺失失败，实现后通过，覆盖资源独立性、边筛选/漂移、输入身份、gallery拒绝、缺图停止、CLI与输出保护。使用合成渐变PNG和人工图夹具，不冒充实际遥感图；真实服务器页面尚未生成返回。
- 新旧相关测试合计33项通过；Bash/zsh均实际运行本地夹具版粘贴入口，确认源代码原样恢复、4例完整内嵌HTML、哈希与退出码。文档链接和diff空白核验通过，没有真实服务器访问。
- 尝试本地合成页面的file://预览，被浏览器URL安全策略拦截；没有绕过，浏览器视觉验收未完成。当前仅能报告结构与资源检查通过。
- [运行说明](docs/prepare_feedback_case_sheet.md)与[中英文图注/讨论草稿](results/seed44_feedback_figure_caption.md)已准备。图注不编造尚未看见的地物或边语义，不把边当attention/因果归因，不用四例估计总体收益。W1-08最终图示验收仍待真实图像检查；没有新增推理、训练、正式指标、提交或推送。

---

## 2026-09-19 — 原图反馈诊断完成：缓存最大差约1e-6，草地/沙滩各纠正1例与破坏1例

- 用户提供完整终端附件，命令 `/usr/bin/time -v python -u /home/featurize/work/pair44_feedback_qnjm_rb7/diagnose_bigearth_feedback.py --input-mode original`；输出 `/home/featurize/work/pair44_feedback_k_7ehbne`。RTX2080 Ti、original、24张不同图片，所有CASE的内部数值/拓扑检查通过，FEEDBACK COMPLETE与Exit status0。报告最大缓存概率差0.000001（六位小数），通过每图19类≤1e-4门槛，24个关注cached/normal打印值一致。原图输入一致性阶段已通过，无需再解压或重复这批推理。
- 与gallery诊断的24个row、类别、标签、cached分数独立核对一致。草地8例：反馈纠正1/破坏1/均对3/均错3；沙滩同为1/1/3/3；湿地0/0/4/4。合计2纠正、2破坏、均对10、均错10；正常和屏蔽均正确12。非总体随机样本，不推导AP/F1或总体有效/无效。
- 全部4个阈值翻转：草地97389负例zero0.9691→normal0.4858（纠正误报），14680正例0.9170→0.4704（增加漏报）；沙滩28450负例0.9972→0.4518（纠正误报），4439正例0.9974→0.2812（增加漏报）。原图结果替代JPEG案例结论中的3/1计数，旧结果保留。反馈对关注logit20降4升；零反馈14项≥0.98，不当作独立无图模型表现。
- 湿地均错仍为2误报2311/2379与2漏报80208/90759，关闭反馈没有纠正它们，但不能因此声称分数/排序无影响。例如2311从0.9990降到0.6908仍误报，80208从0.0015降到0.0007仍漏报。本轮不能解释test海岸湿地额外−12.5 pp。
- 本次GPU2080 Ti，上次gallery3060；输入和设备同时变化，不能声称完成JPEG唯一因素归因。原图当前前向与旧完整val缓存对齐，足以用于这批原图案例干预分析；不追加编码消融或全量训练。原报告的Gallery JPEG提示为通用说明，当前输入以original为准。
- 全命令26.69秒，CPU user/system19.96/3.78秒，主进程最大RSS1309076 KiB（约1.25GiB）；GPU显存未回传。含启动、多次前向与写出，不换算单模型吞吐。无新训练、模型/阈值修改或正式指标变更。
- 附件按字节复制并核对SHA256为 `036f703e09362c26930ad32a6d706319ba0cc98dba9f71c96136f15568d44278`，见[证据](results/evidence/pair44_feedback_k_7ehbne/terminal_return_20260919.txt)。解析24条并重算计数、比对两轮样本身份和全部翻转案例，结果见[原图分析](results/seed44_original_feedback_diagnosis.md)。服务器JSON/PT/图片未独立取回，未声称完成视觉验收。
- 输入对齐与数值干预阶段结束，下一步从已有cases.html检查全部4个翻转案例的真实区域/边；湿地对照可按标签取最小错例row2311/80208。服务器已保存原图Resize224后的无损PNG，保留整目录用于后续图示。W1-08的原图捕获与数值核验完成，论文图的人工检查与整理待完成。没有提交或推送。

---

## 2026-09-18 — 原数据目录仍在：准备直接复用脚本做24张original核对，不再解压

- 用户回传 `ls -lah /home/featurize/data`：`BigEarthNet-S2` 目录存在，mtime Sep18 11:18。只确认目录存在，不能由目录项数117认定数据完整；此前重启不再作为原数据已丢失的推断依据。[目录记录](results/evidence/pair44_feedback_qnjm_rb7/data_directory_return_20260918.txt)。
- 下一步直接执行 `/usr/bin/time -v python -u /home/featurize/work/pair44_feedback_qnjm_rb7/diagnose_bigearth_feedback.py --input-mode original`，复用已部署脚本和原24张清单，无新工具包或解压。省略out自动创建新目录；原图文件/代码身份检查在模型加载前，缺失则停止；全部可读后核对正常预测与原缓存（每图19类最大概率差≤1e-4）并执行原反馈干预检查。
- 前台运行，成功证据改为Input original/24张、FEEDBACK COMPLETE、time的Exit status0；不经过gallery后台包装器，不期待DIAGNOSTIC_EXIT或外层diagnostic.exit/log。结构化身份、逐张分数/图/张量及报告仍保存新目录，终端日志由用户回传。[运行说明](docs/run_bigearth_feedback_diagnostic.md)已更新。
- 本地确认现有脚本支持original、自动新目录和缺图停止路径；只给运行命令，尚未在服务器执行、未确认24张原图完整或缓存一致。失败时保留证据，不通过放宽容差绕过。不更改正式训练/测试指标，无提交推送。

---

## 2026-09-18 — 反馈完整日志返回：内部检查通过，gallery相对原缓存出现显著偏移，先恢复原始案例输入

- 用户回传24条CASE及完整报告，确认RTX3060、gallery、24张不同图片，模型为原GAViT44/K16 attentive_spatial/sparse_hybrid/token_feedback。24例均GATES PASS、48空间+32特征边，capture/reconstruction/zero-branch/restoration检查通过，exit0。只证明该输入下捕获与干预内部一致，不证明gallery等价于原RGB；本地尚未独立取回run.json、张量或图片。
- 全24×19项normal与原缓存最大概率差0.889688；不是AP/平均误差。24项关注分数的最大差为0.4794（row91012），阈值翻转4项：14680（草地正例，0.4704→0.9391）、26944（草地负例，0.6595→0.4693）、91012（沙滩负例，0.7700→0.2906）、97389（草地负例，0.4858→0.7031）。全19类最大差所在类别/行未打印，待原始JSON才能定位。
- 反馈纠正的3例均为误报抑制：26944草地、28450/91012沙滩；反馈破坏的1例是4439真沙滩，zero0.9929→normal0.1885（原缓存0.2812），变为漏报。湿地4个均错分解为2误报2311/2379、2漏报80208/90759；0翻转不等于0作用，11915真阳性logit下降10.9324仍正确，83491负例从0.3197降至0.0495仍正确。
- 24个关注logit有20个normal−zero为负、4个为正，16个zero概率≥0.98。不能把共同训练后截断整支造成的概率变化当作独立无GNN模型能力或架构bug，也不以3纠正/1破坏推断总体AP收益。1.0000和0.0000是打印舍入。
- 追踪源码：原reader→Resize224→ToTensor/Normalize；gallery另经过quality92 JPEG保存/解码。输入改变已确认，但原2080Ti/batch32与当前3060/batch1、依赖差异尚未隔离，不能单凭最大差将原因全归为JPEG。此前采用JPEG省解压的方案保留为工程诊断，不再代替原RGB案例解释。
- 已保存[完整日志整理版](results/evidence/pair44_feedback_qnjm_rb7/full_log_return_20260918.txt)、[24例结构化核验](results/evidence/pair44_feedback_qnjm_rb7/case_analysis_20260918.json)，更新[分析](results/seed44_feedback_diagnosis.md)。独立解析24条、核对唯一row/每类8例、重算干预计数与4项缓存翻转，与报告一致；没有重跑模型或改变正式指标。
- 下一步先 `ls -lah /home/featurize/data` 确认原图/压缩包状态；用户重启不自动视为数据已清空。原图在则直接复验这24张，ZIP在则优先只恢复72个RGB波段并持久化小缓存，路径/格式未确认不启动提取。不再全量验证，不直接修改图模型；恢复原输入并匹配缓存后再解释反馈和制作论文案例图。

---

## 2026-09-18 — 固定案例反馈干预结束：24个类别案例中纠正3个、破坏1个，逐张分数待回传

- 用户回传 `pair44_feedback_qnjm_rb7/diagnostic.log` 末尾，含 `FEEDBACK COMPLETE: .../diagnosis` 与 `DIAGNOSTIC_EXIT 0`；此前后台PID3145。本次程序正常结束；本地未直接访问服务器或独立检查完整run.json/PT/图像。
- 阈值0.5，类别11/14/16各8个案例。草地：反馈纠正1、破坏0、两种设置均对4/均错3；沙滩：2/1/3/2；湿地：0/0/4/4。各行分项之和等于8；合计24个类别案例，纠正3、破坏1、均对11、均错9。正常前向正确14、屏蔽反馈正确12。统计单位是图片×类别，不能未经去重就写成24张图片，也不作为总体准确率/AP/F1。
- 本次执行预定gallery命令以避免实例重启后的完整解压；使用JPEG案例，正常分数可能区别于原RGB缓存。当前末尾片段未含实际GPU、去重图片数、最大缓存概率差和逐张分数，仍待补齐，不能沿用此前实例GPU或本地22张准备检查数。
- 当前支持的结论：反馈在入选案例中有帮助也有损害；湿地没有0.5阈值对错翻转不代表分数或排序不受影响，4个均错案例也不能单凭汇总认定是漏判。选样按成败/标签分层，并非总体样本；共同训练模型的零反馈干预不能当训练消融或Swin对照，不能解释全部test AP退步。
- 已归档[计数与分析](results/seed44_feedback_diagnosis.md)和[规范化终端摘录](results/evidence/pair44_feedback_qnjm_rb7/terminal_excerpt_20260918.txt)。下一步仅 `tail -n 65` 读取现有日志，定位沙滩1个破坏/2个纠正、草地1个纠正以及湿地分数变化，补齐输入差异后再检查图像。真实图像/张量尚未本地检查，论文图可视化验收不勾选；未新增推理、训练、提交或推送。

---

## 2026-09-18 — 固定案例图反馈诊断已准备；实例重启后复用持久化JPEG，GPU结果待回传

- 用户同意继续固定验证案例诊断，并确认已重启实例。默认使用 `pair44_fullval_8Tvk9d/diagnosis` 中持久化的JPEG案例图，避免为本次小检查重新解压原始BigEarthNet。JPEG与原RGB输入不同，不能把其分数/图边当作原评估输入的重现；缓存正常分数、JPEG正常分数与屏蔽反馈分数分开保存。
- 研究问题：同一份GAViT44权重、同一张输入图，保留/屏蔽图反馈是否改变困难类别的预测。主要变量仅为整个 `feedback_proj` 的输出是否置零；固定权重干预用于论文案例讨论，不是重新训练的无GNN消融或Swin baseline，也不能据少量入选案例解释总体test AP下降。没有追加训练或更改阈值/正式结果。
- 新增 `tools/diagnose_bigearth_feedback.py`：每个类别11/14/16的真实标签×原成败分组各选已有第一条，最多24个类别案例，共用图片去重，选样不依赖干预结果。捕获正常前向真实token、region、assignment、图输入边、GAT输出与反馈；校验49 token/16区域/48空间边+32特征入边，记录GAT内部自环。绘图不用重建kNN，不把余弦选边分数或反馈范数当作attention/重要性。
- 运行时四项数值检查：捕获前后输出一致、真实中间量经过原classifier重建一致、屏蔽整个反馈投影后的输出等于原backbone token均值经过原classifier、移除hook后恢复输出；容差atol/rtol均1e-5。所有hook在finally中移除，失败保留产物并停止。真实模型与GPU尚未运行，不能将这些检查记作已通过。
- 复用既有配对工具严格加载GAViT checkpoint，检查metadata、权重与原诊断代码SHA256。每次建立全新 `work/pair44_feedback_*`，保留脚本、日志、退出码、来源/选样/环境、PNG/SVG/PT/JSON、HTML案例页与文本报告；不覆盖正式仓库和已有诊断。完成须有 `FEEDBACK COMPLETE`、`DIAGNOSTIC_EXIT 0` 和run.json complete。
- [可粘贴命令](docs/paste_bigearth_feedback_gallery.sh)已准备，整段复制即可在服务器保存脚本并后台启动，无工具包上传下载。粘贴代码有SHA256检查，打印包含绝对路径的tail命令，不依赖跨连接Shell变量。[运行说明与解释边界](docs/run_bigearth_feedback_diagnostic.md)。尚未在服务器执行。
- 本地11项新增行为测试和16项配对诊断回归测试通过，Python语法、CLI help与diff检查通过；Bash和zsh均实际验证粘贴代码原样恢复、后台包装器exit0以及已有 `pair44_diagnostic_RqV3th` 归档的准备流程（22张去重案例）。该检查仅用本地旧子集案例证明无原始数据/Torch时的准备路径可行，不代表服务器完整val案例数或GPU前向。当前本地没有Torch，后续以服务器真实数值核验为准。没有提交或推送。

---

## 2026-09-17 — 完整val缓存分组完成：内陆组为共同短板，不支持GAViT更依赖海洋背景的简单解释

- 用户回传服务器内联分析完整输出：`RUN complete N 122342 AP_CHECKS 38`、19类per_class.csv和沙滩/海岸湿地水体分组。沿用现有seed44权重、原val、阈值0.5，无新模型推理或训练。
- 从19类结果独立汇总：Swin mAP78.578229%、macro-F1 71.631856%、micro-F1 77.324873%；GAViT78.517128%/72.268942%/77.765520%，差值−0.061101/+0.637086/+0.440647 pp。GAViT AP为10升9降；主要指标接近且略低，不用次要F1改善替代mAP结论。正式test和三seed统计不变。
- 四个互斥水体标签组样本数：仅海洋13998、仅内陆15834、两者都有66、两者都无92444，共122342。仅内陆组沙滩正例15，两模型均命中1个（6.7%），AP8.0000/10.4114%；海岸湿地正例37，命中3/5个（8.1%/13.5%），AP9.8635/21.7833%。两模型均存在低召回，但GAViT组内AP更高，不能用这份val分组支持GAViT更依赖海洋背景的简单解释。
- 仅海洋组沙滩正例354，命中296/271，AP85.8322/86.2094%；海岸湿地正例341，命中151/134，AP56.9266/60.5463%。GAViT组内排序较好，但0.5下命中更少。沙滩总体额外22个漏判分解为仅海洋+25、两者都无−3；海岸湿地总体额外17个漏判分解为仅海洋+17、仅内陆−2、两者都有+1、两者都无+1。属于阈值计数分解，不等于AP差值分解。
- GAViT完整val草地多命中69、也多误报117，AP−1.3311但F1+0.8466 pp；沙滩/海岸湿地则少命中22/17、少误报80/47。不能把所有困难类概括为分数普遍被压低；调阈值不能改变既有分数AP。
- 组间类别比例不同，不横向用AP高低认定背景因果；总体AP不是组AP的加权平均。仅内陆沙滩15例、湿地37例仍限制结论。此前val/test共现差异保持成立，但缺少test逐图预测，海岸湿地test额外−12.5 pp仍未解释。
- [原始回传与提取CSV](results/evidence/pair44_fullval_8Tvk9d/context_return_20260917.txt)已归档；[结构化分析](results/evidence/pair44_fullval_8Tvk9d/context_analysis_20260917.json)记录SHA256、整体和分组数据。核对19类支持数、38组precision/recall/F1及混淆计数、四组可加和关系与共现边际均通过；all AP与per_class打印精度一致。本地未收到全量逐图分数/run.json，未独立重算本次AP排序。
- [完整分析](results/seed44_fullval_context_analysis.md)可用于论文失败讨论。标签统计→抽样→完整val→缓存分组阶段已完成；若继续解释图分支，下一步固定验证案例捕获真实区域/边/反馈，必要时作固定权重反馈屏蔽干预，不能冒充训练消融。该阶段未实施，不据此新增完整训练；没有提交或推送。

---

## 2026-09-17 — 完整val配对诊断完成：草地−1.33、沙滩−0.39、海岸湿地+2.09 pp

- 用户回传完整val报告和`DIAGNOSTIC COMPLETE: /home/featurize/work/pair44_fullval_8Tvk9d/diagnosis`，Shell显示PID5173 done。均匀验证样本122342、补充0、阈值0.5；结合此前日志可确认两模型加载、8张检查、全量前向及报告生成完成。使用现有Swin44/GAViT44权重，本次没有训练。
- 完整val自然草地正例2687，Swin AP57.91%/GAViT56.58%（−1.33 pp）；沙滩426，76.13%/75.74%（−0.39）；海岸湿地610，44.53%/46.62%（+2.09）。三组打印差值独立算术核对通过，支持数与此前完整split统计一致。完整预测/run.json未在本地收到，不能声称已重算此次AP。
- 草地由4096张抽样的+0.85转为完整val的−1.33 pp，与test−3.7同方向；需要修正原抽样推断。海岸湿地val+2.09与test−12.5方向翻转仍存在，值得检查背景组合，但不能直接归因图模块。沙滩两划分均为负，val差距小于test；不依据不同样本分布的AP差作因果判断。
- 本次全19类与整体mAP/F1尚待回传，不用训练metadata最佳val值代替，也不改正式test/三seed结果。已保存[完成记录](results/evidence/pair44_fullval_8Tvk9d/completion_report_20260917.json)，更新[诊断分析](results/seed44_validation_diagnosis.md)、效果摘要和排期。
- 已准备[服务器缓存分组命令](docs/bigearth_cached_context_analysis.md)：只读现有predictions.csv，按真实水体标签四个互斥组统计沙滩/海岸湿地的支持数、AP、TP/FN/FP，同时打印完整per_class.csv以补齐整体指标。无模型/RGB/GPU计算，不需上传工具包。使用既有标准库指标函数，在本地4096缓存验证组计数/混淆计数可加和、all指标与已验证结果一致，并通过Bash/zsh语法核验；完整val场景分组尚待用户在服务器执行。

---

## 2026-09-17 — 完整val配对前向已到122342/122342，等待指标与归档完成证据

- 用户回传`pair44_fullval_8Tvk9d`日志：train/val/test为237871/122342/119825，manifest为122342张uniform；所选RGB路径检查通过，RTX2080 Ti、torch2.2.2+cu121、CUDA12.1。Swin44/GAViT44均加载，8张SMOKE PASS，完整配对前向到122342/122342。此前启动PID为5173，未读取服务器当前进程状态。
- 最后打印配对吞吐205.84 img/s，只对应推理阶段；本次文本没有最终AP/F1、`DIAGNOSTIC COMPLETE`或run.json，不能把前向完成记作全部诊断完成，也不能据此认定卡死或异常退出。
- 当前源码在前向后依次保存两份分数JSON、与sklearn核对AP、生成指标/PR/预测CSV、导出图片并压缩，期间没有阶段进度打印。完整任务结束以run.json的complete及最终日志为准。
- 原始文本已按字节保存到[回传日志](results/evidence/pair44_fullval_8Tvk9d/inference_log_return_20260917.txt)，SHA256及状态摘要见[状态证据](results/evidence/pair44_fullval_8Tvk9d/inference_status_20260917.json)。下一步只检查PID命令行/CPU、产物及日志，不重复启动推理；正式结果表没有添加未回传指标。

---

## 2026-09-17 — 完整val启动命令停在Shell引号解析，改为无引号逐行命令

- 用户回传终端提示`then dquote>`；当前if块仍在等待闭合双引号，不能记作评估已启动，也不是Python或CUDA报错。仅凭聊天中的转义文本不能定位实际缺失的引号字符。
- 指导先Ctrl+C取消当前未提交命令，再创建新的输出目录，使用无引号、无if、无反斜杠续行的单行nohup命令；评估参数不变。目录变量采用`${pair_full:?}`，为空则停止，避免错误落到根目录。
- [运行指南](docs/run_bigearth_pair_diagnostic.md)同步更新。Bash/zsh语法及空目录变量阻止执行已本地核验；未启动服务器任务，仍等待`SMOKE PASS`及实际推理进度证据。没有修改模型、训练代码或已有结果。

---

## 2026-09-16 — 用户同意完整val配对评估，准备复用现有入口的服务器命令

- 根据上次诊断中少数类和背景组合覆盖不足，用户同意先用现有两份seed44最佳权重评估完整val，目的为背景相关性/失败分组分析，不是新增训练或更改test指标。
- 已在[运行指南](docs/run_bigearth_pair_diagnostic.md)补充直接使用服务器现有工具的命令：`--samples 122342 --min-focus-support 0`，其余保持batch32/workers2/pin_memory=False、原split、预处理和阈值0.5；抽样参数seed42不改变checkpoint训练seed44。
- 创建新`pair44_fullval_XXXXXX`目录；先检查清单RGB文件并做8张前向，随后完整val推理和分数归档。直接nohup调用Python，以日志`DIAGNOSTIC COMPLETE`及run.json的complete为完成证据，不期待旧包装器专属的diagnostic.exit。
- 继续复用当前已部署工具，无须上传新包。先回传report.md与完成标记文本，后续从服务器predictions.csv进行标签场景分组。本地参数/选样/语法核验完成；尚未启动服务器任务或获得完整val新指标。

---

## 2026-09-16 — Seed44配对验证诊断成功：小样本未重现test退步，发现标签共现差异

- 用户提供 `/Users/patrick/Downloads/results/`。运行实际发生于09-15，服务器身份 `pair44_diagnostic_RqV3th`，`run.json`为complete；8张前向和4181张配对推理成功，模型训练seed44，诊断抽样seed42。此前失败运行的数据缺失，用户已确认是当时尚未解压数据，不再作为当前阻塞项。
- 服务器项目 `a23a5663e7bef9749143e9bb91932edaea2884bd`，工作区干净；Swin/GAViT运行时checkpoint SHA256匹配此前值。设备RTX2080 Ti、torch2.2.2+cu121；batch32/workers2/pin_memory=False；8张smoke约1.98秒，4181张配对推理循环约17.30秒，配对CUDA allocated峰值752,484,352字节。不能把它当成全命令耗时或单模型效率。
- 原split 237871/122342/119825。均匀val4096张、独立补充85张；阈值0.5。仅均匀组指标：Swin mAP79.3453/macro-F1 72.4939/micro-F1 77.4310，GAViT79.6674/73.4437/77.9789，单位%。不替换正式test或三seed统计。
- 自然草地正例102，AP64.85→65.70（+0.85 pp）；沙滩14，79.13→77.31（−1.82）；海岸湿地25，53.53→54.97（+1.45）。本次小样本未重现test上海岸湿地−12.5、草地−3.7 pp，不能据此否定原test结果。阈值0.5下，两模型草地均漏56张，沙滩均漏4张，海岸湿地漏15/16张；共同困难以漏判为主。
- 完整split标签统计：沙滩正例同时带海洋标签的比例train/val/test=87.5%/83.1%/11.8%，带内陆水体=1.2%/3.5%/84.2%；海岸湿地+海洋=57.3%/58.4%/25.6%，+内陆水体=6.1%/8.5%/46.2%。分母为各split的目标类正例，多标签组可重叠。这是标签共现差异的直接证据，背景依赖及其是否解释GAViT退步仍是假设。
- 离线分组发现随机子集14张沙滩正例全带海洋标签，内陆水体组合为0。含补充组后，沙滩+海洋57张，Swin/GAViT命中45/42；不带海洋7张，命中2/1。这些含补充组计数仅作案例描述，样本少且选择概率不同，不当作总体召回率。
- 已逐行核对4181条manifest/CSV/JSON预测；独立重算38项AP和两模型19类混淆计数/F1，均一致；41条成败案例分组、分数与图片存在性通过核验，人工查看其中10张。59个原始返回文件按字节归档并记录SHA256，见[证据](results/evidence/pair44_diagnostic_RqV3th/analysis.json)；分析和下一步见[验证诊断报告](results/seed44_validation_diagnosis.md)。
- 论文用途：稳定性、数据分布和失败案例讨论。阶段1标签统计、阶段2配对预测/成败图已完成；真实区域/图边/反馈尚未捕获。建议先用原权重补完整val配对缓存及按场景分组的分析，再决定结构消融；本轮没有启动服务器任务/训练、修改模型或提交/推送。

---

## 2026-09-15 — 诊断返回：标签统计完成，首张抽样图缺失导致推理前退出

- 用户提供本地返回目录 `/Users/patrick/Downloads/pair44_diagnostic_6Rcwll/`。`diagnostic_exit=1`，`failure.json`为FileNotFoundError，日志在 `validate_image_paths` 停止；尚未导入Torch、打印设备、加载权重或执行8张smoke，不能作为模型效果分析结果。
- 服务器项目 `a23a5663e7bef9749143e9bb91932edaea2884bd`，工作区干净。原train/val/test CSV为237871/122342/119825行；固定清单4181张=4096随机+85补充，清单SHA256与preflight一致。
- 第一条抽样记录是val数据行53。检查路径 `/home/featurize/data/BigEarthNet-S2/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_42_71/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_42_71_B04.tif` 时未找到文件。现有证据不能区分实例数据被清空、目录位置不同、解压不完整或单文件缺失；不猜测原因，不跳过该样本或改变划分。
- 已完成的标签统计仍可用。自然草地正例train/val/test=6984/2687/2211，沙滩=738/426/152，海岸湿地=670/610/117。原val与test中海岸湿地占比约0.499%/0.098%，沙滩约0.348%/0.127%；可确认它们在当前划分中为低支持类别，但不能把类别稀少直接认定为GAViT相对退步的原因。
- 随机4096张中三类正例分别102/14/25；独立补充85张分别包含3/50/39个正例（多标签可重叠）。当前没有预测分数、PR或错误样本，无法区分漏判与误报。
- 原始日志、退出码、标签统计、配置与清单已按字节校验复制到 [返回证据](results/evidence/pair44_diagnostic_6Rcwll/analysis.json)。下一步仅需服务器终端检查原始数据根目录与缺失patch，输出文本即可。用户已明确不希望反复上传/下载工具包；本轮没有修改诊断代码、制作新包或提交/推送。

---

## 2026-09-15 — 配对诊断工具交付，等待服务器8张前向检查

- 用户确认实现现有Swin44/GAViT44权重诊断。已新增 `tools/diagnose_bigearth_pair.py`、标准库分析模块 `tools/bigearth_pair_core.py`、启动器 `jobs/run_bigearth_pair_seed44.sh`，使用服务器当前项目中的模型与数据读取代码；没有启动服务器任务或训练。
- 固定val随机样本4096、抽样seed42；自然草地/沙滩/海岸湿地的正负标签不足64时单独补充，补充组不进入随机子集AP/PR。保存全19维预测、真实标签、行号、路径、cohort与输入来源；固定阈值0.5，输出逐类AP/F1/precision/recall/TP/FP/FN/TN、类别支持数及共现。缺少正例或负例时AP标为N/A。
- 两模型共享同一输入批次，batch32、workers2、pin_memory=False。先自动检查8张的加载、前向与有限19维输出，再继续完整清单；构造模型不下载预训练权重，加载现有checkpoint/sidecar。真实运行时用sklearn复核AP，保存真实设备、代码摘要、配对吞吐及显存；这些资源数字不当作单模型效率比较。
- 三个目标类按四种成败关系及真实正/负标签分别选样，每组最多2个，固定哈希顺序，不按最大预测分差挑图。产出 `cases.html`/图片、全部计数、预测缓存及 `results.zip`。本次实现预测与成败样本；真实区域、实际图边与反馈幅度仍是后续诊断范围。
- 本地16项标准库行为测试通过：CSV标签/重复路径、抽样复现、补充组隔离、预测错位/异常值、AP并列分数、成败方向、报告、HTML转义、所选文件预检、失败退出与完成归档。压缩包解包后16项再次通过，CLI帮助、Python编译和Bash语法检查通过。本地无PyTorch/NumPy，真实权重前向及图片解码待Featurize内置8张检查，不宣称GPU验证完成。
- [运行指南](docs/run_bigearth_pair_diagnostic.md)与 [独立交付包](deliverables/gavit-pair-diagnostic-20260915.zip)已准备；包20794字节，SHA256 `b01d9eed51bc2cd2f2d009d0e6dcf6219db90a14700f4af2110eae468b4ba884`。输出使用新的 `pair44_diagnostic_XXXXXX` 目录并记录退出码，失败保留原产物；没有提交或推送。

---

## 2026-09-15 — 效果分析复核与现有权重诊断范围

- 独立使用Python标准库复核19类名称/顺序、两份seed44 AP、差值及mAP贡献、seed42/44逐类对照、三seed均值与样本标准差，均通过；比较表中非seed44实验行与本轮修改前Git版本一致。三类净下降贡献87.05%，三seed配对mAP均值+0.1667 pp、配对差值样本SD1.2097 pp，仍不支持稳定增益。
- 当前源码中的attentive_spatial学习区域内权重，区域归属为固定网格。7×7 token分入4×4区域得到9个4-token区域、6个2-token区域、1个1-token区域；按原坐标公式独立计算总数49。token_feedback复制到所属token再平均，图反馈部分等于`sum_k(n_k/49 * region_feedback_k)`。区域汇总系数不同是源码推导结果，不是已证实的失败原因；需结合真实反馈向量检查。
- 可视化入口仍以NWPU ImageFolder读取数据，且诊断路径直接构造kNN边，不能直接解释本次BigEarthNet sparse-hybrid结果。后续应从实际前向捕获区域和graph_reasoning输入边，并校验捕获不改变logits；余弦选边值不等于GAT注意力。
- [效果分析与诊断范围](results/seed44_followup.md)已补充：CSV标签支持数→验证集固定样本配对预测→真实图和反馈。先8张前向校验，再处理4096张均匀验证子集；补充困难类样本独立标记，不冒充整体指标。保留双向分歧及共同成败样本，后续结构判断使用验证集。
- 本轮只完成离线统计、代码检查和文档，没有新增诊断入口、运行服务器推理或训练，也没有提交/推送。原始结果与历史边界继续保留，元数据差异不作为模型效果分析的前置项。

---

## 2026-09-14 — 完整逐类分析：三类贡献87%的净下降，海岸湿地与沙滩收益跨运行翻转

- 用户回传GAViT44完整19类AP、test_exit=0，并要求将重点转向模型效果，元数据差异只留档。两份已回传测试输出对齐后：7类上升、10类下降、2类持平；主要指标仍为mAP。
- Coastal wetlands −12.5 pp、Natural grassland/sparsely vegetated areas −3.7 pp、Beaches/dunes/sands −3.3 pp；按AP差值/19，三类合计贡献−1.0263 pp，约占19类打印AP净差值−1.1789 pp的87.1%。完整AP均值与汇总mAP差−1.2 pp在舍入范围内一致，不改变正式指标或删除类别。
- 对照seed42：海岸湿地Swin30.3/GAViT34.7（+4.4 pp）变为seed44的30.5/18.0（−12.5 pp）；沙滩Swin13.3/GAViT16.8（+3.5 pp）变为13.0/9.7（−3.3 pp）。这两类增益方向不稳定，差异同时包含硬件/执行变化，不能解释为纯seed效应。
- 自然草地GAViT42.1→41.8，Swin42.9→45.5；该类配对差距扩大主要来自baseline改善，两次运行均未见图模块收益。seed43完整逐类指标尚缺，因此只报告42/44的已知类别趋势。
- 完整[seed44逐类表](results/bigearth_seed44_per_class_comparison.csv)、[42/44对照](results/bigearth_seed42_seed44_class_stability.csv)和[分析](results/seed44_followup.md)已整理。下一步优先使用现有权重，在验证集固定样本上比较预测分数、支持数、PR行为、真实分组和实际sparse-hybrid边，检验困难样本退化的表征/反馈假设；本轮未修改训练代码或运行新的服务器任务。
- 记录补充：Swin44 preflight为RTX3080、Python3.11.8、torch2.2.2、CUDA12.1；最后回传验证到epoch27（Loss0.0268、mAP75.7%、F1 71.0%），train.exit缺失，不能认定正常完成30轮。两权重sha256sum匹配当前metadata。GAViT当前best17/78.5171与旧日记19/77.9的差异已存入[回传证据](results/evidence/seed44_reconciliation_20260914.json)，不继续追查；原历史最终轮次/资源保留注明来源，三seed描述统计不变。

---

## 2026-09-14 — Seed44 配对后续：先补原始日志与逐类差值

- 用户要求继续下一步。先基于已归档 Swin44 完整 AP 与 09-10 日记中的 GAViT44 六类 AP 做只读分析；没有启动新训练或测试。
- 已记录的 Coastal wetlands 为 Swin30.5%/GAViT18.0%，差值 −12.5 pp；Beaches/dunes/sands 为13.0%/9.7%，差值 −3.3 pp。按19类等权 macro mAP，分别贡献约 −0.6579/−0.1737 pp，合计 −0.8316 pp。
- 这只是已打印 AP 的算术分解，GAViT 六类来源是历史日记摘录，待完整原始 test.log 复核；其余13类尚缺，不能据此归因于图结构、训练提前终止或硬件，也不能删除低分类别改变正式指标。
- 已整理 [后续分析与服务器取证命令](results/seed44_followup.md)。下一步回传 Swin44 训练末尾与退出记录、历史 preflight、GAViT44 完整测试块及 sidecar、两个权重 SHA256，再更新完整19类配对分析和缺失的训练字段。
- 命令采用不含引号、Shell变量或heredoc的只读形式，避免此前复制时 dquote 问题。seed45 继续保留排期待执行状态。

---

## 2026-09-14 — Swin44 最佳权重测试通过；三 seed 的 mAP 平均配对增益缩小至 +0.17 pp

### 实验身份与证据

- 本条为用户回传结果的归档日期；实际测试日期待服务器时间戳确认。run_stage formal，run_tag rtx3080_seed44_20260910，seed44；metadata 记录 Git a23a566、dirty=false。用户回传当前 Git 为 detached HEAD，符合启动器固定 commit 的安排。
- 最佳 checkpoint：/home/featurize/work/swin44_3080_20260910/swin.pth；sidecar 为 swin.meta.json。metadata 声明 SHA256：36fb172cd01eddde6d343a4de20b1d15e8b5ec50172401895bbb33c7b4b50319；本地无权重实体，未独立重算哈希。
- 模型：Swin-T global pooling baseline，swin_tiny_patch4_window7_224，dropout0.1，19类；参数量 27,535,501 沿用同结构 baseline 的既有记录，本次未回传参数统计。预训练来源按启动器为 bigearth_files/model.safetensors。
- 数据：BigEarthNet-19 RGB，原固定 train/val/test 为 237871/122342/119825；224×224，训练水平/垂直随机翻转，ImageNet normalization。完整文件和 split 哈希为既有 preflight 范围，本轮未重新回传。
- 优化配置：30 epochs、batch32、AdamW lr3e-4/wd1e-4、CosineAnnealingLR、BCEWithLogitsLoss、F1 threshold0.5。训练命令按 [Swin44 启动器](docs/run_swin44_3080.md)；metadata 证实配置，完整实际训练命令与结束状态仍需 train.log/train.exit 补齐。
- 最佳验证 mAP **78.5782%，epoch9**；sidecar 保存时间 2026-09-10T15:27:44+00:00。最终完成 epoch、最终验证指标及 train_exit 尚未在本轮输出中提供，因此比较表的 Epochs（实际完成轮数）和 Final_Val_mAP 留空；不把配置中的30轮当作完成证据。

### 测试命令、结果与资源

```bash
python -u test_bigearth.py --model swin --data_dir bigearth_files/splits --ckpt /home/featurize/work/swin44_3080_20260910/swin.pth --batch_size 32 --seed 44
```

- 用户通过 nohup /usr/bin/time -v 包装上述测试，日志按本会话命令追加到 /home/featurize/work/swin44_3080_20260910/test_direct.log；两次较早的复制尝试停在 Shell 引号解析阶段，没有作为成功测试记录。
- **3745/3745 批完成；Exit status: 0；Test mAP 71.4%，macro-F1 65.7%，micro-F1 76.7%。** 测试使用 epoch9 最佳权重；pin_memory=False 已由用户回传代码行确认。
- 推理循环 7:01、8.88 batch/s；整个命令 wall time **7:55.63**（475.63秒）。/usr/bin/time maximum RSS **1,145,340 KiB**，不能当作整组进程总内存或 GPU 峰值显存。
- 训练按启动器计划使用 RTX3080，实际训练/测试 GPU 与软件版本未随本次输出回传；峰值 GPU 显存、训练耗时、每轮耗时、费用待确认。本次计时不支持与其他设备直接比较模型效率。
- 19类 AP 已保存到 [Swin44 per-class AP](results/bigearth_swin_seed44_per_class_ap.csv)，metadata 和完整回传指标/资源字段见 [回传证据](results/evidence/swin_seed44_20260914.md)。

### 同 seed 配对与三个 seed 的阶段性统计

| Seed | Swin mAP | GAViT mAP | 差值 pp | macro-F1 差值 pp | micro-F1 差值 pp |
|---|---:|---:|---:|---:|---:|
| 42 | 70.9 | 72.0 | +1.1 | +1.8 | +0.1 |
| 43 | 71.1 | 71.7 | +0.6 | +1.6 | +0.8 |
| 44 | 71.4 | 70.2 | -1.2 | -0.5 | +0.4 |

差值统一为 GAViT − Swin。seed44 的 mAP/macro-F1 下降，micro-F1 上升，不能用次要指标掩盖主要指标的负面结果。

| 模型（seeds42/43/44） | Test mAP % | macro-F1 % | micro-F1 % |
|---|---:|---:|---:|
| Swin-T | 71.13 ± 0.25 | 64.67 ± 0.90 | 76.43 ± 0.38 |
| Sparse-hybrid GAViT | 71.30 ± 0.96 | 65.63 ± 0.40 | 76.87 ± 0.21 |

- 统计基于各次测试打印的一位小数，n=3；± 为样本标准差（分母 n−1），不是标准误或置信区间。结构化结果见 [阶段统计表](results/bigearth_multiseed_summary.csv)，原始实验行保留于 results/bigearth_comparison.csv。
- 平均配对差值：mAP +0.17 pp（差值样本SD 1.21 pp），macro-F1 +0.97 pp，micro-F1 +0.43 pp。mAP 两个 seed 为正、一个为负，平均差距较小；当前证据不能支持稳定提升或额外复杂度已被证明值得。
- 这些是描述性阶段统计：GAViT43 实际到epoch21，GAViT44 到epoch26；Swin44 训练结束证据待补。各 seed 硬件/执行条件有差异，不能写成六次完全同协议30轮训练，也不能把SD解释为纯seed随机性或显著性检验。
- 论文用途：多seed核心对比、稳定性和失败案例讨论。记录负面结果，不依据seed44分数事后改变指标或挑选规则。
- 下一步优先补齐 Swin44 训练末尾/退出记录、硬件信息和完整 GAViT44 per-class AP，利用现有权重及日志分析差异；seed45保留原排期状态，本次没有启动额外训练。

---

## 2026-09-10 — GAViT44（seed44，RTX 3080）正式完成：best epoch 19，Test mAP 70.2%

> 2026-09-14 核对提示：本条best epoch19/77.9%与当前服务器metadata epoch17/78.5171%冲突。本条保留为历史原始记录，其最终轮次、资源与当前checkpoint/test.log的对应关系待核实；当前状态以本日顶部核对条目为准。

**实验身份**

- 日期：训练 2026-09-09 06:52 UTC 启动；测试 2026-09-10。run_stage formal，run_tag `rtx3080_seed44_20260909`，seed 44。
- Git commit：`36345f7d5ffccf714992120abe8e4f6b52d438c7`（pin_memory 修复版；服务器 git pull 后 preflight 校验 commit 一致、工作区干净）。
- 完整命令：`python train_bigearth.py --model gavit --data_dir bigearth_files/splits --epochs 30 --batch_size 32 --lr 3e-4 --seed 44 --num_regions 16 --grouping attentive_spatial --edge_type sparse_hybrid --knn_k 5 --gat_hidden 256 --gat_heads 4 --gat_layers 2 --integration token_feedback --dropout 0.1 --run_stage formal --run_tag rtx3080_seed44_20260909 --pretrained_path bigearth_files/model.safetensors --checkpoint_path /home/featurize/work/gavit44_3080_20260909/gavit.pth`（经 runpy 包装器 nohup 启动，见 [docs/run_gavit44_3080.md](docs/run_gavit44_3080.md)）。

**数据**：BigEarthNet-19 RGB；train 237871 / val 122342 / test 119825，三个 split 的 SHA256 与 preflight 记录一致；预训练权重 SHA256 `f7e995…a9d36`；标准增强与阈值 0.5。

**模型**：GAViT v2 sparse-hybrid：K16、attentive_spatial、sparse_hybrid（4n top2）、GAT hidden256/heads4/layers2、token_feedback、dropout 0.1；参数 31.37M；Swin-T 骨干全量微调（freeze_backbone=False）。

**优化**：计划 30 epochs；AdamW lr 3e-4、wd 1e-4；CosineAnnealingLR；BCEWithLogitsLoss；实际完成 26 epoch 后用户手动早停（见"解释"）。

**资源**：RTX 3080（10GB，CUDA 12.1，cuDNN 8902），实例 24GB RAM；训练中观测显存 4272MiB（~4.2GiB）；吞吐 5.4~6.9 it/s；每轮约 19~23 分钟训练 + ~7 分钟验证 ≈ 24 分钟/轮；26 轮约 10.4 小时（待按日志时间戳精确核算）；费用待确认。`memavail.log` 全程稳定 ~21GB——pin_memory 修复后 26 次全量验证无任何内存压力，OOM 根因闭环。

**输出**

- 最佳轮 epoch 19：Val mAP 77.9%，Val macro-F1 72.3%；best checkpoint `gavit.pth` + sidecar `gavit.meta.json`。
- 最终完成轮 epoch 26：Val mAP 76.9%，Val macro-F1 71.9%（按规则同时记录，不隐藏后期过拟合）。
- 测试（best checkpoint，test_exit=0，3745 批 5:20 @11.69 it/s，架构自 sidecar 重建）：**Test mAP 70.2%，macro-F1 65.2%，micro-F1 77.1%**。per-class AP 已记录于 test.log——最低：Beaches/dunes/sands 9.7%、Coastal wetlands 18.0%、Natural grassland 41.8%、Industrial/commercial 48.7%；最高：Marine waters 99.7%、Coniferous forest 93.8%。
- 产物（Featurize 持久盘）：`/home/featurize/work/gavit44_3080_20260909/{train.log,test.log,memavail.log,train.exit,test.exit,gavit.pth,gavit.meta.json}`。

**解释**

- 早停决策：epoch 19 达峰后 Val mAP 缓慢回落（77.9→76.9，约 1 点），用户据此在 epoch 26 完成后手动停训；随后用户在另一台服务器完整重跑全流程，后段轮次指标持续降低，二次独立确认过拟合形态（该重跑为用户自发验证，未纳入正式记录）。best-val-mAP 挑选规则未变，不属于事后改指标。
- val→test 落差（77.9→70.2，约 7.7 点）与历史 run 一致（seed42/43 同样 val 系统性高于 test），属数据集固有现象，不是异常。
- 2080 Ti 的 epoch1 失败段与第二次服务器重跑均不计入任何论文数字；pin_memory 修复为资源层改动，论文按工程修复披露。
- 待办：同一块 3080 上执行 Swin44 配对（启动器 [docs/run_swin44_3080.md](docs/run_swin44_3080.md)），完成后更新对比表并计算 3+ seed 均值/方差。

---

## 2026-09-09 — run4 单变量实验 PASS；pin_memory 修复提交并推送（36345f7）

- run4（仅验证 loader `pin_memory=True→False`）结果：主进程 RSS 全程平稳（~1.66GB，对比 run3 同期已 6.9GB 并线性冲向 23.3GB），MemAvailable 保持 ~22GB，全量 122342 张验证图与原指标计算完成，`diagnostic_exit=0`，`FULL VALIDATION DIAGNOSTIC PASS`。根因实锤：评估 loader 的 pin_memory 锁页主机内存池在小内存实例上耗尽 RAM。
- 修复（hotfix，commit `36345f7d5ffccf714992120abe8e4f6b52d438c7`，分支 codex/sparse-hybrid-4n-top2，已推送 origin）：`train_bigearth.py` 验证 loader 与 `test_bigearth.py` 测试 loader 均改为 `pin_memory=False`（test 集 119825 图同路径同风险）；训练 loader 保持 `pin_memory=True`（epoch1 全程 7434 批 RSS 采样仅 1.9GB，证明训练路径无此问题，单变量原则不动）。纯资源改动，数据、模型、指标、评估顺序不变。
- 本地验证：`py_compile` 通过；`python3 -m unittest tests.test_resource_probe tests.test_training_profile tests.test_experiment_identity tests.test_sparse_hybrid_static_contract` 37 项全部通过。
- GAViT44 正式重启启动器已整理为 [docs/run_gavit44_3080.md](docs/run_gavit44_3080.md)：git pull 拉取修复、preflight 校验新 commit/3080/权重/全量 RGB、独立输出目录 `gavit44_3080_20260909`、退出码落盘 `train.exit`、主机可用内存每 30 秒记录 `memavail.log`、nohup 后台。训练参数与 2080 Ti 失败尝试完全一致（seed44、30 epochs、batch32、lr3e-4、K16 attentive_spatial、sparse_hybrid、token_feedback），run_tag `rtx3080_seed44_20260909`。
- 预计耗时：每轮约 35~40 分钟（训练段 29.5 分钟 + 修复后验证约 7 分钟），30 轮约 18~20 小时。费用与时长待运行记录。
- 披露事项：GAViT44 训练硬件为 RTX 3080（与 Swin44 配对时也用同一块 3080）；2080 Ti 的 epoch1 失败段不计入任何结果；pin_memory 修复属资源层改动，论文中按工程修复说明。
- 本地 research_diary.md 与 docs/run_gavit44_3080.md 暂未提交（避免改变服务器 pull 目标 commit；训练启动后再提交推送）。

---

## 2026-09-09 — 静默中断根因确认：验证阶段主机内存耗尽触发 OOM killer

- 三阶段诊断链条：短验证（70图）通过 → 训练→验证转换（3+3批）通过 → 全量验证（122342图）三次尝试。第一次在 2080 Ti 精确匹配处主动停止（用户已换卡 3080）；第二次前台运行整组进程消失且无 exit_status，判定网页终端 SIGHUP；第三次 nohup 后台仍整组消失，uptime 2h 排除实例重启。
- 第四次（内存监控版，run3）拿到决定性证据：`diagnostic_exit=137`（SIGKILL），且仅 python 主进程被杀、bash/tee/awk 存活写下退出码——OOM killer 单杀最肥进程的典型签名。内核计数实锤：`system.slice/memory.events: oom_kill=4`、`apphub.service: oom_kill=1`。
- 内存轨迹：验证开始后主进程 RSS 以约 8~9MB/batch 线性增长，2.3GB→6.9GB→13.6GB→19.9GB→23.3GB（4.5 分钟），MemAvailable 跌到 ~50MiB 后于 batch 3369/3824 被杀。DataLoader 子进程稳定在 5.2GB。
- 代码排查（commit bb9f1cf）：`evaluate` 在 `torch.no_grad()` 下每批仅保留 `logits.cpu().numpy()`（~2.4KB），数据集读图在 worker 内无缓存——主进程没有与 8~9MB/batch 匹配的逻辑对象。每批 float32 图像 tensor 为 19.25MB，经 `pin_memory=True` 锁页内存池分配；锁页内存不可被内核回收，与"avail 归零且页面缓存无法挽救"的现象一致。当前主嫌疑：PyTorch 2.2.2 锁页主机内存池在该负载下复用失效。
- 死亡史统一解释：Swin43 在 4090 实例（大内存）30 轮无恙；GAViT43 停于 epoch 21、GAViT44 停于 epoch1 训练段结束，均发生在验证窗口——三次静默中断同为验证期主机 OOM，不是平台故障、不是代码逻辑错误。数据集位于 /home/featurize/data 高速盘，位置无问题。
- 进行中：run4 单变量验证实验——同一诊断仅将验证 loader `pin_memory=True→False`（训练 loader 不变），内存监控保留。若 RSS 平稳且 PASS 则实锤 pin_memory 锁页池；修复方案为正式训练代码验证 loader 关闭 pin_memory（纯资源改动，不改变数据/模型/指标语义），是否改动冻结训练代码由用户决定并记录。
- 硬件/环境记录：RTX 3080（9874MiB，10GB 版）、24GB RAM、torch2.2.2+cu121、torch-geometric 2.8.0（WITH_PYG_LIB/WITH_TORCH_SCATTER/WITH_SOFTMAX 均为 False）。诊断脚本本地备份：`/tmp/gavit44-fullval.KTrcwi/`（bg3=内存监控版，bg4=pin_memory=False 单变量版）。

---

## 2026-09-09 — 用户主动换卡 RTX 3080；全量验证诊断重新定向

- 第三阶段全量验证诊断首次执行在 GPU 身份检查处主动停止（`diagnostic_exit=1`）：脚本要求精确匹配失败现场的 `NVIDIA GeForce RTX 2080 Ti`，当前实例已不是该卡。脚本按设计拒绝了在意外硬件上继续，不是训练/验证路径出错。
- 用户确认是有意将实例换成 RTX 3080（原 2080 Ti 实例不再使用；GAViT44 失败运行本就没有可恢复 checkpoint，无产物损失）。
- 影响与披露：诊断目标从"在失败现场同款硬件上复现"变为"验证全量验证代码路径在规模下可完成"；在 3080 上通过不能复现或解释 2080 Ti 上的静默退出，只能作为代码路径无全量缺陷的旁证。
- seed44 配对安排随之确定：GAViT44 与 Swin44 均在同一 RTX 3080 上执行（GAViT44 需重跑），论文披露硬件为 3080；不混用 2080 Ti 的训练段数据。
- 诊断脚本已更新：GPU 打印（名称/显存/CUDA/cuDNN）前移到校验之前，校验改为 `endswith('RTX 3080')`；其余抽取逻辑、样本边界、看门狗与 20 分钟上限不变。本地 bash -n 与内嵌 Python AST 检查通过。
- 待办：用户在新实例重新执行 `/tmp/gavit44-fullval.KTrcwi/diagnose_gavit44_fullval.sh`，回传 `diagnostic.log` 与 `exit_status.txt`；脚本仍只读旧 2080 Ti 目录的 `gavit.meta.json` 作身份参照（Featurize work 目录跨实例持久），不修改其中任何文件。

---

## 2026-09-09 — GAViT44首轮训练段后异常中断（原因待查，不重跑）

- 用户已同意第三阶段全量验证诊断，20分钟预算。已准备 `/tmp/gavit44-fullval.KTrcwi/diagnose_gavit44_fullval.sh`，沿用刚通过的3批短训练（79图，原变换/AdamW/Cosine和训练片段），仅把验证范围扩大为全部122342图。抽取原load_split/train_tf/val_tf/evaluate，原batch32、4 workers、pin_memory、seed44和模型配置保持；验证3824批（末批6图），完整计算原AP/macro-F1，但不输出为论文性能结果，不保存权重或修改正式训练文件。
- 全量诊断输出由服务器mktemp新建 `gavit44_fullvaldiag_XXXXXX`：完整模块/批次轨迹写diagnostic.log，终端显示前三批、每100批和最终批进度及异常；保留Python、tee和显示筛选各自退出码。每批完成后重置120秒无进展堆栈计时，避免在正常长任务中固定每分钟打印误导性Timeout；外层1200秒触发终止，必要时最多10秒强制清理。退出124仅表示此次人为时间限制，137单独不证明OOM；不自动延长、重试或启动正式训练。
- 本地检查通过：Bash语法、Python编译、原代码抽取、完整3824批/122342图、提前结束/错误batch大小/读取异常四种替身迭代检查、进度次数及计时重置、终端筛选不丢异常堆栈、20分钟限制与退出码记录。真实GPU、全量图像解码及全量指标尚未运行；命令交由用户执行。即使全量验证通过，也不能替代7434个训练batch后的状态或解释原后台进程退出。
- 训练→验证短诊断服务器回传通过：09-09 03:25:32—03:25:46 UTC（BEGIN至PASS约14秒），PID24721，目录 `/home/featurize/work/gavit44_transitiondiag_5HPWmR`，`diagnostic_exit=0 log_exit=0`。3批训练的forward/backward/optimizer均完成，训练iterator耗尽、scheduler.step、验证worker启动、3批验证及原AP/macro-F1全部返回。验证了所选79图短训练→70图验证的执行路径，未复现原退出，不据此宣称完整一轮或全量数据通过。回传文件 `/Users/patrick/.codex/attachments/527d985f-44d9-445b-8b9e-292998d76c02/pasted-text.txt`。
- 两次短诊断均通过后的边界：尚未覆盖完整122342张验证图像、全量预测汇总、经过7434个训练batch后的权重/资源状态或原后台运行的外部终止事件；不能把未复现直接解释为平台强杀，也没有根因修复。建议下一步保持短训练及模型配置，仅扩展到完整验证集并保留批次定位日志；拟设20分钟上限，该扩大GPU诊断需用户确认。本轮仅记录证据，未准备/启动第三次GPU诊断、正式重试、换卡或依赖修改，不更新性能表。
- 验证短诊断服务器回传通过：09-09 03:18:38—03:18:57 UTC（BEGIN至PASS约19秒），PID24474，目录 `/home/featurize/work/gavit44_valdiag_KUHfHp`；原evaluate处理batch32/32/6，数据读取、Swin、attentive_spatial、两层GAT、classifier、AP和macro-F1均完成，`diagnostic_exit=0 log_exit=0`。当前环境及val/预训练身份复核通过；PyG的WITH_PYG_LIB/WITH_TORCH_SCATTER/WITH_SOFTMAX为False，但本次所需路径可运行，不因此安装额外依赖。原始回传位于 `/Users/patrick/.codex/attachments/bcfa6752-2702-419f-97c3-c6b66bc0d439/pasted-text.txt`。
- 本次外层bash提示here-document到EOF仍未匹配BASH；内部Python已完整执行、日志和退出码完整，因此不把该警告当模型错误，也不为这条警告重跑已通过的70图验证。该短诊断仅证明新初始化模型下所选70图及三批推理/指标可执行，不证明原故障已解决、不排除其余验证数据或长时间运行问题，也没有覆盖训练后worker/梯度模式转换。
- 下一步针对缺失的训练→验证转换准备 `/tmp/gavit44-transition.LeJuqR/diagnose_gavit44_transition.sh`：复用第一份诊断，新增原训练变换、criterion/AdamW/Cosine设置和首轮验证调用之前的五条原训练语句，仅对独立模型执行3批训练（32/32/15，共79图），再执行原3批验证。train split SHA复核，4 workers/pin_memory/训练shuffle保持；5分钟上限，输出新建gavit44_transitiondiag目录，不写checkpoint、metadata或正式论文指标，不修改训练代码、依赖或自动提交正式重试。Bash/Python语法、原代码抽取、替身验证恰好3次backward/optimizer后scheduler、79/70样本边界和输出隔离检查通过；第二阶段服务器执行待回传。
- 后续补齐的preflight.log（用户回传）：commit `bb9f1cff463c996bc5f72695af538898c91490eb`；Python3.11.8、torch2.2.2、torchvision0.17.2、CUDA运行时12.1、cuDNN8902、timm1.0.27、torch-geometric2.8.0、numpy1.26.4、scikit-learn1.9.0、rasterio1.4.4。预训练SHA256为 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`；train237871，SHA256为 `25734fd9e8dc7fd78fb259e768ae91fe11b297e1caa50a2d57b74b0296413d8f`；val/test数量和SHA与09-08记录一致。仅凭版本不能认定依赖冲突，未升级或降级环境。
- 系统证据补充（用户回传）：09-09 02:59:53 UTC查询，uptime起点为09-08 09:37:37；train.log最后修改09-08 10:49 UTC，不等于实际退出时刻。当前 `/sys/fs/cgroup/system.slice/apphub.service/memory.events` 的low/high/max/oom/oom_kill均0，memory.max为max，memory.peak不存在；dmesg拒绝访问，journalctl提示无权查看系统消息后显示No entries。没有支持OOM的正面证据，也不能用受限日志或当前cgroup快照完全排除历史终止事件。
- 已转向代码路径定位：原evaluate不打印进度，验证后才打印指标及保存权重。源代码抽取的4个控制流检查验证了正常/异常验证时保存的先后关系，以及启动包装器对RuntimeError/SystemExit执行finally并传播异常；这些检查用替身对象，不是CUDA或数据解码实测。未发现启动器设置30分钟超时或吞普通异常。
- 已准备独立分段诊断命令 `/tmp/gavit44-validation.eIZvr2/diagnose_gavit44_validation.sh`：抽取原val_tf/evaluate，只读原metadata、val split与预训练，选验证集前64张及最后6张，batch32、num_workers4、pin_memory=True；记录取batch、Swin/grouping/GAT/classifier和原指标计算的边界。没有backward/optimizer更新或checkpoint写入；输出由服务器mktemp创建在新的gavit44_valdiag目录，保存日志与退出码，5分钟上限只用于此诊断。使用新初始化模型，不能复现丢失的epoch1权重、完整训练→验证worker生命周期或全量数据；通过不代表原故障已解决。Bash语法、Python编译、函数抽取、样本选择及无训练/权重写入静态检查通过；服务器诊断尚未执行。
- 用户检查 `/home/featurize/work/gavit44_2080ti_20260908/train.log`：最后记录为 `Epoch [1/30] Train` 的7434/7434批，耗时29:29、平均4.20it/s。没有第1轮验证指标、第2轮、最终Best Validation mAP或包装器finally资源统计。
- `ps -p 3504`仅输出表头；`gavit.pth`、`gavit.last.train_state.pth`均不存在，只有1.1K的 `gavit.meta.json`（Sep8 10:20）。据此判定本次不是正常完成；仅完成第1轮训练段，未确认第1轮验证及保存成功。保留全部文件，不重新提交，不把该次结果写入性能比较表。
- 当前训练代码在训练前写metadata；每轮结束先运行无进度条的 `evaluate(val_loader)`，输出验证指标后才调用 `save_epoch_artifacts`。因此metadata存在不证明训练完成；中断窗口在训练段末尾到首次验证/保存完成之前，不能精确到某一验证batch。
- 包装器普通Python异常也应执行finally统计；当前日志没有这部分输出。强制终止（含OOM killer/平台操作）、底层崩溃、实例重启或日志写入异常均为待排查候选，没有内核/平台证据前不认定OOM或显存不足。验证代码累计的是19维预测与标签，不能仅因有数组累积就断言28GB RAM不足。
- 当前已知输出中没有可用于恢复本次训练的完整checkpoint。下一步只读查看日志更新时间、输出目录其他文件、系统启动时间、内核OOM/segfault/NVRM记录和可用的cgroup内存事件；计数需结合时间/PID，不把共享cgroup历史事件直接归到3504。旧启动器未单独保存退出码，不能从消失PID事后推回退出原因。
- 已知训练段资源：2080 Ti；历史运行中GPU采样约4.1GiB，非峰值；主进程RSS采样1942188KiB，非全作业RAM峰值。总运行时长、异常时间、峰值资源和实际费用均待确认。没有续训、代码变更、自动换卡或新付费任务。

---

## 2026-09-08 — 用户选择2080 Ti直接训练（启动成功；后续中断见09-09记录）

- 用户随后补全日志：Train237871/Val122342；`Model: GAVIT | 31,366,926 params | Device: cuda`；`Epoch [1/30] Train`已到3338/7434批（45%），训练段已用13:39，tqdm显示剩余13:41、局部速度4.99it/s。结合PID3504/GPU进程证据，现确认GAViT seed44正式GPU训练已成功启动并推进；不需要重启或修改配置。
- 同轮10:34:20资源快照：PID3504存活14:12、STAT=Dl、RSS1942188KiB（仅主进程）；GPU总占用4184MiB，PID占用4180MiB，瞬时利用率0%。这是采样值，不是峰值；单个D/0%快照不足以证明持续I/O卡死，完整进度日志已证明训练在推进。首轮验证、存盘、全作业RAM峰值和总耗时仍待确认。
- 启动初期10:20:13回传：`Preflight passed`，GAViT44启动命令已提交，PID3504在5秒快照中存在；当时只打印参数、GPU4MiB/0%且无计算进程，因此当时仅记为“启动已提交”；后续补全日志才满足GPU启动确认标准。
- 数据身份回传：val122342，SHA256 `1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45`；test119825，SHA256 `cea165a6443fc5ee44de0a3c55f53ce70505c4e25aba9ca8b52d76634df77204`。按所给命令，`Preflight passed`意味着三个split的RGB路径检查均通过；本次粘贴未包含train SHA256或软件版本，待完整preflight.log补齐，未独立读取服务器文件。
- 用户09:39:37回传当前实例：NVIDIA GeForce RTX 2080 Ti、22528MiB、1MiB已用、无计算进程；驱动610.57.04，nvidia-smi显示CUDA UMD13.3（这不是PyTorch实际CUDA运行时版本）。commit bb9f1cf，Git状态命令无改动输出；预训练109M；CSV含表头分别237872/122343/119826；数据目录存在。完整RGB文件与PyTorch实际CUDA能力仍需启动命令检查。
- 正式命令已整理为 [docs/run_gavit44_2080ti.md](docs/run_gavit44_2080ti.md)，直接调用原入口，独立目录 `/home/featurize/work/gavit44_2080ti_20260908/`；配置、seed和输出名在启动前固定。状态以本条顶部最新回传为准，首批已确认，首轮验证与产物待确认。
- 本地检查：启动块通过 `bash -n`，两个内嵌Python片段通过AST语法检查，实际训练入口的argparse定义接受GAViT/seed44/30epochs/batch32/sparse_hybrid/formal参数；原训练代码无本轮差异。这些是命令静态检查，不是GPU运行测试。
- 用户明确决定使用 RTX 2080 Ti 22GB，跳过原独立资源短测。同步更新 [排期](docs/thesis_schedule.md)，旧3060/3080工具不再作为启动入口；模型与训练代码未改。
- 下一实验：BigEarthNet-19 sparse-hybrid GAViT seed44；研究问题为已有seed42/43提升是否在新增独立seed上成立，进入多seed核心对比表。相比既有GAViT固定架构，主要实验变量为seed；同时发生GPU更换，必须披露，不能把跨硬件方差解释成纯seed方差。
- 后续在同一2080 Ti/软件环境补Swin44形成配对；不自动启动Swin或seed45。保持batch32、224输入、30 epochs、AdamW lr3e-4/wd1e-4、CosineAnnealingLR、BCE、threshold0.5、K16 attentive_spatial、sparse_hybrid和token_feedback。
- 冻结训练commit仍为 `bb9f1cff463c996bc5f72695af538898c91490eb`；先检查当前服务器身份及完整数据，输出使用新的独立目录，不覆盖旧结果。保持原精度设置并记录，不因换卡改AMP/TF32。
- 启动策略：直接正式训练，同时观察首批、首轮验证和checkpoint；不再单独跑资源smoke。单次正式任务可能在新硬件/环境首次运行时失败，该风险由本次选择保留。数据与文件防覆盖检查不取消。
- 当前由用户在服务器执行并回传输出，本任务未直接访问服务器/已登录浏览器或代租实例。已确认启动和局部训练速度；完整首轮、全作业RAM、验证/测试指标及产物待回传，没有新论文性能结果或实际费用，不能标为实验完成。

---

## 2026-09-07 — 五周排期落盘与 3080/3060 资源短测准备

- 用户确认开始执行；排期主记录为 [docs/thesis_schedule.md](docs/thesis_schedule.md)，暂按 09-07 至 10-11 五周安排，最终提交日待确认。
- 当前 BigEarthNet 核心比较按最新导师要求至少再增加两个 seed；候选分配为 3080 10GB 配对 Swin44/GAViT44、3060 12GB 配对 Swin45/GAViT45。先以相同 seed42/小样本/配置测资源，不根据测试集分数决定机型。
- 本轮只完成本地 W1-01—W1-03。新增 `tools/resource_probe.py`、`tools/profile_training.py` 和 [运行说明](docs/gpu_resource_probe.md)，未启动服务器短测或正式训练，没有新增论文指标。
- 工具检查完整 train/val/test 的 RGB 路径、split 数量及身份，运行服务器完整测试后抽样 train4096/val1024；各模型一轮 smoke 保存 best、metadata、last-state、训练日志及资源报告。正式命令保持 batch32/30 epochs，需同 GPU/环境通过的配对 smoke 报告；短测不自动启动正式训练。
- 已验证训练代码基线为 `bb9f1cff463c996bc5f72695af538898c91490eb`，训练入口、模型及原身份工具未修改。资源工具必须放在服务器训练目录外；本地新增文件尚未 commit/push。
- 本地验证命令：`python3 -m unittest tests.test_resource_probe tests.test_training_profile tests.test_experiment_identity tests.test_sparse_hybrid_static_contract -v`；37 项通过（15 新增 + 22 既有）。`py_compile`、CLI help/只读 plan、差异空白检查及工具包解包一致性检查通过。
- 独立审查因服务额度限制未运行；已由主执行者核对训练/产物接口。上述测试不代表 CUDA 运行通过。两张 GPU 的显存、主机 RAM、吞吐、总时长及实际费用全部待服务器实测；不沿用小 split RAM 外推全量峰值。
- 工具包：`/private/tmp/gavit-resource-tools.v5MRYv/gavit-resource-tools-20260907.tar.gz`，仅含两个脚本和说明；SHA256 `3730a0f6185477689004fb3f224efd9a5d02553521c33edb1cbdf85b7b053ee6`。临时包若被系统清理，可由上述三个源文件重新打包。
- 保留 seed43 的既有事实：Swin43 完成30轮，GAViT43 停于21轮、best为8轮，用户曾决定不续跑。未改历史结果，不把当前记录表述为四次同协议完整训练。
- 下一步 W1-04：按运行说明在 3080 10GB 上执行配对资源短测，检查真实报告后再安排正式任务。未访问已登录 Featurize 浏览器、租用实例或改动服务器。

---

## 2026-09-02 — BigEarthNet sparse-hybrid 与 Swin 配对验证（seed 43）

### 一、研究问题与实验身份

- 研究问题：seed 42 上 `sparse_hybrid_4n_top2` 的 Test mAP、macro-F1、
  micro-F1 均高于 Swin 后，同一提升方向能否在独立 seed 43 上复现。
- 配对原则：只比较同一 seed 的 Swin43 与 sparse-hybrid GAViT43；固定
  BigEarthNet-19 split、预处理、batch size 32、AdamW、lr `3e-4`、weight
  decay `1e-4`、CosineAnnealingLR 和 threshold 0.5。
- 分支：`codex/sparse-hybrid-4n-top2`；Git commit：`bb9f1cf`；两份 metadata
  均记录 `dirty=false`、seed 43，checkpoint SHA256 校验一致。
- 配对 driver：
  `/home/featurize/work/run_gavit_seed43_pair_bb9f1cf_retry1.sh`；原计划依次
  运行 Swin43 和 GAViT43，使用 `_retry1` 隔离此前失败产物。

### 二、Swin43 完整训练与测试

- 完成 30/30 epochs；参数量 27,535,501；训练设备 NVIDIA GeForce RTX
  4090；峰值 CUDA allocated 3.56 GiB；训练 wall time 5.22 h。
- Best Validation mAP：**78.3459%（epoch 6）**。
- Epoch 30：Val mAP 75.8%、Val macro-F1 71.1%；相对 best 下降约
  2.55 pp，后期过拟合明显。
- Test mAP：**71.1%**；macro-F1：**64.1%**；micro-F1：**76.0%**。
- 测试 wall time 234.20 s；测试峰值 CUDA allocated 0.58 GiB。
- 训练日志：
  `logs/final_multiseed_swin_bb9f1cf_seed43_retry1_train.log`。
- 测试日志：
  `logs/final_multiseed_swin_bb9f1cf_seed43_retry1_test.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_swin_final_multiseed_bb9f1cf_seed43_retry1.pth`；
  同 stem 的 `.meta.json` 和 `.last.train_state.pth` 均存在，last epoch 30。

### 三、Sparse-hybrid GAViT43 训练边界与测试

- 配置：31,366,926 参数；K=16 `attentive_spatial`；48 条四邻接空间边加
  32 条排除空间邻居后的 cosine top-2 特征边；2-layer/4-head GAT（hidden
  256）；`token_feedback`；dropout 0.1。
- 原计划 30 epochs，实际保存到 epoch 21 后进程终止；driver 和训练日志未记录
  traceback、OOM 或 Python 异常，外部终止原因待确认。该运行不得写成完成
  30/30 epochs。
- Best Validation mAP：**78.4728%（epoch 8）**；到 epoch 21 未出现更高
  best。last state epoch 21，记录的 best metric 与 metadata 一致。
- 结合本模型 seed 42 在 epoch 13 达峰后回落 2.31 pp、corrected-kNN seed 42
  在 epoch 17 达峰后回落，以及 Swin43 在 epoch 6 达峰后回落的重复模式，
  用户决定不再为本次运行续跑 epoch 22--30。正式表格必须保留这一提前停止
  边界；测试严格使用 epoch 8 best checkpoint，而不是 epoch 21 last state。
- 第一次 RTX 3060 测试因实例本地 BigEarthNet 尚缺一个 B04 文件而在首个
  batch 前失败；保留失败日志。数据解压完成后使用独立 `retry2` 日志重新测试，
  完成全量推理并得到：Test mAP **71.7%**、macro-F1 **65.7%**、micro-F1
  **76.8%**。
- 训练日志：
  `logs/final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_train.log`。
- 成功测试日志：
  `logs/final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_test_retry2.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_final_multiseed_bb9f1cf_seed43_retry1.pth`；
  同 stem 的 `.meta.json` 和 `.last.train_state.pth` 均存在，checkpoint hash
  校验通过。
- 本次简化测试命令未包裹 wall-time/CUDA 统计；GAViT测试耗时与峰值显存
  仍待确认。2026-10-10已从现有日志回传补齐两模型19类per-class AP，见
  [seed43逐类表](results/bigearth_seed43_per_class_comparison.csv)。

### 四、同 seed 比较与当前结论

| Model (seed 43) | Test mAP | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Swin-T baseline | 71.1% | 64.1% | 76.0% |
| **Sparse-hybrid GAViT** | **71.7%** | **65.7%** | **76.8%** |
| Delta (GAViT - Swin) | **+0.6 pp** | **+1.6 pp** | **+0.8 pp** |

1. seed 43 上三个聚合指标均再次高于同 seed Swin，提升方向与 seed 42
   （+1.1/+1.8/+0.1 pp）一致。当前两 seed 的平均配对增益约为 mAP
   +0.85 pp、macro-F1 +1.70 pp、micro-F1 +0.45 pp；这里只是基于一位小数
   汇总值的阶段性描述，不替代三 seed 均值与标准差。
2. 两个 seed 的 macro-F1 增益最一致；mAP 为中等幅度提升，micro-F1 的增益
   较小且 seed 间波动更明显。仍不能表述为所有类别都改善；seed43 per-class
   AP已于2026-10-10补齐，为10升8降1打印持平。此段其余结论保留历史两seed边界，
   当前四seed判断见[最新分析](results/bigearth_four_seed_class_analysis.md)。
3. 结果支持“提升方向在 seed42/43 上可复现”，但 GAViT43 未完成原定30轮，
   且最终稳定性结论仍缺配对的 Swin44/GAViT44。论文最终表必须披露本次
   epoch21 停止边界，并在 seed44 完成后报告三 seed 均值与标准差。
4. 后续若引入 early stopping，应预先定义 patience，并保持原定最大 epoch
   和学习率调度身份；不能直接把 `--epochs` 改为 20，因为当前
   `CosineAnnealingLR(T_max=args.epochs)` 会同时改变优化轨迹。

---

## 2026-08-17 — BigEarthNet sparse-hybrid 正式训练与测试（seed 42）

### 一、研究问题与实验身份

- 研究假设：在与 corrected cosine-kNN 相同的 80 条有向边预算下，以 48 条
  K=16 四邻接空间边和 32 条排除空间邻居后的 cosine top-2 特征边组成
  `sparse_hybrid_4n_top2`，能比纯 corrected-kNN 提供更有效的区域关系。
- 唯一主要变量是 graph topology；backbone、grouping、region 数、GAT、
  integration、数据划分、优化和 seed 均与 corrected-kNN control 保持一致。
- 论文用途：进入 `results/bigearth_comparison.csv`，作为 BigEarthNet edge
  topology 消融和候选最佳 GAViT；本条仍是单 seed 结果。
- 分支：`codex/sparse-hybrid-4n-top2`；训练 Git commit：`cd39be9`；
  metadata 记录 `dirty=false`。
- run stage：`formal`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654`；seed 42。

### 二、数据、模型与优化

- BigEarthNet-19 固定 split：train 237,871 / val 122,342 / test 119,825。
  正式测试前逐行验证 test CSV 所需 B02/B03/B04 文件：119,825/119,825
  patch 完整，0 missing。
- 预处理：resize 224x224；训练随机水平/垂直翻转；ImageNet mean/std
  normalization。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；dropout 0.1；
  `token_feedback` integration。
- 图：`edge_type=sparse_hybrid`；48 spatial + 32 feature = 80 条唯一有向边；
  feature top-2；`neighbor_to_query`；cosine 只决定拓扑。
- 预训练：`bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，AdamW，lr `3e-4`，weight decay `1e-4`，
  CosineAnnealingLR，BCEWithLogitsLoss；分类阈值 0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type sparse_hybrid \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM、NaN 或中途终止。
- **Best Validation mAP：79.2087%（epoch 13）**；metadata 和 last state 的
  best metric 一致，last state epoch 30。
- Epoch 13 后训练 loss 继续下降，但验证 mAP 总体回落；epoch 17 短暂回到
  78.9%，未超过 best。Epoch 30：loss 0.0834、Val mAP 76.9%、Val macro-F1
  71.8%；相对 best mAP 下降 2.31 pp，存在明确后期过拟合。
- 前五轮 Val mAP：74.3%、76.4%、76.9%、78.0%、78.1%；epoch 7 为 78.8%，
  epoch 13 达峰；epoch 20--30 位于 76.9%--77.7%。测试严格使用 epoch 13
  best checkpoint，而非 epoch 30 last state。
- 训练设备：NVIDIA GeForce RTX 4090（24,564 MiB）；启动阶段约
  14.10 train batch/s；峰值 CUDA allocated 3.65 GiB；总 wall time 6.64 h，
  平均 wall time 约 13.28 min/epoch；实际费用待确认。
- 训练日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.log`。
- Best checkpoint：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth`；
  同 stem 保存 `.meta.json` 和 `.last.train_state.pth`。

### 四、正式测试

正式测试命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("test_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Test wall time: {time.time()-started:.2f} s")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654.pth \
  --batch_size 32 \
  > logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log 2>&1 &
```

- 测试设备：NVIDIA GeForce RTX 3060（12,288 MiB）；3,745 batches；tqdm
  10:33，5.91 batch/s；端到端 wall time 665.55 s（11.09 min）；峰值 CUDA
  allocated 0.60 GiB。
- **Test macro mAP：72.0%；macro-F1：66.0%；micro-F1：76.7%。**
- 测试日志：
  `logs/sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log`。

| Class | Sparse-hybrid AP | vs corrected-kNN | vs Swin |
|---|---:|---:|---:|
| Urban fabric | 86.3% | +0.1 pp | -0.2 pp |
| Industrial or commercial units | 51.1% | +1.6 pp | -1.3 pp |
| Arable land | 93.8% | +0.6 pp | +0.5 pp |
| Permanent crops | 61.2% | +2.3 pp | +2.4 pp |
| Pastures | 85.8% | -0.3 pp | -0.4 pp |
| Complex cultivation patterns | 70.1% | +1.1 pp | +2.1 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.7% | +1.2 pp | +0.7 pp |
| Agro-forestry areas | 87.1% | +1.2 pp | -0.2 pp |
| Broad-leaved forest | 86.0% | +0.4 pp | +0.9 pp |
| Coniferous forest | 93.5% | -0.2 pp | +0.0 pp |
| Mixed forest | 89.8% | +1.1 pp | +0.5 pp |
| Natural grassland and sparsely vegetated areas | 42.1% | -1.9 pp | -0.8 pp |
| Moors, heathland and sclerophyllous vegetation | 61.8% | -1.0 pp | +5.0 pp |
| Transitional woodland, shrub | 78.8% | +0.0 pp | +0.5 pp |
| Beaches, dunes, sands | 16.8% | +6.7 pp | +3.5 pp |
| Inland wetlands | 64.1% | +1.0 pp | +1.4 pp |
| Coastal wetlands | 34.7% | +12.9 pp | +4.4 pp |
| Inland waters | 91.7% | +0.0 pp | +0.8 pp |
| Marine waters | 99.6% | -0.1 pp | +0.1 pp |

### 五、比较、解释与下一步

| Model | Test mAP | Macro-F1 | Micro-F1 |
|---|---:|---:|---:|
| Swin-T baseline | 70.9% | 64.2% | 76.6% |
| Corrected-kNN GAViT | 70.6% | 65.4% | 76.6% |
| **Sparse-hybrid GAViT** | **72.0%** | **66.0%** | **76.7%** |

1. 相对 corrected-kNN，sparse-hybrid 的 test mAP +1.4 pp、macro-F1 +0.6 pp、
   micro-F1 +0.1 pp；相对 Swin 分别为 +1.1、+1.8、+0.1 pp。验证 mAP 也比
   corrected-kNN best 78.5732% 高 0.6355 pp，验证和测试方向一致。
2. mAP 增益主要来自 Coastal wetlands（+12.9 pp vs corrected）、Beaches
   （+6.7 pp）、Permanent crops（+2.3 pp）等困难类别；micro-F1 几乎不变，
   因此当前证据更支持“改善宏观/尾部类别表现”，而不是全面提高样本级预测。
3. Natural grassland（-1.9 pp）和 Moors（-1.0 pp）相对 corrected 下降，
   sparse topology 并非所有类别一致受益，后续应在论文中保留该限制。
4. **seed 42 的研究假设得到支持，sparse-hybrid 暂定为最佳 GAViT。** 单 seed
   不能声称稳定提升；不再搜索新 topology。下一步只为最终对比补齐 selected
   GAViT 与 Swin baseline 的 seeds 43/44，并报告三 seeds 均值与标准差。

---

## 2026-08-14 — sparse_hybrid_4n_top2 Featurize 工程 gate 通过

### 实验身份与验证环境

- 分支：`codex/sparse-hybrid-4n-top2`；Git commit：`a28cbe1`。
- Featurize GPU：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 固定 BigEarthNet-19 split 已核对：train 237,871 / val 122,342 /
  test 119,825；原始图像位于实例本地
  `/home/featurize/data/BigEarthNet-S2`。
- 完整测试命令：`python -m unittest discover -s tests -v`。
- 测试结果：59 tests，1.879 s，全部通过；无 failure、error 或 skipped
  sparse-hybrid test。覆盖了 48/32/80 edge budget、top-2 排除规则、稳定
  tie-break、batch offset、forward/backward finite gradients、checkpoint metadata
  round-trip 和 training-state 恢复约束。

### Smoke 配置与结果

- 数据：一次性 256 train / 128 val split；1 epoch；batch size 32；seed 42。
- 模型：GAViT，31,366,926 参数；Swin-T backbone；K=16
  `attentive_spatial`；2-layer/4-head GAT（hidden 256）；
  `edge_type=sparse_hybrid`；`token_feedback`；dropout 0.1。
- 预训练权重：`bigearth_files/model.safetensors`；AdamW，lr `3e-4`，
  weight decay `1e-4`，CosineAnnealingLR，BCEWithLogitsLoss。
- run stage：`smoke`；run tag：
  `sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255`。
- 训练完成 8/8 batches，tqdm 平均 5.75 batch/s；该短跑吞吐仅用于工程
  估算，不作为正式效率结果。
- Epoch 1 loss：0.3408；Val macro mAP：47.7286%；Val macro-F1：15.9%。
- 峰值 CUDA allocated：3.65 GiB；总 wall time：8.07 s。
- 无 NaN、Inf、OOM 或 traceback；forward、backward、optimizer step、validation
  和 artifact save 均完成。

### 拓扑与产物验证

- metadata 中 `architecture.edge_type=sparse_hybrid`。
- `graph_topology` 明确记录：`name=sparse_hybrid_4n_top2`、
  `spatial_connectivity=4`、`feature_k=2`、48 条空间有向边、32 条特征有向边、
  总计 80 条唯一有向边、`message_direction=neighbor_to_query`、
  `cosine_role=topology_only`。
- Best checkpoint（约 120 MB）：
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.pth`。
- Metadata（约 1.4 KB）：同 stem 的 `.meta.json`；best metric 为
  `val_mAP=47.7286`，best epoch 1。
- Last training state（约 360 MB）：同 stem 的 `.last.train_state.pth`；
  last epoch 1，`best_metric=47.7285909477752`。
- 日志：
  `logs/sparse_hybrid_4n_top2_a28cbe1_smoke_retry_20260814_125255.log`。
- 独立 artifact 校验脚本完成文件存在性、architecture、topology、best epoch 和
  last-state 断言，输出 `SMOKE ARTIFACT VERIFICATION: OK`。

### 异常与结论

- 首次 smoke 在训练 8/8 后、无进度条的 validation 阶段被手动 `Ctrl+C`
  中断，因此只写入初始 metadata（`best=None`），没有 checkpoint 或 last state；
  使用新的 run identity 重试后完整通过。该中断不是模型或拓扑故障。
- **工程 gate 已通过。** Smoke 指标只证明运行路径和产物契约正确，不进入
  `results/bigearth_comparison.csv`，也不作为论文性能证据。
- 下一步按既定漏斗运行固定 seed 42 的 5-epoch proxy；proxy 之前不启动 30-epoch
  formal，也不追加其他拓扑或 integration 变量。

### Fixed 10% proxy 基础设施

- 为避免直接付费运行全量五轮，已批准固定 10% train/val、5 epochs、training
  seed 42 的 proxy 协议；proxy 只检查多轮学习稳定性，不替代全量 30-epoch
  formal，也不进入论文结果表。
- 设计 commit：`2ab6026`；实现 commit：`3f82563`。
- 新增 `baselines/bigearth/prepare_proxy_split.py`：从 seeds 42--141 的 100 个
  均匀随机候选中选择 train/val 共 38 个类别 prevalence 最大偏差最小者；任何
  split 缺少正类的候选 fail closed，分数相同选择较小 seed。
- 预期输出到 `bigearth_files/proxy_10pct_seed42`：23,787 train / 12,234 val、
  `prevalence.json`、输入/输出 SHA256、每类正例数与 prevalence deviation；
  test split 不参与生成或 proxy 训练，已有输出不会被覆盖。
- 本地标准库定向测试：12 tests 全部通过；相关 torch-free 套件：53 tests
  全部通过；生成器和测试通过 `py_compile`，`git diff --check` 通过。
- 本地完整 discover 因环境没有 PyTorch，三个 PyTorch test module 在 import
  阶段报 `ModuleNotFoundError: torch`；这不是代码回归。Featurize 拉取实现后仍须
  运行完整 suite，预期总数由 59 增至 71，全部通过后才能生成 proxy split。
- 生成、独立 hash/row/prevalence 校验和五轮前台训练命令已写入
  `docs/featurize_runbook.md`；实际运行结果记录如下，分布报告 SHA256 尚待补记。

### Fixed 10% proxy 实际运行结果

- Featurize 在 commit `cd39be9` 上运行完整测试：71 tests，1.512 s，全部通过。
- 生成器从 seeds 42--141 中选择 seed 137；train 23,787 / val 12,234；
  train/val 共 38 个类别 prevalence 的最大绝对偏差为 `0.00394910`
  （约 0.395 个百分点）。输出目录：
  `bigearth_files/proxy_10pct_seed42`。独立报告 SHA256 待补记。
- Proxy run stage：`proxy`；run tag：
  `sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423`；训练 seed 42。
- 模型和优化配置与计划一致：GAViT、K=16 `attentive_spatial`、
  `sparse_hybrid` 4-neighbor + top-2、2-layer/4-head GAT、
  `token_feedback`、batch size 32、lr `3e-4`、5 epochs。
- 每轮 744 个训练 batch；训练吞吐依次为 13.68、13.80、13.88、13.89、
  13.84 batch/s。Epoch 1--5 的 loss 为 0.2234、0.1852、0.1673、0.1492、
  0.1330；Val macro mAP 为 65.5%、67.9%、70.7%、73.7%、74.8%；
  Val macro-F1 为 56.3%、60.1%、60.7%、64.7%、67.3%。
- Best Validation mAP：74.8325%（epoch 5）；峰值 CUDA allocated：3.65 GiB；
  总 wall time：345.48 s（5.76 min，平均 69.10 s/epoch）。无 OOM、NaN、
  traceback 或指标退化，loss 单调下降且 mAP/F1 总体持续上升。
- Best checkpoint 保存到
  `checkpoints/best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.pth`；
  训练日志为 `logs/sparse_hybrid_4n_top2_cd39be9_proxy_20260814_133423.log`。
  独立 artifact 校验确认同 stem 的 `.meta.json` 和 `.last.train_state.pth`
  可读取；metadata best 为 `val_mAP=74.8325`、epoch 5，last state epoch 5，
  Git identity 为 commit `cd39be9`、`dirty=false`，输出
  `PROXY ARTIFACT VERIFICATION: OK`。
- **多轮学习稳定性 proxy gate 已通过。** 该结果只支持启动固定 seed 42 的
  全量 30-epoch formal；不与全量 corrected-kNN 指标直接比较，也不进入论文
  结果表。按数据规模和本次 wall time 线性外推，formal 约 5.76 h；这是启动前
  成本估算，不是实际耗时。

---

## 2026-08-13 — sparse_hybrid_4n_top2 本地代码完成，待 Featurize 工程验证

### 研究假设与单一变量

- 研究问题：在与 corrected cosine-kNN 相同的 80 条 pre-GAT 有向边预算下，
  同时编码空间邻接与非局部特征相似性的稀疏图，是否能提供更符合导师建议、
  更有意义的区域关系。
- 唯一改变的主要变量是 graph topology。Swin-T、K=16
  `attentive_spatial`、2-layer/4-head GAT、`token_feedback`、数据划分、优化器、
  loss、scheduler、增强和阈值保持不变。
- 新配置名为 `edge_type=sparse_hybrid`；历史 `edge_type=hybrid` 的 8-neighbor
  加 kNN 直接拼接行为保持不变，不能用于本实验命名。

### 实现与身份

- 设计规格：`docs/superpowers/specs/2026-07-28-sparse-hybrid-graph-experiment-design.md`。
- 实施计划 commit：`d53cb1d`。
- 图构建 commit：`0be936a`。
- GAViT 与四个 train/test CLI 接线 commit：`79faafa`。
- CLI/topology 静态契约收紧 commit：`2c3f85b`。
- 显式 topology metadata 与 resume identity commit：`3bfd8d5`。
- 分支：`codex/sparse-hybrid-4n-top2`。
- 每张 K=16 图构造 48 条四邻接空间边；每个 query 排除 self 和空间邻居后，
  按 cosine similarity 选择 top-2，共 32 条特征边；合计 80 条唯一有向非自环边。
- 全部边遵守 PyG 的 `neighbor -> query` 消息方向。cosine 数值只用于选择拓扑，
  不传入 GAT 作为 message weight；相等分数按较小 region index 决定顺序。
- `sparse_hybrid` 对 K!=16 和 feature_k!=2 fail closed；checkpoint 名称和
  metadata 使用独立的 `edge_type=sparse_hybrid` 身份，并显式保存
  `graph_topology` 的 4-neighbor、feature_k=2、48/32/80 和消息方向快照；
  `--knn_k 5` 只是与 corrected-kNN 控制命令保持兼容，不表示本拓扑选择五条
  feature edges。resume identity 同时绑定该 topology 快照。

### 本地已验证与尚缺验证

- 本地 `python3 -m unittest tests.test_sparse_hybrid_static_contract
  tests.test_experiment_identity -v`：18 tests 全部通过。
- 本地 torch-free training-state 合约：10 tests 全部通过。
- 修改过的 graph/model/train/test/test-suite 文件均通过 `py_compile`；
  `git diff --check` 通过。
- 已加入但**尚未在本地执行**的 PyTorch 测试覆盖：80-edge budget、48/32 分解、
  四邻接无对角线、top-2 排除规则、tie-break、batch offset、无 self/重复/
  cross-image edge、模型 forward/backward finite gradients，以及 metadata
  round-trip logits。
- 原因：本机 Python 环境没有 PyTorch/NumPy，按既定原则未在本地安装；这些
  张量级测试必须在 Featurize 环境运行后才能标记为通过。
- Featurize 必跑命令和 256/128、1-epoch smoke gate 已写入
  `docs/featurize_runbook.md`。待记录：完整测试数量、GPU、峰值显存、throughput、
  首轮耗时、checkpoint/metadata/last-state 路径和 smoke 结果。

**当前结论：代码已实现，但工程 gate 尚未通过；未启动 proxy 或正式训练。**

---

## 2026-08-12 — BigEarthNet corrected-kNN GAViT 正式训练与测试

### 一、研究问题与实验身份

- 研究问题：修正 cosine-kNN 的消息方向为 `selected_neighbor -> query` 后，当前 GAViT v2 是否优于历史错误方向结果，并缩小或逆转相对 Swin baseline 的差距。
- 本次只改变 kNN 消息方向；数据划分、K=16 attentive spatial grouping、k=5、2-layer/4-head GAT、token feedback、优化器和训练轮数保持不变。
- Git commit：`8cb8563`；训练 metadata 记录 `dirty=false`。
- Featurize：NVIDIA GeForce RTX 4090（24,564 MiB）。
- 训练前验证：`python -m unittest discover -s tests -v`，44 tests，全部通过。
- 工程 smoke：256 train / 128 val，1 epoch，batch size 32；完成 forward、backward、validation 和隔离 checkpoint 保存，峰值 CUDA allocated 3.65 GiB。smoke 指标不作为论文性能证据。

### 二、正式配置与命令

- 数据集：BigEarthNet-19；固定 split 为 Train 237,871 / Val 122,342 / Test 119,825。
- 预处理：resize 224×224；训练使用随机水平/垂直翻转；ImageNet mean/std normalization。
- 模型：GAViT v2，31,366,926 参数；Swin-T backbone；K=16 `attentive_spatial`；corrected cosine-kNN `k=5`；GAT hidden=256、4 heads、2 layers；`token_feedback` integration；dropout=0.1。
- 预训练来源：持久化本地 `bigearth_files/model.safetensors`。
- 优化：30 epochs，batch size 32，seed 42，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR，BCEWithLogitsLoss；F1 threshold=0.5。

正式训练命令：

```bash
nohup python -u -c \
'import runpy, torch, time; started=time.time(); runpy.run_path("train_bigearth.py", run_name="__main__"); print(f"Peak CUDA allocated: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"); print(f"Total wall time: {(time.time()-started)/3600:.2f} h")' \
  --model gavit \
  --data_dir bigearth_files/splits \
  --epochs 30 \
  --batch_size 32 \
  --lr 3e-4 \
  --num_regions 16 \
  --knn_k 5 \
  --gat_hidden 256 \
  --gat_heads 4 \
  --gat_layers 2 \
  --grouping attentive_spatial \
  --edge_type knn \
  --integration token_feedback \
  --dropout 0.1 \
  --seed 42 \
  --run_stage formal \
  --run_tag corrected_knn_8cb8563_20260811 \
  --pretrained_path bigearth_files/model.safetensors \
  --checkpoint_path checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log 2>&1 &
```

正式测试命令：

```bash
nohup python -u test_bigearth.py \
  --model gavit \
  --data_dir bigearth_files/splits \
  --ckpt checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth \
  --batch_size 32 \
  > logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log 2>&1 &
```

### 三、训练结果与资源

- 完成 30/30 epochs；无 traceback、OOM 或中途终止。
- **Best Validation mAP：78.5732%（epoch 17）**。
- Epoch 30：loss 0.0817，Val mAP 76.7%，Val macro-F1 71.7%。
- best 后验证 mAP 持续回落，存在明显后期过拟合；测试严格使用 epoch 17 best checkpoint，而不是 epoch 30 last state。
- 峰值 CUDA allocated：3.65 GiB。
- 训练总时长：6.49 h；实际费用待确认。
- 测试：3,745 batches，3:16，19.06 batch/s。

产物：

- 训练日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811.log`
- 测试日志：`logs/bigearth_gavit_corrected_knn_seed42_8cb8563_20260811_test.log`
- Best checkpoint：`checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42_formal_corrected_knn_8cb8563_20260811.pth`（约 120 MB）
- Metadata：同名 `.meta.json`（best epoch 17，Git `8cb8563`，dirty=false）
- Last training state：同名 `.last.train_state.pth`（epoch 30，约 360 MB，仅用于精确恢复，不用于本次测试）

### 四、正式测试结果

- **Test macro mAP：70.6%**
- **Test macro-F1：65.4%**
- **Test micro-F1：76.6%**
- 分类阈值：0.5

| Class | AP | vs Swin |
|---|---:|---:|
| Urban fabric | 86.2% | -0.3 pp |
| Industrial or commercial units | 49.5% | -2.9 pp |
| Arable land | 93.2% | -0.1 pp |
| Permanent crops | 58.9% | +0.1 pp |
| Pastures | 86.1% | -0.1 pp |
| Complex cultivation patterns | 69.0% | +1.0 pp |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 72.5% | -0.5 pp |
| Agro-forestry areas | 85.9% | -1.4 pp |
| Broad-leaved forest | 85.6% | +0.5 pp |
| Coniferous forest | 93.7% | +0.2 pp |
| Mixed forest | 88.7% | -0.6 pp |
| Natural grassland and sparsely vegetated areas | 44.0% | +1.1 pp |
| Moors, heathland and sclerophyllous vegetation | 62.8% | +6.0 pp |
| Transitional woodland, shrub | 78.8% | +0.5 pp |
| Beaches, dunes, sands | 10.1% | -3.2 pp |
| Inland wetlands | 63.1% | +0.4 pp |
| Coastal wetlands | 21.8% | -8.5 pp |
| Inland waters | 91.7% | +0.8 pp |
| Marine waters | 99.7% | +0.2 pp |

### 五、比较与结论

| Model | Test mAP | Macro-F1 | Micro-F1 | 说明 |
|---|---:|---:|---:|---|
| Swin-T baseline | 70.9% | 64.2% | 76.6% | 相同正式 split，seed 42 |
| Historical GAViT v2 | 70.4% | 待确认 | 待确认 | 修正消息方向前的历史结果 |
| Corrected-kNN GAViT v2 | 70.6% | 65.4% | 76.6% | 本次正式结果，seed 42 |

1. 修正消息方向后，GAViT 相对历史结果的 test mAP 提升 0.2 pp，说明方向错误有小幅负面影响，但不是性能差距的主要原因。
2. Corrected GAViT 的 mAP 仍比 Swin 低 0.3 pp；单 seed 下不能声称 graph module 提升总体 mAP。
3. Macro-F1 比 Swin 高 1.2 pp，micro-F1 持平；类别层面改善和退化并存，其中 Moors/... +6.0 pp，但 Coastal wetlands -8.5 pp。该结果值得在后续拓扑实验中继续观察，不能单独作为模型优越性的证据。
4. 本条形成时的下一项论文问题，是按已批准设计实现并验证 `sparse_hybrid_4n_top2`，以 corrected-kNN 结果作为同预算正式对照；该功能后来已于 2026-08-13 在 `codex/sparse-hybrid-4n-top2` 分支完成本地代码，但仍待 Featurize 工程验证。历史 `hybrid` 仍不得冒充该配置。
5. 在 sparse-hybrid 比较完成前，不追加 corrected-kNN 多 seed，也不启动 gated/cross-attention integration 实验。

---

## 2026-07-14 — Featurize BigEarthNet Swin baseline 完整训练

### 一、实验设置

- 平台：Featurize（GPU 具体型号待从实例记录确认）
- 数据集：BigEarthNet-19，多标签分类
- 数据划分：Train 237,871 / Val 122,342 / Test 119,825
- 模型：Swin-T baseline，27,535,501 参数
- 初始化：本地持久化 `bigearth_files/model.safetensors`，避免 Featurize 访问 Hugging Face 的 TLS 失败
- Seed：42
- 训练：30 epochs，batch size 32，AdamW，lr 3e-4，weight decay 1e-4，CosineAnnealingLR
- Loss：BCEWithLogitsLoss
- 验证指标：macro mAP + macro-F1（threshold=0.5）
- 启动入口：`train_bigearth_local.py`（临时为 Swin 注入本地 pretrained file）
- 日志：`logs/bigearth_swin.log`
- Checkpoint：`checkpoints/best_bigearth_swin.pth`，约 106 MB

### 二、训练结果

- 完成 30/30 epochs
- **Best Validation mAP：78.8%**（准确 best epoch 待从完整日志提取）
- Epoch 30 Loss：0.0223
- Epoch 30 Val mAP：75.8%
- Epoch 30 Val macro-F1：71.2%
- 后期 Val mAP 从最佳值回落至 75.8%，存在过拟合迹象；正式比较应使用 best checkpoint，而不是最后一轮权重
- 已观察到 epoch 24–30 单轮训练约 7:52–8:35；峰值显存、完整 wall time 和实际费用待补记

### 三、存储与复现状态

- BigEarthNet 图像位于实例本地 `/home/featurize/data/BigEarthNet-S2`，更换实例后需要重新添加
- metadata、split CSV、本地预训练权重、日志和 checkpoint 位于 `/home/featurize/work/GAViT_Project`，更换实例后保留
- 固定 split 已持久化在 `bigearth_files/splits`，无需随实例重新生成

### 四、测试结果

- 测试样本：119,825
- 测试耗时：2:25（3,745 batches，25.68 it/s）
- 分类阈值：0.5
- **Test macro mAP：70.9%**
- **Test macro-F1：64.2%**
- **Test micro-F1：76.6%**
- 测试日志：`logs/bigearth_swin_test_retry.log`

Per-class AP：

| Class | AP |
|---|---:|
| Urban fabric | 86.5% |
| Industrial or commercial units | 52.4% |
| Arable land | 93.3% |
| Permanent crops | 58.8% |
| Pastures | 86.2% |
| Complex cultivation patterns | 68.0% |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 73.0% |
| Agro-forestry areas | 87.3% |
| Broad-leaved forest | 85.1% |
| Coniferous forest | 93.5% |
| Mixed forest | 89.3% |
| Natural grassland and sparsely vegetated areas | 42.9% |
| Moors, heathland and sclerophyllous vegetation | 56.8% |
| Transitional woodland, shrub | 78.3% |
| Beaches, dunes, sands | 13.3% |
| Inland wetlands | 62.7% |
| Coastal wetlands | 30.3% |
| Inland waters | 90.9% |
| Marine waters | 99.5% |

### 五、测试脚本问题与修复

- 第一次完整推理完成后，`average_precision_score` 报错：预测数组仍为三维。
- 根因：`test_bigearth.py` 未处理 timm Swin 返回的 `(B, H, W, C)` 四维特征，分类器输出成为 `(B, H, W, 19)`；训练脚本已正确进行空间池化。
- 修复：四维特征在维度 `(1, 2)` 上取均值，三维 token 特征在维度 `1` 上取均值。
- 修复后先用 32 条样本 smoke test，再运行完整测试，避免重复浪费 GPU 时间。

### 六、论文状态与下一步

- 本次结果建立了 BigEarthNet 上的 Swin baseline，后续 GAViT 必须使用相同 split、训练设置和主要指标
- [ ] 从完整日志确认 best epoch
- [ ] 补记训练和测试使用的 GPU 型号、峰值显存、完整 wall time 与实际费用
- [ ] 固定 GAViT 正式训练配置，并先完成小规模 smoke test
- [ ] 使用相同 split、epoch、batch size、seed 和指标训练及测试 GAViT

---

## 2026-04-08 — BigEarthNet 训练超时修复：添加断点续训

**问题**：
- BigEarthNet Swin baseline（Job 56169）在 epoch 16/30 因 8h 时间限制被 SLURM 杀掉
- BigEarthNet GAViT（Job 56170）在 epoch 24/30 因 10h 时间限制被杀掉
- 估算：Swin 每 epoch ~29min（总需 ~14.5h），GAViT 每 epoch ~25min（总需 ~12.5h）

**解决方案**：
- `train_bigearth.py` 新增 `--resume` 参数，从已有 best checkpoint 加载模型权重继续训练
- `train_bigearth.py` 新增 `--start_epoch` 参数，控制训练循环起始 epoch
- 注意：仅恢复模型权重，optimizer/scheduler 状态不保存（影响极小，模型已接近收敛）

**作业脚本更新**：
- `run_bigearth_train_swin.sh`：时间 8h → 16h，添加 `--resume --start_epoch 17`
- `run_bigearth_train_gavit.sh`：时间 10h → 6h，添加 `--resume --start_epoch 25`

**下一步**：
- [ ] Push 后在服务器 `git pull` 并重新 sbatch 两个作业
- [ ] 训练完成后运行测试脚本对比 Swin vs GAViT 在 BigEarthNet 上的表现

---

## 2026-04-04 — GAViT v2 训练结果 + BigEarthNet 实验准备

### 一、GAViT v2 训练 & 测试结果

**训练结果**（Job 53893，NVIDIA RTX A5000）：
- Model: GAViT | K=16 | grouping=attentive_spatial | edge=knn | integration=token_feedback | GAT 2L×4H
- 参数量：31,386,920
- **Best Val Acc：96.2%**（epoch 26）
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 波动，train acc epoch 29 达 100%

**测试结果**（Job 54030，NVIDIA Tesla V100）：
- **Test Acc：95.8%**（4526/4725）

**完整对比**：

| 模型 | Val Acc | Test Acc | Params |
|------|---------|----------|--------|
| Swin-T Baseline | ~96.0% | 96.3% | 27.5M |
| GAViT v1 K=9 spatial + GAT 2L | 96.5% | 95.9% | 30.5M |
| GAViT Fusion (backbone+graph concat) | 95.9% | 95.9% | 30.5M |
| GAViT v2 K=16 attentive + token_feedback | 96.2% | 95.8% | 31.4M |

**结论**：三种 GAViT 变体 test acc 均在 95.8-95.9%，与 baseline（96.3%）差 ~0.4%。与教授预判一致——NWPU-RESISC45 单标签数据集 global appearance 信号太强，graph module 增益天然受限。

### 二、BigEarthNet 多标签实验准备

按 Prof Wang 0327 建议，新增 BigEarthNet 实验以展示 graph module 在关系更重要场景的价值。

**完成内容**：

1. **`models/bigearth_dataset.py`** — BigEarthNet Dataset 类
   - 支持 BigEarthNet-S2 格式（读取 B02/B03/B04 TIF 波段 → RGB）
   - 内置 43→19 类标签映射（BigEarthNet-19 标准）
   - 返回 19 维 binary label vector

2. **`baselines/bigearth/prepare_bigearth.py`** — 数据准备脚本
   - 自动下载官方 train/val/test split 列表
   - 生成 split CSV 文件（patch_path + 19 维标签列）

3. **`train_bigearth.py`** — 统一训练脚本
   - 支持 `--model swin`（baseline）和 `--model gavit`（GAViT v2）
   - Loss: BCEWithLogitsLoss（多标签）
   - Metric: mAP（mean Average Precision）+ Macro-F1 @ 0.5

4. **`test_bigearth.py`** — 测试脚本
   - 输出 mAP / macro-F1 / micro-F1 + per-class AP

5. **Slurm 作业脚本**：
   - `jobs/run_bigearth_prepare.sh`
   - `jobs/run_bigearth_train.sh`（依次训练 Swin baseline + GAViT v2）
   - `jobs/run_bigearth_test.sh`

6. **数据下载进行中**
   - 服务器创建了 `/projects/gavitdata/`（SSD，150GB 配额）
   - BigEarthNet-S2-v1.0 从 Zenodo 下载中（~65GB，约 6 小时）
   - 下载链接：`https://zenodo.org/records/12687186/files/BigEarthNet-S2-v1.0.tar.gz?download=1`

### 三、下一步计划（下载完成后按顺序执行）

- [ ] 选择性解压 RGB 波段 + 标签 JSON，删掉 tar.gz
  ```
  tar -xzf BigEarthNet-S2-v1.0.tar.gz --wildcards '*_B02.tif' '*_B03.tif' '*_B04.tif' '*_labels_metadata.json'
  rm BigEarthNet-S2-v1.0.tar.gz
  ```
- [ ] 运行 prepare 脚本生成 split CSV（`sbatch jobs/run_bigearth_prepare.sh`）
- [ ] 训练 Swin baseline + GAViT v2（`sbatch jobs/run_bigearth_train.sh`）
- [ ] 测试两个模型（`sbatch jobs/run_bigearth_test.sh`）
- [ ] 对比结果：NWPU（graph 增益小）vs BigEarthNet（graph 增益预期更大）
- [ ] NWPU 可视化：airport/bridge/church 的 region 分区和 graph 连接
- [ ] 整理所有结果给导师发邮件

---

## 2026-04-04 — GAViT v2 架构改进：AttentiveSpatialGrouping + Token Feedback

**动机（基于 Prof Wang 0327 回复）**：

教授指出当前 GAViT 的两个核心问题：
1. **集成方式不当**：graph module 输出在分类器阶段才与 backbone 拼接（fusion），relational reasoning 没有影响表征本身。应让 graph 输出直接修改 token 特征。
2. **Region 定义太粗糙**：3×3 spatial grid 分辨率不够，小结构被合并；mean pooling 没有区分 token 的重要性。应提高分辨率（4×4 或 5×5）并用 attention weighting 替代均值。

此外教授指出 NWPU-RESISC45 作为单标签数据集，global appearance 已是很强信号，graph module 增益天然有限。建议新增 BigEarthNet（多标签）数据集实验。

**完成内容**：

1. **`models/region_grouping.py` — 新增 `AttentiveSpatialGrouping`**
   - 支持任意完全平方数 K（默认 K=16，即 4×4 grid）
   - 每个 region 内用 lightweight attention（2 层 MLP → scalar score → softmax）加权聚合 token，替代均值 pooling
   - 信息量大的 token 贡献更多，使 region 特征更有语义代表性

2. **`models/gavit.py` — 新增 `token_feedback` 集成模式**
   - GAT 精炼 region features 后，通过 `feedback_proj`（Linear + LayerNorm）映射回 768 维
   - 每个 token 通过 assignments 获取其所属 region 的精炼特征
   - 残差更新：`updated_tokens = original_tokens + region_feedback`
   - 在更新后的 tokens 上做 mean pool → 分类
   - 旧的 `fusion` 模式保留，通过 `--integration` 参数切换

3. **`train_gavit.py` — 新增命令行参数**
   - `--grouping attentive_spatial`（新默认值）
   - `--integration token_feedback`（新默认值）
   - `--num_regions` 默认改为 16
   - Checkpoint 命名包含 integration 类型

**架构对比**：

| 版本 | Region Grouping | 集成方式 | 分类器输入 |
|------|----------------|----------|-----------|
| v1 (legacy fusion) | SpatialGrouping 3×3, mean pool | 末端 concat | backbone_global(768) + graph_global(1024) = 1792 |
| v2 (token feedback) | AttentiveSpatialGrouping 4×4, attention | token-level residual | updated tokens mean pool = 768 |

**实验结果**：

- **Best Val Acc：96.2%**（epoch 26）
- **Test Acc：95.8%**（4526/4725）
- 参数量：31,386,920（全部可训练）
- Checkpoint：`checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth`
- 训练过程：epoch 25-30 val acc 在 95.9%–96.2% 之间波动，epoch 26 达峰；train acc 在 epoch 29 即达 100%，存在轻微过拟合
- 收敛特征：val acc 在 epoch 20 才突破 95.5%，收敛比 v1 慢（v1 epoch 20 已 ~96%），但最终结果持平

**对比 v1（GAViT K=9 spatial + GAT 2L，96.5%）**：
- Val Acc 96.2% vs 96.5%，略低 0.3%
- 尽管 attentive grouping + token feedback 架构更复杂，性能未见提升，可能原因：
  - K=16 比 K=9 区域更细，每个 region 内 token 数减少（~3 个），attention weighting 效果有限
  - Token feedback 引入了更长的梯度路径，30 epoch 内收敛不充分
  - NWPU-RESISC45 单标签数据集对 relational modeling 增益天然受限

**下一步计划**：
- [ ] 在 test set 上评估 v2（运行 `test_gavit.py`，加载 v2 checkpoint）
- [ ] 对比 v2 vs v1 vs Swin-T baseline 的混淆矩阵
- [ ] 可视化 v2 region 分区和 graph 连接（airport, bridge, church）
- [ ] 考虑延长训练至 50 epoch，或调整 lr schedule
- [ ] 准备 BigEarthNet 数据集实验

---

## 2026-03-27 — Backbone-Graph Fusion 架构实验 + Test Set 分析

### 一、Fusion 架构训练

**完成内容**：
- 修改 `gavit.py` 分类器架构：从纯 graph 特征改为 backbone global + graph global 拼接（768 + 1024 = 1792 维）
- 动机：之前的 GAViT 分类器只用 graph module 输出，backbone 全局特征被丢弃，fusion 旨在让两者互补
- 新建 `jobs/run_gavit_fusion.sh`，训练 50 epochs
- Checkpoint：`best_gavit_K9_spatial_knn_fusion.pth`

**训练结果**：
- Best Val Acc：**95.9%**（epoch 47-48）
- 训练过程：Train Acc epoch 42 达到 100%，Val Acc 从 epoch 28 的 94.4% 缓慢爬升至 95.9%

### 二、Fusion 模型 Test Set 分析

对 Fusion 模型跑了和原版 GAViT 相同的 per-class accuracy + 混淆矩阵分析（`compare_models.py --tag fusion`）。

**整体结果**：

| 模型 | Val Acc | Test Acc | vs Baseline (Test) |
|------|---------|----------|-------------------|
| Swin-T Baseline | ~96.0% | 96.3% | — |
| GAViT 原版（纯 graph） | 96.5% | 95.9% | -0.4% |
| GAViT Fusion（backbone+graph） | 95.9% | 95.9% | -0.4% |

两个 GAViT 版本在 test set 上表现完全一致（95.9%），均比 baseline 低 0.4%。但考虑到这种量级的差距（~0.4%）在单次运行中不具备统计显著性，三个模型可视为持平。

**Fusion Per-class 变化（vs Swin Baseline）**：

提升最大：
- roundabout: 97.1% → 100.0%（+2.9%）
- terrace: 95.2% → 98.1%（+2.9%）
- wetland: 89.5% → 91.4%（+1.9%）
- ship: 96.2% → 98.1%（+1.9%）

下降最大：
- bridge: 99.0% → 94.3%（-4.8%）
- airport: 98.1% → 95.2%（-2.9%）
- church: 87.6% → 84.8%（-2.9%）
- medium residential: 96.2% → 93.3%（-2.9%）
- lake: 97.1% → 94.3%（-2.9%）

**Sample-level 分析**：
- Baseline 错 → Fusion 对（修复）：67 samples
- Baseline 对 → Fusion 错（引入）：86 samples
- 净改善：-19 samples

**与原版 GAViT 的 per-class 对比**：
- 两者模式类似：复杂场景（airport, bridge, church）均未改善
- Fusion 波动幅度略小于原版（原版 palace -4.8%/lake -4.8%，Fusion 最大 bridge -4.8%）
- 两者提升的类也类似（ship, wetland, terrace, roundabout）

### 三、关键结论

1. **Fusion vs 原版 GAViT**：两种架构在 test set 上结果一致（95.9%），简单拼接并未带来额外收益
2. **两种 GAViT vs Baseline**：差距仅 0.4%，在单次运行中无统计显著性，三者可视为持平
3. **复杂场景未受益**：airport、bridge、palace 等需要 relational reasoning 的类别，两种 GAViT 均未改善，反而有所下降
4. **核心问题**：graph module 目前尚未展现出对复杂场景分类的独特价值

### 四、下一步计划

- [ ] 整理结果给导师发邮件，请教 graph module 与 backbone 的集成方式
- [ ] 根据导师反馈决定后续方向

---

## 2026-03-26 — Edge 消融实验 + 可视化对比 + 关键讨论

### 一、完成内容

1. **GAT 注意力可视化**（`visualize_graph.py`）
   - 对 `best_gavit_K9_spatial.pth` 生成 2×2 面板（原图 / 区域分配 / kNN图 / GAT注意力）
   - 6 类场景各 2 张，保存至 `results/figures/graph_vis_*.png`

2. **Spatial adjacency 边构建**（`models/graph_construction.py` 新增 `build_spatial_graph`）
   - 3×3 grid 的 8-邻接连边，边权均为 1
   - `gavit.py` 新增 `edge_type` 参数：`knn` / `spatial` / `hybrid`
   - `train_gavit.py` 新增 `--edge_type` 命令行参数

3. **Edge 消融实验**（`jobs/run_edge_ablation.sh`，Job 50315）

| Edge Type | Val Acc | 备注 |
|-----------|---------|------|
| spatial adjacency | **96.2%** | 固定 8-邻接 |
| cosine kNN | 96.1% | 动态边 |
| hybrid (合并) | 96.0% | kNN + spatial |

   **结论**：三种边策略准确率几乎一致，边定义对最终分类影响极小。

4. **Edge comparison 可视化**（`visualize_edge_comparison.py`，Job 50414）
   - 同一张图对比 spatial vs kNN 的 GAT 注意力，1×3 面板
   - 保存至 `results/figures/edge_compare_*.png` 和 `edge_comparison_summary.png`

### 二、关键讨论与反思

**GAT attention 权重的解读问题**：

- ⚠️ **GAT attention ≠ 语义关联度**。attention 权重反映的是"从哪个邻居获取信息最有用"（信息流方向），而不是"哪两个区域语义上相关"。
- 实际观察：stadium 中 R4（球场）对 R0、R2（角落/停车场）的注意力反而高于对 R3、R5（看台），因为差异大的邻居提供更多互补信息。
- 这意味着**用 attention 权重来直接讲"跑道+航站楼→强关联→机场"的故事是不准确的**。

**模型本身没问题**：

- GAViT 的 graph module 确实在帮助分类（96.0% → 96.5%），relational modeling 是有效的
- 只是 attention 权重不适合作为"展示 relational modeling 价值"的主要证据

**更好的展示方向（待验证）**：

- **注意力熵（attention entropy）对比**：复杂场景（airport, stadium）注意力应更集中（低熵），均质场景（forest, desert）注意力更分散（高熵）。"不均匀 vs 均匀"这个对比本身就说明模型根据场景动态调整了信息流。
- **混淆矩阵对比**：Baseline 分错但 GAViT 分对的样本，具体是哪些类
- **Per-class accuracy 对比**：但因为 baseline 已经 96%，差异可能很小，说服力有限

### 三、导师建议执行进度

- ✅ Region grouping 简单设计（spatial + kmeans）
- ✅ 轻量 GNN（2 层 GAT）
- ✅ 三级消融（baseline → +region → +graph）
- ✅ 图连接可视化（已完成，但解读方式需调整）
- ✅ 多种边定义系统对比（spatial / kNN / hybrid，准确率持平）
- ❌ AID 数据集验证

### 四、下一步计划（优先级排序）

- [ ] 计算注意力熵，量化"复杂场景 vs 均质场景"的注意力分布差异
- [ ] 在 test set 上评估最优模型，得到正式 Test Acc
- [ ] 混淆矩阵对比（Baseline vs GAViT）
- [ ] 消融：K 值影响（K=4 vs 9 vs 16）
- [ ] AID 数据集验证
- [ ] 给导师发进度邮件

---

## 2026-03-17 — 项目初始化 & Swin-T Baseline 完成

**完成内容**：
- 初始化项目结构，配置 Python 虚拟环境（PyTorch + timm + PyTorch Geometric）
- 编写 `split_nwpu.py`，将 NWPU-RESISC45 原始数据按 70/15/15 划分为 train/val/test，random seed=42
- 编写 `train_swin_baseline.py`：Swin-T (swin_tiny_patch4_window7_224)，pretrained=True，30 epoch，AdamW lr=3e-4，CosineAnnealingLR
- 编写 `test_swin.py`：加载 checkpoint 在 test set 评估

**实验结果**：
- Val Acc：~96%
- Test Acc：~96%
- 对比基线：本身即为基线

**遇到的问题及解决方案**：
- 无重大问题

**下一步计划**：
- [x] 实现 GAViT 核心模块（`models/` 目录）
- [ ] 训练 GAViT v1 并与 baseline 对比

---

## 2026-03-22 — P2a & P2b 完整实验：GAViT spatial/kmeans + 2-layer GAT

**完成内容**：
- P2a：GAViT K=9 SpatialGrouping + 2-layer GAT（4头，kNN k=5）
- P2b：GAViT K=9 KMeansGrouping + 2-layer GAT（重跑，含 argparse 版）
- 参数量：30,462,375（两组相同）

**实验结果**：
- P2a Best Val Acc：**96.5%**（epoch 30，仍在上升！）
- P2b Best Val Acc：**96.0%**（epoch 28-30）
- Checkpoint：`best_gavit_K9_spatial.pth` / `best_gavit_K9_kmeans.pth`

**消融结论**（完整链条）：

| 模型 | Val Acc | vs Baseline |
|------|---------|-------------|
| Swin-T Baseline | ~96.0% | — |
| + SpatialGrouping (no GNN) | 96.2% | +0.2% |
| + SpatialGrouping + GAT 2L | **96.5%** | **+0.5%** |
| + KMeansGrouping + GAT 2L | 96.0% | ≈0% |

- GAT 在 spatial grouping 基础上带来 +0.3% 提升，证明图推理有效
- KMeans 版性能低于 Spatial 版，原因：kmeans 随机性导致图结构不稳定，GNN 难以学习一致的区域关系
- Spatial+GAT 在 epoch 30 仍在上升（96.4%→96.5%），值得尝试更多 epoch

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 test set 上评估最优模型（`best_gavit_K9_spatial.pth`）
- [ ] 尝试 spatial+GAT 训练 50 epoch，观察是否继续上升
- [ ] 消融：K 的影响（4 vs 9 vs 16）
- [ ] 绘制训练曲线对比图

---

## 2026-03-22 — P2b 实验完成：GAViT K=9 KMeansGrouping + 2-layer GAT

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000）上运行 GAViT 完整模型（kmeans grouping）
- 模型：GAViT | K=9 | grouping=kmeans | GAT 2L×4H | kNN k=5
- 参数量：30,462,375（全部可训练，比 SpatialGrouping 版多约 290 万参数）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.0%**（epoch 29）
- 对比 Swin-T Baseline (~96.0%)：持平
- 对比 SpatialGrouping + no GNN (96.2%)：-0.2%
- 初步结论：kmeans 动态聚类引入了随机性，收敛更慢（epoch 1 train acc 仅 75.9% vs spatial 的 79.7%），最终性能与 baseline 持平

**训练曲线观察**：
- Epoch 1-10：收敛明显慢于 P1（epoch 10 val 92.2% vs P1 的 92.3%）
- Epoch 18-30：缓慢爬升，train acc 趋近 100%，val acc 在 95-96% 区间波动
- Checkpoint 保存至：`checkpoints/best_gavit.pth`（⚠️ 旧脚本名，argparse 未生效）

**注意事项**：
- 服务器拉取了 argparse 版本之前就已提交作业，导致只跑了 kmeans 一组，checkpoint 名为旧版 `best_gavit.pth`
- P2a（spatial + GAT）**尚未跑**，需补跑以完成消融对比

**下一步计划**：
- [ ] push argparse 版 `train_gavit.py` → 服务器 git pull → 补跑 P2a（`--grouping spatial`）
- [ ] 对比四组消融结果，绘制 bar chart
- [ ] 分析 kmeans vs spatial 差异原因

---

## 2026-03-22 — P1 实验完成：Swin + SpatialGrouping K=9（无 GNN）

**完成内容**：
- 在 NTU EEE GPU 服务器（NVIDIA RTX A5000, 24GB）上运行 P1 实验
- 模型：Swin-T + SpatialGrouping（K=9），无图推理层，直接 mean pool → FC 分类
- 参数量：27,555,495（全部可训练）
- 训练配置：30 epoch，AdamW lr=3e-4，CosineAnnealingLR，batch_size=32

**实验结果**：
- Best Val Acc：**96.2%**（epoch 27-28）
- 对比 Swin-T Baseline (~96.0%)：+0.2%（微弱提升）
- 结论：仅加 Region Grouping（无 GNN）相比 baseline 基本持平，说明区域表示本身信息量有限，需要图推理来充分利用区域间关系

**训练曲线观察**：
- Epoch 1-13：快速收敛，val acc 从 87% 升至 94%
- Epoch 14-28：缓慢爬升，train acc 趋近 100%（过拟合迹象）
- Best checkpoint 保存至：`checkpoints/best_region_only_K9_spatial.pth`

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] **P2** 运行完整 GAViT v1（Swin + SpatialGrouping K=9 + 2-layer GAT）
- [ ] 对比三组结果：Baseline vs +Region Grouping vs +Graph Reasoning
- [ ] 保存训练曲线图至 `results/figures/`

---

## 2026-03-17 — GAViT 核心模块实现完成

**完成内容**：
- `models/swin_backbone.py`：封装 Swin-T，去掉分类头，`forward_features` 输出 `(B, 49, 768)` tokens
- `models/region_grouping.py`：实现两种聚类方式
  - `KMeansGrouping`：特征空间 k-means（PyTorch 实现，GPU 兼容），assignment 步骤 detach，averaging 步骤可微，梯度可回传至 backbone
  - `SpatialGrouping`：将 7×7 grid 按空间位置分为 K 个宏区域（K 须为完全平方数），全程可微
- `models/graph_construction.py`：`build_knn_graph`，基于 cosine similarity 构建有向 kNN 图，输出 PyG batch 格式 `edge_index` + `edge_weight` + `batch`
- `models/graph_reasoning.py`：多层 GAT（`GATConv`），含 input projection、残差连接、LayerNorm、GELU、Dropout
- `models/gavit.py`：完整 GAViT 模型，串联以上所有模块
- `train_gavit.py`：训练脚本（默认 K=9, kNN k=5, 2-layer GAT, 4 heads）
- `test_gavit.py`：测试脚本
- `utils.py`：`set_seed`、`accuracy` 工具函数

**实验结果**：
- 模块实现完成，尚未运行训练

**遇到的问题及解决方案**：
- 无

**下一步计划**：
- [ ] 在 GPU 服务器上运行 `train_gavit.py`，获得 GAViT v1 结果
- [ ] 将结果填入 `results/comparison_table.csv`
- [ ] 保存训练曲线至 `results/figures/`

---
