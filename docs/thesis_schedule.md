# GAViT 论文五周目标排期

10-12最新收尾：NWPU43三模型与BigEarth soft43正式训练/test/资源全部归档。BigEarth best第8轮/val mAP78.99004691745016%，test mAP72.867571%；整批6小时22分21.73秒/折算10.64元，加NWPU43共7小时22分39.06秒/12.32元（1.67元/h，准备/短测/空闲另计）。[正式记录](../results/bigearth_soft43_ablation.md)与[资源核验](../results/evidence/bigearth_soft43_complete_20261012/resource_validation.json)齐备，不再索取同批摘要或重复启动。接下来用已有逐类结果、预测与可视化整理论文，明确可支持的主张；最终三seed、公平效率与外部方法缺口仍保留，不自动新起付费训练。

10-12结果解释：soft43三指标均高于旧43对照（mAP72.867571%）。[BigEarth共同43/45表](../results/bigearth_three_model_two_seed_summary.md)：soft−fixed两次mAP升，soft−Swin一升一降；NWPU soft两次低于Swin，不能写成稳定跨任务优势。下方按当时状态保存的启动/工程/待回传步骤均为历史。

10-11最新执行状态：BigEarth soft43当前4090工程96/64单轮已回传exit0/complete，CUDA重载2.38e-7、48+32边/探针归一化通过，资源与记录已归档；正式30轮→best完整test命令已交付，尚未收到PID/GPU/首batch。下一步执行[正式步骤](run_seed43_three_models_20261011.md)，首完整epoch后更新6–8h预算；当前1.67元/h估计10.02–13.36元。保留旧43 saved21轮/硬件/版本限制，不重跑NWPU或旧对照。以下条目为历史状态。

10-11最新正式结果：NWPU43三模型test OA为Swin93.857143、固定93.765873、soft93.626984%，各选轮23/26/29，总exit0，整批1小时0分17.33秒，按1.67元/h约1.68元。固定42正向但43负向，双seed均值差仅+0.039683pp；soft两次负向。42/43硬件分别2080Ti/4090，n=2仍不满足最终三seed。NWPU本批收尾，不因结果差重跑。[正式双seed结果](../results/nwpu_two_seed_summary.md)。BigEarth soft43只读预检已PASS，当前同一4090/软件/精度核对通过，下一步96/64单轮目标GPU工程检查，再接已授权30轮→best完整test；仍无新BigEarth精度，[当前执行步骤](run_seed43_three_models_20261011.md)。以下条目按当时状态保留为历史。

10-11 NWPU43九阶段全部回传STAGE_COMPLETE和总完成标记，训练/测试调度验收已完成；正式分数/选轮/终态exit及资源待紧凑摘要回传。九阶段合计1小时0分15秒，按1.67元/h折算约1.68元，准备/空闲另计。下一项为BigEarth soft43只读预检，不重跑NWPU，见[最新记录](../results/evidence/seed43_nwpu_queue_complete_20261011/summary.json)。

10-11 NWPU43正式Swin校准已回传RTX 4090首batch和30/30完成标记；最大val OA93.095238%、打印首次epoch23，正式test待回传。当前tail为已结束阶段，总run.log的阶段验收与fixed/soft后续需只读确认，不重复启动。[当前进度](../results/nwpu_seed43_calibration_progress.md)。BigEarth仍待本批结束后预检。下方此前状态保留为历史。

10-11 当前4090工程一轮已回传完成，48+32图边及360/90前后向正常；新的正式启动命令先只读核对已存工程产物，再接已授权三模型九阶段，无需再运行smoke。正式首batch/GPU占用尚待回传，BigEarth仍待预检，[执行说明](run_seed43_three_models_20261011.md)。

10-11 NWPU43新版只读预检已回传PASS，RTX 4090及冻结输入符合；先按项目训练前规则完成360/90一轮软分配目标GPU短测，再接已授权三模型九阶段。工程目录独立，不覆盖正式权重，不用短测分数作模型比较。BigEarth预检仍待回传，当前未启动正式训练，见[执行说明](run_seed43_three_models_20261011.md)。

10-11 当前执行范围已由用户明确：补NWPU seed43的Swin/固定4×4/软分配三模型，以及BigEarth soft43；保留原43对照和42/45结果。用户已回传RTX 4090，旧2080 Ti预检误拒绝已修正，四条命令重打包并通过23项本地检查；新版完整服务器预检待回传，训练未启动，[执行说明](run_seed43_three_models_20261011.md)。当前4090计划估计串行7–10h，按用户1.67元/h约11.69–16.70元；12h为20.04元，含恢复数据等可预留25–30元，实际首轮再校准。本批优先于尚未批准的机制训练对照；不自动扩大架构搜索。两seed不等于满足全部多seed要求，BigEarth固定43停止21轮与历史硬件/版本差异继续披露。

10-11 硬件安排更新：用户决定后续补充 seed 统一使用 RTX 4090。同一 seed 的 Swin、固定分组和软分组 GAViT 对照按同硬件/软件/精度配对，记录 GPU UUID，固定数据划分、预训练、训练配方及选轮规则；新增 seed 数量、列表、模型范围和预算在启动前确定。新增统一 4090 的批次与历史四 seed 分别报告，并保留全部历史结果及总汇；不能按 test 成绩挑 seed 或据此认定旧波动源于硬件。旧 2080 Ti 待执行命令须先更新 GPU/资源检查，各数据集原精度设置保持；正在执行的任务不自动中断。这是已确认的硬件决定，本轮仅文档记录，未启动新训练。详见[实验日记](../research_diary.md)。

10-10 当前研究进展：空间先验NWPU seed42单候选已完成校准与缓存收尾，best epoch24、OA92.380952%、macro-F1 92.342474%，均低于旧soft与Swin；对旧soft纠正26/新增错误43，净−17。按冻结规则关闭候选，无需继续服务器补件或扩训，见[最终记录](../results/nwpu_spatial_prior_trial.md)。用户仍优先讨论模型机制，下面旧“优先写作”不是自动恢复写作或启动新训练的指令；目前不据失败结果自动搜索更多配置。

10-06 后续写作完成：[相关工作三主线初稿](thesis_related_work_20261006.md)与12篇[BibTeX](thesis_related_work_20261006.bib)已保存，核对ViT/Swin、遥感ViT/HGTNet/DFAGCN、GAT/ViG与SLIC/SSN/Slot Attention/GroupViT/STViT。正文明确固定成员与注意力权重、S/W、锚点图与内容归属，以及相近汇总/回映的先例，不把组合本身宣称首创或已证实优越。相关工作由“待起稿”改为“初稿完成、待审阅/格式修订”；外部方法实测及其余实验缺口保持。本轮未新训练、改模型或提交/推送。

10-06 当前推进：已形成[三数据集实验与讨论初稿](thesis_experiments_discussion_20261006.md)，串联AID三seed、BigEarth四seed历史统计、NWPU/BigEarth单seed软分配、90例可视化、反馈幅度和FFN负面诊断及效率范围。[导师逐项状态](supervisor_latest_email_received_20260924.md)已纠正“软分配未实现”等旧描述。作者仍无回复，[有界公开材料复查](external_methods_refresh_20261006.md)未取得新完整实现；外部比较、多seed软分组、3×3/4×4受控比较与相关工作章节仍未完成。接下来优先外部材料/协议与文献写作，不自动扩训FFN或重跑负面seed；本轮无GPU任务。下方日期条目为历史，不是新的执行指令。

10-05 最新收尾：FFN完整缓存回传通过，OA与macro-F1均低于对应原版；soft纠正27/新增34，净−7。四格表、配对/逐类与资源证据已归档，候选停止扩训，本批不再索要摘要或补跑；[最终记录](../results/nwpu_ffn_calibration42_comparison.md)。以下“缓存待回传”为历史状态，论文其他未完事项保持独立。

10-05 FFN双校准均30轮/exit0完成回传：Swin+FFN94.047619%、soft+FFN93.174603%，分别较原模型−0.158730/−0.555556pp，冻结继续条件未通过；[四格结果与解释](../results/nwpu_ffn_calibration42_comparison.md)已保存。停止该候选扩训，仅做已有缓存macro-F1/曲线/配对和身份补件；正式test与稳定性未新增，下面“校准待启动”为历史状态。

10-04 最新回传：FFN两组360/90、一轮GPU短测均exit0，初始化/梯度/更新/重载/soft图边字段核对通过；[短测证据](../results/nwpu_ffn_smoke_validation.md)已归档。已交付[两组seed42、30轮校准](run_nwpu_ffn_calibration_20261004.md)，服务器尚待启动回传；旧对照复用，预留约1–1.5小时、首轮修正，不自动扩seed/refit/test。下面早先“待短测”状态保留为历史。

10-04 FFN候选已独立发布：`codex/token-ffn-nwpu-20261003` / `058f1d8`，干净发布checkout144项CPU回归通过，远端SHA一致，原开发目录修改保留。[服务器同步与两组一轮GPU短测命令](run_nwpu_ffn_smoke_20261003.md)已交付，当前服务器未执行；先核验真实短测/成本，之后仅两组NWPU seed42 30轮校准，不自动扩训练范围。

10-03 用户批准有界FFN候选后，本地实现和CPU验收已完成：[冻结协议](run_nwpu_ffn_trial_20261003.md)、[234项回归与发布候选](../results/nwpu_ffn_local_validation.md)。只增加Swin+FFN/soft+FFN两次seed42 NWPU校准，复用旧对照；初始logits/参数/RNG保持、两步梯度及strict重载已验证。9文件补丁已整理，独立Git发布/完整checkout验收和真实GPU短测尚未完成，服务器未启动；先各360/90一轮短测再各30轮，不自动test/refit/扩seed/BigEarth。效果与当前GPU成本待测，原负面结果及外部方法/多seed/公平效率未完范围保持。

10-03 [NWPU反馈幅度诊断](../results/nwpu_feedback_sensitivity_seed42.md)已收尾：run.exit0、1260val、原预测复现及计数核对通过，46.33秒直接读ZIP；fixed/soft四档减弱均OA/F1下降、CE变差，关闭反馈各净少对9张。“简单调小可改善”的有限假设未获支持，到此结束该幅度扫描，不继续门控训练/α搜索/test重跑。结果纳入讨论；原正式表和独立的多seed、外部方法、公平效率范围保持。以下日期保留历史。

10-02 用户确认继续有界诊断：[NWPU固定/软分配校准best的全1260张val反馈强度检查](run_nwpu_feedback_sensitivity_20261002.md)已实现，25项本地检查、真实Swin合成输入CPU重建和单行交付验证通过。服务器尚未启动；预设α=0/0.25/0.5/0.75/1，仅推理并缓存，不新增训练或test、不自动选择α。先核对原预测复现，再根据幅度趋势与类别损益决定是否形成后续训练假设；无收益也收尾。原多seed/外部方法/公平效率范围保持。

10-01 双数据集分析整合完成：[正式消融表、证据分级与中英文正文](../results/soft_assignment_cross_dataset_ablation.md)已核验。NWPU soft−fixed OA−0.234127 pp，BigEarth mAP+0.578683 pp但micro-F1下降；目前不支持稳定整体增益。下一步将表格和已有图稿纳入实验/讨论章节，外部方法材料按既有请求接收核对；多seed与公平效率仍列未完范围，本轮不自动新增付费训练。以下条目按日期保留历史。

10-01 资源归档完成：BigEarth soft45总队列14小时44分18.733秒，训练阶段14小时37分58.069秒，测试阶段5分22.637秒；训练CUDA allocated峰值3.659762 GiB、reserved峰值4.029297 GiB。摘要与冻结身份及既有测试指标一致，固定8张val的3份快照归一化/48+32边检查通过。本轮训练、测试和资源/探针归档已收尾，无需补跑或再次回传同批日志。[完整记录](../results/bigearth_soft45_ablation.md)。这仍是单seed混合结果，多seed、新结构公平效率及外部方法比较另列未完项。

10-01 完成回传：BigEarth soft45已完成30轮及119825张test，总/训练/测试退出码均0。mAP70.836919%、macro-F1 64.864823%、micro-F1 76.383354%；较固定45为+0.578683/+0.569346/−0.319103 pp，仍低于Swin。无需重训/续训/解压；[三模型正式结果](../results/bigearth_soft45_ablation.md)已归档，下一步仅读取现有summary补齐总耗时/显存/探针。单seed结果不代表双数据集多seed要求已完成。

10-01 用户报告实例已归还：BigEarth soft45是否已在此前完成待work产物核验。先查总/阶段退出码、日志、已保存轮次和best/last/test产物；不默认从头重训，不先解压。原队列不再视为持续运行，新指标尚未确认。

09-30 启动回传：BigEarth soft45已在CUDA首轮执行到1201/7434（16%），237871/122342样本、31,466,766参数和原批准2080Ti环境相符。当前训练循环约25分钟/轮的粗估不含验证/存盘；总预算暂维持16–18小时，等完整epoch修正。正式训练中、val/test结果及完成退出码待回传，继续已有队列。

09-30 最新状态：BigEarth soft45服务器全量文件/划分、原对照权重和环境预检已PASS；同型号/驱动2080Ti、物理UUID变化已记录。已交付[单模型30轮→best完整test后台命令](run_bigearth_soft45_20260930.md)，12项本地检查和独立审阅通过，服务器训练待启动。保留旧seed45 Swin/固定结果，新的soft指标待测试，不将预检/命令交付标为实验完成。

09-30 最新执行决定：用户重新确认按导师要求开展BigEarth软分配，当前RTX2080 Ti、数据待恢复。先核验并复用完整30/30的seed45 Swin/固定GAViT，仅补soft_prototype seed45；[冻结方案与单行预检](run_bigearth_soft45_20260930.md)已准备，服务器预检/正式训练尚未执行。暂预留16–18小时训练与测试、另加数据恢复，首轮后修正；单seed不代表完整多seed要求已满足。下方同日撤下安排是较早历史，已被此次明确指令取代。

09-30 用户决定不开展BigEarthNet软分配正式训练：数据规模/耗时成本高，当前收益证据不足，该项从当前执行队列撤下，不准备或启动付费任务。NWPU软分配seed42的正式负面结果及真实可视化保留；不据此宣称软分配在其他数据集无效。导师原邮件明确要求NWPU与BigEarthNet双数据集三模型消融，因此记录为“未执行，论文范围调整待与导师沟通”，不能标为完成或由AID/小样本自动替代。下一步先明确已有证据能支持的论文主张与剩余对照需求，优先归档/写作和外部方法比较材料核对；新增训练须有独立研究问题及预算。

09-30 真实90图册已回传并审阅收尾：90唯一案例/45类/180SVG，全部总览及3纠正、5共同错误的完整S图已检查；主归属变化39.09%，但没有一致的语义区域改善证据，正式test负面结果保持。已保存[分析、全部正反例图和中英文图注](../results/nwpu_regions_visual_review_20260930.md)。本批不再追加捕获或训练；BigEarth正式软分配、多seed、效率和外部方法比较仍按独立未完项处理。

09-29 E05真实90图捕获已完成：run.exit=0，45类各2张val、fixed/soft正确82/85，纠正3/新增0/共同错误5；全部捕获门禁通过，总19分33秒、90张直接来自ZIP。固定案例不替代正式test负面结果。现准备[轻量HTML导出](compact_nwpu_regions_20260929.md)，下一步取回真实图册做语义观察；不用再跑GPU。本批[完成记录](../results/nwpu_regions_server_completion.md)，多seed/BigEarth正式消融及新结构效率等未完范围保持。

09-29 E05 NWPU seed42本批完整收尾：三模型校准/refit/test及缓存复算、配对/部分逐类与混淆分析完成。soft对fixed纠正538/新增597（净−59），对Swin570/586（净−16）；没有整体增益证据，不重跑本seed或继续索取同批摘要。后续多seed/BigEarth正式消融、可视化和效率仍待完成，未新增GPU任务。[收尾记录](../results/nwpu_seed42_closeout.md)。

09-29 E05 NWPU seed42正式六阶段已完成：6300图refit29/29/28及25200test，六阶段/队列exit0；Swin/fixed/soft OA93.932540/94.103175/93.869048%，soft验证优势未延续，全部结果保留。耗时66分05秒；下一步仅缓存配对统计，不启动新训练。双数据集多seed等仍待完成。[正式结果](../results/nwpu_seed42_test_comparison.md)。

09-29用户确认继续，已交付[seed42三模型refit/test六阶段队列](run_nwpu_refit_test42_20260929.md)：先6300图29/29/28轮正式拟合，再统一25200test；同代码/配方、从原预训练重新初始化、Cosine周期30。预计同卡1–1.5小时；本地语法/调度/失败停止/防重复验证通过，服务器尚未启动，当前无新测试指标。

09-29 E05缓存诊断完成：身份一致；soft相对fixed纠正40/破坏31，相对Swin纠正30/破坏36；macro-F1排序同OA。末期无明显崩落，8张非分层探针显示分配更分散，不能证明全局塌缩。建议冻结当前版本，下一批做seed42三模型refit29/29/28及test，预算约1–1.5小时；尚未启动，暂不扩多seed或BigEarth完整训练。[诊断与当前表](../results/nwpu_seed42_cached_diagnosis.md)。

09-29 E05三模型seed42校准已完成：三阶段/队列exit0，各30轮；val OA为Swin94.206349%、fixed93.015873%、soft93.730159%。soft对fixed +0.714286 pp，对Swin −0.476190 pp；只是单seed最佳验证，不是test或稳定提升。队列实测72分26.87秒。下一步只读已有预测/曲线/分配探针；暂不追加refit/test、多seed或调参。[本批结果表](../results/nwpu_seed42_calibration_comparison.md)。以下条目保留历史。

09-28 E05双数据集工程验收收尾：BigEarth缓存/日志/480波段备份核对通过，备份9.62MB。NWPU哨兵缺图，先恢复既有ZIP，再执行[同协议三模型seed42、各30轮校准队列](run_nwpu_ablation42_20260928.md)，串行失败停止、启动前查复用结果。命令/本地调度验证已完成，服务器队列尚未启动，正式效果/多seed未完成。

09-28 E05 BigEarth真实GPU单轮返回：96/64、RTX2080Ti、exit0，重载差4.77e-7、48+32图边；总26.11秒、allocated/reserved3.66/4.04GiB。双数据集训练/重载均已有证据，BigEarth探针张量/备份及日志摘要做一次只读收尾后，进入NWPU三模型seed42校准。37.0547%仅工程短测，不进正式表。[验收与收尾](../results/bigearth_soft_smoke_validation.md)。

09-28 E05用户报告BigEarth数据已恢复；[96/64单轮短测命令](run_bigearth_soft_smoke_20260928.md)交付，包含原图检查、持久化RGB小包、训练/严格重建/图边与资源记录。本地真实模型合成TIF的CPU包装命令验证通过（含缺图/目录占用拒绝），服务器准备与CUDA结果待回传，正式训练未启动。

09-28 E05数据恢复依赖：用户返回data目录不存在，work两层未找到BigEarthNet原包，仅有AID/NWPU及小工具ZIP。下一步重新添加此前BigEarthNet-S2数据；若平台允许只下载则优先取原包后定向恢复480波段，否则按平台流程恢复。固定抽样/划分不变，未启动新训练。详见[恢复说明](check_bigearth_soft_data.md)。

09-28 E05 BigEarth短测预检：96/64抽样的480个原始RGB波段在预期路径全部缺失；冻结CSV和标签断言通过。下一步[定位现有压缩包](check_bigearth_soft_data.md)并优先定向恢复，不换抽样或启动缺图训练；压缩包是否仍在待确认。NWPU工程验收已通过，正式三模型实验尚未启动。

09-28 E05 NWPU软分配工程检查收尾：用户返回SNAPSHOT_CHECK_PASS和真实CUDA的ASSIGNMENT_GRAD_PASS；S/W归一化误差≤2.98e-7，分配专属梯度到达投影/注意力/骨干，权重与原输入哈希一致。与已完成训练/图边/重载证据合并验收，不代表准确率提升。下一步[BigEarthNet原160图存在检查](check_bigearth_soft_data.md)及96/64的1轮BCE短测，随后才安排NWPU同协议三模型校准；保留全部历史结果，未启动正式训练或修改发布代码。

09-28 E05探针摘要通过数值核对：纯训练3.236秒/111.25图每秒，S平均变化0.03925，低范数0，四快照存在。现有摘要不替代张量与S专属梯度验收；已准备并本地验证[两条服务器检查](check_nwpu_soft_final.md)，包括有效路径和损坏/断梯度反例，真实CUDA结果待回传。BigEarth原图短测与正式三模型实验仍未启动。

09-28 E05 NWPU软分配_02训练/重载通过：ed480e6干净、20源码哈希匹配，RTX2080Ti真实360/90、1轮、exit0；48+32图边、重载差3.58e-7。总131.21秒、allocated/reserved3.66/4.04GiB，8val探针摘要完成。40%小样本OA仅作工程记录。下一步读取history与分配探针详情，补缓存张量/分配专属CUDA梯度；BigEarth原图短测及正式三模型比较仍待执行。[工程验收](../results/nwpu_soft_smoke_validation.md)。

09-28 E05 NWPU短测仍待验收：_01先有原图缺失记录，用户解压后重试因目录已存在退出；追加日志包含两段错误，不能认定恢复后仍缺图。已提供同manifest/seed/代码的31500路径只读检查与_02新输出重试，保留失败记录；未修改模型或追加正式训练。[恢复命令](retry_nwpu_soft_smoke_02.md)。

09-28 E05独立代码发布完成：用户授权选项2，`codex/soft-assignment-20260927`已推送`ed480e60fea03ed786fdba70c144de5e815c34f5`，[草稿PR#1](https://github.com/PatrickYxz/GAViT_Project/pull/1)已创建，未合并。候选133项测试通过且源码哈希一致，发布23文件。下一步用户在服务器固定SHA同步、执行NWPU360/90的1轮GPU短测；尚无该运行结果，正式消融待后续。

09-27 E05发布候选整理完成：基于服务器`deb2b45`，限定18个代码/测试文件，候选完整CPU回归133项通过/exit0，源码与先前验收相同。独立分支`codex/soft-assignment-20260927`的发布方式待确认；[补丁与验收](../results/soft_assignment_release_validation.md)已准备，GPU短测尚未启动。

09-27 E05服务器身份通过：RTX2080 Ti 22528 MiB（查询时空闲22009），服务器`deb2b45`干净；NWPU清单SHA256为`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`，预训练SHA256匹配历史。下一步准备基于该版本的独立soft发布候选；尚未发布或启动GPU短测。

09-27 E05数据准备完成：group策略全量prepare回传成功，扫描47.6秒，train5040/val1260/test25200；8组同类重复均未跨划分，31500文件保留、31492个唯一像素图像。固定`nwpu_protocol_20260927_02.json`供三模型共用。下一步获取manifest/预训练SHA256、服务器commit/dirty和GPU，再发布soft独立版本、做360/90的1轮GPU短测。新代码仍未提交/推送，尚无新GPU或模型效果结果。[执行说明](run_soft_assignment_smoke.md)。

09-27 E05完整图像扫描回传：31500图解码与哈希检查完成（46.4秒），首次prepare因机场类同像素图片050/420被reject策略终止，manifest尚未写入。下一步用已实现的同类重复分组策略生成`nwpu_protocol_20260927_02.json`，保留全部图片、seed42/4242与精确份额，同组不跨划分；完整重复报告及成功结果仍待回传，不重新下载或启动训练。[执行说明](run_soft_assignment_smoke.md)。

09-27 E05数据准备更新：用户服务器RESISC45原图ZIP下载及SHA256校验完成，`/home/featurize/data/NWPU-RESISC45`共45类31500图、每类700的断言通过。下一步复用已发布统一入口做逐图检查并生成seed42/4242独立manifest（预期5040/1260/25200）；完整图像校验、新soft代码发布、实际GPU及1轮短测仍待验收。[执行说明](run_soft_assignment_smoke.md)。

09-26 E05本地实现完成：`soft_prototype`局部原型分配、软反馈、严格版本配置、scene/BigEarth入口和固定探针已接通；新增99,840参数。独立审阅发现的中断快照缺失已复现并修复，最终CPU回归190项通过/exit0。见[本地验收](../results/soft_assignment_local_validation.md)与[服务器短测顺序](run_soft_assignment_smoke.md)。下一步确认NWPU数据目录、发布独立固定版本，再跑NWPU与BigEarthNet各1轮原图短测。当前代码未提交/推送；真实GPU、费用、精度和正式三模型/多seed消融仍未完成，不改历史结果表。

09-26用户回复“ok”确认E05设计，[实现计划](superpowers/plans/2026-09-26-local-prototype-soft-assignment.md)已完成，待实施前审阅。已验证现有22项无Torch身份/入口检查；本地旧测试环境缺少Torch等依赖，实施先恢复独立环境，再做可微分组/软反馈/旧checkpoint兼容和双入口验证。当前没有新模型实现或GPU训练，正式消融尚未完成。

09-26用户要求助手明确推荐路线，E05收敛为[局部区域原型软分配](superpowers/specs/2026-09-26-local-prototype-soft-assignment-design.md)：共享投影与当前图片原型的局部cosine分配；取消初稿90%/10%及首批额外冻结对照。先工程检查和短测，再NWPU seed42三模型校准，之后按结果/预算安排BigEarthNet及正式多seed。现阶段设计明确、尚未实现。

09-26按用户建议完成[软分配GitHub源码比较](soft_assignment_github_review_20260926.md)：5仓库11文件静态核对，补充STViT关联stop-gradient、SSN局部可微聚类及原型相似度候选。E05初稿分配器/先验尚未冻结，下一步细化候选及对应对照；没有实现或新训练。

09-25 E05已形成[软分配设计草案](superpowers/specs/2026-09-25-soft-assignment-design.md)，等待设计审阅，尚未实现。推荐局部空间先验引导的软分配，同步区域汇总与反馈；增加NWPU单seed冻结软分配对照的建议及完整验证要求。先工程检查和短测，再按验证集冻结配置与预算，E02等待作者材料的状态不变。

09-25用户确认已发送作者请求邮件，E02进入等待回复与材料阶段，尚未取得新实现或配置。等待期间先推进E05本地设计准备，核对软分配在区域汇总、图反馈及空间边中的一致用法；这是材料等待期间的排期调整，E02仍保留必做状态。本次未启动GPU训练。

09-24按用户“继续”推进E02：公开材料与GitHub直接检索核对完成，同名仓库均未能提供目标方法实现。已准备[作者请求草稿](external_methods_requests_20260924.md)及[接入记录](external_methods_readiness_20260924.md)，当前等待准确源码/配置，未联系作者、未启动GPU。E02保留必做状态，E05本地设计不依赖作者回复，但尚未实施。

09-24收到用户重新提供的[最新导师邮件全文](supervisor_latest_email_received_20260924.md)。确认可学习软分配须实现并在NWPU/BigEarthNet做三模型消融；HGTNet/DFAGCN比较、各数据集内部统一训练/评估协议及多seed统计均属论文必做项。剩余优先级按邮件为外部方法比较→软分配消融→相关工作整合，同时核对可视化覆盖。当前批次收尾不取消这些范围；本次仅校正记录，未启动开发或训练。矩阵旧版“外部方法保留各自配方作为主比较”的安排已修正。

09-24 用户决定先收尾：本轮配对实验、归档和原因解释整理结束，不再追加训练或诊断。[收尾说明](../results/bigearth_closeout_interpretation.md)区分已确认的类别差距、结构假设与尚未建立的因果；E02/E05和E06完整数据处理链等独立事项保留未完成状态，本轮不启动。

09-24 E04收尾：seed45双方30/30，总及六阶段exit0，Swin/GAViT mAP71.6769/70.2582%，差−1.4187 pp；六阶段实测27.8263小时。已核对冻结身份、权重报告链与19类AP均值。[seed45结果](../results/bigearth_seed45_comparison.md)、[四seed汇总](../results/bigearth_four_seed_summary.md)完成；四seed mAP均值约71.27±0.34%/71.04±0.94%，平均差−0.23 pp，两正两负，未显示稳定增益。历史硬件/中断与舍入边界保留，旧分数未改。E04不再追加seed，E02/E05及E06完整数据链等独立研究项仍按原排期保留。

09-23 E04 seed45回传更新：实际RTX2080Ti；GAViT30/30及119825张测试到100%，Test mAP70.3%/macro-F1 64.3%/micro-F1 76.7%，队列打印BIGEARTH SEED45 COMPLETE。已归档[阶段结果](../results/bigearth_seed45_partial.md)；Swin45指标、精确best/身份、总及分阶段exit和实际耗时仍待只读摘要，尚不能汇总四seed配对结果。不再安排本轮训练或换卡，已有历史结果保持。

09-22 用户决定下一步优先E04 BigEarthNet seed45，并确认数据已完整解压。冻结协议的Swin/GAViT各30轮与best测试队列已准备；复用deb2b45中与a23a566一致的BigEarthNet代码，先全RGB预检及双模型1轮smoke。四项本地包装测试和启动语法/失败停止检查通过；尚未收到服务器GPU启动/完成证据，不计入已完成seed。[完整单行启动及协议](run_bigearth_seed45.md)。42/43/44和AID结果保持不变。

09-22 E06模型前向回传完成：seed42 refit配对、RTX2080Ti/FP32/224×224驻留GPU合成输入，四阶段exit0，总88.26秒。Swin/GAViT batch1平均延迟14.891/20.016 ms（+34.41%），batch32吞吐564.892/478.433图/秒（−15.31%），峰值allocated594.482/609.353 MiB（+2.50%）。[效率表与核验范围](../results/aid_efficiency_summary.md)已归档；结合三seed OA均值差+0.0467 pp，当前固定图版本未显示足以支撑开销的稳定收益。原图读取/处理/传输不计时，完整处理链仍待测，不将E06所有范围或论文实验整体标记完成。

09-22 AID准确率与诊断状态：预定三seed指标汇总及限定范围的曲线/重点混淆检查完成。OA差+0.60/+0.30/−0.76 pp；Swin96.6800±0.2946%、GAViT96.7267±0.5707%，均值差+0.0467 pp，未显示稳定整体提升。106条epoch记录确认选轮25/21及T_max30一致；12条混淆行定位44的Commercial→School/StorageTanks及RailwayStation→Industrial错误增多，未建立图模块因果。见[诊断收尾](../results/aid_seed44_curve_confusion_analysis.md)和[三seed主表](../results/aid_multiseed_summary.md)。后续E06模型前向结果见上，E02/E05保留，不因44负面结果重训或挑seed。


09-20 E00完成：AID全量预检、Swin/GAViT在RTX2080Ti的真实GPU短测均exit0，清单/预训练/配方/环境一致，GAViT实际48+32边与保存重载通过。下一步E01先执行Swin seed42的4000/1000、30轮校准，再GAViT同流程及5000训练池重训/最终test；尚未实际启动。见[正式执行记录](run_aid_formal_seed42.md)和[验收表](../results/aid_smoke_validation.md)。


09-20服务器验收更新：AID全量prepare通过，4000/1000/5000，一组重复图均train；Swin seed42在RTX2080Ti上240/60样本、1 epoch、workers2完成，exit0，保存重载差0，Git deb2b45干净。E00的数据与Swin GPU部分通过，下一步同清单/预训练执行GAViT短测，整体仍待双模型验收；未启动正式训练。见[工程验收表](../results/aid_smoke_validation.md)。


更新：2026-09-20。依据本任务中导师邮件及用户确认的执行安排。
时间假设：还有一个多月，暂按 2026-09-07 至 2026-10-11 倒排；10 月 11 日是内部目标，不是已确认的提交截止日。

## 2026-09-14 当前结果更新

09-20 AID本地接入完成：新增统一prepare/train/evaluate入口，18项新测试与相关回归共74项通过；两模型30类合成CPU命令行训练/评估exit0，保存重建差0，真实GAViT48+32边通过。实际AID数据、服务器ImageNet/CUDA和双worker仍待验收；本地双worker失败及超时已留档，未以workers=0成功替代。用户确认服务器尚无AID；代码已在独立分支codex/aid-entry-20260920发布（2ba112a），隔离分支完整89项测试/34个subtest通过，用户服务器拉取及运行待确认。见[运行指南](run_aid.md)、[验证报告](../results/aid_entry_local_validation.md)；E00本地部分完成，整体继续待服务器验收，不启动seed42正式训练。

09-19 W1-09核对完成：AID、HGTNet、DFAGCN论文身份、原始划分/训练流程与当前代码差异已查明；[方法审计](external_methods_audit_20260919.md)与[执行矩阵](experiment_matrix_20260919.md)已落地。HGTNet/DFAGCN官方代码、权重和代码许可证尚未核实，不标记为可直接复现。下一步E00为AID统一入口与双模型smoke；再安排seed42配对。矩阵建议AID50/50训练池/测试、池内验证后重新拟合，新增协议仍待实现冻结。原第二周“接入与smoke完成”目标尚未达到；外部方法正式日期取决于实现可用性，seed45与soft assignment保留。此次未修改训练代码或启动GPU任务。

09-19论文案例图完成：四个原图翻转例已生成[180 mm宽PDF和300 dpi PNG](../output/pdf/seed44_feedback_figure/README.md)，附中英文图注及LaTeX片段。实际图像像素不变，320条捕获图边、分数、字体嵌入、透明度和页面文字边界核验通过，最终渲染已逐行检查，W1-08完成。row4439提供的背景干扰线索仍是假设，未变成因果结论。没有新解压、推理或训练；旧HTML浏览器file://限制未绕过。

09-19原图反馈诊断完成：`pair44_feedback_k_7ehbne`，RTX2080 Ti、24张original，全部内部与拓扑检查通过，缓存最大概率差约1e-6，26.69秒/exit0。草地与沙滩各纠正1、破坏1；湿地无阈值翻转。原图复现与固定案例数值干预阶段结束；W1-08的真实原图图边捕获已获得运行证据，人工检查图像及论文图整理仍待完成，不整体勾选。下一步看已有4个翻转案例与湿地对照，无需重复评估或训练。见[原图报告](../results/seed44_original_feedback_diagnosis.md)。

09-18反馈完整日志：RTX3060/gallery/24张不同图，内部数值与真实图输入边检查通过；但normal与原RGB缓存最大概率差0.889688，关注类别4项阈值翻转。JPEG结果不足以代表原RGB案例，当前优先确认原数据/ZIP并只恢复24张所需72个波段，再核对缓存；不追加全量推理或训练。W1-08的捕获代码已完成GPU工程检查，原RGB匹配与论文图验收仍待完成，见[更新分析](../results/seed44_feedback_diagnosis.md)。

09-18反馈诊断返回：`pair44_feedback_qnjm_rb7` 已完成、exit0，草地/沙滩/湿地各8个类别案例；反馈纠正1/2/0、破坏0/1/0。合计24个图片×类别判断，不当作24张不同图片或总体指标。下一步读取已有逐张分数和JPEG缓存差异，查看具体纠正/破坏案例；实际图像和张量尚未检查，W1-08继续待验收。见[案例结果](../results/seed44_feedback_diagnosis.md)，无需重跑诊断或完整解压。

09-18图反馈诊断准备：用户确认实例已重启，改用 `work` 持久化JPEG案例，避免为小检查再次解压。真实前向捕获、同权重反馈屏蔽与恢复检查已实现，11项新测试+16项回归通过；[粘贴命令与说明](run_bigearth_feedback_diagnostic.md)已准备，GPU运行待回传。JPEG输入区别于原RGB，零反馈不等于训练消融，不写入总体mAP/F1或正式test表。W1-08须在实际图边和数值核验返回后验收。

09-17完整val及缓存分组更新：`pair44_fullval_8Tvk9d`完成122342张配对验证、补充0张，DIAGNOSTIC COMPLETE和全19类已回传。整体mAP78.5782→78.5171%（−0.0611 pp），macro/micro-F1差值+0.6371/+0.4406 pp；草地/沙滩/海岸湿地AP差值−1.33/−0.39/+2.09 pp。按真实水体标签分组已完成，内陆沙滩两模型均命中1/15，内陆湿地命中3/37和5/37，GAViT组内AP更高，不能简单归因为更依赖海洋背景。当前统计阶段完成，详见[完整分析](../results/seed44_fullval_context_analysis.md)；下一研究步骤为固定验证案例的真实图边/反馈诊断，尚未实施，未新增训练。正式test不变。

09-16诊断返回更新：`pair44_diagnostic_RqV3th`的8张前向和4181张配对推理完成，已复核38项AP、混淆计数/F1及41条案例。均匀val4096张mAP为Swin79.35%/GAViT79.67%，草地/沙滩/海岸湿地差值+0.85/−1.82/+1.45 pp；不替换正式test。完整split存在显著的标签共现比例差异：沙滩+海洋在val为83.1%、test为11.8%，+内陆水体分别3.5%/84.2%。当前随机子集14张沙滩正例全带海洋标签，覆盖不足；优先补足验证集场景覆盖并检查背景相关性，再决定结构消融。详见[验证诊断报告](../results/seed44_validation_diagnosis.md)。真实图边/反馈尚未捕获，没有新训练或服务器任务启动。

09-15失败返回历史：`pair44_diagnostic_6Rcwll`完成标签统计和4181张清单后，因首张验证图片B04文件缺失在模型加载前退出（exit1）。用户随后确认当时尚未解压数据；已由上述成功返回解除当前阻塞。用户不希望反复上传下载工具包。

09-15补充（历史）：已独立复核完整逐类表及三seed统计，并明确现有权重诊断范围：标签支持数、验证集固定样本配对预测、实际分组/图边/反馈。当前7×7 token→4×4区域存在4/2/1个token的区域，反馈汇总系数随区域token数变化；这只是待验证的结构线索。详见 [seed44效果分析](../results/seed44_followup.md)。预测与成败样本诊断代码已完成，16项本地行为测试通过；[运行指南](run_bigearth_pair_diagnostic.md)已交付。服务器前向与固定样本结果现已回传，状态以上方09-16条目为准。真实区域/图边/反馈捕获尚未实现，旧脚本不能直接解释sparse-hybrid，不标为已有图可视化结果。

Swin44 最佳权重测试回传成功：best epoch9、Val mAP78.5782%，Test mAP71.4%/macro-F1 65.7%/micro-F1 76.7%，3745/3745批、Exit status0。与 GAViT44 的70.2%/65.2%/77.1%相比，差值为−1.2/−0.5/+0.4 pp。seed42/43/44均已有配对测试分数；阶段统计见 results/bigearth_multiseed_summary.csv。Swin44现已确认RTX3080 preflight，日志验证到epoch27（75.7%/71.0%），train.exit缺失，最终完成状态仍待确认。完整19类差值已归档：7升、10降、2平；三类贡献约87%的净下降，海岸湿地/沙滩在seed42与44之间收益方向翻转。GAViT当前best metadata与旧日记的差异简要留档，按用户要求不作为效果分析前置项。seed45尚无结果记录。

此前主机内存问题的修复及 GAViT44 后续测试见 research_diary.md 的09-09/09-10条目；下方09-09故障描述保留为历史。当前优先使用现有权重做验证集困难样本的预测、分组和实际图边分析，再推进既定实验矩阵。完整per-class AP已齐，本次没有新增训练或调整选取规则。

## 2026-09-09 历史运行异常

GAViT44仅留下epoch1训练段7434/7434、29:29的日志；PID3504已不存在，best与last-state均缺失，只有启动metadata。不是正常训练完成，也没有可确认用于续训的完整权重。先查系统/平台终止证据和输出目录，保留失败产物；不盲目重跑、不自动换卡或启动Swin44。原因未确认，不能归因于28GB RAM或显存不足。

更新（2026-09-09）：用户随后主动将实例换为 RTX 3080（非自动换卡；2080 Ti 失败运行无可恢复产物）。seed44 配对改为 GAViT44/Swin44 均在同一块 3080 上执行，论文披露该硬件。全量验证诊断脚本已重新定向到 3080，其在 3080 上通过不代表复现 2080 Ti 故障。

## 2026-09-08 执行调整（覆盖原 GPU 短测前置安排）

- 用户明确决定直接使用 RTX 2080 Ti 22GB 训练，不再以 3080/3060 两卡配对资源短测作为前置条件；不再要求上传或运行旧资源工具包。
- 本轮按先 GAViT seed44、后同一 GPU/软件环境的 Swin seed44 执行；一次只启动一个正式任务，seed45 不自动排队。
- 直接调用已验证的 `train_bigearth.py`，不修改模型代码、batch32、224 输入、30 epochs、优化器/调度或精度设置。旧包装器只支持3060/3080，不能套用于2080 Ti。
- 仍检查干净训练 commit、CUDA 设备、预训练权重、原 split 数量及全部 RGB 文件；新输出目录保护已有结果。用户选择不再单独执行资源 smoke，GPU 训练/验证及存盘首次运行的风险与资源未知项保留，首个 batch 和首轮运行时核验。
- 2080 Ti 与此前4090属于不同架构，记录真实软件、GPU及精度设置；不擅自开启 AMP/改变 TF32。训练结果和效率对比均披露硬件差异，纯硬件效率比较不使用不同设备的数据。
- 启动时状态：2026-09-08已确认PID3504/GAViT/Device:cuda和训练batch；后续异常以09-09顶部记录为准，启动成功不代表实验完成。

## 当前目标与优先级

1. 当前 BigEarthNet-19 配对比较至少再增加两个 seed，报告 Swin-T 和固定 4×4 sparse-hybrid GAViT 的 mAP、macro-F1、micro-F1 均值及样本标准差。
2. 展示真实区域划分与实际图边，分析 3×3/4×4、空间边/特征边及成功、失败样本。
3. 增加 AID，并纳入 HGTNet、DFAGCN；先核对准确论文、实现及各数据集的比较范围，统一每个数据集内部的训练/评估协议。
4. 实现 learnable soft assignment，在 NWPU-RESISC45 和 BigEarthNet-19 至少比较 Swin、固定 4×4 GAViT、可学习分组 GAViT。
5. 相关工作覆盖遥感 Transformer、图像 GNN、固定网格/k-means/可学习分组，解释 GAViT 的差异与意义。

本轮最新要求将当前 BigEarthNet 核心比较从原来三个 seed 提高为至少四个。新增架构/方法的多 seed 范围在第一周形成实验矩阵，不把所有方法迁移至 BigEarthNet 当作已确认要求。

## 五周安排

| 周 | 日期 | 主要工作 | 交付/验收 |
|---|---|---|---|
| 1 | 09-07—09-13 | seed43 协议边界；2080 Ti 上分阶段执行 seed44 配对，之后安排 seed45；并行可视化适配与相关方法核对 | 运行资源记录、冻结配置、真实图边示例、复现可行性/实验矩阵 |
| 2 | 09-14—09-20 | 完成新增 seed 汇总及可视化；AID/相关方法接入与 smoke；确定 soft assignment 设计 | 当前模型统计表、新增实验统一协议及可运行流程 |
| 3 | 09-21—09-27 | 优先 AID/相关方法正式实验；实现与小规模验证 soft assignment | 新增比较结果、通过工程验证的分组模块 |
| 4 | 09-28—10-04 | 两数据集架构消融及约定的多 seed；统一硬件下测量效率 | 最终比较/消融表、图、资源开销 |
| 5 | 10-05—10-11 | 论文整合、数字/引用核对与导师修改 | 完整初稿及修改余量 |

相关工作及已确定的实验设置从第一周边读边写。正式实验数量与第三、四周容量取决于第一周的复现可行性；发生偏差时更新本表，不能悄悄删除导师必做项。

## 第一周任务状态

- [x] W1-01 将已讨论排期保存到项目文件。
- [x] W1-02 核对本地 seed42/43 记录与资源记录。
- [x] W1-03 准备并本地验证两种 GPU 通用的短测/资源采集工具（37 项本地测试通过；包已解包核验，CUDA 待实测）。
- [x] W1-04 2080 Ti 22GB正式启动确认：RGB预检通过，GAViT44/PID3504/Device:cuda，epoch1进度3338/7434（45%），已观察到GPU约4.1GiB占用及局部4.99it/s。
- [ ] W1-04a 排查首轮训练段后的异常中断：epoch1训练29:29已记录，验证结果/权重缺失，根因及恢复方案待确认；不重跑，不标为首轮验收通过。
- W1-05 原3080/3060双卡资源比较暂不执行；2026-09-08按用户选择取消其前置地位，不标记成实测完成。
- [x] W1-06 42/43/44/45四seed配对测试及描述性统计齐全；45双方30/30、总及六阶段exit0。历史Swin44最终训练退出未确认、GAViT43/44停止21/26轮的边界保留，不影响已取得的最佳权重测试记录，未补造完整训练状态。
- [ ] W1-07 明确 seed43 提前终止结果在最终统计表中的处理方式。
- [x] W1-08 可视化真实 sparse-hybrid 图，区分空间边与特征边。（4例真实素材与图结构核验完成；180 mm宽PDF/300 dpi PNG图稿已渲染检查，附中英文图注和LaTeX片段。）
- [x] W1-09 核对 AID/HGTNet/DFAGCN 原文、代码、许可证与比较范围，形成实验矩阵。（审计完成；外部官方代码与代码许可证未核实，AID接入/服务器smoke待执行，详见09-19矩阵。）

当前记录：Swin43 完成 30 轮；GAViT43 在第 21 轮终止，评估 epoch8 best，用户当时决定不续跑。保留此边界；未确认恢复或另行处理之前，不能标成四次同协议的完整训练。seed42/43 的原始服务器产物需要在服务器另行核验，本地日记不代替当前产物。

## 原 GPU 资源实验设计（2026-09-07，保留历史）

以下候选分配、两卡 smoke 及包装器门槛由上述2026-09-08调整覆盖，仅作为已实现工具的设计背景，不是当前2080 Ti启动要求。

### 问题与候选分配

问题：在保持既有训练配置时，3080 10GB / 3060 12GB 能否完成训练，并以更低的端到端费用支撑后续实验？

| 候选 GPU | 候选正式任务 | 使用条件 |
|---|---|---|
| RTX 3080 10GB | Swin44 → GAViT44 | 两模型资源短测通过，速度、显存、主机内存及磁盘满足要求 |
| RTX 3060 12GB | Swin45 → GAViT45 | 相同条件 |

若短测发现一种机型明显更合算且可用，可让两个 seed 统一使用该机型。GPU 分配只能依据资源/可用性决定，不依据测试集指标决定。若不同 seed 使用不同 GPU，报告硬件映射；标准差包含硬件/执行差异，不能解释成纯 seed 方差。论文中的模型速度/显存比较须另在同一硬件、相同条件下测量。

### 固定配置

- 已验证训练代码身份：`bb9f1cff463c996bc5f72695af538898c91490eb`，分支 `codex/sparse-hybrid-4n-top2`。正式训练不修改模型和训练入口。
- BigEarthNet-19 原 split：train=237871、val=122342、test=119825；所有引用的 B02/B03/B04 文件完整。
- 输入 224×224 RGB，batch=32；相同增强、预训练、软件环境与计算精度设置。
- 正式 seed=44/45，epochs=30，AdamW lr=3e-4、weight_decay=1e-4，CosineAnnealingLR，BCEWithLogitsLoss，F1 threshold=0.5；按验证 mAP 选择 best。
- GAViT：K16 attentive_spatial、sparse_hybrid、48 空间边+32 top2 特征边、GAT hidden256/4heads/2layers、token_feedback、dropout0.1。
- 不为通过显存门槛临时改变 batch、输入、AMP 或正式轮数。需要变更时单独修订协议。

### 短测与测量

- 只从原 train/val 各自确定性抽样 4096/1024 个样本，抽样 seed42；不复制或评估 test 样本。保存源 CSV 与抽样 CSV SHA256。
- 两张 GPU 使用完全相同抽样、模型配置与 seed42，各执行 Swin/GAViT 一轮；一轮 smoke 的调度不代表正式30轮优化轨迹。
- 包装现有 `train_bigearth.py`，采集实际 DataLoader 批次的端到端耗时，剔除每个阶段前10批；不改输入、优化器或模型前向。
- CUDA allocated/reserved 峰值；定期 nvidia-smi 进程显存；Linux 主进程和子进程 RSS 总和（可能重复计算共享页）、PSS 总和及主机 MemAvailable；软件/GPU/代码/预训练身份。
- 完成训练、验证、best、metadata、last-state 检查；失败保留全部日志和失败报告。
- 小 split 的主机内存不能冒充全 split 内存峰值；正式运行继续采集。短测外推要标注不含完整验证/指标计算、启动、存盘、解压及空闲的影响。
- 2026-09-07 查得公开参考价格：4090 ¥1.87/h、3080 ¥0.93/h、3060 ¥0.69/h；实际价格以租用时为准。历史 GAViT 4090 6.64h、Swin 5.22h，只作粗略参考。

### 产物和执行边界

- 工具独立放在服务器 `/home/featurize/work/gavit-resource-tools-20260907/`，训练目录保持原干净 commit；操作说明见 [GPU 资源短测指南](gpu_resource_probe.md)。
- 每次使用新的输出目录，保存输入身份、日志、checkpoint、资源样本与报告；拒绝覆盖。
- 先生成命令计划；运行短测需要显式执行 smoke 命令。短测不会自动启动任何正式训练。
- 本地完成工具和测试；Featurize 负责 GPU 实测。当前任务未授权读取已登录浏览器或自动租用实例；提供可复制命令。
- 本轮先完成 W1-03；其他工作按清单继续，不把五周任务标记成已完成。

参考：`research_diary.md` 的 seed42/43 实验条目；`docs/featurize_runbook.md`；https://featurize.cn/vm/available；https://docs.pytorch.org/docs/2.14/notes/randomness.html 。
