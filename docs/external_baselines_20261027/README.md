# 外部方法独立实现：当前结果与准备历史

准备日期：2026-10-27；完整独立流程更新：2026-10-29；当前状态更新：2026-11-08。

## 当前完成状态

原定最低矩阵HGTNet×NWPU/AID、DFAGCN×NWPU三项独立实现seed42实测均已完成，并回收外部原始预测独立复算。NWPU HGTNet OA94.1190%/macro-F1 94.1053%，DFAGCN OA86.5317%/macro-F1 86.4589%；AID HGTNet OA96.92%/macro-F1 96.7729%。这三项是不同披露配方下的补充比较，统一训练/评估协议及多seed外部均值/SD仍未完成。

AID Swin/GAViT两模型共10份原始评估文件已经回收，包含各5000条预测。逐路径/真值核对后，HGTNet相对固定GAViT纠正75、新增76错误，双方都正确4771、双方都错78，净少对1；相对Swin纠正108、新增79错误，净多对29。分析复用原始完成评估，不重训或重做forward，不把净差1张写成仅1张预测不同。

[三项完成结果、配方与剩余要求](completed_comparison_20261106.md)、[可复制导师更新](../supervisor_external_results_update_20261108.md)、[AID真实配对](../../results/aid_seed42_three_model_paired_analysis_20261106.json)、[完整比较JSON](../../results/external_methods_comparison_20261106.json)为当前入口。DFAGCN×AID不属于本次最低矩阵；46/47缺失F1与原始记录不由本次AID seed42回收填补。

## 2026-10-27至31准备与实施历史

以下段落保留当时的组件、smoke与正式交付状态；其中“尚待运行”等表述属于历史，以以上当前完成状态和核验结果为准。

2026-10-31离线核验：已取回DFAGCN完整test.json，25200条预测/45类各560、全部混淆矩阵、OA和macro-F1独立复算通过，32项核验无异常。指标仍为86.53%/86.46%；新增[逐类误差分析、CSV与图稿](../../results/dfagcn_nwpu_seed42_error_analysis_20261031.md)，并同步汇报。无需为这一步重训；原图、训练历史和模型字节仍未完整回收。

正式结果更新：[DFAGCN NWPU seed42已完成](../../results/dfagcn_nwpu_seed42_20261030.md)，exit0、OA86.53%、macro-F186.46%，选定51轮/C10，完整25200 test。摘要29项核验通过，固定GAViT同seed OA高7.57pp；这是披露不同recipe的完整方法单seed比较，不是原文数值复现、同设备效率或跨seed稳定性结论。[四模型结果CSV](../../results/external_baselines_nwpu_seed42_20261030.csv)与[已更新汇报草稿](../supervisor_progress_report_20261012.md)可直接使用。以下交付/“尚待运行”段落保留为历史记录，以本条和结果证据为当前状态。

2026-10-30最新：用户实际inspect显示metadata省略CUDA后缀、新卡UUID及work16GiB/实例本地349GiB差异；已交付[fix1资源修复包和一行操作](../../output/dfagcn_nwpu_formal_3080_fix1_README_20261030.md)。仅修复门禁与分卷缓存路径，核心配方/算法不变；148项测试和独立审查通过，正式服务器执行仍待回传。

当前硬件按用户2026-10-30的新安排采用RTX3080 10GiB。一次性batch100兼容性检查已[实际回传通过](../../results/evidence/dfagcn_gpu_3080_return_20261030/validation.json)，无需重跑；[正式seed42便携运行包与操作说明](../../output/dfagcn_nwpu_formal_3080_README_20261030.md)已交付，139项相关测试通过、正式训练尚未启动。[正式入口约定](formal_3080_runner_20261030.md)保持现有配方/协议，此前4090工程检查及既有结果保留；独立终审因用量限制未完成，主代理复核与范围见[验收](../../results/evidence/dfagcn_formal_3080_prepare_20261030/validation.json)。

2026-10-30更新：[真实NWPU360/90/450工程检查回传通过](../../results/dfagcn_nwpu_smoke_20261030.md)，exit0、RTX4090、交付身份匹配、重载误差0，不是正式test。先前105项本地测试及预训练CPU检查保留；[便携包说明](../../output/dfagcn_nwpu_smoke_README_20261029.md)作为已完成批次操作记录，下一阶段转完整seed42实验准备。

流程基础：[完整入口与运行说明](pipeline_usage_20261029.md)、[已冻结的独立配方](../../configs/dfagcn_independent_v1.json)、[86项测试与20图完整工程链](../../results/evidence/dfagcn_pipeline_20261029/)。真实NWPU小样本已通过，正式分数仍待运行；下文文献来源和早期准备记录保留。

用户已同意的路线是：原论文报告值先形成文献参考，同时推进有明确假设记录的独立复现。材料准备后，已继续实现DFAGCN第一批独立组件，不以作者或导师回信作为开始条件。本轮未启动付费训练。

2026-10-28新增：[组件代码与使用说明](../../external_baselines/dfagcn/README.md)、[实施记录](implementation_plan_20261028.md)及[验证证据](../../results/evidence/dfagcn_primitives_20261028/)。完整方法和训练比较仍待后续接入；下文原论文数值仍只属于文献报告。

## 1. 本次决定

首篇优先DFAGCN，第一批编码对象是已能从论文确定的特征接口、构图、归一化GCN传播、readout、加权拼接和分阶段数据接口。HGTNet排在后面，先保留其规格与不确定项。

选择依据是可执行结构的明确程度。DFAGCN已确认VGG16多尺度特征、三支图聚合、每支512维输出、加权拼接和LIBLINEAR完整链，也已核对C网格及数据集融合系数。它仍有Table I层配置、池化目标和预处理/权重等缺口，因此不是“完整结构已经冻结”。

HGTNet的未决项更分散：max-relative与average聚合表述、DA门控符号、STB计数到库block的映射、阶段分辨率与k=20可容纳性、跳连融合及部分ImageNet预训练映射。现在直接给它套现有Swin/GAT，会有较多核心结构选择缺乏原文依据。

该次序不依据文献分数高低、不保证DFAGCN训练更便宜，也不意味着取消导师要求的HGTNet比较。

## 2. 本次已交付的材料

- [原论文报告值与可复制论文说明](literature_comparison.md)：逐数据集、逐训练比例列出文献OA，保留来源和打印精度，不使用需要空格对齐的表格。
- [DFAGCN完整链核对及实施前缺口](dfagcn_audit.md)：原文位置、特征层、图、GCN、融合、分类器和有限工程任务。
- [HGTNet模块规格与歧义记录](hgtnet_audit.md)：已确认的形状、尚未确定的配置、明确标注的独立工程候选及最低验证。
- [现有协议、工程入口与阶段验收](protocol_and_acceptance.md)：真实AID/NWPU清单、训练/验证/refit/test隔离，以及不能直接复用的旧训练器限制。
- [全部证据与原文转录记录](../../results/evidence/external_baselines_20261027/)：逐方法JSON、访问范围、当前依赖检查和既有指标文件的保护哈希。

## 3. 可先开展的DFAGCN编码范围

以下对象中的特征接口、图/融合算子和缓存身份接口已在2026-10-28实现。分阶段训练/最终测试门禁仍是后续任务，不能由缓存JSON身份替代。

第一组：固定VGG16的取层接口。224输入导出第一个FC层ReLU后的4096维；448输入导出Conv3-3、Conv4-3、Conv5-3的ReLU前激活。先用形状和激活位置检查接口；只有明确MatConvNet文件、预处理及转换后，才声称预训练数值身份。

第二组：独立的图与融合算子。k=10的二值无向kNN并集、自环只加一次的对称度归一化、逐节点传播和全局均值readout，以及按Fconv5、alpha×Fconv4、beta×Fconv3、gamma×Ffc顺序拼接。三支512维加4096维FC，推导长度为5632。用手算小图和确定性fixture检查，不用训练分数检验代码是否写对。

第三组：清单、缓存和分阶段身份。每条缓存记录样本顺序、原图/manifest/权重/预处理/源码哈希；校准与refit学习得到的产物分开。C、融合系数和停止规则只在train/val内决定，最终测试只接受完整refit身份。

2026-10-29已按用户批准将Table I/池化等未决项固定为披露的独立选择，并接入可执行训练、求解器和停止选择规则。原表等价仍未确认；不需要继续等待回信才能做独立版本的真实小样本。工程fixture与默认49节点完整配方分别保存。

新实现放根目录独立`external_baselines/dfagcn/`，完整训练/评估CLI已实现，身份`dfagcn_independent_v1`。原有Swin/GAViT代码、checkpoint身份和历史结果保持不变。

## 4. 实验问题和预定去向

第一项正式问题是：在相同NWPU冻结训练池/测试集和明确验证选择规则下，完整DFAGCN独立实现与本项目Swin/GAViT的性能及完整推理成本各是什么水平？这是完整方法比较，改变了骨干和处理链，不属于只改变图模块的单变量消融。

文献报告值进入“相关方法的原论文表现”说明；将来的独立实测进入单独标注实现身份和共同协议的外部方法结果。先用无训练算子检查，再用内部最多360/90张smoke排除工程问题；正式单seed完成后才决定后续多seed安排，不按结果是否有利决定保留哪些运行。

完整独立入口和本地合成工程链已完成；真实数据的训练/推理耗时、吞吐、峰值内存与费用仍待测。单元/合成检查耗时不是完整方法效率。本轮没有启动GPU。

## 5. 文献数值的证据范围

本轮从原文正文直接核对到HGTNet四项OA均值和DFAGCN两项OA点值。原表图片与公开PDF访问受限，标准差均未转录，JSON保留`null`；DFAGCN均值/SD单元格本身也未视觉核对。

DFAGCN明确十次随机划分；HGTNet当前可读文字确认五次重复，具体重复机制尚未确认。本项目固定split的多训练seed不与它们混作相同方差来源。

缺失标准差不是模型工程准备的前置条件。核心结构未确认的部分则需要补读原表或在独立版本中逐项披露选择；不能为填满文献表补造数值，也不能为赶进度把一套近似网络登记成作者原实现。

## 6. 2026-10-29至30阶段状态（历史）

材料、完整独立入口和本地工程链已完成。86个unittest方法通过；实际CPU FC6、VGG/图训练/LIBLINEAR/重新拟合/测试已接通，完整49节点45类配置另有前向/反向检查。未验证MatConvNet等价。全仓库仍缺pytest/timm/torch_geometric，本轮185项发现检查有9failure/9error，见[逐项记录](../../results/evidence/dfagcn_pipeline_20261029/repository_failures.json)。

截至2026-10-30，真实数据360/90 smoke和3080一次性兼容性检查均已通过，正式比较尚未启动；未重新测试旧核心模型checkpoint、发送邮件或执行Git提交推送。此节86项为2026-10-29的历史工程链验证，后续119项及本次正式入口验证各有独立证据。
