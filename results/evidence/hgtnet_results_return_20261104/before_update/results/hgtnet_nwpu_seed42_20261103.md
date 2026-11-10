# HGTNet独立实现：NWPU seed42正式结果

状态：**服务器报告完整正式实验完成，摘要身份匹配，逐图文件待回收复算**。不是作者官方数值复现；单seed完整方法结果，不单独归因图模块。

## 结果

- Test25200：OA **94.11904761904762%**、macro-F1 **94.10529172930384%**。OA算术对应23718正确/1482错误；macro-F1为服务器报告，尚未从逐图预测独立复算。
- 校准train5040/val1260、100轮，最佳/选定epoch95；最佳val OA93.0952380952381%（算术1173/1260）、macro-F193.03452094979133%。末轮100的val分数及完整曲线待history，不能由best分数代替final分数。
- Fresh相同初值SHA在pool6300重新训练95轮，随后固定模型评估25200test。315×100＋394×95＝68930次更新，与完整日志/status一致。
- statecomplete/completedtrue/exit0/attempt1，完整模型74619858参数且全部可训练。finalrefit checkpoint独立新模型回读误差0.0，initial/refitinitial SHA相等。
- 程序wall24032.617371428秒＝6小时40分32.617秒；enginewall23899.926034053秒。差额132.691337375秒包含wrapper准备/末尾核验等，未拆项测量。单次完整attempt，仍排除stdlib预检与租卡空闲，未取得账单，不编造费用。
- PyTorch峰值allocated4734618112B＝4.4094567299GiB、reserved5135925248B＝4.783203125GiB。范围含本次engine的训练/诊断/模型回读，不是NVML整卡或纯推理开销。

## 当前同manifest、seed42对比

| 方法 | Test OA% | macro-F1% | 正确数/25200 | 神经网络参数 |
|---|---:|---:|---:|---:|
| Swin-T | 93.932540 | 93.909145 | 23671 | 27,555,495 |
| GAViT fixed4×4 | 94.103175 | 94.089933 | 23714 | 31,386,920 |
| GAViT local soft | 93.869048 | 93.854987 | 23655 | 31,486,760 |
| DFAGCN independent v1 | 86.531746 | 86.458909 | 21806 | 120,436,679* |
| HGTNet independent v1 | 94.119048 | 94.105292 | 23718† | 74,619,858 |

*DF参数范围含冻结VGG与graph/训练CEheads，不含LIBLINEAR；不同于全参数端到端训练。†HGT正确数由报告OA算术推得，尚未逐图复算。DF逐图原件已在10月31日复算；现有Swin/GAViT分数依据当前[正式记录](../.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md)，不是旧约96%协议成绩。

HGT−Swin：OA+0.18650794个百分点、macro-F1+0.19614626个百分点，净多对47张；HGT−fixed：OA+0.01587302个百分点、macro-F1+0.01535827个百分点，净多对4张；HGT−soft：OA+0.25个百分点、净多对63张；HGT−DF：OA+7.58730159个百分点、净多对1912张。

fixed的参数量是HGT的42.0624%，约少57.9376%；HGT约为fixed的2.3774倍参数。这组结果支持描述“当前单seed完整方法的准确率点估计接近，fixed参数更少”，不构成统计等价检验或稳定优势证明。骨干/预训练/优化/增强/epoch/GPU不同，不能把时间或显存差异直接当同条件效率提升。缺少HGT逐图及完整旧模型原始预测，净4张不能解释为只纠正4张且无退化，也不能做HGT的配对显著性结论。

[比较CSV](external_baselines_nwpu_seed42_20261030.csv)已追加HGT一行，原四行字节保留，[追加前快照](evidence/hgtnet_formal_complete_return_20261103/comparison_before_hgtnet.csv)可核验；旧comparison_table.csv/ablation_study.csv历史数字未改。

## 协议与身份

固定seed42/FP32无AMP/TF32/8CPU线程/workers0、batch16/Adam1e-5/WD5e-5/constantLR/CE；精确增强/defaults/最早并列选轮为已披露工程选择，见[配方](../configs/hgtnet_nwpu_formal_v1.json)及[准备说明](../docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)。完整架构和partialSwin-B候选初值不是作者HGTNet权重。

当前UUID GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1；manifestce0cfef2...840e/configaf6ef410...ac25/recipece72e182...af85/初始候选68c6b09e...1a893均与本地交付固定值匹配。包SHAb2bc9077...572e1，Git基线058f1d8b79d0be538ec358eb742d2cd3992921d4，未提交增量由包SHA绑定。[规范化status原值](evidence/hgtnet_formal_complete_return_20261103/status_reported.json)、[43项本地验收](evidence/hgtnet_formal_complete_return_20261103/validation.json)、[独立39项摘要审查](evidence/hgtnet_formal_complete_return_20261103/review/status_audit.json)与[22项完成事件审查](evidence/hgtnet_formal_complete_return_20261103/review/validation.json)保存。

finalrefit SHA716835556fe989739fc39e2d278188739b4d4454a14b5bf4d8d2b9efe060db1c，test.json SHA869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae为服务器原件引用；未下载大模型或test原始文件验证。正常交付程序在测试记录写盘与源码/数据/配方/权重末尾复核后才写summary/exit0/marker并打印COMPLETE，与返回记录一致。

## 下一步离线复核

回收运行目录的test.json/history.json，以及summary/binding/runtime等小metadata，即可独立复算25200预测、完整45类指标/混淆矩阵、训练曲线、最佳/最终差别。后续不需要再训练或占用GPU。导出原件时保留原始字节，以核对返回testSHA；不用回收300MB模型证明本轮流程。
