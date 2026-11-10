# 已完成的外部方法补充比较

实验与原始记录基准日期：2026-11-06；说明更新：2026-11-08。

## 当前范围

导师要求的最低方法×数据集矩阵为HGTNet×NWPU/AID、DFAGCN×NWPU。这三项现均已有训练seed42独立实现正式结果，完成验证选择、完整训练池fresh refit、完整test与原始预测复算。统一训练/评估协议及多seed均值/样本SD仍未完成；不能将最低矩阵的单seed齐备写成全部要求完成。

各方法使用相同数据清单和信息隔离规则，但保留不同骨干、预训练、预处理、优化与预算；此文件是补充比较，不是官方实现、原文分数复现或单变量图模块消融。

## NWPU-RESISC45：固定seed42

数据划分：5040 train、1260 val、6300 refit pool、25200 test。

模型：Swin-T。
测试OA：93.93253968253968%；macro-F1：93.90914546819495%。
正确数：23671/25200；refit轮数：29。
来源级别：existing_reported_integer；证据：`.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md`。

模型：GAViT fixed4x4。
测试OA：94.10317460317461%；macro-F1：94.08993346277097%。
正确数：23714/25200；refit轮数：29。
来源级别：existing_reported_integer；证据：`.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md`。

模型：GAViT local soft。
测试OA：93.86904761904762%；macro-F1：93.854987219354%。
正确数：23655/25200；refit轮数：28。
来源级别：existing_reported_integer；证据：`.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md`。

模型：DFAGCN independent v1。
测试OA：86.53174603174604%；macro-F1：86.45890874660422%。
正确数：21806/25200；refit轮数：51。
来源级别：locally_recomputed_from_returned_test_predictions；证据：`results/evidence/dfagcn_test_predictions_20261031/test.json`。

模型：HGTNet independent v1。
测试OA：94.11904761904762%；macro-F1：94.10529172930384%。
正确数：23718/25200；refit轮数：95。
来源级别：locally_recomputed_from_returned_test_predictions；证据：`results/evidence/hgtnet_results_return_20261104/files/test.json`。

HGTNet比固定GAViT高0.01587302pp OA、净多对4张。NWPU这两者逐图原件未齐，不能由净差推断纠正/新增错误。HGTNet与DFAGCN已有真实配对：双方都正确21303张、HGTNet纠正2415张、新增503张错误、双方都错979张，净多对1912张。

## AID：固定seed42

数据划分：4000 train、1000 val、5000 refit pool、5000 test。

模型：Swin-T。
测试OA：96.34%；macro-F1：96.22088514124282%。
正确数：4817/5000；校准轮数：30；refit轮数：21。
参数：27543960；骨干/初始化：Swin-T。
证据：`results/evidence/external_results_integration_20261106/aid_core_raw/swin`。

模型：GAViT fixed4x4。
测试OA：96.94%；macro-F1：96.7811271561119%。
正确数：4847/5000；校准轮数：30；refit轮数：20。
参数：31375385；骨干/初始化：Swin-T。
证据：`results/evidence/external_results_integration_20261106/aid_core_raw/gavit`。

模型：HGTNet independent v1。
测试OA：96.92%；macro-F1：96.77286814931982%。
正确数：4846/5000；校准轮数：100；refit轮数：96。
参数：74612163；骨干/初始化：partial torchvision Swin-B initialization。
证据：`results/evidence/hgtnet_aid_formal_complete_20261106/raw/test.json`。

Swin/GAViT各5份、共10份原始完成评估文件已回收在`results/evidence/external_results_integration_20261106/aid_core_raw/`，SHA回收记录与原始CSV、metrics、run、退出记录保留。两模型各5000条预测经独立复算与旧摘要一致，并与HGTNet原始预测逐项核对路径和真值后配对。没有重新执行模型。固定GAViT历史macro-F1为96.7811271561119%，原始预测重算浮点结果96.78112715611192%，差异约2.84e-14pp属于浮点表示；保留原归档精度，不将其当成分数变化。

逐图配对方向：reference为参照，candidate为候选；corrects为参照错/候选对，breaks为参照对/候选错。

比较：swin → gavit。
双方都正确：4754；候选纠正：93；候选新增错误：63；双方都错误：90。
候选净正确变化：30；OA变化：0.6pp。

比较：swin → hgtnet。
双方都正确：4738；候选纠正：108；候选新增错误：79；双方都错误：75。
候选净正确变化：29；OA变化：0.58pp。

比较：gavit → hgtnet。
双方都正确：4771；候选纠正：75；候选新增错误：76；双方都错误：78。
候选净正确变化：-1；OA变化：-0.02pp。

固定GAViT比HGTNet高0.02pp OA、0.0082590068pp macro-F1，净多对1张。真实75/76双向计数不能简化为仅1张预测不同；也不能据单seed接近分数声称显著优势、等效或稳定排名。

## 选择、参数与资源边界

HGTNet AID完整100轮校准，best96唯一达到972/1000、OA97.20%、macro-F1 97.074557%；final100为971/1000、OA97.10%、macro-F1 96.990538%。从与校准相同初始状态fresh refit96轮，接受轨迹55048步。历史保留100+96轮、30类全部统计与混淆，不以refit训练100%代替test。

固定GAViT在AID为31,375,385参数，HGTNet74,612,163，少57.95%；NWPU分别31,386,920与74,619,858，少57.94%。参数描述不等同速度或费用优势。DFAGCN的120,436,679参数口径包括冻结VGG和图分支/CE heads，不包含LIBLINEAR，不与all-trainable参数直接混称。

内部Swin/GAViT为Swin-T、224直接resize、batch32/AdamW/30轮cosine校准；HGTNet为部分torchvision Swin-B初值、256resize后224crop、batch16/Adam/100轮且无scheduler；DFAGCN为冻结torchvision VGG16、224/448输入、GCN训练与LIBLINEAR。MatConvNet等价及原方法全部实现身份未建立。

硬件为历史核心seed42的RTX2080Ti、DFAGCN NWPU的RTX3080、HGTNet NWPU的RTX3090和AID的RTX3060。AID恢复attempt2墙钟4986.588285893秒，peak allocated4731174400B、reserved5175771136B，仅覆盖该调用（含诊断/重载）；全实验墙钟、全程峰值与费用未知。模型二进制SHA由服务器强status核对，重载检查也在服务器完成；本地回收记录/收据不称本地模型重载。

## 仍需完成

外部主比较的统一训练/评估协议，以及多个完整训练seed的均值与样本SD；不同骨干和配方不能只因split相同便称统一。

完整内部结果、架构消融、综合讨论和结论的论文接入，以及符合口径的同设备效率对照。

原内部汇报子集仍为42/43/46/47，其他历史正负seed继续保留；46/47缺失F1、原始预测与运行记录仍未回收，不从本次AID seed42文件填补。DFAGCN×AID不是当前最低矩阵，本轮不新增该训练。

## 可复制与机器记录

[11月8日导师更新](../supervisor_external_results_update_20261108.md)、[英文草稿](../supervisor_email_draft_20261012.md)。

[统一比较JSON](../../results/external_methods_comparison_20261106.json)、[AID配对JSON](../../results/aid_seed42_three_model_paired_analysis_20261106.json)、[AID完整离线分析](../../results/hgtnet_aid_seed42_analysis_20261106.md)。
