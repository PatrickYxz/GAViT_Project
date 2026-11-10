# seed43补充状态（更新2026-10-12）

当前：NWPU43三模型九阶段与总exit0、正式OA/macro-F1、选轮及资源已回传并核验；[正式结果](nwpu_seed43_test_comparison.md)、[42/43双seed表](nwpu_two_seed_summary.md)。整批1小时0分17.33秒，按1.67元/h折算约1.68元。BigEarth soft43完成正式30轮与119825 test、总/训练/test各exit0，best为第8轮/val mAP78.99004691745016%，完整资源已核验；[正式结果](bigearth_soft43_ablation.md)、[43/45共同seed表](bigearth_three_model_two_seed_summary.md)。整批6小时22分21.73秒、折算约10.64元；两批正式队列共7小时22分39.06秒/约12.32元，恢复数据、短测及空闲另计。此次补实验已完整归档。

| 数据集 | 模型 | 处理 | 新test指标 |
|---|---|---|---|
| NWPU43 | Swin | 完成；校准30轮、选23、refit23、test25200 | OA93.857143/macro-F1 93.834530% |
| NWPU43 | 固定4×4 | 完成；校准30轮、选26、refit26、test25200 | OA93.765873/macro-F1 93.743513% |
| NWPU43 | soft_prototype | 完成；校准30轮、选29、refit29、test25200 | OA93.626984/macro-F1 93.587821% |
| BigEarth43 | Swin | 保留已有30轮/best6与test | mAP71.1/macro-F1 64.1/micro-F1 76.0% |
| BigEarth43 | 固定4×4 | 保留已保存21轮/best8与test | mAP71.7/macro-F1 65.7/micro-F1 76.8% |
| BigEarth43 | soft_prototype | 正式30轮、best8与119825 test完成，三个exit0；资源已归档 | mAP72.867571/macro-F1 66.673892/micro-F1 76.905971% |

固定NWPU42高于Swin，但43未复现；双seed平均ΔOA仅+0.039683pp。soft两seed均低于Swin，平均−0.146825pp。42/43分别2080Ti/4090，n=2描述汇总不能认定稳定收益或把差距仅归因seed/GPU。BigEarth预检回传见[转录](evidence/seed43_bigearth_preflight_20261011/user_return.json)，完整资源见[核验](evidence/bigearth_soft43_complete_20261012/resource_validation.json)；不再请求同批摘要或重复运行。

不改历史主表分数、不补造近似数据，保留负面与正面结果。BigEarth43历史版本/硬件/轮数不完全一致；新的soft结果不得用来声称已消除这些限制。本批训练/test/资源已补齐，现在两个数据集各有两seed软分配证据，最终三seed、公平效率与外部方法实测等要求仍有缺口。

BigEarth soft43比旧Swin43和fixed43三指标均高；共同43/45中soft对fixed的mAP两次升，对Swin一升一降。旧43对照为打印一位小数，差值近似；不能把本次正向结果写成跨数据集稳定优势。
