# Swin / GAViT 验证样本诊断

均匀验证样本：4096；独立补充样本：85。
AP/PR只使用均匀验证样本，非正式测试集结果。成败分类阈值为0.5。

| 类别 | 正例数 | Swin AP % | GAViT AP % | 差值 pp |
|---|---:|---:|---:|---:|
| Natural grassland and sparsely vegetated areas | 102 | 64.85 | 65.70 | 0.85 |
| Beaches, dunes, sands | 14 | 79.13 | 77.31 | -1.82 |
| Coastal wetlands | 25 | 53.53 | 54.97 | 1.45 |

完整19类：per_class.csv；配对分数：predictions.csv；成败样本：cases.html / cases.csv。
样本计数：case_counts.csv；PR曲线数据：pr_curves.csv。
本次不绘制区域或图边；图反馈是否导致错误，需要后续真实前向诊断。
