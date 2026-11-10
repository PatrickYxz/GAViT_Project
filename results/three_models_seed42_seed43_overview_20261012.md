# 三数据集三模型seed42/43正式测试汇总（2026-10-12）

18个组合中15个有已归档正式test结果。缺AID soft42/43和BigEarthNet soft42；不以seed45代替42，不填入小样本或验证分数。固定模型采用attentive_spatial、sparse_hybrid 4n top2与token_feedback；软模型采用soft_prototype。

分数单位为%，差值单位为百分点（pp）。AID/NWPU主指标是OA，BigEarthNet主指标是19类macro-mAP。

## AID：test OA

| seed | 模型 | OA % | macro-F1 % | micro-F1 % |
|---|---|---:|---:|---:|
| 42 | Swin baseline | 96.3400 | 96.2209 | — |
| 42 | 固定4×4 GAViT | 96.9400 | 96.7811 | — |
| 42 | 软分配 GAViT | — | — | — |
| 43 | Swin baseline | 96.8600 | 96.6842 | — |
| 43 | 固定4×4 GAViT | 97.1600 | 96.9559 | — |
| 43 | 软分配 GAViT | — | — | — |

每次test样本数：5000。

## NWPU-RESISC45：test OA

| seed | 模型 | OA % | macro-F1 % | micro-F1 % |
|---|---|---:|---:|---:|
| 42 | Swin baseline | 93.9325 | 93.9091 | — |
| 42 | 固定4×4 GAViT | 94.1032 | 94.0899 | — |
| 42 | 软分配 GAViT | 93.8690 | 93.8550 | — |
| 43 | Swin baseline | 93.8571 | 93.8345 | — |
| 43 | 固定4×4 GAViT | 93.7659 | 93.7435 | — |
| 43 | 软分配 GAViT | 93.6270 | 93.5878 | — |

每次test样本数：25200。

## BigEarthNet-19：test mAP

| seed | 模型 | mAP % | macro-F1 % | micro-F1 % |
|---|---|---:|---:|---:|
| 42 | Swin baseline | 70.9 | 64.2 | 76.6 |
| 42 | 固定4×4 GAViT | 72.0 | 66.0 | 76.7 |
| 42 | 软分配 GAViT | — | — | — |
| 43 | Swin baseline | 71.1 | 64.1 | 76.0 |
| 43 | 固定4×4 GAViT | 71.7 | 65.7 | 76.8 |
| 43 | 软分配 GAViT | 72.8676 | 66.6739 | 76.9060 |

每次test样本数：119825。

## 相对Swin的主指标变化

| 数据集 | seed | 固定−Swin pp | soft−Swin pp |
|---|---:|---:|---:|
| AID | 42 | +0.6000 | — |
| AID | 43 | +0.3000 | — |
| NWPU-RESISC45 | 42 | +0.1706 | -0.0635 |
| NWPU-RESISC45 | 43 | -0.0913 | -0.2302 |
| BigEarthNet-19 | 42 | +1.1000 | — |
| BigEarthNet-19 | 43 | +0.6000 | +1.7676 |

固定：AID42/43均高于Swin；NWPU42高于、43低于；BEN42/43均高于。软分配：NWPU42/43均低于Swin；BEN43高于Swin及固定，另外三组暂无已归档正式结果。这里只描述所列数据集×seed组合，不把跨任务组合当同一任务的独立重复。

## 精度、协议与解释范围

BEN旧42/43对照只保存一位小数，差值为近似。BEN固定43保存21轮/best8，Swin43 full30/best6，soft43 full30/best8，代码与历史硬件不同；不能把差值解释为完全受控的分组因果收益。NWPU42/43使用2080Ti/4090。AID/NWPU是各自验证选轮后重新拟合训练池再test，BEN是验证最佳checkpoint的test；单标签micro-F1没有在这些摘要中单独归档，表中不推导补填。

OA与mAP不跨数据集平均。本表按用户要求仅列42/43，全部历史seed仍须保留，不能用这份子集替代完整多seed结果。参数量、资源与逐类结果仍分别保存在各原报告，本次未新增训练或推理。

## 来源

[AID42](aid_seed42_comparison.md)、[AID43](aid_seed43_comparison.md)、[AID完整三seed](aid_multiseed_summary.md)；[NWPU42/43](nwpu_two_seed_summary.md)；[BEN历史四seed](bigearth_four_seed_summary.md)、[BEN soft43](bigearth_soft43_ablation.md)。

[精确数值、来源SHA与完整差值JSON](three_models_seed42_seed43_overview_20261012.json)。
