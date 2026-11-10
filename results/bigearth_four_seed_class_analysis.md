# BigEarthNet固定分组：四seed逐类AP对照

2026-10-10。补齐seed43现有测试日志中的19类AP；没有重新训练或推理。

[原始回传](evidence/bigearth_seed43_per_class_20261010/server_return.txt)、[解析与来源SHA](evidence/bigearth_seed43_per_class_20261010/summary.json)、[seed43 CSV](bigearth_seed43_per_class_comparison.csv)、[四seed CSV](bigearth_four_seed_per_class_comparison.csv)、[聚合指标与运行边界](bigearth_four_seed_summary.md)。

## seed43结果

Swin/GAViT的mAP为71.1/71.7%，macro-F1为64.1/65.7%，micro-F1为76.0/76.8%；差值分别为+0.6/+1.6/+0.8 pp。逐类10升、8降、1打印持平。

最大提升：Moors, heathland and sclerophyllous vegetation +4.2 pp；Permanent crops +3.3 pp；Inland wetlands +2.9 pp；Coastal wetlands +1.6 pp；Natural grassland and sparsely vegetated areas +1.3 pp。最大下降为Beaches, dunes, sands −2.7 pp，其余下降为−0.1至−0.4 pp。

## 四seed方向

差值均为GAViT−Swin，单位pp。均值和样本SD由各seed配对差计算，属于描述性汇总。

| Class | seed42 | seed43 | seed44 | seed45 | 均值 ± SD | 升/降/打印持平 |
|---|---:|---:|---:|---:|---:|---:|
| Urban fabric | -0.20 | +0.60 | -0.30 | +0.10 | +0.05 ± 0.40 | 2/2/0 |
| Industrial or commercial units | -1.30 | +0.60 | -0.90 | +0.06 | -0.39 ± 0.87 | 2/2/0 |
| Arable land | +0.50 | -0.20 | +0.20 | +0.14 | +0.16 ± 0.29 | 3/1/0 |
| Permanent crops | +2.40 | +3.30 | -1.50 | -2.54 | +0.42 ± 2.87 | 2/2/0 |
| Pastures | -0.40 | -0.10 | +0.10 | -0.07 | -0.12 ± 0.21 | 1/3/0 |
| Complex cultivation patterns | +2.10 | -0.40 | -1.30 | +1.25 | +0.41 ± 1.54 | 2/2/0 |
| Land principally occupied by agriculture, with significant areas of natural vegetation | +0.70 | +0.20 | +0.00 | +0.45 | +0.34 ± 0.30 | 3/0/1 |
| Agro-forestry areas | -0.20 | +0.00 | -0.40 | +0.04 | -0.14 ± 0.20 | 1/2/1 |
| Broad-leaved forest | +0.90 | -0.10 | +0.80 | -0.66 | +0.23 ± 0.75 | 2/2/0 |
| Coniferous forest | +0.00 | -0.30 | +0.50 | -0.05 | +0.04 ± 0.33 | 1/2/1 |
| Mixed forest | +0.50 | +0.90 | -0.20 | -0.69 | +0.13 ± 0.71 | 2/2/0 |
| Natural grassland and sparsely vegetated areas | -0.80 | +1.30 | -3.70 | -0.49 | -0.92 ± 2.07 | 1/3/0 |
| Moors, heathland and sclerophyllous vegetation | +5.00 | +4.20 | -0.30 | -0.43 | +2.12 ± 2.89 | 2/2/0 |
| Transitional woodland, shrub | +0.50 | +0.30 | +0.20 | -1.09 | -0.02 ± 0.72 | 3/1/0 |
| Beaches, dunes, sands | +3.50 | -2.70 | -3.30 | -0.65 | -0.79 ± 3.08 | 1/3/0 |
| Inland wetlands | +1.40 | +2.90 | +0.00 | +0.36 | +1.17 ± 1.30 | 3/0/1 |
| Coastal wetlands | +4.40 | +1.60 | -12.50 | -23.19 | -7.42 ± 12.85 | 2/2/0 |
| Inland waters | +0.80 | -0.10 | +0.10 | +0.60 | +0.35 ± 0.42 | 3/1/0 |
| Marine waters | +0.10 | -0.10 | +0.10 | -0.11 | -0.00 ± 0.12 | 2/2/0 |

没有类别四次全部提升或四次全部下降。此前只用42/44/45判断的Arable land、Inland waters三次正向和Natural grassland三次负向，在seed43分别出现负向、负向和正向，不能外推为四seed一致。

农业与自然植被混合区、内陆湿地是3次提升、1次打印持平；自然草地、海滩沙丘沙地是1次提升、3次下降；海岸湿地是2次提升、2次下降，但后两次降幅很大，四次平均−7.42 pp、样本SD约12.85 pp。这些可作为失败分析线索，不能称为显著稳定的类别收益或退化。

## 精度与证据边界

seed42/43/44只保留一位小数AP，seed45使用完整返回精度；接近零的差值谨慎解释，打印持平不证明真实AP相等。seed43打印AP均值为71.110526/71.736842%，配对均值差为+0.626316 pp，与日志71.1/71.7%相容；正式聚合表继续使用日志报告值。

Swin43回传含测试峰值allocated 0.58 GiB、wall 234.20 s；GAViT43片段未含计时/显存，不填补。此片段不重新证明权重身份、退出状态或逐图预测；沿用此前已记录的测试身份和边界。GAViT43仍是epoch8 best测试、训练到epoch21停止，不能写为30/30。跨seed硬件与历史训练完成条件不同，不能把所有波动唯一归因于随机seed。

AP是整个类别的排序指标，不是某一张场景图像的准确率；没有取得逐图预测/分组/图反馈证据，无法仅凭本表解释哪个具体场景的图结构导致得失。
