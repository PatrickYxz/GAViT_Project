# AID 三seed类别变化

2026-09-22。每格为该类GAViT正确数−Swin正确数，同一测试集和支持数。正负不能当作相同图片在各seed的重复行为，原始逐样本预测尚未本地取得。不是AP。

| 类别 | 每seed support | seed42净正确 | seed43净正确 | seed44净正确 | 平均类准确率差pp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Airport | 180 | +1 | +3 | -4 | +0.0000 |
| BareLand | 155 | -2 | +3 | +0 | +0.2151 |
| BaseballField | 110 | +1 | +0 | +0 | +0.3030 |
| Beach | 200 | +0 | +0 | +1 | +0.1667 |
| Bridge | 180 | +2 | +4 | -1 | +0.9259 |
| Center | 130 | +1 | +4 | +3 | +2.0513 |
| Church | 120 | +1 | +0 | +0 | +0.2778 |
| Commercial | 175 | +23 | +3 | -21 | +0.9524 |
| DenseResidential | 205 | -2 | +3 | +5 | +0.9756 |
| Desert | 150 | +1 | +1 | -1 | +0.2222 |
| Farmland | 185 | +0 | +1 | +1 | +0.3604 |
| Forest | 125 | +0 | +0 | -1 | -0.2667 |
| Industrial | 195 | -6 | -1 | -1 | -1.3675 |
| Meadow | 140 | +2 | -1 | -4 | -0.7143 |
| MediumResidential | 145 | +1 | -2 | -1 | -0.4598 |
| Mountain | 170 | +2 | -1 | +0 | +0.1961 |
| Park | 175 | +7 | -1 | -5 | +0.1905 |
| Parking | 195 | +0 | +0 | +0 | +0.0000 |
| Playground | 185 | +0 | +0 | -1 | -0.1802 |
| Pond | 210 | +1 | +1 | +1 | +0.4762 |
| Port | 190 | +0 | +3 | -2 | +0.1754 |
| RailwayStation | 130 | +2 | +3 | -10 | -1.2821 |
| Resort | 145 | -5 | +0 | +5 | +0.0000 |
| River | 205 | +2 | +0 | -3 | -0.1626 |
| School | 150 | -2 | +0 | -3 | -1.1111 |
| SparseResidential | 150 | +2 | +1 | +1 | +0.8889 |
| Square | 165 | -3 | -4 | +3 | -0.8081 |
| Stadium | 145 | +1 | -3 | +0 | -0.4598 |
| StorageTanks | 180 | -1 | -2 | +0 | -0.5556 |
| Viaduct | 210 | +1 | +0 | +0 | +0.1587 |

Commercial +23/+3/−21、RailwayStation +2/+3/−10、Park +7/−1/−5，收益依赖seed。三次都正向的类为Center、Pond、SparseResidential；三次都负向为Industrial。小计数和多类事后查看不支持类别级显著性或结构因果结论，不能据此选模型。

来源：[seed42逐类核验](aid_seed42_per_class_analysis.json)、[seed43](aid_seed43_per_class_analysis.json)、[seed44](aid_seed44_per_class_analysis.json)。全局结果见[三seed汇总](aid_multiseed_summary.md)。
