# BigEarthNet seed43/45 三模型描述汇总（2026-10-12）

| seed | Swin mAP% | fixed mAP% | soft mAP% | soft−Swin约 pp | soft−fixed约 pp |
|---|---:|---:|---:|---:|---:|
| 43 | 71.1000 | 71.7000 | 72.8676 | +1.7676 | +1.1676 |
| 45 | 71.6769 | 70.2582 | 70.8369 | -0.8400 | +0.5787 |

| 模型 | mAP均值±样本SD % | macro-F1均值±样本SD % | micro-F1均值±样本SD % |
|---|---:|---:|---:|
| Swin-T | 71.3885 ± 0.4080 | 64.5921 ± 0.6959 | 76.3661 ± 0.5178 |
| 固定4×4 GAViT | 70.9791 ± 1.0195 | 64.9977 ± 0.9931 | 76.7512 ± 0.0690 |
| soft_prototype GAViT | 71.8522 ± 1.4359 | 65.7694 ± 1.2792 | 76.6447 ± 0.3695 |

seed43旧Swin/fixed仅一位小数，表中相关差、均值与SD为打印中心值计算的近似；多写小数不代表精度提高。soft两次均有精确回传指标。SD为ddof=1，配对差先按同seed计算再求汇总，n=2不作显著性或稳定性结论。

soft−fixed mAP平均约+0.8731pp，两次均升；macro-F1两次均升，micro-F1一升一降、均差约-0.1066pp。soft−Swin mAP平均约+0.4638pp，43升45降。当前支持BigEarth上soft相对固定的两次主要指标改善观察，尚未建立稳定优于Swin或跨数据集一致优势。

43对照fixed仅保存21轮且历史代码/训练-test硬件不同；45三模型完整30轮，旧对照deb2b45/soft ed480e6、2080Ti型号相同但物理UUID不同。43soft4090、45soft2080Ti，不能把变化仅归因seed或GPU。相同119825 test不累计成更多独立测试样本，不能拿Swin/fixed四seed均值和soft两seed均值冒充共同seed配对。

共同类别观察：seed43与45，soft相对fixed均改善海岸湿地、沙滩/沙丘/沙地，均降低草地/稀疏植被；只作为有限类别描述，AP与罕见类支持量/机制仍需现有预测分析。NWPU soft两个seed均低于Swin；OA不与mAP跨任务平均。最终三seed要求未满足，不按结果筛选seed或自行新起训练。

来源：[seed43](bigearth_soft43_ablation.md)、[seed45](bigearth_soft45_ablation.md)；[精确/打印精度分列CSV](bigearth_three_model_two_seed_comparison.csv)、[结构化统计](bigearth_three_model_two_seed_summary.json)。

seed43资源现已归档：best第8轮/val mAP78.99004691745016%，30轮训练及完整test整批6小时22分21.73秒，按1.67元/h折算约10.64元；资源口径与证据见[正式记录](bigearth_soft43_ablation.md)。资源补齐不改变上表指标或n=2限制，也不构成三模型公平速度比较。
