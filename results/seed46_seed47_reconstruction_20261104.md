# seed46/47历史结果补录

2026-11-04。用户再次确认已有主指标；原实验平台和记录已丢失，完整F1不可回收。本文件由先前已保存的数值表补录，保留全部正负结果，不作为原始平台运行日志。

AID / Swin-T Baseline
指标：Test_Acc(%)
seed46：96.56%
seed47：95.72%
均值±样本SD：96.14±0.59%
平均对Swin差值：0.00个百分点
macro-F1：未提供
BEN micro-F1：未提供

AID / GAViT (fixed grouping)
指标：Test_Acc(%)
seed46：97.12%
seed47：96.08%
均值±样本SD：96.60±0.74%
平均对Swin差值：0.46个百分点
macro-F1：未提供
BEN micro-F1：未提供

NWPU-RESISC45 / Swin-T Baseline
指标：Test_Acc(%)
seed46：93.76%
seed47：93.68%
均值±样本SD：93.72±0.06%
平均对Swin差值：0.00个百分点
macro-F1：未提供
BEN micro-F1：未提供

NWPU-RESISC45 / GAViT (fixed grouping)
指标：Test_Acc(%)
seed46：93.96%
seed47：94.19%
均值±样本SD：94.08±0.16%
平均对Swin差值：0.36个百分点
macro-F1：未提供
BEN micro-F1：未提供

NWPU-RESISC45 / GAViT soft_prototype
指标：Test_Acc(%)
seed46：93.54%
seed47：93.55%
均值±样本SD：93.55±0.01%
平均对Swin差值：-0.18个百分点
macro-F1：未提供
BEN micro-F1：未提供

BigEarthNet-19 / Swin-T Baseline
指标：Test_mAP(%)
seed46：71.0%
seed47：70.6%
均值±样本SD：70.80±0.28%
平均对Swin差值：0.00个百分点
macro-F1：未提供
BEN micro-F1：未提供

BigEarthNet-19 / GAViT (fixed grouping)
指标：Test_mAP(%)
seed46：71.8%
seed47：71.1%
均值±样本SD：71.45±0.49%
平均对Swin差值：0.65个百分点
macro-F1：未提供
BEN micro-F1：未提供

BigEarthNet-19 / GAViT soft_prototype
指标：Test_mAP(%)
seed46：71.4%
seed47：70.3%
均值±样本SD：70.85±0.78%
平均对Swin差值：0.05个百分点
macro-F1：未提供
BEN micro-F1：未提供

AID soft：未运行。

均值、样本SD和配对差由各自原始打印精度计算，最后统一显示两位小数。F1、逐图预测、混淆矩阵、运行时间、UUID、checkpoint与退出码未知，JSON保持null；没有从其他seed移植。既有历史F1按其实际seed范围单列。

[机器记录](seed46_seed47_reconstruction_20261104.json)、[原有数值表](multiseed_test_results.csv)、[原算术核对](seed46_seed47_audit_20261012.md)。补录和缺失状态不再作为AID及其他后续工作的前置阻塞。
