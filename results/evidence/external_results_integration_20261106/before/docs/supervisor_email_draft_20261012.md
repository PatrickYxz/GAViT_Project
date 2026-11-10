# 导师邮件草稿（2026-10-17更新）

草稿仅供审阅和复制发送。主表采用用户指定的seed42/43/46/47；其他历史seed及限制见[进度汇报](supervisor_progress_report_20261012.md)。BEN soft42正式结果及逐类对比已补齐；新增46/47原始工件与辅助指标尚待补齐。

Subject: GAViT progress update: four-seed comparison and soft-grouping ablations

Dear Professor Wang,

I have evaluated Swin-T and fixed-grid GAViT on AID, NWPU-RESISC45, and BigEarthNet-19. This update summarizes four seeds per dataset, alongside the soft-assignment ablation on NWPU and BigEarthNet.

The following results use seeds 42, 43, 44, and 45(mean ± sample standard deviation, n=4). All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding.

Dataset / metric: AID / OA (%)
Swin-T: 96.37 ± 0.48
Fixed-grid GAViT: 96.83 ± 0.51
Mean gain (pp): +0.46

Dataset / metric: NWPU-RESISC45 / OA (%)
Swin-T: 93.81 ± 0.11
Fixed-grid GAViT: 94.00 ± 0.19
Mean gain (pp): +0.20

Dataset / metric: BigEarthNet-19 / mAP (%)
Swin-T: 70.90 ± 0.22
Fixed-grid GAViT: 71.65 ± 0.39
Mean gain (pp): +0.75

Within this set, fixed-grid GAViT improves over Swin in all four AID and BigEarthNet seeds, and in three of the four NWPU seeds. NWPU seed 43 remains negative. 

The soft-assignment ablation covers the same four seeds for all three models on both NWPU and BigEarthNet. AID soft assignment has not been evaluated.

Dataset / metric: NWPU-RESISC45 / OA (%)
n: 4
Swin-T: 93.81 ± 0.11
Fixed-grid GAViT: 94.00 ± 0.19
Soft-assignment GAViT: 93.65 ± 0.15

Dataset / metric: BigEarthNet-19 / mAP (%)
n: 4
Swin-T: 70.90 ± 0.22
Fixed-grid GAViT: 71.65 ± 0.39
Soft-assignment GAViT: 71.19 ± 1.25

The soft-assignment variant is below both controls in all four NWPU seeds. On BigEarthNet it exceeds Swin in seeds 43 and 44 and falls below it in seeds 42 and 45, averaging +0.29 percentage points. It exceeds fixed grouping only in seed 43 and averages −0.46 points relative to fixed grouping, with a larger sample standard deviation. Per-class results also reverse across seeds: coastal wetlands and beaches improve with soft assignment in seed 43 but deteriorate in seed 42.

I have completed a trained 3×3 versus 4×4 comparison on four predetermined BigEarthNet validation inputs using the seed-42 checkpoints, with actual graph edges, within-region pooling weights and paired predictions. The 4×4 model has higher reference test mAP (72.00% versus 71.08%). In the four examples it corrects three label decisions but introduces one missed label; some original errors persist. The figures retain every example and both outcome directions. These diagnostic cases do not estimate population gains or isolate an individual edge's causal contribution.

I also retain the fixed/soft GAViT visualizations on 90 predetermined NWPU validation images using seed-42 calibration-best checkpoints: 82 images are correct for both models, soft corrects three fixed-grid errors, introduces no new errors, and both remain wrong on five images. Semantic region quality has not been quantitatively assessed.

The related-work and experiment/discussion drafts are in place, together with a same-device model-forward efficiency comparison. The principal outstanding experimental requirement is the comparison with HGTNet on NWPU/AID and DFAGCN on NWPU. My latest archived material audit has not yielded the complete implementations needed for a reliable reproduction. Published scores will be kept separate from measurements under our evaluation protocol.

Could you advise on an acceptable reproduction route if the authors' complete implementations cannot be obtained? If an independent implementation is appropriate, I would like to agree on the assumptions and verification requirements before proceeding with multi-seed training. I would also appreciate your feedback on framing the thesis around conditional gains, the fixed-versus-soft ablation, robustness across seeds, and the additional computational cost.

I plan to finalize the experiment records and missing metrics first, address the remaining comparisons according to your feedback, and update the thesis chapters accordingly. The accompanying report includes the selected four-seed summary, the complete historical record, the available auxiliary metrics, and the remaining limitations.

Best regards,
Xingze
