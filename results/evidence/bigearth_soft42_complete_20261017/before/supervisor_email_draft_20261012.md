# 导师邮件草稿（2026-10-13更新）

草稿仅供审阅和复制发送。主表采用用户指定的seed42/43/46/47；其他历史seed及限制见[进度汇报](supervisor_progress_report_20261012.md)。新增46/47原始工件与辅助指标尚待补齐。

Subject: GAViT progress update: four-seed comparison and soft-grouping ablations

Dear Professor Wang,

I have evaluated Swin-T and fixed-grid GAViT on AID, NWPU-RESISC45, and BigEarthNet-19. This update summarizes four seeds per dataset, alongside the soft-assignment ablation on NWPU and BigEarthNet.

The main table below uses seeds 42, 43, 44, and 45(mean ± sample standard deviation, n=4). This is a set of the existing experiments;  All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding.

| Dataset / metric | Swin-T | Fixed-grid GAViT | Mean gain (pp) |
|---|---:|---:|---:|
| AID / OA (%) | 96.37 ± 0.48 | 96.83 ± 0.51 | +0.46 |
| NWPU-RESISC45 / OA (%) | 93.81 ± 0.11 | 94.00 ± 0.19 | +0.20 |
| BigEarthNet-19 / mAP (%) | 70.90 ± 0.22 | 71.65 ± 0.39 | +0.75 |

Within this subset, fixed-grid GAViT improves over Swin in all four AID and BigEarthNet seeds, and in three of the four NWPU seeds. NWPU seed 43 remains negative. 

The soft-assignment ablation below uses the same seeds for all three models within each row. NWPU has all four requested seeds. BigEarthNet currently has soft results for seeds 43, 44, and 45; soft seed 42 is unavailable, so its ablation uses n=3. AID soft assignment has not been evaluated.

| Dataset / metric | n | Swin-T | Fixed-grid GAViT | Soft-assignment GAViT |
|---|---:|---:|---:|---:|
| NWPU-RESISC45 / OA (%) | 4 | 93.81 ± 0.11 | 94.00 ± 0.19 | 93.65 ± 0.15 |
| BigEarthNet-19 / mAP (%) | 3 | 70.90 ± 0.26 | 71.53 ± 0.38 | 71.52 ± 1.29 |

The soft-assignment variant is below both controls in all four NWPU seeds. On BigEarthNet it improves over fixed grouping in seed 43 but is worse in seeds 44 and 45; the paired mean difference is close to zero and the available controls are rounded, so this remains a mixed outcome for this implementation. 

I have completed and reviewed the fixed/soft region overlays and graph-construction visualizations on a predetermined 90-image NWPU validation subset. The figures illustrate changes in region assignments and feature-edge connectivity, with three corrected predictions and five jointly incorrect cases retained for qualitative analysis. Semantic region quality has not been quantitatively assessed. A controlled 3×3 versus 4×4 comparison remains incomplete.

The related-work and experiment/discussion drafts are in place, together with a same-device model-forward efficiency comparison. The principal outstanding experimental requirement is the comparison with HGTNet on NWPU/AID and DFAGCN on NWPU. My latest archived material audit has not yielded the complete implementations needed for a reliable reproduction. Published scores will be kept separate from measurements under our evaluation protocol.

Could you advise on an acceptable reproduction route if the authors' complete implementations cannot be obtained? If an independent implementation is appropriate, I would like to agree on the assumptions and verification requirements before proceeding with multi-seed training. I would also appreciate your feedback on framing the thesis around conditional gains, the fixed-versus-soft ablation, robustness across seeds, and the additional computational cost.

I plan to finalize the experiment records and missing metrics first, address the remaining comparisons according to your feedback, and update the thesis chapters accordingly. The accompanying report includes the selected four-seed summary, the complete historical record, the available auxiliary metrics, and the remaining limitations.

Best regards,
Xingze
