# 导师邮件草稿（2026-10-12）

草稿仅供审阅和复制发送。最新46/47属于用户回传分数，原始运行工件尚待补齐；完整正负seed与限制见[进度汇报](supervisor_progress_report_20261012.md)。

Subject: GAViT progress update: additional seeds and soft-grouping ablations

Dear Lipo,

I have evaluated Swin-T and fixed-grid GAViT on AID, NWPU-RESISC45, and BigEarthNet-19, using at least four random seeds per dataset. I have also implemented and evaluated the local-prototype soft-assignment module on NWPU and BigEarthNet.

The additional seeds 46 and 47 were run on RTX 4090 instances. Their reported primary metrics are summarized below (mean ± sample standard deviation, n=2). All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding. The NWPU seed46/47 scores were reported rounded to two decimal places; further digits are not available. I corrected two AID transcription errors: fixed GAViT seed46 is 97.12%, and Swin seed47 is 95.72%. I am still collecting the original run artifacts and the missing F1 metrics for this batch before finalizing the thesis tables.

| Dataset / metric | Swin-T | Fixed-grid GAViT | Soft-assignment GAViT |
|---|---:|---:|---:|
| AID / OA (%) | 96.14 ± 0.59 | 96.60 ± 0.74 | Not evaluated |
| NWPU / OA (%) | 93.72 ± 0.06 | 94.08 ± 0.16 | 93.55 ± 0.01 |
| BigEarthNet-19 / mAP (%) | 70.80 ± 0.28 | 71.45 ± 0.49 | 70.85 ± 0.78 |

The fixed-grid model improves over Swin in both additional seeds on each dataset. However, I have retained all earlier negative seeds in the progress report. The complete record does not yet establish a consistent improvement across datasets and seeds, and some historical runs differ in hardware, code version, or completed epochs. The latest BigEarthNet values are rounded to one decimal, so the +0.05-point mean difference between soft assignment and Swin in this batch does not establish an improvement.

The soft-assignment variant is below both controls in all four recorded NWPU seeds. On BigEarthNet, it improves over fixed grouping for seeds 43 and 45 but is worse for seeds 46 and 47. I will report this as a mixed ablation outcome for this specific implementation. The fixed/soft region overlays and actual construction-edge visualizations have also been reviewed on a predetermined 90-image NWPU validation subset. They show changes in grouping, but do not yet demonstrate a consistent improvement in semantic coherence. A controlled 3×3 versus 4×4 comparison remains incomplete.

The related-work and experiment/discussion drafts are in place, together with a same-device model-forward efficiency comparison. The principal outstanding experimental requirement is the comparison with HGTNet on NWPU/AID and DFAGCN on NWPU. My latest archived material audit has not yielded the complete implementations needed for a reliable reproduction. Published scores will be kept separate from measurements under our evaluation protocol.

Could you advise on an acceptable reproduction route if the authors' complete implementations cannot be obtained? If an independent implementation is appropriate, I would like to agree on the assumptions and verification requirements before proceeding with multi-seed training. I would also appreciate your feedback on framing the thesis around conditional gains, the fixed-versus-soft ablation, robustness across seeds, and the additional computational cost.

I plan to finalize the experiment records and missing metrics first, address the remaining comparisons according to your feedback, and update the thesis chapters accordingly. The accompanying report includes all currently recorded seeds, the available auxiliary metrics, and the remaining limitations.

Best regards,
Xingze
