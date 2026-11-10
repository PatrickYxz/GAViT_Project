# 导师完整seed结果澄清草稿

日期：2026-11-09。根据用户转贴的导师回复及本地归档拟定；仅为草稿，没有发送。

写作要点：上次仅列42/43/46/47，须补上已完成但未列的结果；不要把早已完成的历史实验改述为收到本次建议后新做的实验。AID原42/43/44同协议三seed的固定−Swin OA差为+0.60/+0.30/−0.76pp，平均+0.046667pp。BEN44/45负结果存在，但历史代码、硬件及训练终止差异需另列，不作为严格同条件seed方差。保留固定版为最终模型与评价它对Swin的稳定性是两项独立判断；soft的负面消融及可能机制按真实证据表述。

导师新增要求先登记为待核对方案：DFAGCN代码/权重联系和NWPU四seed，HGTNet原文数值与协议注，AID soft四seed，三模型参数/FLOPs、效率表及相关工作差异。用户当前问题是是否补正汇报，不视作租卡、联系作者或运行实验的授权。当前DFAGCN/HGTNet独立seed42结果均保留，不因新邮件而删除。

## 英文草稿

Subject: Clarification on the complete seed results and thesis framing

Dear Professor Wang,

Thank you for your detailed guidance. Before proceeding, I need to correct the scope of my previous update. I reported the results for seeds 42, 43, 46 and 47, but did not include all completed runs. I should have made this selection and the omitted outcomes clear.

For the originally planned AID seeds 42, 43 and 44, the fixed-grid GAViT minus Swin-T OA differences are +0.60, +0.30 and -0.76 percentage points, respectively, giving a mean difference of approximately +0.047 pp. The historical BigEarthNet runs for seeds 44 and 45 also show negative differences. Some historical runs differ in execution conditions, so I will distinguish comparable repetitions from historical comparisons rather than pool them as equivalent trials.

These additional results mean that the complete record does not currently support a claim of consistent improvement across all completed runs. I will provide the full per-seed results, training conditions and record availability, with means and sample standard deviations for explicitly defined groups. The +0.46 pp and +0.75 pp gains in my earlier update describe the reported four-seed subset.

I understand your recommendation to present soft assignment as a negative ablation where evaluated and to retain the fixed-grid architecture as the final model. I would appreciate your guidance on revising the performance claims and remaining experimental plan in light of the complete results. I apologise for the incomplete presentation in my previous update.

Best regards,
Xingze
