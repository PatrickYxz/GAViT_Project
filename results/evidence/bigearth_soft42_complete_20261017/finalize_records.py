"""Reconcile current status and advisor prose after the archived formal BEN42 return."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import json
import shlex
import statistics

ROOT = Path('/Users/patrick/Projects/GAViT_Project')
E = Path(__file__).resolve().parent
WT = ROOT / '.worktrees/sparse-hybrid-4n-top2'

def read(p):
    return json.loads(p.read_text())

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def edit(p, replacements):
    text = p.read_text()
    for old, new in replacements:
        assert text.count(old) == 1, (p, old[:90], text.count(old))
        text = text.replace(old, new, 1)
    p.write_text(text)

def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))

summary = read(E/'summary.json')
train = summary['stages']['gavit_train']
metrics = summary['stages']['gavit_test']['metrics']
four = read(ROOT/'results/ben_four_seed_three_model_20261017.json')
progress_path = ROOT/'results/supervisor_progress_data_20261012.json'
progress = read(progress_path)
f1rows = [r for r in progress['rows'] if r['dataset']=='BEN' and r.get('soft_macro_f1') is not None]
f1 = {'seeds':[r['seed'] for r in f1rows], 'n':len(f1rows)}
for key in ('macro_f1','micro_f1'):
    values = [r['soft_'+key] for r in f1rows]
    f1[key] = {'mean':statistics.mean(values),'sample_sd_ddof1':statistics.stdev(values)}
progress['ben_soft_available_auxiliary_summary'] = f1
save(progress_path, progress)
f1text = [fmt(f1[k]['mean'])+'±'+fmt(f1[k]['sample_sd_ddof1']) for k in ('macro_f1','micro_f1')]

status_path = ROOT/'results/bigearth_soft42_status_20261013.json'
status = read(status_path)
status.update(status='formal_complete_return_reconciled', completion_verified=True,
    server_training_started=True, server_training_start_verified=True,
    training_start_verification_scope='Retrospective verification from completed full30 formal train/test receipts, not a live first-batch observation',
    formal_test_metrics=metrics, missing_formal_soft_seed=None,
    existing_soft_metric_seeds=[42,43,46,47], best=train['best'], final_epoch=30,
    final_val_mAP_percent=four['soft42_final_val_mAP_percent'],
    checkpoint_sha256=train['checkpoint_sha256'], resources=four['resources'],
    scheduler_repair_required=False, scheduler_repair_applied_on_server=True,
    scheduler_repair_confirmation='Both returned formal stage launcher SHAs match the locally repaired worker 09722f10542bed449d2e80df0041387e113cab58db403b3979e07f192194f767; failed historical smoke/dispatch evidence retained',
    evidence_scope='User-returned queue/train/test exit0, complete summary and log reconciled locally; server checkpoint and prediction-array bytes not independently downloaded',
    latest_evidence_received_date='2026-10-17',
    completion_evidence='results/evidence/bigearth_soft42_complete_20261017/summary.json',
    completion_validation='results/evidence/bigearth_soft42_complete_20261017/validation.json',
    next_action='Use archived formal result and completed four-seed/per-class reports; preserve negative result and collect46/47 original artifacts/F1; no repeat soft42 training')
save(status_path, status)

edit(ROOT/'docs/run_ben_soft42_20261013.md', [
('用户已授权补跑。**2026-10-17：用户报告实例已退还，是否完整训练和测试仍待持久化工件核对。** 先读取`work`目录中的结果，不重新启动、预检或解压数据；本地尚无soft42正式分数。',
 '用户已授权的补跑已完成。**2026-10-17：回传总队列、训练和测试三项exit0，summary complete、30轮及119825张/19类测试均已核对。** 正式test mAP **70.17%**、macro-F1 **64.05%**、micro-F1 **76.80%**；best epoch10。详见[完成结果与四seed汇总](ben_four_seed_three_model_20261017.md)和[核对记录](../results/evidence/bigearth_soft42_complete_20261017/validation.json)。下文为历史操作及原读取方式，不重复训练。'),
('## 当前只读检查：查看已保存结果','## 历史只读检查命令（此次回传已经核对完成）')])

edit(ROOT/'docs/supervisor_email_draft_20261012.md', [
('# 导师邮件草稿（2026-10-13更新）','# 导师邮件草稿（2026-10-17更新）'),
('新增46/47原始工件与辅助指标尚待补齐。','BEN soft42正式结果及逐类对比已补齐；新增46/47原始工件与辅助指标尚待补齐。'),
('The main table below uses seeds 42, 43, 44, and 45(mean ± sample standard deviation, n=4). This is a set of the existing experiments;  All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding.',
 'The main table below uses seeds 42, 43, 46, and 47 (mean ± sample standard deviation, n=4). This reporting subset was selected from existing experiments; the full historical record is retained in the accompanying report. All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding.'),
('The soft-assignment ablation below uses the same seeds for all three models within each row. NWPU has all four requested seeds. BigEarthNet currently has soft results for seeds 43, 44, and 45; soft seed 42 is unavailable, so its ablation uses n=3. AID soft assignment has not been evaluated.',
 'The soft-assignment ablation now covers the same four seeds (42, 43, 46, and 47) for all three models on both NWPU and BigEarthNet. The missing BigEarthNet soft seed42 run has completed 30 epochs and the full 119,825-image test, with test mAP of 70.17%. AID soft assignment has not been evaluated.'),
('| BigEarthNet-19 / mAP (%) | 3 | 70.90 ± 0.26 | 71.53 ± 0.38 | 71.52 ± 1.29 |',
 '| BigEarthNet-19 / mAP (%) | 4 | 70.90 ± 0.22 | 71.65 ± 0.39 | 71.19 ± 1.25 |'),
('The soft-assignment variant is below both controls in all four NWPU seeds. On BigEarthNet it improves over fixed grouping in seed 43 but is worse in seeds 44 and 45; the paired mean difference is close to zero and the available controls are rounded, so this remains a mixed outcome for this implementation.',
 'The soft-assignment variant is below both controls in all four NWPU seeds. On BigEarthNet it exceeds Swin in seeds43/46 and falls below it in42/47, averaging +0.29 percentage points. It exceeds fixed grouping only in seed43 and averages −0.46 points relative to fixed grouping, with a larger sample standard deviation. Per-class results also reverse across seeds: coastal wetlands and beaches improve in soft43 but deteriorate in soft42. These are mixed outcomes for this implementation. Historical code/hardware differences remain, and the original artifacts and auxiliary metrics for seeds46/47 still need to be collected.')])

edit(ROOT/'docs/supervisor_progress_report_20261012.md', [
('更新日期：2026-10-13。','更新日期：2026-10-17。'),
('NWPU42/43三模型；BEN43/45三模型均有归档；新增46/47双数据集三模型主指标',
 'NWPU42/43三模型；BEN42/43/45三模型均有归档，soft42已完成full30及完整test；新增46/47双数据集三模型主指标'),
('NWPU在指定四seed上结果齐全。BEN缺少soft42，因此当前仅在43/46/47三seed上汇总三模型；不使用seed45补位，也不把三seed软分配均值与四seed对照均值直接相减。AID未运行soft。',
 'NWPU与BEN在指定42/43/46/47四seed上均已有三模型主指标。BEN soft42正式30轮与完整119825张测试已核对，test mAP70.1745705651681%；seed45继续保留在历史补充，不替换指定四seed。AID未运行soft。46/47主指标为用户确认回传值，原始工件仍待收回。'),
('| BigEarthNet-19 | 43/46/47 | 3 | 70.90±0.26 | 71.53±0.38 | 71.52±1.29 | +0.62 | -0.01 |',
 '| BigEarthNet-19 | 42/43/46/47 | 4 | 70.90±0.22 | 71.65±0.39 | 71.19±1.25 | +0.29 | -0.46 |\n\n在指定BEN四seed中，soft对Swin为2升2降（43/46正、42/47负）；对固定版仅43为正。soft样本SD1.25pp高于固定版0.39pp，不据均值+0.29pp宣称稳定提升。'),
('| 42 | 70.90 | 72.00 | — | +1.10 | — | 历史归档* |',
 '| 42 | 70.90 | 72.00 | 70.17 | +1.10 | -0.73 | 历史对照*；soft正式full30/test已归档 |'),
('*历史42–44主指标原始回传保留一位小数，本表补尾零显示两位；',
 '*历史42–44的Swin/固定对照主指标原始回传保留一位小数，本表补尾零显示两位；soft42/43使用正式JSON完整精度统计再显示；'),
('43三模型存在代码/历史硬件差异，soft43正式使用当前4090、full30；这些限制不被相同seed消除。',
 '42/43三模型存在代码/历史硬件差异；soft42/43分别在记录UUID的RTX4090上full30。soft42 best10/val78.75%、final30/val75.71%，测试最佳验证checkpoint。这些限制不被相同seed消除。'),
('避免将BEN soft的四seed均值与Swin的六seed均值直接相减。',
 '避免将BEN soft的五seed均值与Swin的六seed均值直接相减。'),
('| BEN | 43/45/46/47 | 71.09±0.44 | 71.21±0.71 | 71.35±1.11 | +0.26 | +0.14 |',
 '| BEN | 42/43/45/46/47 | 71.06±0.39 | 71.37±0.71 | 71.12±1.09 | +0.06 | -0.26 |'),
('BEN soft43/45相对固定的mAP提高，而46/47下降；',
 'BEN soft43/45相对固定的mAP提高，而42/46/47下降；'),
('| BEN | soft | 43/45 | 65.77±1.28 | 76.64±0.37 |',
 '| BEN | soft | 42/43/45 | '+f1text[0]+' | '+f1text[1]+' |'),
('软分配GAViT对Swin（BEN42）：缺少正式test mAP和逐类AP，当前留空；小样本工程短测、NWPU soft42或其他BEN seed不补入此处。',
 '软分配GAViT对Swin（BEN42）：天然草地/稀疏植被+6.24pp、多年生作物+2.76pp、荒原/石南灌丛/硬叶植被+2.64pp、阔叶林+1.04pp；海岸湿地−19.64pp、沙滩/沙丘/沙地−4.43pp、工业或商业区−2.32pp。总体mAP−0.73pp。该次正式test macro-F1为64.05%、micro-F1为76.80%。海岸湿地、沙滩和草地的变化方向与soft43相反，不能写成跨seed一致的类别改善。'),
('## 7. 可直接复用的归档材料\n',
 '## 7. 可直接复用的归档材料\n\n- [BEN soft42完成及指定四seed三模型汇总](ben_four_seed_three_model_20261017.md)、[四seed CSV](../results/ben_four_seed_three_model_20261017.csv)、[soft42正式回传核对](../results/evidence/bigearth_soft42_complete_20261017/validation.json)。\n')])

entry = '''## 2026-10-17 — BEN soft42正式完成；补齐指定四seed并保留负面结果

- 对用户回传的总队列/训练/test三项exit0、summary complete、final30、最佳验证checkpoint、119825张/19类测试和完成标记完成31项身份/数值核对；[原文](results/evidence/bigearth_soft42_complete_20261017/server_return.txt)、[summary](results/evidence/bigearth_soft42_complete_20261017/summary.json)、[核对边界](results/evidence/bigearth_soft42_complete_20261017/validation.json)保存。该核对基于回传，未独立下载服务器权重/预测数组或运行CUDA。
- 实验bigearth_soft42_20261013_01，seed42、模型ed480e60fea03ed786fdba70c144de5e815c34f5，已修复调度worker SHA09722f10542bed449d2e80df0041387e113cab58db403b3979e07f192194f767与两阶段一致。原调度失败历史保留。RTX4090 UUID GPU-44825cf0-bd98-bfdc-efbc-0100238a6789，FP32/noAMP、matmulTF32false/cuDNNtrue、8线程。
- 问题为补齐指定42/43/46/47的BEN soft缺项及逐类比较；沿用soft43配方，主要变量为训练seed42，物理GPU身份按实记录。RGB/19标签、原划分237871/122342/119825及SHA、原ImageNet权重；预处理/增强沿用冻结配方，未重建split。K16局部原型dim128/radius1/temp0.1/1次迭代、48空间+32特征边、GAT256/4头/2层、token_feedback/dropout0.1，31466766参数；AdamW lr3e-4/wd1e-4、Cosine30、batch32/workers4、BCE、F1阈值0.5。原命令和完整metadata见summary argv；由训练seed变化不能排除历史对照代码/硬件混杂。
- best10/val mAP78.75%，final30/val75.70723466069043%；test mAP70.1745705651681%、macro-F1 64.04828509185678%、micro-F1 76.7999586513447%。checkpoint SHA23314ea561f753b1aad73025148d7b9788bbdddc4f38f49afeca59a6aec91c35在train/meta/test/metrics一致，路径为/home/featurize/work/bigearth_soft42_20261013_01/gavit_train/gavit.pth；日志为同根run.log和gavit_train/run.log，曲线/探针留在服务器阶段目录。
- 训练20619.28s、完整test155.17s、队列20808.02s（5h46m48s）；有效训练处理量346.09图/s包含validation/IO/探针，不是纯前向吞吐。训练峰值allocated3.66GiB；按旧价1.67元/h计运行部分约9.65元，实际账单/准备空闲未知。
- 相比Swin42 mAP−0.73pp、固定42−1.83pp；soft对Swin草地+6.24、多年生作物+2.76，但海岸湿地−19.64、沙滩−4.43、工业区−2.32pp。与soft43方向反转，补足失败分析，不删除、不为了正向结果重跑。指定四seed Swin70.90±0.22、固定71.65±0.39、soft71.19±1.25%；soft对Swin2升2降/均值+0.29pp，对固定仅43正/均值−0.46pp。用户指定子集及历史44/45均保留，46/47原工件/F1/逐类仍缺。
- [四seed汇总](docs/ben_four_seed_three_model_20261017.md)、[CSV](results/ben_four_seed_three_model_20261017.csv)、[逐seed固定/soft分别对Swin的全部类别](docs/ben_seed42_seed43_vs_swin_20261013.md)、导师中文汇报/英文草稿及当前状态已同步。研究工作树bigearth_comparison.csv仅追加一个正式soft42条目，旧行字节不变；主表CSV由artifact-tool author/inspect/render，统计先按原精度计算再四舍五入两位。其余AID/NWPU/seed43记录与46/47 CSV保持。无新训练、外部发送、Git提交推送。

正式阶段命令（保存的argv；冻结调度负责入口加载、精度和资源记录）：

'''
for stage in ('gavit_train','gavit_test'):
    entry += '```bash\npython -B -u '+shlex.join(summary['stages'][stage]['argv'])+'\n```\n\n'
entry += '---\n\n'
for diary in (ROOT/'research_diary.md', WT/'research_diary.md'):
    text = diary.read_text()
    marker = '## 2026-10-17 — 用户报告实例已退还；等待持久化结果核对'
    assert text.count(marker)==1
    addition = entry if diary.parent==ROOT else entry.replace('](results/','](../../results/').replace('](docs/','](../../docs/')
    diary.write_text(text.replace(marker,addition+marker,1))

print(json.dumps({'status':'records_updated','BEN_soft_F1':f1,'formatted_F1':f1text},ensure_ascii=False))
