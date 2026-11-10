import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

root = Path('/Users/patrick/Projects/GAViT_Project')
wt = root / '.worktrees/sparse-hybrid-4n-top2'
data_path = root / 'results/supervisor_progress_data_20261012.json'
report_path = root / 'docs/supervisor_progress_report_20261012.md'
email_path = root / 'docs/supervisor_email_draft_20261012.md'
evidence = root / 'results/evidence/seed46_seed47_reported_20261012'
data = json.loads(data_path.read_text())
old_data = json.loads(data_path.read_text())
report = report_path.read_text()
email = email_path.read_text()
visual = report.split('## 5. 方法、可视化与效率\n', 1)[1].split('## 6.', 1)[0]
original_csv_hash = hashlib.sha256((root / 'results/multiseed_test_results.csv').read_bytes()).hexdigest()
requested = [42, 43, 46, 47]

def D(x):
    return Decimal(str(x))

def fmt(x, signed=False):
    value = D(x).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return ('+' if signed and value >= 0 else '') + format(value, '.2f')

def stats(values):
    v = [D(x) for x in values]
    m = sum(v) / len(v)
    sd = (sum((x-m)**2 for x in v)/(len(v)-1)).sqrt()
    return {'n':len(v), 'mean':float(m), 'sample_sd_ddof1':float(sd), 'mean_decimal':str(m), 'sample_sd_decimal':str(sd)}

def ms(s):
    return fmt(s['mean_decimal']) + '±' + fmt(s['sample_sd_decimal'])

chosen = [r for r in data['rows'] if r['seed'] in requested]
assert len(chosen) == 12
summaries = []
soft_summaries = []
for dataset in ['AID', 'NWPU', 'BEN']:
    rs = [r for r in chosen if r['dataset'] == dataset]
    assert [r['seed'] for r in rs] == requested
    summaries.append({'dataset':dataset, 'seeds':requested, 'metric':rs[0]['metric'],
        'swin':stats([r['swin'] for r in rs]), 'fixed':stats([r['fixed'] for r in rs]),
        'paired_fixed_minus_swin_pp':stats([D(r['fixed'])-D(r['swin']) for r in rs]),
        'fixed_positive_seeds':[r['seed'] for r in rs if r['fixed']>r['swin']],
        'scope':'user-requested post-result reporting subset; other historical seeds retained'})
    complete = [r for r in rs if r['soft'] is not None]
    if complete:
        soft_summaries.append({'dataset':dataset, 'seeds':[r['seed'] for r in complete],
            'n':len(complete), 'requested_seeds':requested,
            'missing_soft_seeds':[r['seed'] for r in rs if r['soft'] is None],
            'swin':stats([r['swin'] for r in complete]), 'fixed':stats([r['fixed'] for r in complete]),
            'soft':stats([r['soft'] for r in complete]),
            'paired_soft_minus_swin_pp':stats([D(r['soft'])-D(r['swin']) for r in complete]),
            'paired_soft_minus_fixed_pp':stats([D(r['soft'])-D(r['fixed']) for r in complete]),
            'scope':'complete three-model same-seed subset of requested seeds'})

data['reporting_selection'] = {'requested_seeds':requested, 'source':'direct user instruction 2026-10-13: 应该使用seed42434647四个seed汇报',
    'selection_after_existing_results':True, 'historical_seeds_not_removed':True,
    'not_a_complete_predefined_seed_stability_summary':True,
    'missing_soft_results':{'AID':requested, 'BEN':[42]},
    'auxiliary_F1_missing_for_new_seeds':[46,47]}
data['selected_four_seed_rows'] = chosen
data['selected_four_seed_summaries'] = summaries
data['selected_soft_same_seed_summaries'] = soft_summaries

main_lines = []
email_main_lines = []
for s in summaries:
    metric = 'mAP' if s['dataset']=='BEN' else 'OA'
    label = 'BigEarthNet-19' if s['dataset']=='BEN' else 'NWPU-RESISC45' if s['dataset']=='NWPU' else 'AID'
    delta = fmt(s['paired_fixed_minus_swin_pp']['mean_decimal'], True)
    main_lines.append(f"| {label} | {metric} | 4 | {ms(s['swin'])} | {ms(s['fixed'])} | {delta} |")
    email_main_lines.append(f"| {label} / {metric} (%) | {ms(s['swin']).replace('±',' ± ')} | {ms(s['fixed']).replace('±',' ± ')} | {delta} |")
soft_lines = []
email_soft_lines = []
for s in soft_summaries:
    label = 'BigEarthNet-19' if s['dataset']=='BEN' else 'NWPU-RESISC45'
    cells = [ms(s[m]) for m in ['swin','fixed','soft']]
    soft_lines.append(f"| {label} | {'/'.join(map(str,s['seeds']))} | {s['n']} | {' | '.join(cells)} | {fmt(s['paired_soft_minus_swin_pp']['mean_decimal'],True)} | {fmt(s['paired_soft_minus_fixed_pp']['mean_decimal'],True)} |")
    email_soft_lines.append(f"| {label} / {'mAP' if s['dataset']=='BEN' else 'OA'} (%) | {s['n']} | {' | '.join(c.replace('±',' ± ') for c in cells)} |")

selected_section = '''## 2. 本次汇报主表：seed42、43、46、47

本次主表按用户指定采用42、43、46、47四个seed，报告均值±样本SD（n−1）。这是在已有结果上确定的汇报子集；全部历史seed及其正负结果保留在第3节，不能将本表称为完整预设seed的稳定性证据。历史硬件/代码/训练轮数差异及新增46/47工件待核验的限制仍适用。

| 数据集 | 指标 | n | Swin均值±SD % | 固定GAViT均值±SD % | 固定−Swin均值 pp |
|---|---|---:|---:|---:|---:|
''' + '\n'.join(main_lines) + '''

在这个四seed子集中，固定版AID与BEN的四次配对主指标均为正；NWPU为三次正、seed43负。不得据此改写为三个数据集每次均提升，或忽略历史44/45来证明总体稳定。

### 软分配：只在有三模型结果的相同seed上比较

NWPU在指定四seed上结果齐全。BEN缺少soft42，因此当前仅在43/46/47三seed上汇总三模型；不使用seed45补位，也不把三seed软分配均值与四seed对照均值直接相减。AID未运行soft。

| 数据集 | 三模型共同seeds | n | Swin均值±SD % | 固定均值±SD % | soft均值±SD % | soft−Swin均值 pp | soft−固定均值 pp |
|---|---|---:|---:|---:|---:|---:|---:|
''' + '\n'.join(soft_lines) + '''

新46/47的F1尚未提供，指定四seed的完整辅助指标表待补齐。现有历史F1分数及其统计范围见第4节，不能直接作为本次四seed汇总。

### 新增RTX4090批次：seed46/47（补充）
'''
assert '## 2. 最新RTX4090批次：seed46/47' in report
report = report.replace('## 2. 最新RTX4090批次：seed46/47', selected_section.rstrip())
report = report.replace('日期：2026-10-12。本文为导师汇报草稿，汇总当前已归档结果及最新用户回传，尚待原始工件补齐的部分单独标注。',
    '更新日期：2026-10-13。本文为导师汇报草稿，主表采用用户指定的seed42/43/46/47；全量历史结果另列。尚待原始工件补齐的部分单独标注。')
report = report.replace('## 3. 全部已记录主指标：保留正负seed', '## 3. 完整历史记录（本次四seed主表之外的补充）')
report = report.replace('下表包含用户回传的新46/47与全部既有正负seed，给出当前完整账面的变化。', '下表为全量历史统计，范围不同于第2节指定的四seed主表，包含用户回传的新46/47与全部既有正负seed。')
old_sentence = '最终均值的seed范围应提前固定，保留已获得的负面结果；只选42/43/46/47四个正向seed会排除AID44和BEN44/45，不能支持总体稳定性。'
new_sentence = '本次进度主表按用户指定采用42/43/46/47，其他历史结果继续保留。该子集为已有结果上的汇报选择，不能代替全部预设seed的稳定性分析；论文最终统计范围和选择理由需明确说明。'
assert old_sentence in report
report = report.replace(old_sentence, new_sentence)
assert report.split('## 5. 方法、可视化与效率\n', 1)[1].split('## 6.', 1)[0] == visual

email = '''# 导师邮件草稿（2026-10-13更新）

草稿仅供审阅和复制发送。主表采用用户指定的seed42/43/46/47；其他历史seed及限制见[进度汇报](supervisor_progress_report_20261012.md)。新增46/47原始工件与辅助指标尚待补齐。

Subject: GAViT progress update: four-seed comparison and soft-grouping ablations

Dear Lipo,

I have evaluated Swin-T and fixed-grid GAViT on AID, NWPU-RESISC45, and BigEarthNet-19. This update summarizes four seeds per dataset, alongside the soft-assignment ablation on NWPU and BigEarthNet.

The main table below uses seeds 42, 43, 46, and 47 (mean ± sample standard deviation, n=4). This is a selected subset of the existing experiments; other historical seeds are retained in the accompanying report, so the subset is not presented as evidence of overall stability. All scores and summaries are displayed to two decimal places, with statistics calculated before display rounding.

| Dataset / metric | Swin-T | Fixed-grid GAViT | Mean gain (pp) |
|---|---:|---:|---:|
''' + '\n'.join(email_main_lines) + '''

Within this subset, fixed-grid GAViT improves over Swin in all four AID and BigEarthNet seeds, and in three of the four NWPU seeds. NWPU seed43 remains negative. Historical runs differ in some hardware, code versions, or completed epochs, so these are descriptive comparisons rather than a controlled estimate of seed variability.

The soft-assignment ablation below uses the same seeds for all three models within each row. NWPU has all four requested seeds. BigEarthNet currently has soft results for seeds43/46/47; soft42 is unavailable, so its ablation uses n=3. AID soft assignment has not been evaluated.

| Dataset / metric | n | Swin-T | Fixed-grid GAViT | Soft-assignment GAViT |
|---|---:|---:|---:|---:|
''' + '\n'.join(email_soft_lines) + '''

The soft-assignment variant is below both controls in all four NWPU seeds. On BigEarthNet it improves over fixed grouping in seed43 but is worse in seeds46/47; the paired mean difference is close to zero and the available controls are rounded, so this remains a mixed outcome for this implementation. The additional seeds46/47 were run on RTX4090 instances, and I am collecting their original run artifacts and missing F1 metrics before finalizing the complete tables.

The fixed/soft region overlays and actual construction-edge visualizations have been reviewed on a predetermined 90-image NWPU validation subset. They show changes in grouping, but do not yet demonstrate a consistent improvement in semantic coherence. A controlled 3×3 versus 4×4 comparison remains incomplete.

The related-work and experiment/discussion drafts are in place, together with a same-device model-forward efficiency comparison. The principal outstanding experimental requirement is the comparison with HGTNet on NWPU/AID and DFAGCN on NWPU. My latest archived material audit has not yielded the complete implementations needed for a reliable reproduction. Published scores will be kept separate from measurements under our evaluation protocol.

Could you advise on an acceptable reproduction route if the authors' complete implementations cannot be obtained? If an independent implementation is appropriate, I would like to agree on the assumptions and verification requirements before proceeding with multi-seed training. I would also appreciate your feedback on framing the thesis around conditional gains, the fixed-versus-soft ablation, robustness across seeds, and the additional computational cost.

I plan to finalize the experiment records and missing metrics first, address the remaining comparisons according to your feedback, and update the thesis chapters accordingly. The accompanying report includes the selected four-seed summary, the complete historical record, the available auxiliary metrics, and the remaining limitations.

Best regards,
Xingze
'''
email = email.replace('seeds43/46/47','seeds 43, 46, and 47').replace('soft42','soft seed42').replace('seed43','seed 43').replace('seeds46/47','seeds 46 and 47').replace('RTX4090','RTX 4090')

assert data['rows'] == old_data['rows']
assert data['summaries'] == old_data['summaries']
assert data['soft_three_model_same_seed_summaries'] == old_data['soft_three_model_same_seed_summaries']
assert original_csv_hash == hashlib.sha256((root/'results/multiseed_test_results.csv').read_bytes()).hexdigest()
for filename, content in [('email_before_four_seed_scope_20261013.md',email_path.read_bytes()),('report_before_four_seed_scope_20261013.md',report_path.read_bytes())]:
    dest = evidence/filename
    assert not dest.exists(), dest
    dest.write_bytes(content)
data_path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
report_path.write_text(report)
email_path.write_text(email)

entry = '''## 2026-10-13 — 按用户指定用42/43/46/47四seed作为导师汇报主表

- 用户明确指定本次汇报使用seed42、43、46、47。主表从46/47两seed补充汇总改为该四seed的Swin/固定版均值±样本SD，并统一两位小数。AID为96.37±0.48%→96.83±0.51%（配对均值+0.46pp），NWPU为93.81±0.11%→94.00±0.19%（+0.20pp），BEN为70.90±0.22%→71.65±0.39%（+0.75pp）。NWPU43仍负，不能把四seed解释为每次均优于baseline。
- 软分配按相同seed完整结果比较：NWPU四seed齐全；BEN soft42缺正式归档，仅43/46/47三seed的三模型配对，不用45替换42或与四seed均值直接相减。AID soft未运行。新46/47 F1及原始工件仍待补。汇报JSON新增指定子集与统计，原全量rows/summary、原CSV及历史44/45保留；邮件和中文汇报明确本次是已获得结果上的汇报子集，不作为全部预设seed的稳定性证明。
- 可视化和效率分析段落沿用，没有新deadline表述、外部发送或GPU工作。更改前两份草稿保留于证据目录，全部新统计均从原始可得精度计算后再显示。

---

'''
for p in [root/'research_diary.md',wt/'research_diary.md']:
    text = p.read_text()
    index = text.index('## 2026-10-')
    p.write_text(text[:index]+entry+text[index:])
print(json.dumps({'status':'four_seed_scope_updated','main':summaries,'soft':soft_summaries,'full_history_preserved':True},ensure_ascii=False))
