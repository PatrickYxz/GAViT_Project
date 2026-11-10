"""Independent stdlib-only audit. Writes exclusively inside this review folder."""
from decimal import Decimal, ROUND_HALF_UP, localcontext
from pathlib import Path
import csv
import hashlib
import json
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent
checks = {}


def check(name, value):
    checks[name] = bool(value)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(rel):
    return (ROOT / rel).read_text()


def display(value):
    return str(value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def numeric_tokens(text):
    return re.findall(r'[-−+]?\d+(?:\.\d+)?', text)


def section(text, start, end):
    return text[text.index(start):text.index(end)]


source = ROOT / 'results/multiseed_test_results.csv'
record_path = ROOT / 'results/seed46_seed47_reconstruction_20261104.json'
record = json.loads(record_path.read_text())
record_md = read('results/seed46_seed47_reconstruction_20261104.md')
rows = [r for r in csv.DictReader(source.read_text().splitlines()) if r['Test_seed46']]
check('source_hash_matches_record_and_historical_audit',
      sha(source) == record['source_sha256'] and
      sha(source) in read('results/seed46_seed47_audit_20261012.md'))
check('sixteen_records_eight_summaries',
      len(record['records']) == 16 and len(record['two_seed_descriptive_summaries']) == 8)
by_record = {(r['dataset'], r['model'], r['training_seed']): r for r in record['records']}
by_summary = {(r['dataset'], r['model']): r for r in record['two_seed_descriptive_summaries']}
baseline = {}
rounding = []
with localcontext() as context:
    context.prec = 60
    for row in rows:
        key = row['Dataset'], row['Model']
        values = [Decimal(row[f'Test_seed{seed}']) for seed in (46, 47)]
        # For n=2, sample SD has the independent closed form |x1-x2|/sqrt(2).
        mean = (values[0] + values[1]) / 2
        sd = abs(values[0] - values[1]) / Decimal(2).sqrt()
        if row['Model'] == 'Swin-T Baseline':
            baseline[row['Dataset']] = values
        differences = [v - b for v, b in zip(values, baseline[row['Dataset']])]
        delta = sum(differences) / 2
        for i, seed in enumerate((46, 47)):
            r = by_record[(*key, seed)]
            check(f'exact_source_value:{key}:{seed}',
                  Decimal(str(r['reported_value_percent'])) == values[i] and
                  r['source_printed_value'] == row[f'Test_seed{seed}'])
            check(f'exact_paired_difference:{key}:{seed}',
                  Decimal(str(r['delta_vs_swin_pp'])) == differences[i])
        s = by_summary[key]
        check(f'exact_mean_and_delta:{key}',
              Decimal(str(s['mean_percent'])) == mean and
              Decimal(str(s['mean_delta_vs_swin_pp'])) == delta)
        check(f'independent_sample_sd:{key}',
              abs(Decimal(str(s['sample_sd_percent'])) - sd) < Decimal('1e-15') and
              abs(statistics.stdev(map(float, values)) - float(sd)) < 1e-13)
        shown = f'{display(mean)}±{display(sd)}'
        check(f'half_up_display:{key}', s['display_mean_sd_percent'] == shown)
        expected = (f"{key[0]} / {key[1]}\n指标：{row['Metric']}\n"
                    f"seed46：{row['Test_seed46']}%\nseed47：{row['Test_seed47']}%\n"
                    f"均值±样本SD：{shown}%\n平均对Swin差值：{display(delta)}个百分点")
        check(f'markdown_values_and_display:{key}', expected in record_md)
        rounding.append({'dataset': key[0], 'model': key[1], 'mean': str(mean),
                         'sample_sd': str(sd), 'display': shown,
                         'mean_delta_pp': str(delta), 'display_delta_pp': display(delta)})

unknown = ['macro_f1_percent', 'micro_f1_percent', 'per_class_metrics', 'predictions',
           'confusion_matrix', 'run_exit_code', 'actual_commit', 'checkpoint_sha256',
           'actual_gpu_uuid', 'actual_software_versions', 'best_epoch', 'final_epoch',
           'peak_memory_bytes', 'wall_seconds']
check('all_224_unknown_measurement_fields_are_null',
      all(r[k] is None for r in record['records'] for k in unknown))
check('all_records_labeled_historical_user_report_no_raw_artifacts',
      all(r['source_type'] == 'user_reported_historical_result' and
          r['raw_artifacts_available'] is False for r in record['records']))
check('aid_soft_not_run', record['not_run'] == [
    {'dataset': 'AID', 'model': 'GAViT soft_prototype', 'status': 'not_run_user_reported'}])
check('lost_originals_and_f1_no_longer_progress_prerequisite',
      record['availability_update']['original_platform_and_records_lost_user_confirmed'] is True and
      record['availability_update']['complete_f1_not_available_user_confirmed'] is True and
      record['availability_update']['artifact_recovery_is_progress_prerequisite'] is False)
validation = json.loads((BASE / 'validation.json').read_text())
check('record_validation_current_and_internally_consistent',
      validation['record_sha256'] == sha(record_path) and
      validation['source_sha256'] == sha(source) and
      validation['count'] == len(validation['checks']) == 28 and
      validation['passed'] is True and all(validation['checks'].values()))

md_path = ROOT / 'docs/supervisor_progress_report_20261012.md'
plain_path = ROOT / 'docs/supervisor_progress_report_20261012_plaintext.txt'
md, plain = md_path.read_text(), plain_path.read_text()
old = (BASE / 'before_update/supervisor_progress_report_20261012.md').read_text()
old_plain = (BASE / 'before_update/supervisor_progress_report_20261012_plaintext.txt').read_text()
check('all_markdown_plaintext_numeric_tokens_match', numeric_tokens(md) == numeric_tokens(plain))
check('plaintext_has_no_tabs_or_markdown_table_rows',
      '\t' not in plain and not any(line.startswith('|') for line in plain.splitlines()))
prefixes = ('- Swin', '- 固定', '- soft', '- macro-F1', '- micro-F1', '- n：',
            '- seeds：', '- 已有F1', '- 三模型共同')
for name, previous, current, start, end in [
        ('markdown', old, md, '## 2.', '## 6.'),
        ('plaintext', old_plain, plain, '2. 本次', '6. 外部')]:
    before = [line for line in section(previous, start, end).splitlines()
              if line.startswith(prefixes)]
    after = [line for line in section(current, start, end).splitlines()
             if line.startswith(prefixes)]
    check(f'{name}_all_old_internal_metric_bullets_byte_identical', before == after)
    # Section 5 carries all existing methods, diagnostic and efficiency numbers.
    sec_start = '## 5.' if name == 'markdown' else '5. 方法'
    check(f'{name}_method_diagnostic_and_efficiency_section_byte_identical',
          section(previous, sec_start, end) == section(current, sec_start, end))
report_validation = json.loads((BASE / 'report_update_validation.json').read_text())
check('report_validation_hashes_current',
      report_validation['source_sha256'] == sha(md_path) and
      report_validation['plaintext_sha256'] == sha(plain_path))
check('english_email_validation_is_process_fact_without_byte_claim',
      'english_email_unchanged' not in report_validation and
      report_validation.get('english_email_edited_in_this_update') is False)

hgt = json.loads(read('results/hgtnet_nwpu_seed42_analysis_20261104.json'))
comparison = list(csv.DictReader(read('results/external_baselines_nwpu_seed42_20261030.csv').splitlines()))
hgt_row = next(r for r in comparison if r['model'] == 'HGTNet independent v1')
fixed_row = next(r for r in comparison if r['model'] == 'GAViT fixed4x4')
check('hgt_oa_and_f1_csv_matches_analysis',
      float(hgt_row['oa_percent']) == hgt['metrics']['oa_percent'] and
      float(hgt_row['macro_f1_percent']) == hgt['metrics']['macro_f1_percent'])
check('hgt_two_decimal_values_match_report',
      f"HGTNet独立实现：OA{hgt['metrics']['oa_percent']:.2f}%，macro-F1 {hgt['metrics']['macro_f1_percent']:.2f}%" in md)
check('hgt_eight_decimal_values_match_report',
      f"OA{hgt['metrics']['oa_percent']:.8f}%、macro-F1{hgt['metrics']['macro_f1_percent']:.8f}%" in md)
check('hgt_test_counts_match_report',
      f"{hgt['samples']}条预测重算得到{hgt['correct']}张正确、{hgt['errors']}张错误" in md)
history = hgt['history']
check('hgt_history_counts_and_unique_selection_match_report',
      f"完整{history['calibration_epochs']}轮校准和{history['refit_epochs']}轮fresh refit历史已核对" in md and
      f"验证集唯一最佳为{history['best_epoch']}轮，末轮{history['final_validation']['epoch']}" in md and
      history['max_tied_epochs'] == [95])
oa_delta = hgt['metrics']['oa_percent'] - float(fixed_row['oa_percent'])
check('hgt_oa_delta_and_net4_correct',
      f'高{oa_delta:.8f}个百分点OA' in md and
      hgt['correct'] - int(fixed_row['correct']) == 4 and '净正确数只差4张' in md)
fixed_params = int(fixed_row['neural_parameters'])
hgt_params = int(hgt_row['neural_parameters'])
reduction = (1 - fixed_params / hgt_params) * 100
check('hgt_params_and_reduction_match_report',
      f'固定版{fixed_params:,}参数，HGTNet{hgt_params:,}，前者少{reduction:.2f}%' in md)
paired = hgt['paired_vs_dfagcn']
check('hgt_paired_counts_match_report_and_sum25200',
      f"双方都正确{paired['both_correct']}、HGT纠正{paired['corrected']}、HGT新增错误{paired['regressed']}、双方都错误{paired['both_wrong']}，净多对{paired['corrected']-paired['regressed']}张" in md and
      sum(paired.values()) == 25200)
for line in old.splitlines():
    if line.startswith(('- Swin-T：OA', '- 固定4×4 GAViT：OA',
                        '- 局部软分配GAViT：OA', '- DFAGCN独立实现：OA')):
        check(f'existing_single_seed_score_row_unchanged:{line}', line in md.splitlines())
check('independent_recipe_single_seed_boundary_retained',
      '不同骨干、预训练、优化、增强和训练预算' in md and
      '不归因单个图算子' in md and '不能仅因split相同就宣称全部训练条件一致' in md)

aid = read('docs/external_baselines_20261027/hgtnet_aid_preparation_20261104.md')
old_aid = read('.worktrees/sparse-hybrid-4n-top2/docs/run_aid_formal_seed42.md')
manifest_sha = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
check('aid_manifest_identity_matches_historical_formal_record',
      manifest_sha in aid and manifest_sha in old_aid and
      '/home/featurize/work/aid_protocol_20260920_03.json' in aid and
      '/home/featurize/data/AID/AID' in aid)
check('aid_counts_and_smoke_scope_consistent',
      'train4000、val1000、pool5000、test5000' in aid and
      '共240/60' in aid and '30类、240/60、test0' in aid)
check('aid_recovery_and_future_scope_boundaries_explicit',
      '本阶段不再以回收46/47作为推进前置条件' in aid and
      '不表示训练入口已实现或GPU训练已通过' in aid and
      '不能由增加一个AID单seed自动替代' in aid)

findings = []
if '近期需要收回新增批次的summary.json/run.json' in md:
    findings.append({'priority': 'P2', 'file': str(md_path), 'line': 397,
                     'detail': 'Obsolete demand to recover lost 46/47 original logs.'})
quality = []
if '\n逐图复核新增：25200条唯一预测' in md:
    quality.append({'priority': 'P3', 'file': str(md_path), 'line': 447,
                    'detail': 'Name DFAGCN in this inherited paragraph following the new HGTNet paragraphs; the 21806-correct metrics themselves are unchanged.'})

result = {'review': 'independent focused record and report audit',
          'passed': all(checks.values()) and not findings,
          'checks_passed': sum(checks.values()), 'checks_total': len(checks),
          'checks': checks, 'failed_checks': [k for k, v in checks.items() if not v],
          'findings': findings, 'quality_observations': quality,
          'independent_rounding': rounding,
          'report_numeric_tokens_compared': len(numeric_tokens(md)),
          'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                            (source, record_path, md_path, plain_path,
                             BASE / 'report_update_validation.json')},
          'limits': ['No build_record execution; no large tests, network, GPU, installs or Git.',
                     'No reviewed files were edited. Writes are confined to this review directory.',
                     'Unknown log artifacts have no fabricated references; logs are absent from this schema rather than supplied.',
                     'English-email edited-in-update flag is a process fact from the root, not an independently established byte-preservation claim.']}
(HERE / 'independent_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('passed', 'checks_passed', 'checks_total',
                                       'failed_checks', 'findings', 'quality_observations')},
                 ensure_ascii=False, indent=2))
