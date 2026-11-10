"""Compile only previously reported metrics; unknown measurements remain null."""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import csv
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
source = ROOT / 'results/multiseed_test_results.csv'
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == '639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988'
rows = list(csv.DictReader(raw.decode().splitlines()))
records, summaries, not_run = [], [], []
references = {}
checks = {}


def check(name, value):
    checks[name] = bool(value)
    assert value, name


def display(value):
    return str(value.quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def new_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


blocks = []
for row in rows:
    if not row['Test_seed46'] and not row['Test_seed47']:
        not_run.append({'dataset': row['Dataset'], 'model': row['Model'], 'status': 'not_run_user_reported'})
        continue
    values = [Decimal(row[f'Test_seed{seed}']) for seed in (46, 47)]
    mean = sum(values) / 2
    sd = (sum((v - mean) ** 2 for v in values)).sqrt()
    check(f"mean:{row['Dataset']}:{row['Model']}", abs(mean - Decimal(row['Mean'])) <= Decimal('.0005'))
    check(f"sd:{row['Dataset']}:{row['Model']}", abs(sd - Decimal(row['Std_sample_n-1'])) <= Decimal('.0005'))
    if row['Model'] == 'Swin-T Baseline':
        references[row['Dataset']] = values
    differences = [value - baseline for value, baseline in zip(values, references[row['Dataset']])]
    delta = sum(differences) / 2
    check(f"delta:{row['Dataset']}:{row['Model']}", abs(delta - Decimal(row['Delta_vs_Swin_mean'])) <= Decimal('.0005'))
    summaries.append({'dataset': row['Dataset'], 'model': row['Model'], 'metric': row['Metric'], 'n': 2,
                      'mean_percent': float(mean), 'sample_sd_percent': float(sd),
                      'mean_delta_vs_swin_pp': float(delta), 'display_mean_sd_percent': f'{display(mean)}±{display(sd)}'})
    for seed, value, diff in zip((46, 47), values, differences):
        records.append({'dataset': row['Dataset'], 'model': row['Model'], 'training_seed': seed,
                        'metric': row['Metric'], 'reported_value_percent': float(value),
                        'source_printed_value': row[f'Test_seed{seed}'], 'delta_vs_swin_pp': float(diff),
                        'source_type': 'user_reported_historical_result', 'raw_artifacts_available': False,
                        'macro_f1_percent': None, 'micro_f1_percent': None, 'per_class_metrics': None,
                        'predictions': None, 'confusion_matrix': None, 'run_exit_code': None,
                        'actual_commit': None, 'checkpoint_sha256': None, 'actual_gpu_uuid': None,
                        'actual_software_versions': None, 'best_epoch': None, 'final_epoch': None,
                        'peak_memory_bytes': None, 'wall_seconds': None})
    blocks.append(f"{row['Dataset']} / {row['Model']}\n指标：{row['Metric']}\nseed46：{row['Test_seed46']}%\nseed47：{row['Test_seed47']}%\n均值±样本SD：{display(mean)}±{display(sd)}%\n平均对Swin差值：{display(delta)}个百分点\nmacro-F1：未提供\nBEN micro-F1：未提供\n")

check('eight_result_rows_sixteen_seed_values', len(summaries) == 8 and len(records) == 16)
check('aid_soft_not_run_retained', len(not_run) == 1 and not_run[0]['dataset'] == 'AID')
unknown_fields = ['macro_f1_percent', 'micro_f1_percent', 'per_class_metrics', 'predictions', 'confusion_matrix',
                  'run_exit_code', 'actual_commit', 'checkpoint_sha256', 'actual_gpu_uuid', 'actual_software_versions',
                  'best_epoch', 'final_epoch', 'peak_memory_bytes', 'wall_seconds']
check('no_inferred_missing_measurements', all(r[key] is None for r in records for key in unknown_fields))
check('source_file_unchanged', source.read_bytes() == raw)
report = {'schema': 'historical-metric-transcription-v1', 'date': '2026-11-04',
          'status': 'USER_REPORTED_VALUES_TRANSCRIBED_ORIGINAL_ARTIFACTS_LOST',
          'source': str(source.relative_to(ROOT)), 'source_sha256': digest,
          'availability_update': {'source': 'direct_user_message_2026-11-04',
                                  'original_platform_and_records_lost_user_confirmed': True,
                                  'complete_f1_not_available_user_confirmed': True,
                                  'known_results_reaffirmed_by_user': True,
                                  'artifact_recovery_is_progress_prerequisite': False},
          'meaning': 'A dated transcription of known reported scores, not original platform logs or recovered predictions.',
          'records': records, 'two_seed_descriptive_summaries': summaries, 'not_run': not_run,
          'auxiliary_reporting': 'Use only actually available historical F1 values with their own seed scope; do not insert 46/47 F1 into a four-seed aggregate.',
          'numeric_policy': 'Calculate from original printed decimals, sample SD n-1, ROUND_HALF_UP only for display.',
          'interpretation': 'Historical code, hardware and evidence differences remain; descriptive summaries do not identify pure seed variance or establish statistical equivalence.'}
target = ROOT / 'results/seed46_seed47_reconstruction_20261104.json'
new_json(target, report)
text = '# seed46/47历史结果补录\n\n2026-11-04。用户再次确认已有主指标；原实验平台和记录已丢失，完整F1不可回收。本文件由先前已保存的数值表补录，保留全部正负结果，不作为原始平台运行日志。\n\n'
text += '\n'.join(blocks)
text += '\nAID soft：未运行。\n\n均值、样本SD和配对差由各自原始打印精度计算，最后统一显示两位小数。F1、逐图预测、混淆矩阵、运行时间、UUID、checkpoint与退出码未知，JSON保持null；没有从其他seed移植。既有历史F1按其实际seed范围单列。\n\n'
text += '[机器记录](seed46_seed47_reconstruction_20261104.json)、[原有数值表](multiseed_test_results.csv)、[原算术核对](seed46_seed47_audit_20261012.md)。补录和缺失状态不再作为AID及其他后续工作的前置阻塞。\n'
with target.with_suffix('.md').open('x', encoding='utf-8') as stream:
    stream.write(text)
new_json(HERE / 'validation.json', {'passed': True, 'count': len(checks), 'checks': checks,
                                  'source_sha256': digest, 'records': len(records),
                                  'record_sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
print(json.dumps({'passed': True, 'checks': len(checks), 'records': len(records), 'source_unchanged': True}))
