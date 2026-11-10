"""Independent archive/identity/prediction audit using exact rational metrics.

Read JSON only; never load model, images or executable archive content.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = Path('/Users/patrick/Downloads/hgtnet_seed42_results_20261103.tar.tar')
EXPECTED_TEST_SHA = '869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae'
EXPECTED_MEMBERS = {'test.json', 'history.json', 'summary.json', 'binding.json',
                    'initialization.json', 'checkpoint_reload.json', 'attempts/0001/runtime.json'}
checks = {}
with tarfile.open(SOURCE, 'r:*') as archive:
    members = archive.getmembers()
    checks['exact_seven_unique_regular_members'] = (len(members) == 7
        and len({m.name for m in members}) == 7
        and {m.name for m in members} == EXPECTED_MEMBERS
        and all(m.isfile() and not m.issym() and not m.islnk() for m in members))
    checks['bounded_archive_sizes'] = sum(m.size for m in members) < 20 * 1024 ** 2
    checks['normalized_relative_members'] = all(not PurePosixPath(m.name).is_absolute()
        and '..' not in PurePosixPath(m.name).parts and '\\' not in m.name
        and PurePosixPath(m.name).as_posix() == m.name for m in members)
    assert all(checks.values())
    raws = {m.name: archive.extractfile(m).read() for m in members}
test = json.loads(raws['test.json'])
binding = json.loads(raws['binding.json'])
summary = json.loads(raws['summary.json'])
initialization = json.loads(raws['initialization.json'])
reload_record = json.loads(raws['checkpoint_reload.json'])
prior_status = json.loads((ROOT/'results/evidence/hgtnet_formal_complete_return_20261103/status_reported.json').read_text())
checks['returned_test_raw_sha'] = hashlib.sha256(raws['test.json']).hexdigest() == EXPECTED_TEST_SHA
checks['summary_same_test_sha'] = summary['test_sha256'] == summary['test']['sha256'] == EXPECTED_TEST_SHA
checks['test_schema_and_scope'] = (test['schema'] == 'hgtnet-formal-evaluation-v1'
    and test['input_scope'] == 'real_nwpu' and test['phase'] == 'test'
    and test['metrics']['scope'] == 'formal_test')
checks['summary_completed_real_formal'] = summary['completed'] is True and summary['formal_training_started'] is True and summary['input_scope'] == 'real_nwpu'
checks['ordered_45_class_names'] = (len(binding['class_names']) == 45
    and binding['class_names'] == sorted(binding['class_names'])
    and len(set(binding['class_names'])) == 45)
names = binding['class_names']; index = {name: i for i, name in enumerate(names)}
paths, supplied_truth, predictions = test['paths'], test['labels'], test['predictions']
checks['25200_unique_ordered_examples'] = (len(paths) == len(supplied_truth) == len(predictions) == 25200
    and len(set(paths)) == 25200 and paths == sorted(paths))
truth = []
for path in paths:
    p = PurePosixPath(path)
    assert not p.is_absolute() and '..' not in p.parts and '\\' not in path
    assert p.as_posix() == path and len(p.parts) == 2 and p.parts[0] in index
    truth.append(index[p.parts[0]])
checks['labels_rebuilt_from_class_directories'] = supplied_truth == truth and all(type(v) is int for v in supplied_truth)
checks['predictions_valid_integer_classes'] = all(type(v) is int and 0 <= v < 45 for v in predictions)
checks['test_paths_match_frozen_binding_order'] = paths == [r['path'] for r in binding['records']['test']]
checks['test_labels_match_frozen_binding_order'] = truth == [r['label'] for r in binding['records']['test']]
checks['each_class_support560'] = Counter(truth) == Counter({i: 560 for i in range(45)})
confusion = [[0 for _ in names] for _ in names]
for actual, predicted in zip(truth, predictions): confusion[actual][predicted] += 1
checks['full_confusion2025_entries_match'] = confusion == test['metrics']['confusion_matrix']
checks['exact45_per_class_metric_rows'] = len(test['metrics']['per_class']) == 45
checks['confusion_integer_nonnegative'] = all(type(v) is int and v >= 0 for row in test['metrics']['confusion_matrix'] for v in row)
correct = sum(actual == predicted for actual, predicted in zip(truth, predictions))
oa = float(Fraction(correct, len(paths)) * 100)
per_class = []; exact_f1s = []
for i, name in enumerate(names):
    tp = confusion[i][i]; support = sum(confusion[i]); predicted = sum(row[i] for row in confusion)
    recall = Fraction(tp, support) if support else Fraction(0)
    precision = Fraction(tp, predicted) if predicted else Fraction(0)
    exact_f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
    exact_f1s.append(exact_f1)
    row = dict(class_index=i, class_name=name, support=support, predicted_count=predicted,
        tp=tp, fn=support-tp, fp=predicted-tp, precision_percent=float(100*precision),
        accuracy_recall_percent=float(100*recall), f1_percent=float(100*exact_f1))
    reported = test['metrics']['per_class'][i]
    checks['class%d_identity_support_correct' % i] = (reported['class'] == name
        and reported['support'] == support and reported['correct'] == tp)
    checks['class%d_recall_f1' % i] = (math.isclose(reported['accuracy_percent'], row['accuracy_recall_percent'], rel_tol=0, abs_tol=1e-10)
        and math.isclose(reported['f1_percent'], row['f1_percent'], rel_tol=0, abs_tol=1e-10))
    per_class.append(row)
macro = float(sum(exact_f1s, Fraction(0)) / 45 * 100)
checks['23718_correct_1482_errors'] = correct == 23718 and len(paths)-correct == 1482
checks['OA_recomputed_match'] = math.isclose(oa, test['metrics']['accuracy_percent'], rel_tol=0, abs_tol=1e-10)
checks['rational_macroF1_recomputed_match'] = math.isclose(macro, test['metrics']['macro_f1_percent'], rel_tol=0, abs_tol=1e-10)
checks['summary_test_metrics_equal'] = test['metrics'] == summary['test_metrics'] == summary['test']['metrics']
checks['status_test_metrics_match'] = all(prior_status['test_metrics'][k] == test['metrics'][k] for k in ['accuracy_percent','macro_f1_percent','scope'])
checks['summary_reference_rows_loss_seconds'] = (summary['test']['rows'] == 25200
    and summary['test']['loss'] == test['loss'] and summary['test']['seconds'] == test['seconds'])
checks['test_runtime_loss_positive_finite'] = all(math.isfinite(test[k]) and test[k] > 0 for k in ['seconds','loss'])
for key in ['manifest_sha256','recipe_sha256']:
    checks[key+'_cross_records'] = test[key] == binding[key] == summary[key] == prior_status[key]
for key in ['config_sha256','weights_sha256']:
    checks[key+'_cross_records'] = binding[key] == summary[key] == prior_status[key]
checks['epoch95_references'] = test['selected_epoch'] == summary['selected_epoch'] == summary['best_epoch'] == summary['refit_epochs'] == 95 and summary['final_epoch'] == 100
checks['refit_checkpoint_reference_consistent'] = (test['refit_checkpoint_sha256'] == summary['checkpoints']['final_refit']['sha256']
    == summary['checkpoint_reload']['sha256'] == reload_record['sha256'] == prior_status['checkpoints']['final_refit']['sha256'])
checks['refit_initial_state_reference_consistent'] = (initialization['equal'] is True
    and initialization['calibration_state_sha256'] == initialization['refit_state_sha256']
    == summary['initial_state_sha256'] == summary['refit_initial_state_sha256']
    == reload_record['metadata']['initial_state_sha256'])
checks['reload_original_matches_summary'] = reload_record == summary['checkpoint_reload'] == prior_status['checkpoint_reload']
checks['snapshot_metadata_counts'] = binding['counts'] == summary['counts'] == dict(train=5040,val=1260,pool=6300,test=25200)

df_path = ROOT/'results/evidence/dfagcn_test_predictions_20261031/test.json'
df_raw = df_path.read_bytes();df = json.loads(df_raw)
checks['DF_reference_original_sha'] = hashlib.sha256(df_raw).hexdigest() == '771dee2dcd1e676e4884c36654cc8fbcc4c067811362dbf9f3e997c903ccd4e9'
df_record = {k:v for k,v in df.items() if k != 'record_sha256'}
checks['DF_record_seal'] = hashlib.sha256(json.dumps(df_record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest() == df['record_sha256']
checks['HGT_DF_same_manifest_paths_classes'] = (df['manifest_sha256'] == test['manifest_sha256']
    and df['sample_paths'] == paths and [r['class_name'] for r in df['metrics']['per_class']] == names)
assert checks['HGT_DF_same_manifest_paths_classes']
paired = Counter();paired_per_class = {name:Counter() for name in names}
for actual, hgt, baseline in zip(truth, predictions, df['predictions']):
    key = ('both_correct' if hgt == actual else 'HGT_regressed') if baseline == actual else ('HGT_corrected' if hgt == actual else 'both_wrong')
    paired[key] += 1;paired_per_class[names[actual]][key] += 1
checks['paired_all25200'] = sum(paired.values()) == 25200
checks['paired_net1912'] = paired['HGT_corrected'] - paired['HGT_regressed'] == 1912
checks['paired_HGT_correct_sum'] = paired['both_correct'] + paired['HGT_corrected'] == correct
checks['paired_DF_correct_sum'] = paired['both_correct'] + paired['HGT_regressed'] == 21806
checks['paired_both_error_directions_retained'] = paired['HGT_corrected'] > 0 and paired['HGT_regressed'] > 0
record = dict(schema='hgtnet-return-independent-prediction-audit-v1',analysis_date='2026-11-04',
    passed=all(checks.values()),checks_count=len(checks),checks=checks,
    archive_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),test_sha256=hashlib.sha256(raws['test.json']).hexdigest(),
    metrics_recomputed=dict(samples=25200,correct=correct,errors=25200-correct,oa_percent=oa,macro_f1_percent=macro,
        reported_macro_f1_percent=test['metrics']['macro_f1_percent'],absolute_error_macro_f1=abs(macro-test['metrics']['macro_f1_percent']),absolute_tolerance=1e-10),
    confusion_matrix=confusion,per_class=per_class,HGT_vs_DF_paired=dict(paired),
    HGT_vs_DF_paired_per_class={name:dict(counts) for name,counts in paired_per_class.items()},
    limits=['Original manifest/image/model/checkpoint bytes were not uploaded; supplied binding and naming rule are independently checked as data references.',
        'Initial/refit state equality and numeric reload remain references in returned original metadata, not an independent model-byte reload.',
        'Both trained models are one seed with different recipes/backbones; paired counts are descriptive, not module causality or seed stability.',
        'No HGT-vs-GAViT paired correction/regression inferred from the four-example net difference; GAViT full predictions not supplied here.'])
(HERE/'validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:record[k] for k in ['passed','checks_count','metrics_recomputed','HGT_vs_DF_paired']},indent=2))
if not record['passed']:raise SystemExit('audit failed: '+str([k for k,v in checks.items() if not v]))
