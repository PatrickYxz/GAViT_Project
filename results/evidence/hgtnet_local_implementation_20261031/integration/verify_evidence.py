"""Independent stdlib audit of the saved local receipts and current source bytes."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(relative):
    return json.loads((EVIDENCE / relative).read_text())


summary = read('full_cpu_01/summary.json')
training = read('full_cpu_01/training_summary.json')
pretraining = read('full_cpu_01/pretraining.json')
suite = read('integration/related_suite.json')
preservation = read('preservation/audit.json')
checks = {}
checks['local_complete'] = summary['completed'] is True and training['completed'] is False
checks['no_formal_or_accuracy'] = summary['formal_training_started'] is False and summary['accuracy_metrics'] is None
checks['full_parameters'] = summary['parameters']['total'] == 74619858 == sum(summary['parameters']['groups'].values())
checks['actual_cpu_fp32'] = summary['runtime']['device'] == 'cpu' and summary['runtime']['dtype'] == 'float32'
checks['different_reload_pid'] = summary['runtime']['pid'] != summary['reload']['runtime']['pid']
checks['both_reload_errors_zero'] = summary['reload']['max_abs_errors'] == {'batch1': 0.0, 'batch2': 0.0}
checks['training_receipt_bound'] = digest(EVIDENCE / 'full_cpu_01/training_summary.json') == summary['reload']['training_summary_sha256']
checks['checkpoint_bytes'] = digest(EVIDENCE / 'full_cpu_01/model.pt') == summary['checkpoint']['sha256']
checks['fixture_bytes'] = digest(EVIDENCE / 'full_cpu_01/inference_fixture.pt') == summary['fixture_sha256']
cfg_raw = json.dumps(summary['config'], sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
checks['config_bytes'] = hashlib.sha256(cfg_raw).hexdigest() == summary['config_sha256']
for name, expected in summary['source_identity'].items():
    checks['source:' + name] = digest(ROOT / 'external_baselines/hgtnet' / name) == expected
checks['all_gradients'] = summary['gradients']['passed'] and not summary['gradients']['missing'] and not summary['gradients']['nonfinite']
checks['all_451_parameter_tensors'] = sum(group['tensors'] for group in summary['gradients']['groups'].values()) == 451
checks['all_five_groups_updated'] = len(summary['updated_groups']) == 5 and all(group['changed'] for group in summary['updated_groups'].values())
checks['candidate_actual_load'] = pretraining['mode'] == 'applied' and pretraining['source_identity_verified'] is True and len(pretraining['loaded']) == 318
checks['candidate_identity'] = pretraining['source']['sha256'] == '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
checks['candidate_bytes'] = digest(pretraining['source']['path']) == pretraining['source']['sha256']
checks['candidate_not_original'] = pretraining['is_original_hgtnet_weights'] is False
checks['candidate_all_gates'] = not any(pretraining[key] for key in ['missing', 'unexpected', 'shape_mismatch', 'dtype_mismatch', 'nonfinite', 'invalid'])
checks['related_tests'] = suite['successful'] and suite['tests_run'] == 235 and not (suite['failures'] or suite['errors'] or suite['skipped'])
for group in preservation['groups'].values():
    for record in group['records']:
        checks['preserved:' + record['path']] = digest(ROOT / record['path']) == record['expected_sha256']
for batch in [1, 2]:
    trace = summary[f'batch{batch}_trace']
    checks[f'full_batch{batch}_trace'] = (trace['input'] == [batch, 3, 224, 224]
        and trace['stage_outputs'] == [[batch, 256, 28, 28], [batch, 512, 14, 14], [batch, 512, 7, 7]]
        and trace['graph_outputs'] == [[batch, 256, 7, 7], [batch, 512, 5, 5]]
        and [len(group) for group in trace['graph_indices']] == [6, 2]
        and all(record['self_edges'] == 0 and record['indices_shape'] == [batch, nodes, 20]
                for group, nodes in zip(trace['graph_indices'], [49, 25]) for record in group))
package = ROOT / 'external_baselines/hgtnet'
record = {'scope': 'saved local synthetic CPU receipt audit, not remote or dataset verification',
          'passed': all(checks.values()), 'checks': checks,
          'passed_checks': sum(checks.values()), 'total_checks': len(checks),
          'package_python_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sorted(package.glob('*.py'))},
          'supporting_files_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
              sorted(list((ROOT / 'tests').glob('test_hgtnet_*.py')) +
                     [ROOT / 'configs/hgtnet_independent_v0.json',
                      EVIDENCE / 'review/independent_review.md', EVIDENCE / 'integration/run_related_tests.py',
                      EVIDENCE / 'integration/commands_and_exits.json'])},
          'git_head': '058f1d8b79d0be538ec358eb742d2cd3992921d4',
          'summary_sha256': digest(EVIDENCE / 'full_cpu_01/summary.json'),
          'related_tests': suite, 'preservation_report': str(EVIDENCE / 'preservation/audit.json')}
print(json.dumps(record, sort_keys=True, indent=2))
sys.exit(0 if record['passed'] else 1)
