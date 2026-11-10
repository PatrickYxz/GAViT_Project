"""Narrow standard-library P2 follow-up; never runs a model or GPU operation."""
import ast
import builtins
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'PIL'}:
        raise RuntimeError('ML import prohibited in P2 status check: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from external_baselines.hgtnet import aid_entry
from tests.test_hgtnet_aid_entry import EntryTests, complete_fixture

tempfile.tempdir = str(HERE)
names = [
    'test_status_needs_exit_progress_summary_checkpoint_and_current_sources',
    'test_skeletal_or_tampered_completion_receipts_stay_incomplete',
    'test_missing_or_malformed_required_completion_artifacts_stay_incomplete',
    'test_end_check_receipt_requires_actual_end_identity_selected_images_and_checkpoint',
]
stream = io.StringIO()
suite = unittest.TestSuite(EntryTests(name) for name in names)
run = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
(HERE/'p2_delta_tests.log').write_text(stream.getvalue())

# Preserve the original failed fixture; make a new source-current copy so that
# rejection cannot be explained only by the old entry hash in source_identity.
original = HERE/'status_partial_fixture'
updated = HERE/'status_partial_current_source_fixture'
updated.mkdir(exist_ok=False)
for path in original.iterdir():
    if path.is_file():
        (updated/path.name).write_bytes(path.read_bytes())
summary = json.loads((updated/'summary.json').read_text())
summary['source_identity'] = aid_entry._source_identity()
(updated/'summary.json').write_text(json.dumps(summary))
partial = aid_entry.status(updated)

# Read already generated full-model CPU evidence solely for the engine/entry
# schema correspondence. Its CPU scope must remain invalid for real completion.
integration = HERE.parent/'integration'
actual = json.loads((integration/'full_model_final_result.json').read_text())
actual_v = json.loads((integration/'full_model_final_validation.json').read_text())
actual_dir = Path(actual_v['cpu_fixture_output'])
actual_first = json.loads((actual_dir/'first_step.json').read_text())
with tempfile.TemporaryDirectory(dir=HERE) as temporary:
    fixture = Path(temporary)/'complete'
    positive = complete_fixture(fixture)
    engine_schema_checks = {
        'result_fields_required_by_entry_present': set(positive).intersection({
            'completed','training_started','formal_training_started','counts','optimizer_steps','best_epoch',
            'final_epoch','protocol','history','val_metrics','checkpoint','first_batch_gradients',
        }).issubset(actual),
        'history_fields_match': set(actual['history']) == set(positive['history']),
        'checkpoint_receipt_fields_match': set(actual['checkpoint']) == set(positive['checkpoint']),
        'gradients_fields_match': set(actual['first_batch_gradients']) == set(positive['first_batch_gradients']),
        'five_real_gradient_groups': set(actual['first_batch_gradients']['groups']) == aid_entry.GRADIENT_GROUPS,
        'first_step_fields_match': set(actual_first) == set(json.loads((fixture/'first_step.json').read_text())),
        'actual_first_step_diagnostic_passed': actual_first['diagnostic_passed'] is True,
        'actual_30class_metrics_fields_match': set(actual['val_metrics']) == set(positive['val_metrics']),
        'protocol_matches_actual_engine': aid_entry._same_json(actual['protocol'], aid_entry.PROTOCOL),
        'actual_CPU_fixture_not_real_completion': not aid_entry._completion_evidence_ok(actual_dir, actual),
    }
entry_path = ROOT/'external_baselines/hgtnet/aid_entry.py'
checks = dict(
    four_targeted_P2_regressions=run.wasSuccessful(),
    original_counterexample_with_current_sources_rejected=not partial['completed'],
    original_counterexample_current_sources_match=partial['source_identity_matches'] is True,
    no_ML_import=not any(name in sys.modules for name in ('torch','torchvision','numpy','PIL')),
    entry_AST_valid=ast.parse(entry_path.read_text()) is not None,
    **engine_schema_checks,
)
result = dict(
    schema='hgtnet-aid-independent-P2-followup-v1',
    passed=all(checks.values()), checks=checks,
    entry_sha256=hashlib.sha256(entry_path.read_bytes()).hexdigest(),
    current_source_identity=aid_entry._source_identity(),
    targeted_tests=dict(count=run.testsRun,failures=len(run.failures),errors=len(run.errors)),
    original_counterexample_state=partial['state'],
    source_engine_evidence=str(integration/'full_model_final_result.json'),
    actual_engine_schema_scope='Existing actual CPU full-model artifacts only; no rerun or conversion into real-AID evidence.',
    no_GPU_or_training_performed=True,
)
(HERE/'p2_delta_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
raise SystemExit(0 if result['passed'] else 1)
