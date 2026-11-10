"""Consolidate independent review evidence without any model/GPU execution."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
load=lambda name:json.loads((HERE/name).read_text())
focus=load('focused_suite.json');post=load('checkpoint_postfix_validation.json')
counter=load('failure_counter_final_validation.json');pool=load('partial_pool_post_validation.json')
transaction=load('best_transaction_validation.json')
best_packet=load('best_packet_post_validation.json')
paths=sorted((ROOT/'external_baselines/hgtnet').glob('formal_*.py'))
sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
unchanged=[name for name in sources if not name.endswith(('formal_data.py','formal_engine.py'))]
checks=dict(focused50_passed=focus['passed'] and focus['tests']==50 and not focus['failures'] and not focus['errors'] and not focus['skipped'],
    focused_sources_unchanged_during_run=focus['source_unchanged_during_run'],
    three_unchanged_sources_still_same=all(focus['source_hashes'][name]==sources[name] for name in unchanged),
    checkpoint_postfix8_pass=post['passed'] and post['checks_count']==8,
    checkpoint_postfix_sources_current=all(sources[name]==sha for name,sha in post['sources'].items()),
    complete_training_steps_preserved_on_validation_failure=counter['all_completed_steps_observable'] and counter['actual_adam_steps']==2,
    failure_counter_source_current=counter['engine_sha256']==sources['external_baselines/hgtnet/formal_engine.py'],
    partial_pool_staged_to_new_target=pool['model_stage_reached'] and 'pool_attempt_' in pool['pool_roots'][0],
    partial_pool_data_preserved=pool['partial_directory_preserved'],
    pool_entry_source_current=pool['entry_sha256']==sources['external_baselines/hgtnet/formal_entry.py'],
    best_slot_transaction7_pass=transaction['passed'] and transaction['checks_count']==7,
    best_slot_source_current=transaction['engine_sha256']==sources['external_baselines/hgtnet/formal_engine.py'],
    final_data13_pass='Ran 13 tests' in (HERE/'data_final_independent.log').read_text() and (HERE/'data_final_independent.log').read_text().rstrip().endswith('OK'),
    final_engine15_pass='Ran 15 tests' in (HERE/'engine_final_independent.log').read_text() and (HERE/'engine_final_independent.log').read_text().rstrip().endswith('OK'),
    best_packet_matches_selected_after_replay=best_packet['best_packet_matches_selected_model_metrics'] and best_packet['final_file_correct']==best_packet['selected_best_correct']==0,
    best_packet_source_current=best_packet['engine_sha256']==sources['external_baselines/hgtnet/formal_engine.py'])
result=dict(schema='hgtnet-independent-final-review-v1',passed=all(checks.values()),checks=checks,checks_count=len(checks),
    source_hashes=sources,unresolved_important_findings=[],
    test_scope='independent focused50 suite on stable sources, then final data13 and final engine15 tests; checkpoint/protocol/entry unchanged; not claiming one53-test invocation',
    additional_scope='8 post-fix checkpoint checks, injected validation failure accounting, actual small ZIP resumed staging,7 actual CPU best-slot transaction checks and best-packet replay variation proof',
    not_claimed='full model CPU receipt, full HGT/DF regression suite, target3090 formal execution, NWPU scores, billing or bitwise deterministic GPU resume')
(HERE/'final_validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=result['passed'],checks=result['checks_count'],failed=[k for k,v in checks.items() if not v],source_hashes=sources),indent=2))
assert result['passed']
