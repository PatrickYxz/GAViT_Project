"""Independent stdlib review of user-reported formal startup/first epoch."""
import ast
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
SOURCE=Path('/Users/patrick/.codex/attachments/acdadd1b-130a-4b01-b2c1-8295d42ad6a8/已粘贴的文本.txt')
raw=SOURCE.read_text();lines=raw.splitlines()
device=json.loads(next(line.split(' ',1)[1] for line in lines if line.startswith('HGTNET_CUDA_DEVICE_PROBE ')))
events=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('HGTNET_FORMAL_PROGRESS ')]
first=events[0];complete=[e for e in events if e['stage']=='epoch_complete'];last=events[-1]
epoch1=complete[0];groups=first['gradients']['groups'];checks={}
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
checks['actual_local_bundle_sha']=hashlib.sha256(bundle.read_bytes()).hexdigest()=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1'
with zipfile.ZipFile(bundle) as z:
    engine=z.read('external_baselines/hgtnet/formal_engine.py').decode()
    checks['local_engine_matches_delivered_source']=engine==(ROOT/'external_baselines/hgtnet/formal_engine.py').read_text()
    run=next(n for n in ast.parse(engine).body if isinstance(n,ast.FunctionDef) and n.name=='run_formal')
    code=ast.get_source_segment(engine,run)
    a=code.index("_snapshot(output,state,model,optimizer,'calibration',epoch)")
    b=code.index("_history_file(output,state)",a)
    c=code.index("_notify(on_progress,state,'epoch_complete',phase='calibration'",b)
    checks['epoch_complete_after_snapshot_history']=a<b<c
    train=next(n for n in ast.parse(engine).body if isinstance(n,ast.FunctionDef) and n.name=='_train_epoch')
    text=ast.get_source_segment(engine,train)
    checks['training_complete_has_no_validation_or_snapshot']='_evaluate(' not in text and '_snapshot(' not in text
    checks['training_complete_after_sample_and_finite_checks']=text.index("if seen!=expected")<text.index("_notify(callback,state,'training_epoch_complete'")
checks['device_actual_cuda0_report']=device==dict(available=True,cuda='12.1',device_count=1,name='NVIDIA GeForce RTX 3090',torch='2.2.2+cu121',total_memory_bytes=25295323136)
checks['first_update_success']=first['device']=='cuda:0' and first['formal_training_started'] is True and first['optimizer_step_succeeded'] is True and first['post_update_checks_passed'] is True
checks['five_gradient_groups']=set(groups)=={'dual_attention','graph','head','patch_embed','transformer'}
checks['451_gradient_tensors']=sum(g['tensors'] for g in groups.values())==451
checks['all_group_norms_finite_positive']=all(math.isfinite(g['l2']) and g['l2']>0 for g in groups.values())
checks['no_missing_nonfinite_gradients']=first['gradients']['missing']==[] and first['gradients']['nonfinite']==[] and first['gradients']['passed'] is True
checks['five_representative_parameter_updates']=first['updated_groups']==dict.fromkeys(groups,True)
checks['same_pid_attempt']=all(e['pid']==2661 and e['attempt']==1 for e in events)
checks['only_epoch1_complete']=len(complete)==1 and epoch1['epoch']==1 and epoch1['optimizer_steps']==315
train_events=[e for e in events if e['stage']=='training_epoch_complete']
checks['two_training_segments5040each']=[(e['epoch'],e['optimizer_steps'],e['epoch_optimizer_steps'],e['samples_seen']) for e in train_events]==[(1,315,315,5040),(2,630,315,5040)]
checks['epoch2_not_validation_complete']=last['stage']=='training_epoch_complete' and last['epoch']==2 and last['completed'] is False
checks['no_final_completion_marker']='HGTNET_NWPU_FORMAL_COMPLETE' not in raw
checks['only_calibration_phase']=all(e['phase']=='calibration' for e in events)
checks['train_rate_matches']=math.isclose(5040/epoch1['train_seconds'],epoch1['train_images_per_second'],rel_tol=0,abs_tol=1e-12)
checks['remaining99_epoch_mean_matches']=math.isclose(epoch1['epoch_wall_seconds']*99,epoch1['estimated_remaining_phase_seconds'],rel_tol=0,abs_tol=1e-9)
correct=epoch1['val_accuracy_percent']*1260/100
train_correct=epoch1['train_accuracy_percent']*5040/100
checks['val_correct974_arithmetic']=math.isclose(correct,974,rel_tol=0,abs_tol=1e-9)
checks['online_train_correct2415_arithmetic']=math.isclose(train_correct,2415,rel_tol=0,abs_tol=1e-9)
derived=dict(epoch1_train_seconds=epoch1['train_seconds'],epoch1_validation_seconds=epoch1['validation_seconds'],
    epoch1_wall_seconds=epoch1['epoch_wall_seconds'],epoch1_other_wall_seconds=epoch1['epoch_wall_seconds']-epoch1['train_seconds']-epoch1['validation_seconds'],
    reported_remaining_calibration_hours=epoch1['estimated_remaining_phase_seconds']/3600,
    first_epoch_conditional_calibration100_total_hours=100*epoch1['epoch_wall_seconds']/3600,
    epoch1_val_correct_arithmetic=round(correct),epoch1_train_correct_online_arithmetic=round(train_correct),
    epoch2_training_seconds=last['train_seconds'],epoch2_train_images_per_second_arithmetic=5040/last['train_seconds'])
record=dict(schema='hgtnet-formal-start-independent-review-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),derived=derived,
    disposition='Program-reported actual CUDA training startup accepted, epoch1 full calibration epoch complete; epoch2 training only. No anomaly in supplied events; continue unchanged.',
    source='user-pasted terminal output, not direct remote retrieval',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    not_confirmed=['current remote GPU process inventory/peak memory','remote files, checkpoints or run.exit bytes','epoch2 validation or complete epoch wall','final test scores','population performance or stable model superiority'],
    time_scope='First completed epoch-based current calibration phase estimate only; remaining refit/test and other startup/billing excluded. Other-wall subtraction is unsplit checks/serialization/storage/history overhead, not proof of one bottleneck.',
    metric_scope='macro-F1 server-reported, no prediction recomputation. Online train accuracy differs from frozen epoch-end validation scope.')
(HERE/'validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(passed=record['passed'],checks=len(checks),failed=[k for k,v in checks.items() if not v],derived=derived),indent=2))
if not record['passed']:raise SystemExit(1)
