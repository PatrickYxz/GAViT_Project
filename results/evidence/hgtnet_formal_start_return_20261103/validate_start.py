"""Validate user-returned startup/epoch events, not remote live status."""
import hashlib,json,math,shlex,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from tools.package_hgtnet_nwpu_smoke import _read_bundle
raw=(HERE/'reported_output.txt').read_text();lines=raw.splitlines()
command=shlex.split(lines[0]);args={command[i]:command[i+1] for i in range(3,len(command),2)}
device=json.loads(next(s.split(' ',1)[1] for s in lines if s.startswith('HGTNET_CUDA_DEVICE_PROBE ')))
events=[json.loads(s.split(' ',1)[1]) for s in lines if s.startswith('HGTNET_FORMAL_PROGRESS ')]
first=events[0];epochs=[s for s in events if s['stage']=='epoch_complete'];e=epochs[0];last=events[-1]
checks={}
checks['v1_command']=command[:3]==['/environment/miniconda3/bin/python','/home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1.pyz','run']
checks['current_uuid']=args['--gpu-uuid']=='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1'
checks['formal_output']=args['--output']=='/home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/runs/nwpu_seed42_01'
checks['actual_cuda']=device==dict(available=True,cuda='12.1',device_count=1,name='NVIDIA GeForce RTX 3090',torch='2.2.2+cu121',total_memory_bytes=25295323136)
checks['first_cuda_step']=first['stage']=='first_optimizer_step' and first['optimizer_steps']==1 and first['device']=='cuda:0' and first['samples_seen']==16
checks['formal_attempt1_pid2661']=all(s['attempt']==1 and s['pid']==2661 and s['formal_training_started'] is True and s['training_started'] is True for s in events)
checks['unfinished_calibration']=all(s['completed'] is False and s['phase']=='calibration' for s in events)
groups={'dual_attention':30,'graph':120,'head':2,'patch_embed':4,'transformer':295}
checks['451_finite_gradient_tensors']=first['gradients']['passed'] is True and first['gradients']['missing']==[] and first['gradients']['nonfinite']==[] and {k:v['tensors'] for k,v in first['gradients']['groups'].items()}==groups
checks['positive_group_norms']=all(math.isfinite(v['l2']) and v['l2']>0 for v in first['gradients']['groups'].values())
checks['five_updated_groups']=first['updated_groups']==dict.fromkeys(groups,True) and first['post_update_checks_passed'] is True
checks['epoch1_complete_only']=len(epochs)==1 and e['epoch']==1 and e['optimizer_steps']==315
train_epochs=[s for s in events if s['stage']=='training_epoch_complete']
checks['both_full_training_counts']=[(s['epoch'],s['optimizer_steps'],s['epoch_optimizer_steps'],s['samples_seen']) for s in train_epochs]==[(1,315,315,5040),(2,630,315,5040)]
checks['latest_epoch2_training_only']=last is train_epochs[-1] and last['epoch']==2
checks['epoch1_throughput']=math.isclose(5040/e['train_seconds'],e['train_images_per_second'],rel_tol=1e-12)
checks['finite_timings_losses']=all(math.isfinite(e[k]) and e[k]>0 for k in ['train_loss','train_seconds','validation_seconds','epoch_wall_seconds','train_images_per_second'])
other=e['epoch_wall_seconds']-e['train_seconds']-e['validation_seconds']
checks['wall_scopes']=other>0
checks['remaining99epochs']=math.isclose(e['estimated_remaining_phase_seconds'],99*e['epoch_wall_seconds'],abs_tol=1e-8)
correct=e['val_accuracy_percent']*1260/100;train_correct=e['train_accuracy_percent']*5040/100
checks['val_correct_integer']=math.isclose(correct,round(correct),abs_tol=1e-8) and round(correct)==974
checks['online_train_correct_integer']=math.isclose(train_correct,round(train_correct),abs_tol=1e-8) and round(train_correct)==2415
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
checks['local_delivered_bundle_sha']=hashlib.sha256(bundle.read_bytes()).hexdigest()=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1'
manifest,_,payload=_read_bundle(bundle)
checks['formal3090_entry']=manifest['runner']=='external_baselines/hgtnet/formal_entry.py' and manifest['target_gpu']=='3090'
checks['local_engine_payload_exact']=payload['external_baselines/hgtnet/formal_engine.py']==(ROOT/'external_baselines/hgtnet/formal_engine.py').read_bytes()
derived=dict(epoch1_val_correct_arithmetic=974,epoch1_online_train_correct_arithmetic=2415,
    epoch1_other_wall_seconds=other,epoch1_wall_minutes=e['epoch_wall_seconds']/60,
    reported_remaining_calibration_hours=e['estimated_remaining_phase_seconds']/3600,
    conditional_100_calibration_epochs_hours=100*e['epoch_wall_seconds']/3600,
    epoch2_train_images_per_second=5040/last['train_seconds'],
    estimate_scope='at first epoch event, current calibration phase only; no server timestamp to infer present remaining time; excludes refit/test and is not guaranteed/billed time',
    other_wall_scope='combined dataset/loader setup, checks, checkpoint/history IO and synchronization; not separately profiled')
record=dict(schema='hgtnet-formal-startup-return-v1',state='SERVER_REPORTED_FORMAL_STARTED_EPOCH1_COMPLETE_EPOCH2_TRAIN_COMPLETE',
    source='user-pasted terminal output, archived byte-exact; no direct server access',
    raw_sha256=hashlib.sha256((HERE/'reported_output.txt').read_bytes()).hexdigest(),command=lines[0].strip(),
    device_reported=device,progress_reported=events,derived=derived,checks=checks,checks_count=len(checks),passed=all(checks.values()),
    completed=False,formal_training_started=True,formal_benchmark_result=False,
    not_directly_retrieved=['serverruntime/source/snapshot bytes','GPU compute-process query','current allocated/reserved peak','final summary/run.exit/test predictions'],
    metric_boundary='epoch1 validation/online training only; macroF1 reported, not recomputed from per-image predictions')
(HERE/'validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
(ROOT/'results/hgtnet_nwpu_formal_progress_20261103.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=record['passed'],failed=[k for k,v in checks.items() if not v],derived=derived),indent=2))
if not record['passed']:raise SystemExit(1)
