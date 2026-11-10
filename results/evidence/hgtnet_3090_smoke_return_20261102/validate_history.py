"""Arithmetic audit of user-pasted history, not remote file retrieval."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
history=json.loads((HERE/'history_reported.json').read_text())
status=json.loads((HERE/'status_reported.json').read_text())
checks={}
checks['one_epoch_list']=isinstance(history,list) and len(history)==1
h=history[0]
checks['epoch1']=type(h['epoch']) is int and h['epoch']==status['best_epoch']==status['final_epoch']==1
checks['steps180']=type(h['optimizer_steps']) is int and h['optimizer_steps']==status['optimizer_steps']==180
checks['batch2_train360']=2*h['optimizer_steps']==status['counts']['train']==360
checks['finite_positive_losses_times_throughput']=all(type(h[key]) in (int,float) and math.isfinite(h[key]) and h[key]>0
    for key in ('train_loss','val_loss','train_seconds','validation_seconds','train_images_per_second'))
rate=status['counts']['train']/h['train_seconds']
checks['training_rate_matches_count_and_seconds']=math.isclose(rate,h['train_images_per_second'],rel_tol=0,abs_tol=1e-10)
train_correct=h['train_accuracy_percent']*360/100
checks['train_accuracy_valid_arithmetic']=0<=h['train_accuracy_percent']<=100 and math.isclose(train_correct,round(train_correct),abs_tol=1e-10)
other=status['wall_seconds']-h['train_seconds']-h['validation_seconds']
checks['phase_times_within_whole_smoke_wall']=0<other<status['wall_seconds']
derived=dict(train_correct_streaming_arithmetic=round(train_correct),train_images_per_second_recomputed=rate,
    train_plus_validation_seconds=h['train_seconds']+h['validation_seconds'],
    other_smoke_phase_seconds=other,validation_images_per_second_arithmetic=90/h['validation_seconds'],
    same_batch2_no_augmentation_conditional_train5040_seconds=5040/rate,
    same_batch2_no_augmentation_conditional_refit6300_seconds=6300/rate,
    condition='constant observed batch2 rate only, no startup/val/checkpoint/test/IO changes; not batch16 or formal ETA')
receipt=dict(source='user-pasted cat history output; no original file/checkpoint/prediction bytes retrieved',
    history_sha256=hashlib.sha256((HERE/'history_reported.json').read_bytes()).hexdigest(),
    checks=checks,passed=all(checks.values()),checks_count=len(checks),derived=derived,
    metric_scope='training accuracy is accumulated pre-update across evolving weights; val is epoch-end model on distinct samples; gap does not identify its cause')
(HERE/'history_validation.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
if not receipt['passed']:raise SystemExit('history audit failed')
result_path=ROOT/'results/hgtnet_nwpu_smoke_3090_20261102.json'
result=json.loads(result_path.read_text())
result.update(history_reported=history,history_derived=derived,history_evidence=str((HERE/'history_validation.json').relative_to(ROOT)))
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2,sort_keys=True))
