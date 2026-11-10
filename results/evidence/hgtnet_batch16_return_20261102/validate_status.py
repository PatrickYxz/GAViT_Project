import hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
s=json.loads((HERE/'status_reported.json').read_text());previous=json.loads((HERE/'validation.json').read_text())
checks={}
expected=dict(schema='hgtnet-batch16-capacity-probe-v1',state='complete',completed=True,exit_code=0,
    batch_size=16,counts=dict(train=16,val=0,test=0),formal_training_started=False,
    optimizer_steps=4,expected_processed_images=64,processed_images=64,expected_unique_images=16,
    unique_images=16,warmup_steps=1,measured_steps=3,target_gpu='3090',training_started=True,
    physical_gpu_uuid='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
    config_sha256='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25',
    manifest_sha256='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e',
    weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
for key,value in expected.items():checks[key]=type(s[key]) is type(value) and s[key]==value
checks['timings_match_progress']=s['step_seconds']==previous['derived']['measured_step_seconds']
checks['median_recomputed']=s['median_step_seconds']==statistics.median(s['step_seconds'])
checks['rate_recomputed']=s['compute_images_per_second']==16/s['median_step_seconds']
checks['allocator_peaks_valid']=all(type(s[k]) is int for k in ['cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']) and 0<s['cuda_peak_allocated_bytes']<=s['cuda_peak_reserved_bytes']<previous['device_reported']['total_memory_bytes']
checks['wall_valid']=math.isfinite(s['wall_seconds']) and s['wall_seconds']>sum(s['step_seconds'])+previous['derived']['warmup_seconds']
checks['progress_matches']=s['progress']==previous['progress_reported'][-1]
derived={k.removeprefix('cuda_peak_')+'_gib':s[k]/2**30 for k in ['cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']}
derived.update(wall_seconds=s['wall_seconds'],memory_scope='one-model four-step allocator including Adam/gradients/activations/diagnostics; not total GPU/process memory or a formal epoch')
receipt=dict(checks=checks,checks_count=len(checks),passed=all(checks.values()),derived=derived,
    source='user-returned read-only status; original remote summary/files not directly retrieved',
    status_sha256=hashlib.sha256((HERE/'status_reported.json').read_bytes()).hexdigest())
(HERE/'status_validation.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
record=json.loads((ROOT/'results/hgtnet_batch16_probe_3090_20261102.json').read_text())
record.update(state='SERVER_REPORTED_BATCH16_PASS_STATUS_IDENTITIES_MATCH',status_reported=s,status_derived=derived,pending=[],status_validation='results/evidence/hgtnet_batch16_return_20261102/status_validation.json')
(ROOT/'results/hgtnet_batch16_probe_3090_20261102.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=receipt['passed'],derived=derived),indent=2))
if not receipt['passed']:raise SystemExit(1)
