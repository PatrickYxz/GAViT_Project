"""Check user-reported bounded GPU progress and derive scoped timing arithmetic."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
raw=(HERE/'reported_output.txt').read_text()
lines=raw.splitlines()
device=json.loads(next(v.removeprefix('HGTNET_CUDA_DEVICE_PROBE ') for v in lines if v.startswith('HGTNET_CUDA_DEVICE_PROBE ')))
progress=[json.loads(v.removeprefix('HGTNET_BATCH16_PROBE_PROGRESS ')) for v in lines if v.startswith('HGTNET_BATCH16_PROBE_PROGRESS ')]
steps=progress[:-1];first=steps[0]
durations=[v['elapsed_seconds'] for v in steps if v['phase']=='measured']
checks={}
checks['exact4_update_events']=len(steps)==4
checks['steps_sequential']=[v['optimizer_steps'] for v in steps]==[1,2,3,4]
checks['samples16_to64']=[v['samples_seen'] for v in steps]==[16,32,48,64]
checks['warmup_then3_measured']=[v['phase'] for v in steps]==['warmup','measured','measured','measured']
checks['fixed_batch16']=all(type(v['batch_size']) is int and v['batch_size']==16 for v in steps)
checks['all_success_receipts']=all(v['optimizer_step_completed'] is True and v['optimizer_step_succeeded'] is True and v['post_update_checks_passed'] is True for v in steps)
checks['disposable_only']=all(v['formal_training_started'] is False for v in steps)
checks['finite_positive_loss_time']=all(math.isfinite(v[k]) and v[k]>0 for v in steps for k in ['loss','elapsed_seconds'])
checks['same_pid5457']=all(v['pid']==5457 for v in progress)
checks['completion_progress']=progress[-1]['completed'] is True and progress[-1]['stage']=='complete' and progress[-1]['optimizer_steps']==4
expected_path='/home/featurize/work/hgtnet_batch16_probe_3090_20261102/runs/batch16_seed42_01/summary.json'
checks['completion_marker_path']=lines[-1]=='HGTNET_BATCH16_PROBE_COMPLETE '+expected_path
checks['runtime_3090']=device['available'] is True and device['device_count']==1 and device['name']=='NVIDIA GeForce RTX 3090'
checks['actual_torch_cuda_report']=device['torch']=='2.2.2+cu121' and device['cuda']=='12.1'
checks['actual_cuda_bytes_integer']=type(device['total_memory_bytes']) is int and device['total_memory_bytes']==25295323136
checks['cuda_nominal_allowance_prior_physical']=24576*2**20-2**30<=device['total_memory_bytes']<=24576*2**20
groups={'dual_attention':30,'graph':120,'head':2,'patch_embed':4,'transformer':295}
checks['exact5_gradient_groups']=set(first['gradients']['groups'])==set(groups)
checks['451_trainable_parameter_tensors']=sum(v['tensors'] for v in first['gradients']['groups'].values())==451
checks['group_tensor_counts']=all(first['gradients']['groups'][k]['tensors']==n for k,n in groups.items())
checks['gradient_pass']=first['gradients']['passed'] is True and first['gradients']['missing']==[] and first['gradients']['nonfinite']==[]
checks['group_norms_finite_positive']=all(math.isfinite(v['l2']) and v['l2']>0 for v in first['gradients']['groups'].values())
checks['all5_groups_updated']=first['updated_groups']==dict.fromkeys(groups,True)
bundle=ROOT/'output/hgtnet_batch16_probe_3090_20261102.pyz'
checks['delivered_local_bundle_sha']=hashlib.sha256(bundle.read_bytes()).hexdigest()=='9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799'
with zipfile.ZipFile(bundle) as archive:
    manifest=json.loads(archive.read('delivery_manifest.json'))
    source=archive.read('external_baselines/hgtnet/batch_probe.py').decode()
    engine=archive.read('external_baselines/hgtnet/batch_probe_engine.py').decode()
    checks['entry_target_fixed']=manifest['runner']=='external_baselines/hgtnet/batch_probe.py' and manifest['target_gpu']=='3090'
    checks['local_current_entry_exact']=source==(ROOT/'external_baselines/hgtnet/batch_probe.py').read_text()
    checks['local_current_engine_exact']=engine==(ROOT/'external_baselines/hgtnet/batch_probe_engine.py').read_text()
    function=next(v for v in ast.parse(source).body if isinstance(v,ast.FunctionDef) and v.name=='run_probe')
    text=ast.get_source_segment(source,function)
    checks['source_complete_after_summary_exit_and_marker']=text.index("COMMON._write(args.output/'summary.json',result)")<text.index("(args.output/'run.exit').open('x')")<text.index("(args.output/'completion.marker').open('x')")<text.index("print('HGTNET_BATCH16_PROBE_COMPLETE '")
    checks['source_exact4_completion_required']="result.get('optimizer_steps') != 4" in text
    checks['source_end_input_recheck']='DATA.validate_selected(data_root,records)' in text and "COMMON._source_identity()!=sources" in text and 'COMMON._sha(args.weights)!=COMMON.WEIGHTS_SHA' in text
    checks['compute_timing_scoped']='compute_timing_scope=COMPUTE_SCOPE' in engine and 'excludes image IO' in engine
median=statistics.median(durations)
derived=dict(measured_step_seconds=durations,median_step_seconds=median,mean_step_seconds=statistics.mean(durations),
    median_compute_images_per_second=16/median,measured_aggregate_compute_images_per_second=48/sum(durations),
    warmup_seconds=first['elapsed_seconds'],cuda_exposed_mib=device['total_memory_bytes']/2**20,
    cuda_exposed_gib=device['total_memory_bytes']/2**30,
    conditional_compute_calibration_epoch315_seconds=315*median,
    conditional_compute_refit_epoch394_seconds=394*median,
    conditional_compute_max100_calibration_plus100_refit_hours=70900*median/3600,
    estimate_scope='only conditional update compute scaled from three reused-input steps; not end-to-end epoch/runtime/billing; excludes IO, augmentations, validation, saving, refit startup and final test; actual full pipeline unmeasured')
record=dict(schema='hgtnet-batch16-server-reported-marker-v1',
    state='SERVER_REPORTED_BATCH16_4_UPDATES_COMPLETE_SUMMARY_PENDING',
    source='user-pasted run output; no direct server files retrieved',
    reported_output_sha256=hashlib.sha256((HERE/'reported_output.txt').read_bytes()).hexdigest(),
    command=lines[0],device_reported=device,progress_reported=progress,derived=derived,
    checks=checks,checks_count=len(checks),passed=all(checks.values()),
    capacity_scope='this fixed16 FP32 full-model short probe completed; no long-run memory guarantee or method-performance result',
    pending=['summary/status peak allocated/reserved bytes','whole probe wall_seconds','summary config/manifest/weights identity','reported run.exit value'],
    no_rerun_needed=True,formal_benchmark_result=False)
(HERE/'validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
result_path=ROOT/'results/hgtnet_batch16_probe_3090_20261102.json'
result_path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=record['passed'],failed=[k for k,v in checks.items() if not v],derived=derived),indent=2))
if not record['passed']:raise SystemExit(1)
