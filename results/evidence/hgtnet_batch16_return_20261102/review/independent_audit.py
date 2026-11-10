"""Independent stdlib audit: pasted output and delivered source, never server access."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
raw=(HERE.parent/'reported_output.txt').read_text()
lines=raw.splitlines()
device=json.loads(next(line.split(' ',1)[1] for line in lines if line.startswith('HGTNET_CUDA_DEVICE_PROBE ')))
progress=[json.loads(line.split(' ',1)[1]) for line in lines if line.startswith('HGTNET_BATCH16_PROBE_PROGRESS ')]
steps=[row for row in progress if 'phase' in row]
measured=[row['elapsed_seconds'] for row in steps if row['phase']=='measured']
first=steps[0];groups=first['gradients']['groups'];median=statistics.median(measured)
bundle=ROOT/'output/hgtnet_batch16_probe_3090_20261102.pyz'
with zipfile.ZipFile(bundle) as archive:
    wrapper=archive.read('external_baselines/hgtnet/batch_probe.py').decode()
    engine=archive.read('external_baselines/hgtnet/batch_probe_engine.py').decode()
    manifest=json.loads(archive.read('delivery_manifest.json'))
checks={
 'bundle_sha':hashlib.sha256(bundle.read_bytes()).hexdigest()=='9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799',
 'runner':manifest['runner']=='external_baselines/hgtnet/batch_probe.py',
 'target3090':manifest['target_gpu']=='3090',
 'cuda_available':device['available'] is True,
 'one_cuda_device':device['device_count']==1,
 'cuda_name':device['name']=='NVIDIA GeForce RTX 3090',
 'torch_and_cuda':device['torch']=='2.2.2+cu121' and device['cuda']=='12.1',
 'actual_cuda_memory_bytes':device['total_memory_bytes']==25295323136,
 'exact_four_steps':len(steps)==4 and [row['optimizer_steps'] for row in steps]==[1,2,3,4],
 'one_warmup_three_measured':[row['phase'] for row in steps]==['warmup','measured','measured','measured'],
 'batch16_all':all(row['batch_size']==16 for row in steps),
 'processed64':[row['samples_seen'] for row in steps]==[16,32,48,64],
 'true_update_receipts':all(row['optimizer_step_completed'] and row['optimizer_step_succeeded'] and row['post_update_checks_passed'] for row in steps),
 'no_formal_training':all(row['formal_training_started'] is False for row in steps),
 'positive_finite_times_and_losses':all(math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0 and math.isfinite(row['loss']) for row in steps),
 'five_groups':set(groups)=={'dual_attention','graph','head','patch_embed','transformer'},
 '451_gradient_tensors':sum(group['tensors'] for group in groups.values())==451,
 'finite_positive_group_norms':all(math.isfinite(group['l2']) and group['l2']>0 for group in groups.values()),
 'gradient_report_passed':first['gradients']['passed'] is True and first['gradients']['missing']==[] and first['gradients']['nonfinite']==[],
 'all_five_group_updates':set(first['updated_groups'])==set(groups) and all(first['updated_groups'].values()),
 'completion_progress':progress[-1]['completed'] is True and progress[-1]['optimizer_steps']==4,
 'completion_marker':lines[-1]=='HGTNET_BATCH16_PROBE_COMPLETE /home/featurize/work/hgtnet_batch16_probe_3090_20261102/runs/batch16_seed42_01/summary.json',
 'source_summary_before_completion':wrapper.index("COMMON._write(args.output/'summary.json',result)")<wrapper.index("print('HGTNET_BATCH16_PROBE_COMPLETE '"),
 'source_exit0_before_completion':wrapper.index("stream.write('0\\n')")<wrapper.index("print('HGTNET_BATCH16_PROBE_COMPLETE '"),
 'source_identity_and_data_rechecked':"COMMON._source_identity()!=sources" in wrapper and 'DATA.validate_selected(data_root,records)' in wrapper,
 'source_exact4_and_finalfinite':"result.get('optimizer_steps') != 4" in wrapper and "if not _finite_parameters(model): raise ValueError('Final probe parameters nonfinite')" in engine,
}
derived=dict(measured_step_seconds=measured,median_step_seconds=median,mean_step_seconds=statistics.mean(measured),
 compute_images_per_second=16/median,calibration_steps_per_epoch=math.ceil(5040/16),refit_steps_per_epoch=math.ceil(6300/16),
 max_calibration_refit_steps=100*(math.ceil(5040/16)+math.ceil(6300/16)),
 conditional_calibration_train_compute_seconds=315*median,conditional_refit_train_compute_seconds=394*median,
 conditional_max_100plus100_train_compute_seconds=70900*median,
 conditional_max_100plus100_train_compute_hours=70900*median/3600,
 actual_cuda_memory_mib=device['total_memory_bytes']/2**20,
 actual_cuda_memory_gib=device['total_memory_bytes']/2**30)
result=dict(schema='hgtnet-batch16-return-independent-audit-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),derived=derived,
 accepted='server-reported batch16 completed marker and four actual optimizer success receipts; local source identity matches delivered package',
 timing_scope='three compute-only steps on already device-resident repeated16 inputs; extrapolations conditional on unchanged rate, no formal ETA',
 not_confirmed=['server original summary/status/run.exit bytes','peak allocated/reserved memory','wall time','current physical NVML inventory','formal training accuracy or budget'],
 disposition='no engineering anomaly found in provided output; no smoke/probe rerun justified; read existing status for remaining fields')
(HERE/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks_count=len(checks),passed=result['passed'],derived=derived),indent=2))
if not result['passed']:raise SystemExit(1)
