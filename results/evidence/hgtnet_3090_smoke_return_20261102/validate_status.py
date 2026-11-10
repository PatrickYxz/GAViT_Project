"""Audit user-reported status against delivered identities; no GPU/model import."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
status=json.loads((E/'status_reported.json').read_text())
bundle=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz'
checks={
    'complete_state':status['completed'] is True and status['state']=='complete',
    'exit_zero':type(status['exit_code']) is int and status['exit_code']==0,
    'real_training_observed':status['training_started'] is True,
    'not_formal_training':status['formal_training_started'] is False,
    'complete_progress':status['progress']['stage']=='complete' and status['progress']['completed'] is True and status['progress']['training_started'] is True,
    'epoch1_best_and_final':status['best_epoch']==status['final_epoch']==1,
    'expected_360_90_test0':status['counts']=={'train':360,'val':90,'test':0},
    'expected_180_optimizer_steps':status['optimizer_steps']==180,
    'physical_gpu_matches_reported_nvidia':status['target_gpu']=='3090' and status['physical_gpu_uuid']=='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
    'zero_reload_error':status['checkpoint']['reload_max_abs_error']==0.0,
    'checkpoint_full_sha_format':re.fullmatch('[0-9a-f]{64}',status['checkpoint']['sha256']) is not None,
    'cuda_peaks_positive_and_ordered':0<status['cuda_peak_allocated_bytes']<=status['cuda_peak_reserved_bytes']<24*1024**3,
    'wall_positive':math.isfinite(status['wall_seconds']) and status['wall_seconds']>0,
    'expected_output':status['output']=='/home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01',
    'expected_schema':status['schema']=='hgtnet-nwpu-smoke-v2',
    'bundle_sha':hashlib.sha256(bundle.read_bytes()).hexdigest()=='90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42',
}
with zipfile.ZipFile(bundle) as archive:
    config=json.loads(archive.read('configs/hgtnet_independent_v0.json'))
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    checks['canonical_config_identity']=status['config_sha256']==config_sha
    checks['seed42_class45']=config['seed']==42 and config['num_classes']==45
    provenance=json.loads(archive.read('bundled_weights_provenance.json'))
    checks['candidate_weight_identity']=status['weights_sha256']==provenance['sha256']
    tree=ast.parse(archive.read('external_baselines/hgtnet/nwpu_data.py'))
    manifest_sha=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MANIFEST_SHA' for t in node.targets))
    checks['manifest_identity']=status['manifest_sha256']==manifest_sha
    checks['bundle_profile3090']=json.loads(archive.read('delivery_manifest.json'))['target_gpu']=='3090'
    manifest=json.loads(archive.read('delivery_manifest.json'))
    for name,item in manifest['files'].items():
        data=archive.read(name);checks['payload:'+name]=hashlib.sha256(data).hexdigest()==item['sha256']
        if name!='bundled_weights_provenance.json':checks['local_source:'+name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==item['sha256']
accuracy=status['smoke_val']['accuracy_percent'];macro_f1=status['smoke_val']['macro_f1_percent']
correct=round(accuracy*90/100)
checks['smoke_accuracy_integer_consistency']=0<=correct<=90 and math.isclose(100*correct/90,accuracy,rel_tol=0,abs_tol=1e-10)
checks['macro_f1_range_only']=math.isfinite(macro_f1) and 0<=macro_f1<=100
record={'state':'SERVER_REPORTED_REAL_NWPU_3090_SMOKE_PASS_STATUS_IDENTITIES_MATCH',
        'passed':all(checks.values()),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),
        'source':'user-pasted status JSON, normalized locally; no direct server file reads',
        'derived':{'val_correct_arithmetic':correct,'val_errors_arithmetic':90-correct,
                   'allocated_mib':status['cuda_peak_allocated_bytes']/1024**2,
                   'allocated_gib':status['cuda_peak_allocated_bytes']/1024**3,
                   'reserved_mib':status['cuda_peak_reserved_bytes']/1024**2,
                   'reserved_gib':status['cuda_peak_reserved_bytes']/1024**3},
        'not_independently_recomputed':['macro_f1_from_predictions','per_class_metrics','checkpoint_bytes','server_source_snapshot_bytes','complete_runtime_versions'],
        'timing_scope':'whole smoke after stdlib preflight, including imports/recovery/initialization/train/validation/save/reload, not training-only throughput or rental bill',
        'memory_scope':'PyTorch allocator whole smoke including update diagnostics and checkpoint reload, not whole GPU/process memory',
        'formal_benchmark_result':False}
print(json.dumps(record,indent=2,sort_keys=True));raise SystemExit(0 if record['passed'] else 1)
