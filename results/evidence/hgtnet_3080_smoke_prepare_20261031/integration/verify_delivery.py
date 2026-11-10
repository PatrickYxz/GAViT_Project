"""Stdlib audit of actual delivery, local receipts and protected prior artifacts."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[4]
E=Path(__file__).resolve().parents[1]
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()
def read(path):return json.loads(Path(path).read_text())
checks={}
old=ROOT/'results/evidence/hgtnet_local_implementation_20261031'
for group in read(old/'preservation/audit.json')['groups'].values():
    for record in group['records']:checks['old:'+record['path']]=sha(ROOT/record['path'])==record['expected_sha256']
for path,expected in read(old/'integration/validation.json')['package_python_sha256'].items():
    checks['old_core:'+path]=sha(ROOT/path)==expected
checks['old_config']=sha(ROOT/'configs/hgtnet_independent_v0.json')==read(old/'integration/validation.json')['supporting_files_sha256']['configs/hgtnet_independent_v0.json']
bundle=ROOT/'output/hgtnet_nwpu_smoke_3080_20261031.pyz'
record=read(str(bundle)+'.manifest.json')
checks['bundle_sha']=sha(bundle)==record['sha256']
checks['bundle_size']=bundle.stat().st_size==record['bytes']
with zipfile.ZipFile(bundle) as archive:
    for name,item in record['manifest']['files'].items():
        data=archive.read(name)
        checks['payload:'+name]=hashlib.sha256(data).hexdigest()==item['sha256'] and len(data)==item['bytes']
        if name!='bundled_weights_provenance.json':checks['repo_payload:'+name]=sha(ROOT/name)==item['sha256']
    checks['no_large_weight_in_payload']=not record['manifest']['weight_file_included'] and not any(n.endswith(('.pth','.pt')) for n in archive.namelist())
fixture=read(E/'full_model_cpu_fixture_02/summary.json')
checks['full_cpu_fixture_only']=fixture['completed'] and fixture['input_scope']=='cpu_fixture' and fixture['counts']=={'train':4,'val':2,'test':0}
checks['full_model_parameters']=fixture['parameters']['total']==74619858 and fixture['candidate_actual_loaded_tensors']==318
checks['actual_training_reload']=fixture['optimizer_steps']==2 and fixture['checkpoint']['reload_max_abs_error']==0.0
checks['all_gradient_groups']=fixture['first_batch_gradients']['passed'] and len(fixture['first_batch_gradients']['groups'])==5
checks['fixture_checkpoint_sha']=sha(E/'full_model_cpu_fixture_02/smoke_epoch1.pt')==fixture['checkpoint']['sha256']
for path,expected in fixture['source_identity'].items():checks['fixture_source:'+path]=sha(ROOT/path)==expected
suite=read(E/'integration/related_suite.json')
checks['related_suite']=suite['successful'] and suite['tests_run']==285 and not(suite['failures'] or suite['errors'] or suite['skipped'])
guard=read(E/'integration/bundle_inspect_guard.json')
checks['actual_pyz_no_ml']=guard['exit_code']==0 and not guard['ml_modules_loaded'] and guard['inspect']['ready'] is False
checks['bundle_did_not_start_run']=not (bundle.with_suffix('')/'runs').exists()
weights=ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth'
checks['candidate_file_sha']=sha(weights)=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
report={'scope':'local production delivery/CPU fixture audit only; actual target GPU and real NWPU smoke pending',
        'passed':all(checks.values()),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),
        'historical_paths_preserved':52,'prior_hgtnet_python_files_preserved':9,
        'bundle':{k:record[k] for k in ['path','sha256','bytes','bundle_id']},
        'candidate_weights':{'path':str(weights),'sha256':sha(weights),'bytes':weights.stat().st_size},
        'suite':suite,'formal_training_started':False,'target_gpu_verified':False,
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'external_baselines/hgtnet').glob('*.py'))},
        'builder_sha256':sha(ROOT/'tools/package_hgtnet_nwpu_smoke.py'),
        'runner_sha256':sha(ROOT/'tools/run_hgtnet_nwpu_smoke.py')}
print(json.dumps(report,indent=2,sort_keys=True));sys.exit(0 if report['passed'] else 1)
