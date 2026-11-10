"""Read-only hardware-compatibility delivery and preservation audit."""
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
old=ROOT/'output/hgtnet_nwpu_smoke_3080_20261031.pyz'
checks['old_3080_bundle_unchanged']=sha(old)=='e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78'
allowed_changes={'external_baselines/hgtnet/resources.py','tools/run_hgtnet_nwpu_smoke.py'}
preserved_payload=[]
with zipfile.ZipFile(old) as archive:
    previous=json.loads(archive.read('delivery_manifest.json'))
    for name,item in previous['files'].items():
        if name in allowed_changes or name=='bundled_weights_provenance.json':continue
        checks['preserved_payload:'+name]=sha(ROOT/name)==item['sha256'];preserved_payload.append(name)
historical=read(ROOT/'results/evidence/hgtnet_local_implementation_20261031/preservation/audit.json')
for group in historical['groups'].values():
    for item in group['records']:checks['historical:'+item['path']]=sha(ROOT/item['path'])==item['expected_sha256']
bundle=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102.pyz'
record=read(str(bundle)+'.manifest.json')
checks['new_bundle_sha']=sha(bundle)==record['sha256']
checks['3090_manifest_target']=record['manifest']['target_gpu']=='3090'
with zipfile.ZipFile(bundle) as archive:
    checks['current_bootstrap']=hashlib.sha256(archive.read('__main__.py')).hexdigest()==sha(ROOT/'tools/package_hgtnet_nwpu_smoke.py')
    for name,item in record['manifest']['files'].items():
        checks['payload:'+name]=hashlib.sha256(archive.read(name)).hexdigest()==item['sha256']
        if name!='bundled_weights_provenance.json':checks['repo_source:'+name]=sha(ROOT/name)==item['sha256']
    checks['pretraining_metadata_unchanged']=archive.read('bundled_weights_provenance.json')==zipfile.ZipFile(old).read('bundled_weights_provenance.json')
suite=read(E/'integration/targeted_suite.json')
checks['hardware_and_delivery_suite']=suite['successful'] and suite['tests_run']==39 and not(suite['errors'] or suite['failures'] or suite['skipped'])
guard=read(E/'integration/bundle_inspect_guard.json')
checks['actual_final_bundle_target3090']=guard['inspect']['target_gpu']=='3090'
checks['actual_final_bundle_no_ml_imports']=guard['exit_code']==0 and not guard['ml_modules_loaded']
checks['status_not_started']=read(E/'integration/actual_bundle_status.json')['state']=='not_started'
checks['no_gpu_run_created']=not (bundle.with_suffix('')/'runs').exists()
weight=ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth'
checks['candidate_weights_unchanged']=sha(weight)=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
report={'scope':'local hardware compatibility only; user NVIDIA inventory does not prove actual CUDA/smoke',
        'passed':all(checks.values()),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),
        'bundle':{k:record[k] for k in ['path','sha256','bytes','bundle_id']},'target_gpu':'3090',
        'preserved_local_payloads':preserved_payload,'historical_artifacts_preserved':52,'suite':suite,
        'full_suite_attempt':read(E/'integration/full_suite_attempt.json'),
        'numeric_regression_limitation':'temporary local Python prefix no longer contains torch/numpy; no current full numerical pass claimed. Model/data/engine bytes preserved.',
        'reported_gpu':read(E/'user_gpu_inventory.json'),'actual_target_cuda_verified':False,'formal_training_started':False}
print(json.dumps(report,indent=2,sort_keys=True));sys.exit(0 if report['passed'] else 1)
