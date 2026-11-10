"""Audit the one-source CUDA memory-gate fix and unchanged numeric pipeline."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[4];E=Path(__file__).resolve().parents[1]
def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()
def read(path):return json.loads(Path(path).read_text())
checks={};old=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102.pyz'
checks['original3090_bundle_preserved']=sha(old)=='33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff'
preserved=[]
with zipfile.ZipFile(old) as archive:
    manifest=json.loads(archive.read('delivery_manifest.json'))
    for name,item in manifest['files'].items():
        if name in ('tools/run_hgtnet_nwpu_smoke.py','bundled_weights_provenance.json'):continue
        checks['unchanged:'+name]=sha(ROOT/name)==item['sha256'];preserved.append(name)
    checks['builder_unchanged']=sha(ROOT/'tools/package_hgtnet_nwpu_smoke.py')==manifest['bootstrap_sha256']
bundle=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz';record=read(str(bundle)+'.manifest.json')
checks['fix1_bytes']=sha(bundle)==record['sha256'] and bundle.stat().st_size==record['bytes']
checks['profile3090']=record['manifest']['target_gpu']=='3090'
with zipfile.ZipFile(bundle) as archive:
    for name,item in record['manifest']['files'].items():
        checks['payload:'+name]=hashlib.sha256(archive.read(name)).hexdigest()==item['sha256']
        if name!='bundled_weights_provenance.json':checks['current_source:'+name]=sha(ROOT/name)==item['sha256']
    with zipfile.ZipFile(old) as previous:
        checks['weight_provenance_unchanged']=archive.read('bundled_weights_provenance.json')==previous.read('bundled_weights_provenance.json')
suite=read(E/'integration/targeted_suite.json')
checks['45_target_tests']=suite['successful'] and suite['tests_run']==45 and not(suite['errors'] or suite['failures'] or suite['skipped'])
guard=read(E/'integration/bundle_inspect_guard.json')
checks['actual_pyz_stdlib_only']=guard['exit_code']==0 and not guard['ml_modules_loaded'] and guard['inspect']['target_gpu']=='3090'
checks['new_scratch_namespace']='fix1' in guard['inspect']['scratch_root']
checks['new_output_namespace']='fix1' in guard['inspect']['output']
checks['not_started']=read(E/'integration/actual_bundle_status.json')['state']=='not_started'
checks['no_run_directory']=not (bundle.with_suffix('')/'runs').exists()
historical=read(ROOT/'results/evidence/hgtnet_local_implementation_20261031/preservation/audit.json')
for group in historical['groups'].values():
    for item in group['records']:checks['historical:'+item['path']]=sha(ROOT/item['path'])==item['expected_sha256']
checks['candidate_weights_unchanged']=sha(ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth')=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
report={'scope':'local stdlib gate/package fix; CUDA boundary fixtures are not real training evidence',
        'passed':all(checks.values()),'checks':checks,'passed_checks':sum(checks.values()),'total_checks':len(checks),
        'bundle':{k:record[k] for k in ['path','sha256','bytes','bundle_id']},'unchanged_local_payloads':preserved,
        'suite':suite,'actual_target_smoke_verified':False,'formal_training_started':False,
        'numerical_suite_limitation':'previous local temporary interpreter lacks Torch/NumPy; no repeated full numeric pass claimed, unchanged numeric source bytes verified'}
print(json.dumps(report,indent=2,sort_keys=True));sys.exit(0 if report['passed'] else 1)
