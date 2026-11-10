"""Audit current payload identity and protected historical bytes without training."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tools.package_hgtnet_nwpu_smoke import _read_bundle, PROBE_RUNNER, WEIGHTS
from external_baselines.hgtnet.config import load_config, config_sha256

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
checks={}
here=Path(__file__).resolve().parent
path=ROOT/'output/hgtnet_batch16_probe_3090_20261102.pyz'
record=json.loads((here/'final_bundle_build.json').read_text())
manifest,raw,payload=_read_bundle(path)
checks['production_sha']=sha(path)==record['sha256']
checks['production_size']=path.stat().st_size==record['bytes']
checks['production_bundle_id']=manifest['bundle_id']==record['bundle_id']
checks['probe_runner']=manifest['runner']==PROBE_RUNNER
checks['profile3090']=manifest['target_gpu']=='3090'
checks['bootstrap_current']=hashlib.sha256(zipfile.ZipFile(path).read('__main__.py')).hexdigest()==sha(ROOT/'tools/package_hgtnet_nwpu_smoke.py')
for name,data in payload.items():
    if name!='bundled_weights_provenance.json': checks['payload_current:'+name]=(ROOT/name).read_bytes()==data
old=ROOT/'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz'
with zipfile.ZipFile(old) as archive:
    previous=json.loads(archive.read('delivery_manifest.json'))
    for name in previous['files']:
        checks['old_payload_kept:'+name]=archive.read(name)==payload[name]
checks['only_two_payload_additions']=set(payload)-set(previous['files'])=={
    'external_baselines/hgtnet/batch_probe.py','external_baselines/hgtnet/batch_probe_engine.py'}
protected={
 'output/hgtnet_nwpu_smoke_3080_20261031.pyz':'e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78',
 'output/hgtnet_nwpu_smoke_3090_20261102.pyz':'33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff',
 'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz':'90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42'}
for name,expected in protected.items():checks['old_bundle_kept:'+name]=sha(ROOT/name)==expected
audit=json.loads((ROOT/'results/evidence/hgtnet_local_implementation_20261031/preservation/audit.json').read_text())
for group in audit['groups'].values():
    for item in group['records']:checks['historical_kept:'+item['path']]=sha(ROOT/item['path'])==item['expected_sha256']
weights=ROOT/'checkpoints/hgtnet_pretrained_20261031'/WEIGHTS['filename']
checks['candidate_full_sha']=sha(weights)==WEIGHTS['sha256']
checks['candidate_size']=weights.stat().st_size==WEIGHTS['bytes']
checks['config_canonical']=config_sha256(load_config(ROOT/'configs/hgtnet_independent_v0.json'))=='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25'
suite=json.loads((here/'related_suite.json').read_text())
checks['related333_pass']=suite['tests_run']==333 and suite['successful'] and all(suite[k]==0 for k in ['failures','errors','skipped'])
guard=json.loads((here/'actual_bundle_stdlib.json').read_text())
checks['actual_pyz_stdlib_exit0']=guard['exit_code']==0 and guard['ml_modules_loaded']==[]
checks['no_runs_created']=not (path.with_suffix('')/'runs').exists()
history=json.loads((ROOT/'results/evidence/hgtnet_3090_smoke_return_20261102/history_validation.json').read_text())
checks['history_arithmetic_pass']=history['passed']
review=json.loads((here.parent/'review/validation.json').read_text())
receipt=dict(schema='hgtnet-batch16-delivery-audit-v1',checks=checks,count=len(checks),passed=all(checks.values()),
    artifact={k:record[k] for k in ['path','bytes','sha256','bundle_id']},
    source_scope='current production/new-source identity and explicitly listed historical SHA values',
    not_claimed='GPU batch16 memory, GPU timing, formal recipe/training, or formal metrics')
(here/'validation.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(count=len(checks),passed=receipt['passed'],failed=[k for k,v in checks.items() if not v]),indent=2))
if not receipt['passed']:raise SystemExit(1)
