import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT));HERE=Path(__file__).resolve().parent
from tools.package_hgtnet_nwpu_smoke import _read_bundle,FORMAL_RUNNER,WEIGHTS
from external_baselines.hgtnet.config import load_config,config_sha256
from external_baselines.hgtnet.formal_protocol import load_recipe,recipe_sha256,RECIPE
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
checks={};bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
record=json.loads((HERE/'final_bundle_build.json').read_text());manifest,raw,payload=_read_bundle(bundle)
checks['package_sha']=sha(bundle)==record['sha256'];checks['package_bytes']=bundle.stat().st_size==record['bytes']
checks['package_id']=manifest['bundle_id']==record['bundle_id'];checks['runner']=manifest['runner']==FORMAL_RUNNER
checks['profile']=manifest['target_gpu']=='3090'
with zipfile.ZipFile(bundle) as z:checks['bootstrap_current']=hashlib.sha256(z.read('__main__.py')).hexdigest()==sha(ROOT/'tools/package_hgtnet_nwpu_smoke.py')
for name,value in payload.items():
    if name!='bundled_weights_provenance.json':checks['source_current:'+name]=(ROOT/name).read_bytes()==value
with zipfile.ZipFile(ROOT/'output/hgtnet_batch16_probe_3090_20261102.pyz') as z:
    old=json.loads(z.read('delivery_manifest.json'))
    for name in old['files']:checks['protected_previous_payload:'+name]=z.read(name)==payload[name]
    checks['only_formal_additions']=set(payload)-set(old['files'])=={
        'external_baselines/hgtnet/formal_data.py','external_baselines/hgtnet/formal_protocol.py',
        'external_baselines/hgtnet/formal_checkpoints.py','external_baselines/hgtnet/formal_engine.py',
        'external_baselines/hgtnet/formal_entry.py','configs/hgtnet_nwpu_formal_v1.json'}
protected={
 'output/hgtnet_nwpu_smoke_3080_20261031.pyz':'e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78',
 'output/hgtnet_nwpu_smoke_3090_20261102.pyz':'33e6f0a9ab449b314ba62b6961389db2d7464039c598b7775f7fef881bda19ff',
 'output/hgtnet_nwpu_smoke_3090_20261102_fix1.pyz':'90c60dabcd185ffd0a1b575f8544e7363a469c4e3616528955cb3645c2ae7c42',
 'output/hgtnet_batch16_probe_3090_20261102.pyz':'9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799'}
for name,expected in protected.items():checks['old_package:'+name]=sha(ROOT/name)==expected
baseline=json.loads((ROOT/'results/evidence/hgtnet_local_implementation_20261031/preservation/audit.json').read_text())
for group in baseline['groups'].values():
    for item in group['records']:checks['historical:'+item['path']]=sha(ROOT/item['path'])==item['expected_sha256']
weights=ROOT/'checkpoints/hgtnet_pretrained_20261031'/WEIGHTS['filename']
checks['candidate_complete_sha']=sha(weights)==WEIGHTS['sha256'];checks['candidate_size']=weights.stat().st_size==WEIGHTS['bytes']
checks['config_canonical']=config_sha256(load_config(ROOT/'configs/hgtnet_independent_v0.json'))=='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25'
checks['formal_recipe']=load_recipe(ROOT/'configs/hgtnet_nwpu_formal_v1.json')==RECIPE
suite=json.loads((HERE/'related_suite.json').read_text())
checks['related_tests_pass']=suite['tests_run']>=384 and suite['successful'] and all(suite[k]==0 for k in ['failures','errors','skipped'])
guard=json.loads((HERE/'actual_bundle_stdlib.json').read_text())
checks['actual_package_stdlib_only']=guard['exit_code']==0 and guard['ml_modules_loaded']==[]
checks['no_local_runs']=not (bundle.with_suffix('')/'runs').exists()
fixture=HERE.parent/'full_model_cpu_fixture_03';f=json.loads((fixture/'summary.json').read_text());binding=json.loads((fixture/'binding.json').read_text())
checks['full_model_cpu_complete']=f['completed'] is True and f['input_scope']=='cpu_fixture' and f['formal_training_started'] is False
checks['full_model_parameter_count']=f['parameters']['total']==f['parameters']['trainable']==74619858
checks['full_model_initial_equal']=f['initial_state_sha256']==f['refit_initial_state_sha256']
checks['full_model_reload_zero']=f['checkpoint_reload']['reload_max_abs_error']==0.0
checks['full_model_4_updates']=f['optimizer_steps']==4 and f['selected_epoch']==1 and f['final_epoch']==2
checks['full_model_test_file_sha']=sha(f['test_path'])==f['test_sha256']
checks['full_model_test2']=len(json.loads(Path(f['test_path']).read_text())['predictions'])==2
for name,value in binding['source_identity'].items():checks['fixture_current_source:'+name]=sha(ROOT/name)==value
review=json.loads((HERE.parent/'review/final_validation.json').read_text())
checks['independent_review_pass']=review['passed'] is True
probe=json.loads((ROOT/'results/evidence/hgtnet_batch16_return_20261102/status_validation.json').read_text())
checks['actual_batch16_status_accepted']=probe['passed'] is True
receipt=dict(schema='hgtnet-formal-local-delivery-validation-v1',count=len(checks),passed=all(checks.values()),checks=checks,
    package={k:record[k] for k in ['path','bytes','sha256','bundle_id']},recipe_canonical_sha256=recipe_sha256(RECIPE),
    scope='new local package and CPU fixture verified; real3090 formal training/test not started here')
(HERE/'validation.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(count=len(checks),passed=receipt['passed'],failed=[k for k,v in checks.items() if not v],recipe_sha256=recipe_sha256(RECIPE)),indent=2))
if not receipt['passed']:raise SystemExit(1)
