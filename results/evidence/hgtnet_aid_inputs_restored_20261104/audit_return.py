"""Audit the returned successful restore/300-image inventory; no command execution."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
source=Path('/Users/patrick/.codex/attachments/c8ae8f0e-ac49-44e5-9da5-bd5573546b5c/已粘贴的文本.txt')
raw=source.read_bytes(); text=raw.decode(); start=text.index('{')
p,_=json.JSONDecoder().raw_decode(text[start:]); m=p['manifest']; checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name
def new_json(path,value):
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n')
check('archive_OK_reported','/home/featurize/work/AID_visionflan_20260920.zip: OK' in text[:start])
check('input_assets_verified',p['schema']=='hgtnet-aid-input-inventory-v1' and p['input_assets_verified'] is True and p['status']=='INPUT_ASSETS_VERIFIED_SMOKE_PREPARATION_PENDING')
check('manifest_identity',m['identity_verified'] is True and m['structure_verified'] is True and not m['errors'] and m['status']=='verified' and m['actual_sha256']==m['expected_sha256']=='30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22')
check('original_counts',m['counts']==dict(train=4000,val=1000,pool=5000,test=5000))
check('thirty_classes',len(m['class_names'])==len(set(m['class_names']))==30)
check('selected_counts',m['selected_counts']==dict(train=240,val=60,test=0))
for split,count in [('train',8),('val',2)]:
    paths=m['selected_paths'][split]
    check('selection:'+split,len(paths)==count*30 and len(set(paths))==len(paths) and paths==sorted(paths) and Counter(x.split('/')[0] for x in paths)=={name:count for name in m['class_names']})
previous=json.loads((ROOT/'results/evidence/hgtnet_aid_inputs_return_20261104/inventory_reported.json').read_text())
check('same_selection_as_missing_inventory',m['selected_paths']==previous['manifest']['selected_paths'])
check('all300_hashes_reported_verified',m['image_checks']==dict(checked_count=300,verified_count=300,missing_count=0,invalid_count=0,missing=[],invalid=[]))
check('original_duplicate_groups_retained',m['duplicate_groups']==previous['manifest']['duplicate_groups'])
check('no_test_image_reads',m['test_image_bytes_read']==0 and p['boundaries']['test_images_opened'] is False)
check('weights_identity',p['weights']['identity_verified'] is True and p['weights']['status']=='verified' and p['weights']['actual_sha256']==p['weights']['expected_sha256']=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
check('same3060',p['gpu']==previous['gpu'])
check('metadata_only',p['package_versions']['method']=='importlib.metadata_only_no_ML_imports')
check('no_training',p['training_started'] is False and p['training_entry_available'] is False and p['boundaries']['model_or_CUDA_computation_performed'] is False and p['boundaries']['new_split_created'] is False)
disk={r['role']:r for r in p['disk']}
check('data_root_restored',disk['data_root']['requested_path_exists'] is True and disk['data_root']['measured_path']=='/home/featurize/data/AID/AID')
with (HERE/'reported_output.txt').open('xb') as stream:stream.write(raw)
new_json(HERE/'inventory_reported.json',p)
new_json(HERE/'validation.json',dict(passed=True,count=len(checks),checks=checks,attachment_sha256=hashlib.sha256(raw).hexdigest(),scope='User-returned archive OK and inventory, not direct target access or local image bytes.'))
current=ROOT/'results/hgtnet_aid_inputs_20261104.json';shutil.copyfile(current,HERE/'before_inputs_status.json')
value=json.loads(current.read_text());value.update(status='SERVER_REPORTED_AID_SMOKE_INPUTS_VERIFIED_GPU_SMOKE_PENDING',image_check=dict(checked=300,verified=300,missing=0,invalid=0,default_root_exists=True),evidence=str(HERE.relative_to(ROOT)),checks_passed=len(checks),next_action='Prepare independent30class/240train/60val/1epoch/batch2 smoke for actual RTX3060; no formal training or test.')
value['original_archive_recovery_reference'].update(current_file_existence_verified=True,current_file_sha_verified=True,current_verification_source='User-returned sha256sum OK in chained restore command')
value['disk_free_bytes']['instance_local_ancestor']=disk['data_root']['free_bytes']
current.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps({'passed':True,'checks':len(checks),'verified_images':300,'training_started':False}))
