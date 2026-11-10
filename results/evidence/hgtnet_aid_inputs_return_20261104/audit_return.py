"""Archive and audit the user's input inventory; never execute pasted commands."""
from collections import Counter
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path('/Users/patrick/.codex/attachments/777b33ca-345a-40eb-a88f-6371f4258201/已粘贴的文本.txt')
raw = SOURCE.read_bytes()
text = raw.decode()
packet, end = json.JSONDecoder().raw_decode(text[text.index('{'):])
checks = {}


def check(name, ok):
    checks[name] = bool(ok)
    assert ok, name


def write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


m, w = packet['manifest'], packet['weights']
expected_m = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
expected_w = '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
check('schema', packet['schema'] == 'hgtnet-aid-input-inventory-v1')
check('reported_incomplete', packet['status'] == 'INPUT_CHECK_INCOMPLETE' and packet['input_assets_verified'] is False)
check('reported_manifest_identity', m['identity_verified'] is True and m['structure_verified'] is True and not m['errors'] and
      m['actual_sha256'] == m['expected_sha256'] == expected_m and m['size_bytes'] == 2710270)
check('counts', m['counts'] == dict(train=4000, val=1000, pool=5000, test=5000))
check('thirty_classes', len(m['class_names']) == len(set(m['class_names'])) == 30 and m['class_names'] == sorted(m['class_names']))
per = m['per_class_counts']
check('perclass_names', [r['name'] for r in per] == m['class_names'] and [r['label'] for r in per] == list(range(30)))
for split, count in [('train', 4000), ('val', 1000), ('test', 5000)]:
    check('perclass_total:' + split, sum(r[split] for r in per) == count)
check('perclass_original_ratios', all(r['test'] == r['train'] + r['val'] and r['val'] * 4 == r['train'] for r in per))
selected = m['selected_paths']['train'] + m['selected_paths']['val']
check('selected_240_60', m['selected_counts'] == dict(train=240, val=60, test=0) and
      len(m['selected_paths']['train']) == 240 and len(m['selected_paths']['val']) == 60 and len(set(selected)) == 300)
for split, count in [('train', 8), ('val', 2)]:
    check('selected_each_class:' + split, Counter(p.split('/')[0] for p in m['selected_paths'][split]) ==
          {name: count for name in m['class_names']})
im = m['image_checks']
check('all300_missing_not_invalid', m['status'] == 'missing_images' and im['missing_count'] == len(im['missing']) == 300 and
      set(im['missing']) == set(selected) and im['checked_count'] == im['verified_count'] == im['invalid_count'] == 0 and im['invalid'] == [])
check('no_test_image_reads', m['test_image_bytes_read'] == 0 and packet['boundaries']['test_images_opened'] is False)
dups = m['duplicate_groups']
check('original_duplicate_retained', dups['extra_copies'] == 1 and dups['unique_images'] == 9999 and
      len(dups['groups']) == 1 and dups['groups'][0]['paths'] == ['Industrial/industrial_203.jpg', 'Industrial/industrial_205.jpg'] and
      dups['groups'][0]['split'] == 'train' and dups['groups'][0]['label'] == 12)
check('reported_initial_weights_identity', w['identity_verified'] is True and w['status'] == 'verified' and
      w['size_bytes'] == 351654207 and w['actual_sha256'] == w['expected_sha256'] == expected_w)
gpu = packet['gpu']['devices']
check('reported_gpu_inventory', packet['gpu']['status'] == 'inventoried' and packet['gpu']['returncode'] == 0 and len(gpu) == 1 and
      gpu[0]['name'] == 'NVIDIA GeForce RTX 3060' and gpu[0]['memory_total_mib'] == 12288 and
      gpu[0]['uuid'] == 'GPU-7acce72a-93cd-e56f-35d8-5d33ecf6655e')
disk = {row['role']: row for row in packet['disk']}
check('data_root_absent', disk['data_root']['requested_path'] == '/home/featurize/data/AID/AID' and
      disk['data_root']['requested_path_exists'] is False)
check('disk_capacity_values', disk['manifest']['free_bytes'] == 9767485440 and disk['data_root']['free_bytes'] == 349589377024)
check('not_training', packet['training_started'] is False and packet['training_entry_available'] is False and
      packet['boundaries']['model_or_CUDA_computation_performed'] is False and packet['boundaries']['new_split_created'] is False)
check('versions_are_metadata_only', packet['package_versions']['method'] == 'importlib.metadata_only_no_ML_imports' and
      packet['package_versions']['python'] == '3.11.8')
check('delivered_package_unchanged', hashlib.sha256((ROOT/'output/hgtnet_aid_inputs_20261104.pyz').read_bytes()).hexdigest() ==
      '1cd2ab967cdbc386b294b397c6804bf06c5796c01332c709e44646f98d03f036')
with (HERE/'reported_output.txt').open('xb') as stream:
    stream.write(raw)
write_json(HERE/'inventory_reported.json', packet)
write_json(HERE/'validation.json', {'passed': True, 'count': len(checks), 'checks': checks,
                                  'attachment_sha256': hashlib.sha256(raw).hexdigest(),
                                  'scope': 'Audit of user-returned inspector JSON and local delivery identity, not direct server access or original data bytes.'})
archive = {'path': '/home/featurize/work/AID_visionflan_20260920.zip', 'bytes': 2628474154,
           'expected_sha256': 'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7',
           'historical_user_reported_sha_verified': True, 'current_file_existence_verified': False,
           'current_file_sha_verified': False, 'source': '.worktrees/sparse-hybrid-4n-top2/docs/download_aid.md',
           'historical_confirmation': 'research_diary.md entry 2026-09-20 AID压缩包哈希确认'}
result = {'date': '2026-11-04', 'status': 'SERVER_REPORTED_AID_MANIFEST_WEIGHTS_PASS_IMAGES_MISSING',
          'source_type': 'user_returned_input_inventory', 'counts': m['counts'], 'selected_counts': m['selected_counts'],
          'manifest_sha256_reported_verified': expected_m, 'initial_weights_sha256_reported_verified': expected_w,
          'image_check': {'checked': 0, 'verified': 0, 'missing': 300, 'invalid': 0, 'default_root_exists': False},
          'gpu_inventory': gpu[0], 'runtime_CUDA_verified': False,
          'disk_free_bytes': {'work': disk['manifest']['free_bytes'], 'instance_local_ancestor': disk['data_root']['free_bytes']},
          'original_archive_recovery_reference': archive,
          'next_action': 'If the old ZIP is present and current full SHA matches, restore original images to instance-local data and rerun the same inspector; retain original manifest.',
          'scope': 'Manifest/initial weight verification is reported by the server inspector; original byte files not downloaded locally. GPU information is nvidia-smi inventory, not Torch/CUDA/model verification.',
          'training_started': False, 'formal_training_started': False, 'new_test_metrics': None,
          'evidence': str(HERE.relative_to(ROOT)), 'checks_passed': len(checks)}
write_json(ROOT/'results/hgtnet_aid_inputs_20261104.json', result)
print(json.dumps({'passed': True, 'checks': len(checks), 'status': result['status'],
                  'missing': 300, 'gpu': gpu[0]['name'], 'archive_current_availability': 'unknown'}))
