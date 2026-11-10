"""Read-only production review, with isolated stdlib zipapp boundary probes."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from tools import package_hgtnet_nwpu_smoke as delivery

HERE=Path(__file__).resolve().parent
old=ROOT/'output/hgtnet_nwpu_smoke_3080_20261031.pyz'
record={'scope':'local independent stdlib review; no actual CUDA/training',
        'old_bundle_sha256':hashlib.sha256(old.read_bytes()).hexdigest()}
assert record['old_bundle_sha256']=='e798372589f83359f6d33d3c78d48ea1a82a760310b23f26aa0d6e689835ce78'
with zipfile.ZipFile(old) as archive:
    manifest=json.loads(archive.read('delivery_manifest.json'))
    same=[];changed=[]
    for name in manifest['files']:
        path=ROOT/name
        if path.is_file():
            (same if path.read_bytes()==archive.read(name) else changed).append(name)
    record.update(unchanged_payloads=same,changed_payloads=changed)
assert set(changed)=={'external_baselines/hgtnet/resources.py','tools/run_hgtnet_nwpu_smoke.py'}

bundle=HERE/'independent_target3090_final.pyz'
if not bundle.exists():delivery.build_bundle(bundle,target_gpu='3090')
guard=HERE/'guard';guard.mkdir(exist_ok=True)
(guard/'sitecustomize.py').write_text('''import builtins
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:
        raise AssertionError('ML import forbidden in independent inspect/status: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=checked
''')
binary=HERE/'bin';binary.mkdir(exist_ok=True)
query=binary/'nvidia-smi'
query.write_text('''#!/bin/sh
case "$1" in
  --query-gpu=*) printf 'NVIDIA GeForce RTX 3090, GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1, 24576, 24123, 610.57.04\\n' ;;
  --query-compute-apps=*) ;;
  *) exit 1 ;;
esac
''')
query.chmod(0o755)
env=dict(os.environ,HGTNET_TARGET_GPU='3080',PYTHONPATH=str(guard),
         PATH=str(binary)+os.pathsep+os.environ.get('PATH',''))
for action in ['inspect','status']:
    result=subprocess.run([sys.executable,str(bundle),action],capture_output=True,text=True,env=env)
    assert result.returncode==0,(action,result.stderr)
    value=json.loads(result.stdout)
    (HERE/(action+'_actual_source_bundle.json')).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    if action=='inspect':
        assert value['target_gpu']=='3090'
        assert value['selected_gpu']['uuid']=='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1'
        assert value['selected_gpu']['memory_total_mib']==24576
        assert '3090' in value['scratch_root'] and '3080' not in value['scratch_root']
        assert 'target_gpu' not in value['blockers']
        assert value['formal_training_started'] is False
        record['inspect_fixture_gpu_selected']=value['selected_gpu']
    else:assert value['state']=='not_started' and not value['completed']
assert not (bundle.with_suffix('')/'runs').exists()
record.update(actual_source_zipapp_inspect_status_without_ml=True,automatic_training=False,
              actual_gpu_or_cuda_verified=False)
(HERE/'audit.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
