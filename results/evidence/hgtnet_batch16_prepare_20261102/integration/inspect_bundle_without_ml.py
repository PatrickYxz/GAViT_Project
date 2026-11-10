"""Actual delivered zipapp inspect/status, with all ML imports forbidden."""
import builtins
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[4]
bundle=ROOT/'output/hgtnet_batch16_probe_3090_20261102.pyz'
original=builtins.__import__
def guarded(name,*pos,**kw):
    if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:
        raise AssertionError('ML import during stdlib boundary: '+name)
    return original(name,*pos,**kw)
reports={}
for action in ['inspect','status']:
    stream=io.StringIO();sys.argv=[str(bundle),action];code=None
    builtins.__import__=guarded
    try:
        with contextlib.redirect_stdout(stream):
            try:runpy.run_path(str(bundle),run_name='__main__')
            except SystemExit as exc:code=exc.code
    finally:builtins.__import__=original
    assert code==0,stream.getvalue()
    reports[action]=json.loads(stream.getvalue())
    assert reports[action]['schema']=='hgtnet-batch16-capacity-probe-v1'
loaded=[name for name in ['torch','torchvision','numpy','PIL'] if name in sys.modules]
assert not loaded
assert reports['inspect']['counts']==dict(train=16,val=0,test=0)
assert reports['inspect']['expected_optimizer_steps']==4 and reports['inspect']['target_gpu']=='3090'
assert reports['inspect']['output']==str(bundle.with_suffix('')/'runs/batch16_seed42_01')
assert reports['status']['state']=='not_started' and not reports['status']['completed']
assert not (bundle.with_suffix('')/'runs').exists()
assert not reports['inspect']['ready'] # Local machine lacks target resources, not a GPU validation.
print(json.dumps(dict(scope='actual production pyz stdlib inspect/status; target resources absent locally',
    exit_code=0,ml_modules_loaded=loaded,reports=reports),indent=2,sort_keys=True))
