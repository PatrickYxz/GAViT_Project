"""Exercise actual delivered formal zipapp without loading ML or starting a run."""
import builtins,contextlib,io,json,runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
original=builtins.__import__;reports={}
def guarded(name,*a,**kw):
    if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError('Unexpected ML import '+name)
    return original(name,*a,**kw)
for action,tail in [('inspect',[]),('status',[]),('resume',['--help'])]:
    stream=io.StringIO();sys.argv=[str(bundle),action,*tail];code=None;builtins.__import__=guarded
    try:
        with contextlib.redirect_stdout(stream):
            try:runpy.run_path(str(bundle),run_name='__main__')
            except SystemExit as exc:code=exc.code
    finally:builtins.__import__=original
    assert code==0,stream.getvalue()
    reports[action]=stream.getvalue() if action=='resume' else json.loads(stream.getvalue())
assert reports['inspect']['schema']=='hgtnet-nwpu-formal-v1' and reports['inspect']['target_gpu']=='3090'
assert not reports['inspect']['ready'] # Missing target resources on localCPU is expected.
assert reports['status']['state']=='not_started' and not reports['status']['completed']
assert '--gpu-uuid' in reports['resume']
assert not (bundle.with_suffix('')/'runs').exists()
assert not any(k in sys.modules for k in ['torch','torchvision','numpy','PIL'])
print(json.dumps(dict(exit_code=0,ml_modules_loaded=[],reports=reports,
    scope='actual production zipapp inspect/status/resume-help stdlib only; no target GPU or training'),indent=2,sort_keys=True))
