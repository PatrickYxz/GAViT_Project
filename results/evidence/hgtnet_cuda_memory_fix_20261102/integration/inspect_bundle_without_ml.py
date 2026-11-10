"""Execute the actual production pyz inspect while forbidding all ML imports."""
import argparse
import builtins
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys

parser=argparse.ArgumentParser();parser.add_argument('--bundle',required=True,type=Path);args=parser.parse_args()
bundle=args.bundle.resolve();original=builtins.__import__
def guarded(name,*pos,**kw):
    if name.split('.')[0] in {'torch','torchvision','numpy','PIL'}:raise AssertionError('ML import during stdlib inspection: '+name)
    return original(name,*pos,**kw)
builtins.__import__=guarded
stream=io.StringIO();sys.argv=[str(bundle),'inspect'];exit_code=None
try:
    with contextlib.redirect_stdout(stream):
        try:runpy.run_path(str(bundle),run_name='__main__')
        except SystemExit as exc:exit_code=exc.code
finally:builtins.__import__=original
assert exit_code==0,stream.getvalue()
report=json.loads(stream.getvalue())
loaded=[name for name in ['torch','torchvision','numpy','PIL'] if name in sys.modules]
assert not loaded and not report['ready']
assert not (bundle.with_suffix('')/'runs').exists()
print(json.dumps({'scope':'actual source pyz stdlib-only inspection; missing target resources expected',
                  'exit_code':exit_code,'ml_modules_loaded':loaded,'inspect':report},indent=2))
