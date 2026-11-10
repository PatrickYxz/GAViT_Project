"""Execute the exact release bundle inspect/status while blocking ML imports."""
import builtins
from pathlib import Path
import runpy
import sys

BLOCKED={'torch','torchvision','numpy','PIL','timm','scipy','sklearn'}
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
    if name.split('.')[0] in BLOCKED:
        raise AssertionError('ML import during stdlib action: '+name)
    return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
ROOT=Path(__file__).resolve().parents[4]
bundle=ROOT/'output/hgtnet_aid_smoke_3060_20261104.pyz'
sys.argv=[str(bundle),sys.argv[1]]
try:
    runpy.run_path(str(bundle),run_name='__main__')
finally:
    assert not any(name.split('.')[0] in BLOCKED for name in sys.modules)
