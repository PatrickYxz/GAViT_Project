"""Execute the exact release bundle inspect/status while blocking ML imports."""
import importlib.abc
from pathlib import Path
import runpy
import sys

BLOCKED={'torch','torchvision','numpy','PIL','timm','scipy','sklearn'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('ML import during stdlib action: '+fullname)
sys.meta_path.insert(0,Guard())
ROOT=Path(__file__).resolve().parents[4]
bundle=ROOT/'output/hgtnet_aid_smoke_3060_20261104.pyz'
sys.argv=[str(bundle),sys.argv[1]]
runpy.run_path(str(bundle),run_name='__main__')
