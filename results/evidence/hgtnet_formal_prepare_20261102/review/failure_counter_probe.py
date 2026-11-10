"""Count actual tiny Adam updates before a deliberately invalid validation order."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from external_baselines.hgtnet import formal_protocol as p,formal_engine as engine
HERE=Path(__file__).resolve().parent
label=sys.argv[1] if len(sys.argv)>1 else 'pre'
r=copy.deepcopy(p.RECIPE);r.update(max_epochs=2,batch_size=2,eval_batch_size=2)
b=dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=r,recipe_sha256=p.recipe_sha256(r),
    class_names=['c%02d'%i for i in range(45)],counts=dict(train=4,val=2,pool=6,test=2),
    records={k:[dict(path=k+str(i),label=i) for i in range(2)] for k in ['val','test']},
    manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
class Tiny(nn.Module):
    def __init__(self):super().__init__();self.head=nn.Linear(3,45)
    def forward(self,x):return self.head(x.mean((2,3)))
events=[];actual_steps=0;original=torch.optim.Adam.step
def counted(self,*args,**kwargs):
    global actual_steps
    result=original(self,*args,**kwargs);actual_steps+=1;return result
def loaders(split,epoch):
    count=b['counts'][split];labels=torch.arange(count)%2
    if split=='val':labels=1-labels
    images=torch.randn(count,3,224,224,generator=torch.Generator().manual_seed(1))
    return DataLoader(TensorDataset(images,labels),batch_size=2,shuffle=split in ('train','pool'),generator=torch.Generator().manual_seed(42+epoch))
error=None
with tempfile.TemporaryDirectory() as d,patch.object(torch.optim.Adam,'step',counted):
    try:engine.run_formal(Tiny,loaders,Path(d).resolve(),binding=b,device='cpu',on_progress=events.append)
    except ValueError as exc:error=str(exc)
observed=max([v['optimizer_steps'] for v in events]+[0])
record=dict(error=error,actual_adam_steps=actual_steps,last_observed_optimizer_steps=observed,
    all_completed_steps_observable=actual_steps==observed,events=events,
    engine_sha256=hashlib.sha256((ROOT/'external_baselines/hgtnet/formal_engine.py').read_bytes()).hexdigest(),
    scope='small CPU injected validation failure; no GPU/NWPU result')
(HERE/('failure_counter_'+label+'_validation.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True));assert actual_steps==2 and error and 'order' in error
