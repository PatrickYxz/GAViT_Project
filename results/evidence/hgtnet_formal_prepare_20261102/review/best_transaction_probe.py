"""Actual CPU validation improvement plus interruption before epoch commit."""
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
from torch.utils.data import DataLoader,TensorDataset
from external_baselines.hgtnet import formal_protocol as p,formal_engine as e,formal_checkpoints as c
HERE=Path(__file__).resolve().parent
r=copy.deepcopy(p.RECIPE);r.update(max_epochs=2,batch_size=2,eval_batch_size=2)
b=dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=r,recipe_sha256=p.recipe_sha256(r),
    class_names=['c%02d'%i for i in range(45)],counts=dict(train=2,val=2,pool=4,test=2),
    records={k:[dict(path=k+str(i),label=0) for i in range(2)] for k in ['val','test']},
    manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Linear(3,45)
        with torch.no_grad():
            self.head.weight.zero_();self.head.bias.fill_(-10);self.head.bias[0]=0;self.head.bias[1]=0.00004
    def forward(self,x):return self.head(x.mean((2,3)))
def loaders(split,epoch):
    count=b['counts'][split]
    ds=TensorDataset(torch.full((count,3,224,224),0.2),torch.zeros(count,dtype=torch.long))
    return DataLoader(ds,batch_size=2,shuffle=split in ('train','pool'),generator=torch.Generator().manual_seed(42+epoch))
class Interrupted(RuntimeError):pass
real_save=e.save_snapshot
def fail_second_epoch(output,payload):
    if payload['phase']=='calibration' and payload['epoch']==2:raise Interrupted('after new best file but before new epoch commit')
    return real_save(output,payload)
checks={}
with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as d:
    full=Path(a).resolve();partial=Path(d).resolve()
    baseline=e.run_formal(Tiny,loaders,full,binding=b,device='cpu')
    checks['fixture_has_real_later_validation_improvement']=[row['val_correct'] for row in baseline['calibration_history']]==[0,2]
    checks['selected_later_epoch2']=baseline['selected_epoch']==2
    try:
        with patch.object(e,'save_snapshot',side_effect=fail_second_epoch):e.run_formal(Tiny,loaders,partial,binding=b,device='cpu')
    except Interrupted:checks['injected_precommit_failure']=True
    payload,receipt=c.load_snapshot(partial,b)
    checks['previous_epoch1_still_recoverable']=payload['epoch']==1 and payload['best_epoch']==1
    old_best=payload['checkpoints']['best_calibration']
    checks['old_best_slot_bytes_preserved']=hashlib.sha256(Path(old_best['path']).read_bytes()).hexdigest()==old_best['sha256']
    resumed=e.run_formal(Tiny,loaders,partial,binding=b,device='cpu',resume=True)
    checks['resumed_selection_and_test_equal']=resumed['selected_epoch']==2 and resumed['test_metrics']==baseline['test_metrics']
    states=[]
    for out,result in [(full,baseline),(partial,resumed)]:
        model=Tiny();rec=result['checkpoints']['final_refit'];c.load_model(model,rec['path'],expected_sha256=rec['sha256'],binding=b);states.append(c.state_sha256(model))
    checks['final_refit_state_equal']=states[0]==states[1]
record=dict(schema='hgtnet-best-slot-transaction-review-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),
    engine_sha256=hashlib.sha256((ROOT/'external_baselines/hgtnet/formal_engine.py').read_bytes()).hexdigest(),
    scope='actual small CPU optimization/evaluation, two best slots and injected interrupted save; no full HGT/GPU/NWPU claim')
(HERE/'best_transaction_validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True));assert record['passed']
