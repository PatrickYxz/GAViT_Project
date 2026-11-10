"""Inject changed replay validation after an interrupted best/epoch transaction."""
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
from external_baselines.hgtnet import formal_protocol as p,formal_engine as e
HERE=Path(__file__).resolve().parent
label=sys.argv[1] if len(sys.argv)>1 else 'pre'
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
real_save=e.save_snapshot;real_evaluate=e._evaluate
def fail_second_epoch(output,payload):
    if payload['phase']=='calibration' and payload['epoch']==2:raise Interrupted('after uncommitted best predictions')
    return real_save(output,payload)
def replay_validation_not_improved(model,loader,binding,split,device):
    packet,reference=real_evaluate(model,loader,binding,split,device)
    if split=='val':
        packet=copy.deepcopy(packet);packet['predictions']=[1]*len(packet['labels'])
        packet['metrics']=e._metrics(packet['labels'],packet['predictions'],binding['class_names'])
        packet['metrics']['scope']='cpu_fixture_validation'
    return packet,reference
with tempfile.TemporaryDirectory() as d:
    output=Path(d).resolve()
    try:
        with patch.object(e,'save_snapshot',side_effect=fail_second_epoch):e.run_formal(Tiny,loaders,output,binding=b,device='cpu')
    except Interrupted:pass
    uncommitted=json.loads((output/'best_val_predictions.json').read_text())
    with patch.object(e,'_evaluate',side_effect=replay_validation_not_improved):
        result=e.run_formal(Tiny,loaders,output,binding=b,device='cpu',resume=True)
    final=json.loads((output/'best_val_predictions.json').read_text())
    file_correct=sum(x==y for x,y in zip(final['labels'],final['predictions']))
    expected_correct=int(round(result['best_val_metrics']['accuracy_percent']*b['counts']['val']/100))
    record=dict(schema='hgtnet-best-packet-replay-review-v1',selected_epoch=result['selected_epoch'],
        uncommitted_best_correct=sum(x==y for x,y in zip(uncommitted['labels'],uncommitted['predictions'])),
        final_file_correct=file_correct,selected_best_correct=expected_correct,
        best_packet_matches_selected_model_metrics=file_correct==expected_correct,
        engine_sha256=hashlib.sha256((ROOT/'external_baselines/hgtnet/formal_engine.py').read_bytes()).hexdigest(),
        scope='actual small CPU training with controlled replay-validation variation, not observed GPU nondeterminism; no NWPU claim')
(HERE/('best_packet_'+label+'_validation.json')).write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
assert record['selected_epoch']==1 and record['uncommitted_best_correct']==2
