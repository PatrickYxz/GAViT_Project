"""Post-fix independent CPU checks; preserve original pre-fix probe separately."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from external_baselines.hgtnet import formal_protocol as p, formal_checkpoints as c
HERE=Path(__file__).resolve().parent
recipe=copy.deepcopy(p.RECIPE);recipe.update(max_epochs=2,batch_size=2,eval_batch_size=2)
binding=dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=recipe,
    recipe_sha256=p.recipe_sha256(recipe),class_names=['c%02d'%i for i in range(45)],
    counts=dict(train=2,val=2,pool=4,test=2),records={k:[dict(path=k+str(i),label=i) for i in range(2)] for k in ['val','test']},
    manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
checks={}
def rejects(key,function):
    try:function();checks[key]=False
    except ValueError:checks[key]=True
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary).resolve();model=nn.Linear(3,45)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=5e-5)
    model(torch.randn(2,3)).sum().backward();optimizer.step()
    base=dict(schema='hgtnet-formal-snapshot-v1',binding=binding,generation=1,phase='calibration',epoch=1,
        model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
    c.save_snapshot(root,base)
    saved,receipt=c.load_snapshot(root,binding)
    checks['valid_epoch_snapshot_roundtrip']=saved['phase']=='calibration' and saved['epoch']==1
    wrong=copy.deepcopy(base);wrong.update(generation=999,epoch=999)
    rejects('out_of_recipe_epoch_rejected',lambda:c.save_snapshot(root,wrong))
    wrong=copy.deepcopy(base);wrong['generation']=2
    rejects('generation_epoch_mismatch_rejected',lambda:c.save_snapshot(root,wrong))
    wrong=copy.deepcopy(base);wrong['optimizer_state']['param_groups'][0]['lr']=0.001
    rejects('changed_adam_lr_rejected_before_save',lambda:c.save_snapshot(root,wrong))
    wrong=copy.deepcopy(base);wrong['optimizer_state']['state'][0]['step'].add_(1)
    rejects('adam_step_not_completed_epoch_rejected',lambda:c.save_snapshot(root,wrong))
    fresh=nn.Linear(3,45);c.apply_model_state(fresh,saved['model_state'])
    restored=torch.optim.Adam(fresh.parameters(),lr=1e-5,weight_decay=5e-5)
    restored.load_state_dict(saved['optimizer_state']);checks['restored_optimizer_valid']=c.validate_optimizer(restored,recipe)
    first=next(iter(restored.state.values()));first['exp_avg']=torch.zeros(1)
    rejects('wrong_moment_shape_rejected',lambda:c.validate_optimizer(restored,recipe))
    optimizer.zero_grad(set_to_none=True);model(torch.randn(2,3)).sum().backward();optimizer.step()
    latest=dict(base,generation=2,epoch=2,model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
    rec=c.save_snapshot(root,latest)
    recorded=json.loads(Path(rec['path']+'.json').read_text());recorded['epoch']=1
    Path(rec['path']+'.json').write_text(json.dumps(recorded))
    previous,_=c.load_snapshot(root,binding)
    checks['receipt_payload_epoch_mismatch_falls_back']=previous['generation']==1
record=dict(schema='hgtnet-checkpoint-postfix-review-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),
    sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [ROOT/'external_baselines/hgtnet/formal_protocol.py',ROOT/'external_baselines/hgtnet/formal_checkpoints.py']},
    scope='actual small CPU tensors; storage/restore boundaries only; no target GPU or benchmark')
(HERE/'checkpoint_postfix_validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True));assert record['passed']
