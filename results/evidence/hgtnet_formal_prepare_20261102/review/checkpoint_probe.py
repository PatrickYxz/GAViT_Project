"""Small CPU audit of safe storage boundaries; never a target GPU claim."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from external_baselines.hgtnet import formal_protocol as p, formal_checkpoints as c

HERE=Path(__file__).resolve().parent
recipe=copy.deepcopy(p.RECIPE);recipe.update(max_epochs=2,batch_size=2,eval_batch_size=2)
binding=dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=recipe,
    recipe_sha256=p.recipe_sha256(recipe),class_names=['c%02d'%i for i in range(45)],
    counts=dict(train=2,val=2,pool=4,test=2),records={k:[dict(path=k+str(i),label=i) for i in range(2)] for k in ['val','test']},
    manifest_sha256='a'*64,config_sha256='b'*64,weights_sha256='c'*64,source_identity={'fixture.py':'d'*64})
checks={};observations={}
checks['fixture_binding_valid']=p.validate_binding(binding)==binding
bad=copy.deepcopy(binding);bad['recipe']['amp']=0;bad['recipe_sha256']=p.recipe_sha256(bad['recipe'])
try:p.validate_binding(bad);checks['integer_amp_rejected']=False
except ValueError:checks['integer_amp_rejected']=True
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary).resolve();model=nn.Linear(3,45);optimizer=torch.optim.Adam(model.parameters(),lr=1e-5,weight_decay=5e-5)
    model(torch.randn(2,3)).sum().backward();optimizer.step()
    payload=dict(schema='hgtnet-formal-snapshot-v1',binding=binding,generation=1,phase='calibration',epoch=1,
        model_state=model.state_dict(),optimizer_state=optimizer.state_dict())
    c.save_snapshot(root,payload)
    saved,receipt=c.load_snapshot(root,binding)
    fresh=nn.Linear(3,45);c.apply_model_state(fresh,saved['model_state'])
    checks['full_model_state_equal']=c.state_sha256(fresh)==c.state_sha256(model)
    fresh_optimizer=torch.optim.Adam(fresh.parameters(),lr=1e-5,weight_decay=5e-5)
    fresh_optimizer.load_state_dict(saved['optimizer_state'])
    checks['adam_state_nonempty']=bool(fresh_optimizer.state_dict()['state'])
    checks['adam_profile_equal']=fresh_optimizer.param_groups[0]['lr']==1e-5 and fresh_optimizer.param_groups[0]['weight_decay']==5e-5
    changed=copy.deepcopy(payload);changed['generation']=2;changed['epoch']=2
    changed['optimizer_state']['param_groups'][0]['lr']=0.001
    changed_receipt=c.save_snapshot(root,changed)
    changed_saved,_=c.load_snapshot(root,binding)
    fresh_optimizer.load_state_dict(changed_saved['optimizer_state'])
    observations['storage_accepts_finite_changed_adam_lr']=fresh_optimizer.param_groups[0]['lr']==0.001
    # Storage intentionally cannot prove optimizer semantics; engine must reject.
    Path(changed_receipt['path']).write_bytes(b'corrupt')
    recovered,_=c.load_snapshot(root,binding)
    checks['corrupted_newest_falls_back']=recovered['generation']==1
    invalid=copy.deepcopy(payload);invalid.update(generation=3,epoch=999)
    c.save_snapshot(root,invalid)
    invalid_saved,_=c.load_snapshot(root,binding)
    observations['storage_accepts_epoch_above_recipe']=invalid_saved['epoch']==999

record=dict(schema='hgtnet-independent-checkpoint-review-v1',checks=checks,passed=all(checks.values()),
    observations_requiring_engine_semantic_validation=observations,
    source_hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
        [ROOT/'external_baselines/hgtnet/formal_protocol.py',ROOT/'external_baselines/hgtnet/formal_checkpoints.py',ROOT/'external_baselines/hgtnet/formal_data.py']},
    scope='small actual CPU tensors and storage only; no GPU, NWPU, whole formal trainer or benchmark claim')
(HERE/'checkpoint_validation.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
assert record['passed']
