"""Safe finite model reload and alternating completed-epoch recovery snapshots."""
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch

from .aid_formal_protocol import validate_binding

def _sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def _check_path(path):
    path=Path(path)
    if any(p.is_symlink() for p in [path,*path.parents]):raise ValueError('Checkpoint paths must not be symlinks')
    path.parent.mkdir(parents=True,exist_ok=True)
    return path

def _sync_directory(path):
    # File fsync alone does not durably publish the rename after a power loss.
    descriptor=os.open(str(Path(path).parent),os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def atomic_json(path,value):
    path=_check_path(path);temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as stream:
            json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);_sync_directory(path)
    finally:temporary.unlink(missing_ok=True)

def _cpu(value):
    if isinstance(value,torch.Tensor):
        if value.is_meta or value.layout!=torch.strided or not torch.isfinite(value).all().item():raise ValueError('Checkpoint tensors must be actual finite dense tensors')
        return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    if value is None or type(value) in (str,int,float,bool):
        if isinstance(value,float) and not __import__('math').isfinite(value):raise ValueError('Nonfinite checkpoint metadata')
        return value
    raise ValueError('Unsupported checkpoint metadata type')

def _atomic_torch(path,payload):
    path=_check_path(path);temporary=path.with_name(path.name+'.tmp.'+uuid.uuid4().hex)
    try:
        value=_cpu(payload)
        with temporary.open('xb') as stream:torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path);_sync_directory(path)
    finally:temporary.unlink(missing_ok=True)
    return dict(path=str(path.absolute()),sha256=_sha(path),bytes=path.stat().st_size)

def state_sha256(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        h.update(json.dumps([name,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def apply_model_state(model,state):
    target=model.state_dict()
    if not isinstance(state,dict) or set(state)!=set(target):raise ValueError('Model state keys mismatch')
    for name,value in state.items():
        if (not isinstance(value,torch.Tensor) or value.is_meta or value.layout!=torch.strided
                or value.shape!=target[name].shape or value.dtype!=target[name].dtype
                or not torch.isfinite(value).all().item()):raise ValueError('Model state tensor mismatch: '+name)
    model.load_state_dict(state,strict=True)

def save_model(model,path,*,binding,metadata):
    b=validate_binding(binding)
    receipt=_atomic_torch(path,dict(schema='hgtnet-aid-formal-model-v1',binding=b,metadata=metadata,state_dict=model.state_dict()))
    atomic_json(str(path)+'.json',receipt)
    return receipt

def load_model(model,path,*,expected_sha256,binding):
    path=_check_path(path)
    if _sha(path)!=expected_sha256:raise ValueError('Model file SHA256 mismatch')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!='hgtnet-aid-formal-model-v1' or payload.get('binding')!=validate_binding(binding):raise ValueError('Model checkpoint binding mismatch')
    apply_model_state(model,payload['state_dict'])
    return dict(loaded=True,sha256=expected_sha256,metadata=payload['metadata'])

def _optimizer_policy(groups,recipe):
    if not isinstance(groups,list) or len(groups)!=1:raise ValueError('One frozen Adam parameter group required')
    expected=dict(lr=recipe['lr'],weight_decay=recipe['weight_decay'],eps=recipe['eps'],
        amsgrad=False,maximize=False,foreach=None,capturable=False,differentiable=False,fused=None)
    group=groups[0]
    for key,value in expected.items():
        if key not in group or type(group[key]) is not type(value) or group[key]!=value:
            raise ValueError('Restored Adam policy differs from recipe: '+key)
    if list(group.get('betas',[]))!=recipe['betas']:raise ValueError('Restored Adam betas differ')
    return group

def validate_optimizer(optimizer,recipe):
    """Call after restore; Adam.load_state_dict replaces constructor policy."""
    if not isinstance(optimizer,torch.optim.Adam):raise ValueError('Actual optimizer must be Adam')
    group=_optimizer_policy(optimizer.param_groups,recipe)
    for parameter in group['params']:
        state=optimizer.state.get(parameter)
        if not isinstance(state,dict) or set(state)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Complete Adam moments required')
        step=state['step']
        if (not isinstance(step,torch.Tensor) or step.numel()!=1 or not torch.isfinite(step).all().item()
                or float(step)<1 or float(step)!=int(float(step))):raise ValueError('Invalid Adam step')
        for name in ['exp_avg','exp_avg_sq']:
            value=state[name]
            if (not isinstance(value,torch.Tensor) or value.shape!=parameter.shape or value.dtype!=parameter.dtype
                    or value.is_meta or value.layout!=torch.strided or not torch.isfinite(value).all().item()):
                raise ValueError('Restored Adam moment differs from parameter: '+name)
    return True

def _validate_snapshot(payload):
    b=validate_binding(payload['binding']);generation=payload.get('generation');epoch=payload.get('epoch')
    phase=payload.get('phase');limit=b['recipe']['max_epochs']
    if phase=='refit':
        selected=payload.get('selected_epoch')
        if type(selected) is not int or not 1<=selected<=limit:raise ValueError('Selected refit budget invalid')
        limit=selected
    if (payload.get('schema')!='hgtnet-aid-formal-snapshot-v1' or phase not in ('calibration','refit')
            or type(epoch) is not int or not 1<=epoch<=limit or type(generation) is not int
            or generation!=(epoch if phase=='calibration' else b['recipe']['max_epochs']+epoch)):
        raise ValueError('Completed-epoch snapshot identity required')
    state=payload.get('optimizer_state',{});group=_optimizer_policy(state.get('param_groups'),b['recipe'])
    ids=group.get('params');states=state.get('state')
    if (not isinstance(ids,list) or not ids or any(type(p) is not int for p in ids) or len(set(ids))!=len(ids)
            or not isinstance(states,dict) or set(states)!=set(ids)):
        raise ValueError('Completed-epoch Adam state identities mismatch')
    count=b['counts']['train' if phase=='calibration' else 'pool'];batch=b['recipe']['batch_size']
    expected_steps=epoch*((count+batch-1)//batch)
    for value in states.values():
        if not isinstance(value,dict) or set(value)!={'step','exp_avg','exp_avg_sq'}:raise ValueError('Incomplete Adam state')
        step=value['step']
        if not isinstance(step,torch.Tensor) or step.numel()!=1 or float(step)!=expected_steps:
            raise ValueError('Adam state steps differ from completed epoch')
    return b

def save_snapshot(output,payload):
    b=_validate_snapshot(payload);generation=payload['generation']
    path=Path(output)/'recovery'/('slot%d.pt'%(generation%2))
    receipt=_atomic_torch(path,payload)
    receipt.update(generation=generation,phase=payload['phase'],epoch=payload['epoch'],binding_sha256=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest())
    atomic_json(str(path)+'.json',receipt)
    return receipt

def load_snapshot(output,binding):
    b=validate_binding(binding);valid=[]
    for slot in [0,1]:
        path=Path(output)/'recovery'/('slot%d.pt'%slot)
        try:
            _check_path(path);receipt=json.loads(Path(str(path)+'.json').read_text())
            if _sha(path)!=receipt['sha256']:continue
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload.get('binding')!=b:continue
            _validate_snapshot(payload)
            if any(payload[key]!=receipt[key] for key in ['generation','phase','epoch']):continue
            expected_binding_sha=hashlib.sha256(json.dumps(b,sort_keys=True).encode()).hexdigest()
            if receipt.get('binding_sha256')!=expected_binding_sha:continue
            if receipt.get('path')!=str(path.absolute()) or receipt.get('bytes')!=path.stat().st_size:continue
            _cpu(payload) # Reject invalid optimizer or metadata tensors before mutation.
            valid.append((payload['generation'],payload,receipt))
        except (OSError,ValueError,KeyError,TypeError,RuntimeError,EOFError,__import__('pickle').UnpicklingError):continue
    if not valid:raise ValueError('No compatible completed-epoch recovery snapshot')
    _,payload,receipt=max(valid,key=lambda item:item[0])
    return payload,receipt
