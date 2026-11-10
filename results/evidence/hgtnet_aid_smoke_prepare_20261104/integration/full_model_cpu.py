"""Actual30-class Tensor fixture; never an AID or target-GPU benchmark."""
from pathlib import Path
import hashlib
import json
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,TensorDataset
from external_baselines.hgtnet.config import load_config,config_sha256
from external_baselines.hgtnet.model import HGTNet,parameter_counts
from external_baselines.hgtnet.pretrained import load_pretrained_file
from external_baselines.hgtnet.smoke import _validate_trace
from external_baselines.hgtnet import aid_engine
from external_baselines.hgtnet.aid_entry import SOURCE_FILES

HERE=Path(__file__).resolve().parent
def dump(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
started=time.perf_counter();torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(42)
sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}
cfg=load_config(ROOT/'configs/hgtnet_aid_independent_v0.json')
assert cfg['num_classes']==30
model=HGTNet(cfg)
pretrained=load_pretrained_file(model,ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth',
    '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
assert pretrained['source_identity_verified'] is True and pretrained['mode']=='applied'
model.pretraining_loaded=True
images=torch.randn(2,3,224,224,generator=torch.Generator().manual_seed(42));labels=torch.tensor([0,1],dtype=torch.long)
model.eval()
with torch.no_grad():
    logits,trace=model.forward_with_trace(images)
_validate_trace(trace,2,30)
assert list(logits.shape)==[2,30] and torch.isfinite(logits).all()
del logits
names=[f'fixture_class{i:02}' for i in range(30)]
binding=dict(schema='hgtnet-aid-smoke-binding-v1',input_scope='cpu_fixture',config=cfg,config_sha256=config_sha256(cfg),
    class_names=names,counts=dict(train=2,val=2,test=0),runtime=dict(device='cpu',torch=str(torch.__version__),cpu_threads=2),
    val_paths=['fixture_class00/val.jpg','fixture_class01/val.jpg'],val_labels=[0,1],
    train_paths=['fixture_class00/train.jpg','fixture_class01/train.jpg'],train_labels=[0,1],
    manifest_sha256='0'*64,weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893',
    source_identity=sources,pretraining=pretrained)
train=DataLoader(TensorDataset(images,labels),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(42),drop_last=False)
val=DataLoader(TensorDataset(images.clone(),labels.clone()),batch_size=2,shuffle=False,drop_last=False)
output=Path(tempfile.mkdtemp(prefix='gavit-aid-fullmodel-cpu-20261104-',dir='/private/tmp'))
events=[]
result=aid_engine.run_epoch_smoke(model,train,val,output,binding=binding,device='cpu',model_factory=lambda:HGTNet(cfg),on_progress=events.append)
checks={
    '30class_full_trace':list(trace['logits'])==[2,30],
    'actual_candidate_loaded':pretrained['source_identity_verified'] is True and pretrained['mode']=='applied',
    'full_parameter_count':parameter_counts(model)['total']==74612163,
    'actual_optimizer_update':result['optimizer_steps']==1 and result['training_started'] is True,
    'all_five_groups':set(result['first_batch_gradients']['groups'])=={'patch_embed','transformer','dual_attention','graph','head'} and result['first_batch_gradients']['passed'],
    'complete_and_reload':result['completed'] is True and result['checkpoint']['reload_max_abs_error']<=1e-6,
    '30x30_cm':len(result['val_metrics']['confusion_matrix'])==30 and all(len(r)==30 for r in result['val_metrics']['confusion_matrix']),
    'fixture_only':result['input_scope']=='cpu_fixture' and result['formal_training_started'] is False,
    'source_unchanged':sources=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in SOURCE_FILES},
}
assert all(checks.values()),checks
dump(HERE/'full_model_release_result.json',result)
dump(HERE/'full_model_release_validation.json',dict(passed=True,count=len(checks),checks=checks,source_identity=sources,
    parameters=parameter_counts(model),pretraining=pretrained,trace=trace,wall_seconds=time.perf_counter()-started,
    cpu_fixture_output=str(output),scope='Two generated Tensor inputs, one real CPU update, full30-class HGTNet and actual candidate initialization; no AID image or GPU measurement.'))
print(json.dumps(dict(passed=True,checks=len(checks),parameters=parameter_counts(model)['total'],optimizer_steps=result['optimizer_steps'],reload_error=result['checkpoint']['reload_max_abs_error'],wall_seconds=time.perf_counter()-started,scope='CPU_fixture_only')))
