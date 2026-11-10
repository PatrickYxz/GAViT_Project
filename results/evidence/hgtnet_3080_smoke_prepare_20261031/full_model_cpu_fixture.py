"""Exercise the actual full HGTNet engine on six generated JPEGs, never NWPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from external_baselines.hgtnet.config import load_config,config_sha256
from external_baselines.hgtnet.model import HGTNet,parameter_counts
from external_baselines.hgtnet.nwpu_data import make_dataset
from external_baselines.hgtnet.nwpu_engine import run_epoch_smoke
from external_baselines.hgtnet.pretrained import load_pretrained_file

args=argparse.ArgumentParser();args.add_argument('--output',required=True,type=Path);args=args.parse_args()
output=args.output.absolute();output.mkdir(parents=True,exist_ok=False)
(output/'.gitignore').write_text('*.pt\nimages/\n')
torch.set_num_threads(1);torch.manual_seed(42)
started=time.perf_counter(); rng=np.random.default_rng(42)
rows={'train':[],'val':[]};images=output/'images'
for split,labels in [('train',[0,1,2,3]),('val',[0,1])]:
    for index,label in enumerate(labels):
        relative=f'class{label:02}/{split}_{index:03}.jpg'; path=images/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,256,(256,256,3),dtype=np.uint8)).save(path,quality=85)
        rows[split].append({'path':relative,'label':label,'split':split,
                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(output/'fixture_rows.json').write_text(json.dumps({'scope':'generated_jpeg_cpu_fixture_only','rows':rows},indent=2)+'\n')
cfg=load_config(ROOT/'configs/hgtnet_independent_v0.json');model=HGTNet(cfg)
weights=ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth'
pretraining=load_pretrained_file(model,weights,'68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
model.pretraining_loaded=True
source_paths=list((ROOT/'external_baselines/hgtnet').glob('*.py'))+[ROOT/'tools/run_hgtnet_nwpu_smoke.py',ROOT/'tools/run_dfagcn_nwpu_smoke.py']
sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)}
binding={'schema':'hgtnet-nwpu-smoke-binding-v1','input_scope':'cpu_fixture',
         'config':cfg,'config_sha256':config_sha256(cfg),'counts':{'train':4,'val':2,'test':0},
         'class_names':[f'class{i:02}' for i in range(45)],'manifest_sha256':None,
         'source_identity':sources,'pretraining':pretraining,'runtime':{'device':'cpu','cpu_threads':1,
         'torch':str(torch.__version__),'python':sys.version},
         'val_paths':[r['path'] for r in rows['val']],'val_labels':[r['label'] for r in rows['val']]}
train=DataLoader(make_dataset(images,rows['train']),batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(42))
val=DataLoader(make_dataset(images,rows['val']),batch_size=2,shuffle=False)
result=run_epoch_smoke(model,train,val,output,binding=binding,device='cpu',model_factory=lambda:HGTNet(cfg))
assert result['completed'] is True and result['optimizer_steps']==2
assert result['checkpoint']['reload_max_abs_error']==0.0
assert result['first_batch_gradients']['passed'] and len(result['first_batch_gradients']['groups'])==5
result.update(parameters=parameter_counts(model),source_identity=sources,fixture_record=str(output/'fixture_rows.json'),
              candidate_sha256=pretraining['source']['sha256'],candidate_actual_loaded_tensors=len(pretraining['loaded']),
              verification_wall_seconds=time.perf_counter()-started,
              scope='generated_JPEG_full_HGTNet_CPU_engineering_fixture_not_real_NWPU_or_GPU')
(output/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'completed':result['completed'],'input_scope':result['input_scope'],'parameters':result['parameters'],
                  'optimizer_steps':result['optimizer_steps'],'reload_error':result['checkpoint']['reload_max_abs_error'],
                  'wall_seconds':result['verification_wall_seconds']},indent=2))
