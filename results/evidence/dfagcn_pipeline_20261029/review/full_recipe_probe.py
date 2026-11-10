"""One real CPU forward/backward for declared 7x7/256/1024→512 dimensions."""
import json
from pathlib import Path
import torch
from external_baselines.dfagcn.config import load_recipe
from external_baselines.dfagcn.model import DFAGCNBranches

recipe=load_recipe('configs/dfagcn_independent_v1.json')
torch.manual_seed(42)
model=DFAGCNBranches(recipe,45)
features={name:torch.randn(2,49,width) for name,width in [('conv3',256),('conv4',512),('conv5',512)]}
features['fc']=torch.rand(2,4096)
logits=model(features)
loss=sum(torch.nn.functional.cross_entropy(value,torch.tensor([0,44])) for value in logits.values())
loss.backward()
for branch in model.branches.values():
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
model.eval()
with torch.no_grad():
    fused=model.fused_features(features)
assert fused.shape==(2,5632) and torch.isfinite(fused).all()
record={'scope':'synthetic numerical tensor fixture, not NWPU training/evaluation',
        'recipe':'configs/dfagcn_independent_v1.json','batch':2,'pooled_nodes':49,'classes':45,
        'logit_shapes':{name:list(value.shape) for name,value in logits.items()},
        'all_parameter_gradients_finite':True,'fused_shape':list(fused.shape),'loss':float(loss.detach())}
Path('results/evidence/dfagcn_pipeline_20261029/review/full_recipe_probe.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
