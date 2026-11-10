"""Actual official torchvision checkpoint vs canonical VGG numerical check."""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torchvision.models import vgg16
from PIL import Image

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from external_baselines.dfagcn.extraction import load_vgg_adapter, _preprocess

output=Path(__file__).resolve().parent
metadata=json.loads((output/'weights_provenance.json').read_text())
weights=Path(metadata['local_path'])
torch.set_num_threads(1)
started=time.monotonic()
adapter=load_vgg_adapter(weights,metadata['sha256'],device='cpu')
state=torch.load(weights,map_location='cpu',weights_only=True)
with torch.device('meta'):
    reference=vgg16(weights=None)
reference.load_state_dict(state,strict=True,assign=True)
reference.eval()
image_path=ROOT/'results/evidence/dfagcn_pipeline_20261029/smoke_run_02/images/class_a/00.png'
with Image.open(image_path) as image:
    x224=_preprocess(image,224).unsqueeze(0)
    x448=_preprocess(image,448).unsqueeze(0)
captured={}
def capture(name):
    def hook(module,args,result):
        captured[name]=result.detach().clone()
    return hook
handles=[reference.classifier[1].register_forward_hook(capture('fc'))]
for index,name in [(14,'conv3'),(21,'conv4'),(28,'conv5')]:
    handles.append(reference.features[index].register_forward_hook(capture(name)))
with torch.no_grad():
    reference(x224)
    reference.features(x448)
    actual=adapter(x224,x448)
for handle in handles:handle.remove()
results={}
for name in ('fc','conv3','conv4','conv5'):
    value=actual[name]
    error=float((value-captured[name]).abs().max())
    assert error<=1e-6,(name,error)
    assert torch.isfinite(value).all()
    results[name]={'shape':list(value.shape),'reference_max_abs_error':error,
                   'min':float(value.min()),'max':float(value.max()),
                   'negative_values':int((value<0).sum()),
                   'tensor_sha256':hashlib.sha256(value.numpy().tobytes()).hexdigest()}
assert results['fc']['shape']==[1,4096]
assert all(results[name]['negative_values']>0 for name in ('conv3','conv4','conv5'))
report={'schema':'dfagcn-official-vgg16-cpu-verification-v1','passed':True,
        'checkpoint_sha256':metadata['sha256'],'source_url':metadata['source_url'],
        'python':sys.version,'torch':torch.__version__,'device':'cpu','cpu_threads':1,
        'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
        'results':results,'elapsed_seconds':time.monotonic()-started,
        'scope':'Real official torchvision VGG16 vs canonical torchvision model; no MatConvNet, realNWPU or CUDA equivalence claim'}
(output/'pretrained_cpu_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
