"""Full unchanged HGTNet+actual candidate, tiny artificial JPEG staged workflow."""
import copy,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from external_baselines.hgtnet import formal_data as data
from external_baselines.hgtnet.formal_protocol import RECIPE,recipe_sha256,validate_binding
from external_baselines.hgtnet.formal_engine import run_formal
from external_baselines.hgtnet.model import HGTNet,parameter_counts
from external_baselines.hgtnet.config import load_config
from external_baselines.hgtnet.pretrained import load_pretrained_file
from tools import run_hgtnet_nwpu_smoke as common

HERE=Path(__file__).resolve().parent.parent/'full_model_cpu_fixture_03'
output=HERE/'run'
if output.exists():raise FileExistsError('Use a distinct fixture output; no overwrite')
output.mkdir();images=HERE/'images';images.mkdir()
torch.set_num_threads(1);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
selected={k:[] for k in ['train','val','test']}
for split in selected:
    for label in [0,1]:
        rowpath='c%02d/%s_%d.jpg'%(label,split,label)
        yy,xx=np.indices((256,256));offset=11*len(split)+31*label
        pixels=np.stack([(xx+offset)%256,(yy*2+offset)%256,(xx+yy+offset)%256],axis=-1).astype('uint8')
        path=images/rowpath;path.parent.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(path,format='JPEG',quality=90)
        selected[split].append(dict(path=rowpath,label=label,split=split,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
selected['pool']=sorted(selected['train']+selected['val'],key=lambda r:r['path'])
recipe=copy.deepcopy(RECIPE);recipe.update(max_epochs=2,batch_size=2,eval_batch_size=2)
binding=validate_binding(dict(schema='hgtnet-formal-binding-v1',input_scope='cpu_fixture',recipe=recipe,
    recipe_sha256=recipe_sha256(recipe),class_names=['c%02d'%i for i in range(45)],
    counts={k:len(v) for k,v in selected.items()},
    manifest_sha256=hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest(),
    config_sha256=common.CONFIG_SHA,weights_sha256=common.WEIGHTS_SHA,source_identity=common._source_identity(),
    records={k:[dict(path=r['path'],label=r['label']) for r in selected[k]] for k in ['val','test']}))
(HERE/'binding.json').write_text(json.dumps(binding,indent=2,sort_keys=True)+'\n')
weights=ROOT/'checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth'
cfg=load_config(ROOT/'configs/hgtnet_independent_v0.json');pretraining=[];timeline=[];datasets={};parameter_reports=[]
def factory():
    model=HGTNet(cfg)
    report=load_pretrained_file(model,weights,common.WEIGHTS_SHA)
    model.pretraining_loaded=True;pretraining.append(report);parameter_reports.append(parameter_counts(model))
    assert parameter_reports[-1]['total']==parameter_reports[-1]['trainable']==74619858
    return model
def loaders(split,epoch):
    timeline.append(dict(kind='loader',split=split,epoch=epoch))
    if split=='test' and not any(v.get('stage')=='before_test' for v in timeline):raise AssertionError('Premature test')
    if split not in datasets:datasets[split]=data.make_dataset(images,selected[split],training=split in ('train','pool'))
    return torch.utils.data.DataLoader(datasets[split],batch_size=2,shuffle=split in ('train','pool'),
        generator=torch.Generator().manual_seed(42+epoch),num_workers=0,pin_memory=False,drop_last=False)
def progress(value):
    timeline.append(dict(value,kind='progress'))
    print('CPU_FIXTURE_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
started=time.perf_counter()
result=run_formal(factory,loaders,output,binding=binding,device='cpu',on_progress=progress)
(HERE/'engine_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
assert result['completed'] and not result['formal_training_started'] and result['input_scope']=='cpu_fixture'
assert result['initial_state_sha256']==result['refit_initial_state_sha256']
assert result['checkpoint_reload']['reload_max_abs_error']<=1e-6
test_packet=json.loads(Path(result['test_path']).read_text())
assert len(test_packet['predictions'])==2 and result['counts']==dict(train=2,val=2,pool=4,test=2)
assert hashlib.sha256(Path(result['test_path']).read_bytes()).hexdigest()==result['test_sha256']
assert result['cuda_peak_allocated_bytes'] is None
assert common._source_identity()==binding['source_identity'],'Source changed during fixture verification'
result.update(verification_wall_seconds=time.perf_counter()-started,
    local_runtime=dict(python=sys.version,torch=str(torch.__version__),device='cpu',threads=torch.get_num_threads()),
    pretraining_reports=pretraining,parameters=parameter_reports[0],timeline=timeline,
    explicit_scope='full unchanged74619858-parameter HGTNet plus actual Swin-B candidate; tiny artificialJPEG CPU fixture only, no realNWPU/GPU/performance claim')
(HERE/'summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(completed=result['completed'],input_scope=result['input_scope'],
    selected_epoch=result['selected_epoch'],optimizer_steps=result['optimizer_steps'],
    reload_error=result['checkpoint_reload']['reload_max_abs_error'],seconds=result['verification_wall_seconds']),indent=2))
