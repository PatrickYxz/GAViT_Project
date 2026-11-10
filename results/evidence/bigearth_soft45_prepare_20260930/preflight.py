"""Check existing controls and restored RGB data; never start training."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Disable PYTHONOPTIMIZE before running experiment checks')
print('BIGEARTH SOFT45 PRECHECK START (no training)', flush=True)
assert Path('/home/featurize/data/BigEarthNet-S2').is_dir(), 'Restore BigEarthNet-S2 to /home/featurize/data/BigEarthNet-S2 first'
root = Path('/home/featurize/work/bigearth_seed45_20260922_01')
repo = Path('/home/featurize/work/GAViT_SOFT_20260927')
old_tool = root / 'run_seed45.py'
sha = hashlib.sha256(old_tool.read_bytes()).hexdigest()
assert sha == '8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369', 'Original gate script changed'
spec = importlib.util.spec_from_file_location('original_bigearth_gate', old_tool)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
reference_commit = b.COMMIT
b.REPO = repo
b.COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
b.identity()
assert (root / 'run.exit').read_text().strip() == 'exit_code=0'
assert json.loads((root / 'summary.json').read_text())['status'] == 'complete'
expected = {'swin': '78cf048b6e2690e40789d13c78030c2350b238c899024a632347af7f0997ed8c',
            'gavit': '4268de448d7fabee4fde6db042241efbf38459953f6108337b95a3025313940c'}
reports, controls = {}, {}
for model in ('swin', 'gavit'):
    print('CHECKING CONTROL: ' + model, flush=True)
    train = root / (model + '_train')
    test = root / (model + '_test')
    assert all((p / 'run.exit').read_text().strip() == 'exit_code=0' for p in (train, test))
    r = json.loads((train / 'run.json').read_text())
    t = json.loads((test / 'run.json').read_text())
    m = json.loads((train / (model + '.meta.json')).read_text())
    metrics = json.loads((test / 'metrics.json').read_text())
    assert r['status'] == t['status'] == 'complete' and r['final_epoch'] == 30
    assert r['model'] == t['model'] == m['model'] == model
    assert r['seed'] == t['seed'] == m['training']['seed'] == 45
    assert r['commit'] == t['commit'] == reference_commit
    assert m['git']['commit'] == reference_commit[:7] and not m['git']['dirty']
    assert m['execution']['run_stage'] == 'formal' and m['dataset'] == 'BigEarthNet-19'
    assert r['pretrained_sha256'] == t['pretrained_sha256'] == b.PRETRAINED_SHA
    assert r['splits'] == t['splits'] == {k: list(v) for k,v in b.FROZEN.items()}
    assert r['metadata'] == m
    assert all(m['training'][k] == v for k,v in dict(epochs=30,batch_size=32,lr=0.0003,
        weight_decay=0.0001,optimizer='AdamW',scheduler='CosineAnnealingLR',
        loss='BCEWithLogitsLoss',f1_threshold=0.5).items())
    assert m['architecture']['num_classes'] == 19 and m['architecture']['dropout'] == 0.1
    if model == 'gavit':
        assert all(m['architecture'][k] == v for k,v in dict(num_regions=16,knn_k=5,
            gat_hidden=256,gat_heads=4,gat_layers=2,grouping='attentive_spatial',
            edge_type='sparse_hybrid',integration='token_feedback').items())
    assert b.digest(train / (model + '.pth')) == expected[model] == r['checkpoint_sha256'] == m['checkpoint_sha256'] == metrics['checkpoint_sha256']
    assert metrics == t['metrics'] and metrics['samples'] == 119825 and metrics['f1_threshold'] == 0.5
    assert r['precision'] == t['precision']
    assert r['hardware'] == t['hardware']
    reports[model] = r
    controls[model] = dict(final_epoch=r['final_epoch'],checkpoint_sha256=expected[model],
        best=m['best'],test_metrics={k:metrics[k] for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')})
print('EXISTING CONTROLS PASS: both full 30 epochs; checkpoint hashes and recipe match', flush=True)
assert reports['swin']['hardware'] == reports['gavit']['hardware']
assert reports['swin']['precision'] == reports['gavit']['precision']
b.idle_gpu()
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required'
torch.set_num_threads(8)
hardware = dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
    threads=torch.get_num_threads(),python=sys.version,
    packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},
    gpu_identity=b.output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader']))
precision = dict(dtype=str(torch.get_default_dtype()),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
    cudnn_tf32=torch.backends.cudnn.allow_tf32,amp=False)
reference = reports['gavit']['hardware']
assert all(hardware[k] == reference[k] for k in ('gpu','cuda','cudnn','threads','python','packages')), 'Hardware/software changed; inspect before launch'
assert precision == reports['gavit']['precision'], 'Precision changed'
print('ENVIRONMENT PASS ' + json.dumps(hardware), flush=True)
seen, scans = set(), {}
for name, (count, digest) in b.FROZEN.items():
    row = b.scan_split(b.SPLITS / (name + '.csv'), count, digest, seen)
    row.pop('examples')
    scans[name] = row
# Check for existing soft formal runs under work and one level of run subdirectories.
existing = []
work = Path('/home/featurize/work')
for parent in work.iterdir():
    if not parent.is_dir() or 'bigearth' not in parent.name.lower():
        continue
    for meta_path in list(parent.glob('*.meta.json')) + list(parent.glob('*/*.meta.json')):
        meta = json.loads(meta_path.read_text())
        if (meta.get('dataset') == 'BigEarthNet-19' and meta.get('architecture', {}).get('grouping') == 'soft_prototype'
                and meta.get('training', {}).get('seed') == 45 and meta.get('execution', {}).get('run_stage') == 'formal'):
            existing.append(str(meta_path))
assert not existing, 'Existing formal soft45 artifacts; preserve and inspect: ' + str(existing)
report = dict(status='BIGEARTH_SOFT45_PREFLIGHT_PASS',controls=controls,splits=scans,
    hardware=hardware,precision=precision,physical_gpu_identity_changed=hardware['gpu_identity']!=reference['gpu_identity'],
    reference_gpu_identity=reference['gpu_identity'],
    soft_commit=b.COMMIT,reference_commit=reports['gavit']['commit'],new_training_started=False,
    data_scope='All frozen split paths and RGB files exist; no replacement split or merged RGB images',
    existing_run_search_scope='work directories whose names include bigearth, root and one child level')
print(json.dumps(report,indent=2),flush=True)
