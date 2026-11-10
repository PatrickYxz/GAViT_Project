"""Read-only identity/data gate for the authorized soft43 supplement."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if sys.flags.optimize:
    raise RuntimeError('Disable PYTHONOPTIMIZE')
old_tool = Path('/home/featurize/work/bigearth_seed45_20260922_01/run_seed45.py')
assert hashlib.sha256(old_tool.read_bytes()).hexdigest() == '8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369', 'Frozen helper changed/missing'
spec = importlib.util.spec_from_file_location('frozen_data_gate', old_tool)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.REPO = Path('/home/featurize/work/GAViT_SOFT_20260927')
b.COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
b.identity()
b.idle_gpu()
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required'
torch.set_num_threads(8)
import importlib.metadata
hardware = dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
    threads=torch.get_num_threads(), python=sys.version,
    packages={n: importlib.metadata.version(n) for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},
    gpu_identity=b.output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader']))
precision = dict(dtype=str(torch.get_default_dtype()), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32, amp=False)
reference = json.loads(Path('/home/featurize/work/bigearth_soft45_20260930_01/preflight.json').read_text())
assert all(hardware[k] == reference['hardware'][k] for k in ('gpu','cuda','cudnn','threads','python','packages')), 'Environment differs from validated soft45; inspect before paid launch'
assert precision == reference['precision'], 'Precision differs from soft45'
controls = {}
control_names = {
    'swin': ('best_bigearth_swin_final_multiseed_bb9f1cf_seed43_retry1', 'final_multiseed_swin_bb9f1cf_seed43_retry1_test.log', 30, 6),
    'gavit': ('best_bigearth_gavit_sparse_hybrid_4n_top2_final_multiseed_bb9f1cf_seed43_retry1', 'final_multiseed_sparse_hybrid_bb9f1cf_seed43_retry1_test_retry2.log', 21, 8),
}
base = Path('/home/featurize/work/GAViT_Project')
for model, (stem, test_log, final_epoch, best_epoch) in control_names.items():
    ckpt = base / 'checkpoints' / (stem + '.pth')
    meta = json.loads(ckpt.with_suffix('.meta.json').read_text())
    assert meta['dataset'] == 'BigEarthNet-19' and meta['model'] == model
    assert meta['git'] == {'commit': 'bb9f1cf', 'dirty': False}, 'Wrong control commit'
    assert meta['training'] == dict(seed=43, epochs=30, batch_size=32, lr=.0003, weight_decay=.0001,
        optimizer='AdamW', scheduler='CosineAnnealingLR', loss='BCEWithLogitsLoss', f1_threshold=.5), 'Changed control recipe'
    assert meta['execution']['run_stage'] == 'formal' and meta['best']['epoch'] == best_epoch
    assert b.digest(ckpt) == meta['checkpoint_sha256'], 'Control weight/metadata hash mismatch'
    if model == 'gavit':
        assert all(meta['architecture'][k] == v for k, v in dict(num_regions=16, knn_k=5, gat_hidden=256,
            gat_heads=4, gat_layers=2, dropout=.1, grouping='attentive_spatial', edge_type='sparse_hybrid', integration='token_feedback').items())
    state = torch.load(ckpt.with_suffix('.last.train_state.pth'), map_location='cpu', weights_only=True)
    assert state['epoch'] == final_epoch, 'Control saved epoch differs from record'
    del state
    log = (base / 'logs' / test_log).read_text()
    expected = ('71.1%', '64.1%', '76.0%') if model == 'swin' else ('71.7%', '65.7%', '76.8%')
    for marker, value in zip(('Test mAP   (macro): ', 'Test F1    (macro): ', 'Test F1    (micro): '), expected):
        assert marker + value in log, 'Control test differs from archived scores'
    controls[model] = dict(final_epoch=final_epoch, best=meta['best'], checkpoint_sha256=meta['checkpoint_sha256'],
        test_log=str(base / 'logs' / test_log), test_log_sha256=b.digest(base / 'logs' / test_log), metrics_printed=expected)
existing = []
for parent in base.parent.iterdir():
    if parent.is_dir() and ('bigearth' in parent.name.lower() or parent == base):
        for p in list(parent.glob('*.meta.json')) + list(parent.glob('*/*.meta.json')):
            m = json.loads(p.read_text())
            if (m.get('dataset') == 'BigEarthNet-19' and m.get('training', {}).get('seed') == 43
                and m.get('architecture', {}).get('grouping') == 'soft_prototype' and m.get('execution', {}).get('run_stage') == 'formal'):
                existing.append(str(p))
assert not existing, 'Existing soft43 artifacts: preserve/inspect for reuse: ' + str(existing)
assert Path('/home/featurize/data/BigEarthNet-S2').is_dir(), 'Restore BigEarthNet-S2 first'
seen, scans = set(), {}
for name, (count, sha) in b.FROZEN.items():
    scans[name] = b.scan_split(b.SPLITS / (name + '.csv'), count, sha, seen)
    scans[name].pop('examples')
report = dict(status='BIGEARTH_SOFT43_PREFLIGHT_PASS', seed=43, controls=controls, splits=scans,
    hardware=hardware, precision=precision, soft_commit=b.COMMIT, reference_commit='bb9f1cf',
    reference_gpu_identity=reference['hardware']['gpu_identity'],
    physical_gpu_identity_changed=hardware['gpu_identity'] != reference['hardware']['gpu_identity'],
    new_training_started=False, comparison_limitations='Fixed43 saved epoch21/best8; Swin43 epoch30/best6; historical 4090 train and 3060 fixed test; new soft43 uses full30 on current2080Ti, different code. Same seed does not remove these confounders.')
print(json.dumps(report, indent=2), flush=True)
