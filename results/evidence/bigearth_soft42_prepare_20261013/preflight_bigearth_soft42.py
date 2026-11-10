"""Read-only inventory and frozen-protocol gate for the authorized BEN soft42 run."""
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys

WORK = Path('/home/featurize/work')
REPO = WORK / 'GAViT_SOFT_20260927'
COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
OLD_TOOL = WORK / 'bigearth_seed45_20260922_01/run_seed45.py'
OLD_SHA = '8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def existing_soft42(work=WORK):
    """Inspect BEN run folders and GAViT checkpoint folders without loading weights."""
    found = dict(formal_or_partial=[], engineering_smoke=[], unreadable=[])
    for parent in sorted(work.iterdir()):
        if not parent.is_dir() or not ('bigearth' in parent.name.lower() or 'gavit' in parent.name.lower()):
            continue
        for directory, dirs, files in os.walk(parent, followlinks=False):
            current = Path(directory)
            depth = len(current.relative_to(parent).parts)
            dirs[:] = sorted(d for d in dirs if d not in
                ('.git', '.venv', 'venv', '__pycache__', 'data', 'bigearth_files', 'site-packages', 'node_modules', '.worktrees'))
            if depth >= 4:
                dirs[:] = []
            for name in sorted(files):
                path_hint = str(current/name).lower()
                if name.endswith('.pth') and 'soft' in path_hint and '42' in path_hint:
                    stem = name[:-4].removesuffix('.last.train_state')
                    if not (current/(stem+'.meta.json')).exists():
                        bucket = 'engineering_smoke' if 'smoke' in path_hint else 'formal_or_partial'
                        found[bucket].append(dict(path=str(current/name),status='unidentified weights without sidecar',
                            stage=None,checkpoint=str(current/name),best=None))
                if not (name.endswith('.meta.json') or name in ('run.json', 'summary.json')):
                    continue
                p = current / name
                try:
                    obj = json.loads(p.read_text())
                    require(isinstance(obj, dict), 'JSON report is not an object')
                except (ValueError, OSError, RuntimeError) as exc:
                    found['unreadable'].append(dict(path=str(p), error=str(exc)))
                    continue
                candidates = [obj]
                if isinstance(obj.get('metadata'), dict):
                    candidates.append(obj['metadata'])
                if isinstance(obj.get('stages'), dict):
                    candidates += [v for v in obj['stages'].values() if isinstance(v, dict)]
                for m in candidates:
                    seed = m.get('training', {}).get('seed', m.get('seed'))
                    arch = m.get('architecture', {})
                    if seed != 42 or arch.get('grouping') != 'soft_prototype':
                        continue
                    if m.get('dataset') != 'BigEarthNet-19' and not ('splits' in m and m.get('stage') in ('gavit_train','gavit_test')):
                        continue
                    row = dict(path=str(p), status=m.get('status'), stage=m.get('execution', {}).get('run_stage'),
                               checkpoint=m.get('checkpoint'), best=m.get('best'))
                    bucket = 'engineering_smoke' if row['stage'] == 'smoke' else 'formal_or_partial'
                    if row not in found[bucket]:
                        found[bucket].append(row)
    return found


def load_base():
    require(OLD_TOOL.is_file(), 'Missing frozen run_seed45.py; restore it from persistent work artifacts, do not retrain45')
    require(hashlib.sha256(OLD_TOOL.read_bytes()).hexdigest() == OLD_SHA, 'Frozen base helper changed')
    spec = importlib.util.spec_from_file_location('frozen_data_gate', OLD_TOOL)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.REPO, b.COMMIT = REPO, COMMIT
    return b


def controls42(b, torch):
    base = WORK / 'GAViT_Project'
    names = dict(
        swin=('best_bigearth_swin', 'bigearth_swin_test_retry.log', ('70.9%', '64.2%', '76.6%')),
        gavit=('best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654',
               'sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log',
               ('72.0%', '66.0%', '76.7%')))
    result = {}
    for model, (stem, log_name, expected) in names.items():
        ckpt = base / 'checkpoints' / (stem + '.pth')
        log_path = base / 'logs' / log_name
        require(ckpt.is_file() and log_path.is_file(), 'Missing existing42 control: ' + str(ckpt) + ' / ' + str(log_path))
        log = log_path.read_text()
        for marker, value in zip(('Test mAP   (macro): ', 'Test F1    (macro): ', 'Test F1    (micro): '), expected):
            require(marker + value in log, 'Control42 test metrics differ from archive: ' + str(log_path))
        sha = b.digest(ckpt)
        record = dict(checkpoint=str(ckpt), checkpoint_sha256=sha, test_log=str(log_path),
                      test_log_sha256=b.digest(log_path), metrics_printed=list(expected))
        sidecar = ckpt.with_suffix('.meta.json')
        if sidecar.exists():
            m = json.loads(sidecar.read_text())
            require(m['dataset'] == 'BigEarthNet-19' and m['model'] == model and m['training']['seed'] == 42,
                    'Wrong control42 metadata: ' + str(sidecar))
            require(m['checkpoint_sha256'] == sha, 'Control42 checkpoint/sidecar hash mismatch')
            record.update(metadata=m, metadata_sha256=b.digest(sidecar))
        else:
            require(model == 'swin', 'Fixed42 metadata missing; inspect persistent artifacts')
            record['metadata_status'] = 'legacy baseline has no sidecar; historical protocol and rounded log only'
        if model == 'gavit':
            expected_arch = dict(num_classes=19, num_regions=16, knn_k=5, gat_hidden=256, gat_heads=4,
                                 gat_layers=2, dropout=.1, grouping='attentive_spatial', edge_type='sparse_hybrid', integration='token_feedback')
            require(all(m['architecture'][k] == v for k,v in expected_arch.items()), 'Wrong fixed42 architecture')
            require(m['git'] == dict(commit='cd39be9', dirty=False), 'Wrong fixed42 code')
            require(m['best']['epoch'] == 13 and m['execution']['run_stage'] == 'formal', 'Wrong fixed42 best checkpoint')
            require(m['training'] == dict(seed=42, epochs=30, batch_size=32, lr=.0003, weight_decay=.0001,
                optimizer='AdamW', scheduler='CosineAnnealingLR', loss='BCEWithLogitsLoss', f1_threshold=.5), 'Wrong fixed42 recipe')
            state = torch.load(ckpt.with_suffix('.last.train_state.pth'), map_location='cpu', weights_only=True)
            require(state['epoch'] == 30, 'Fixed42 final epoch differs from archived30')
            record['final_epoch'] = state['epoch']
            del state
        result[model] = record
    return result


def preflight():
    require(not sys.flags.optimize, 'Disable PYTHONOPTIMIZE')
    inventory = existing_soft42()
    require(not inventory['unreadable'], 'Unreadable existing report(s); preserve/inspect: ' + json.dumps(inventory['unreadable']))
    if inventory['formal_or_partial']:
        return dict(status='BIGEARTH_SOFT42_EXISTING_RESULT_REVIEW_REQUIRED', seed=42, inventory=inventory,
                    new_training_started=False, next_action='Inspect existing weights/test reports and reuse if valid; do not relaunch')
    b = load_base()
    b.identity()
    b.idle_gpu()
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required')
    torch.set_num_threads(8)
    hardware = dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        threads=torch.get_num_threads(), python=sys.version,
        packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},
        gpu_identity=b.output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader']))
    precision = dict(dtype=str(torch.get_default_dtype()), matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                     cudnn_tf32=torch.backends.cudnn.allow_tf32, amp=False)
    rows = list(csv.reader(hardware['gpu_identity'].splitlines(), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == 3, 'Expected one GPU UUID/name/driver')
    uuid, gpu, driver = [v.strip() for v in rows[0]]
    require(hardware['gpu'] == gpu == 'NVIDIA GeForce RTX 4090' and uuid.startswith('GPU-') and driver, 'Expected RTX4090')
    reference = json.loads((WORK / 'bigearth_soft43_20261011_01/preflight.json').read_text())
    require(all(hardware[k] == reference['hardware'][k] for k in ('cuda','cudnn','threads','python','packages')),
            'Software differs from validated soft43; inspect before launch')
    require(precision == reference['precision'], 'Precision differs from soft43')
    controls = controls42(b, torch)
    require(Path('/home/featurize/data/BigEarthNet-S2').is_dir(), 'Restore BigEarthNet-S2 on instance-local data first')
    seen, scans = set(), {}
    for name, (count, sha) in b.FROZEN.items():
        scans[name] = b.scan_split(b.SPLITS / (name + '.csv'), count, sha, seen)
        scans[name].pop('examples')
    return dict(status='BIGEARTH_SOFT42_PREFLIGHT_PASS', seed=42, controls=controls, inventory=inventory,
        splits=scans, hardware=hardware, precision=precision, soft_commit=COMMIT,
        reference_gpu_identity=reference['hardware']['gpu_identity'], reference_gpu_model=reference['hardware']['gpu'],
        physical_gpu_identity_changed=hardware['gpu_identity'] != reference['hardware']['gpu_identity'],
        gpu_model_changed=hardware['gpu'] != reference['hardware']['gpu'], new_training_started=False,
        comparison_limitations='Reuse legacy Swin42 and cd39be9 fixed42 controls; soft42 uses frozen ed480e6 on current4090. Historical code/GPU and baseline provenance limitations remain; same seed does not erase them.')


if __name__ in ('__main__', 'soft42_preflight'):
    sys.dont_write_bytecode = True
    report = preflight()
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
