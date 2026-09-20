"""Shared smoke, calibration, refit and evaluation lifecycle for single-label scenes."""
import csv
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import random
import resource
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from experiment_identity import (assert_clean_git_state, get_git_state, graph_topology_identity,
    load_checkpoint_metadata, sha256_file, write_checkpoint_metadata)
from utils import save_checkpoint as _legacy_save_checkpoint, save_training_state, set_seed
from .data import load_manifest, select_records, write_json
from .metrics import classification_metrics
from .modeling import architecture, build_model, capture_graph_once, image_transform, transform_identity


@dataclass
class RunConfig:
    model: str
    manifest: str
    data_root: str
    output: str
    phase: str = 'smoke'
    device: str = 'cuda'
    seed: int = 42
    batch_size: int = 32
    workers: int = 2
    pretrained_path: str | None = None
    random_init: bool = False
    train_per_class: int = 8
    val_per_class: int = 2
    calibration: str | None = None


def training_identity(cfg):
    return {'seed': cfg.seed, 'batch_size': cfg.batch_size, 'workers': cfg.workers, 'optimizer': 'AdamW',
            'lr': 3e-4, 'weight_decay': 1e-4, 'betas': [0.9, 0.999], 'eps': 1e-8,
            'scheduler': 'CosineAnnealingLR', 'scheduler_t_max': 30, 'eta_min': 0.0,
            'loss': 'CrossEntropyLoss', 'label_smoothing': 0.0,
            'transforms': transform_identity(), 'precision': 'float32_no_amp',
            'tf32': False, 'drop_last': False, 'pin_memory': False}


class _Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text):
        self.log.write(text)
        self.log.flush()
        try:
            self.stream.write(text)
        except (BrokenPipeError, OSError):
            pass
        return len(text)

    def flush(self):
        self.log.flush()
        try:
            self.stream.flush()
        except (BrokenPipeError, OSError):
            pass


@contextmanager
def _run_output(output, phase):
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with (out / 'run.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f'OUTPUT {out}; PHASE {phase}; preflight starting', flush=True)
            try:
                yield out, start
            except BaseException as exc:
                traceback.print_exc()
                write_json(out / 'run.json', {'status': 'failed', 'phase': phase,
                    'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.perf_counter() - start}, replace=True)
                (out / 'run.exit').write_text('exit_code=1\n')
                raise


def _code_identity():
    root = Path(__file__).resolve().parents[1]
    files = [root / 'run_scene.py', root / 'utils.py', root / 'experiment_identity.py',
             *sorted((root / 'scene_classification').glob('*.py')), *sorted((root / 'models').glob('*.py'))]
    return {'git': get_git_state(str(root)),
            'files_sha256': {str(p.relative_to(root)): sha256_file(str(p)) for p in files}}


def _save_checkpoint(model, path, metadata):
    _legacy_save_checkpoint(model, str(path), metadata)
    sidecar = load_checkpoint_metadata(str(path))
    # The legacy helper uses cwd; this entry also supports absolute-path invocation.
    sidecar['git'] = metadata['code']['git']
    write_checkpoint_metadata(str(path), sidecar)


class SceneDataset(Dataset):
    def __init__(self, root, rows, training):
        self.root, self.rows, self.transform = Path(root), rows, image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.root / row['path']) as image:
            inputs = self.transform(image.convert('RGB'))
        return inputs, row['label'], row['path']


def _seed_worker(_index):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _loader(root, rows, cfg, training):
    generator = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(SceneDataset(root, rows, training), batch_size=cfg.batch_size,
                      shuffle=training, num_workers=cfg.workers, pin_memory=False,
                      drop_last=False, worker_init_fn=_seed_worker, generator=generator)


def _device(name):
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only cpu or cuda are supported by this protocol')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; use --device cpu only for local smoke')
    return device


def _hardware(device):
    return {'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
            'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'python': sys.version, 'torch_num_threads': torch.get_num_threads(),
            'packages': {p: importlib.metadata.version(p)
                         for p in ('torch', 'torchvision', 'timm', 'torch-geometric', 'numpy', 'pillow')}}


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _predict(model, loader, device, names):
    model.eval()
    truth, predictions, rows = [], [], []
    with torch.no_grad():
        for images, labels, paths in loader:
            logits = model(images.to(device))
            if logits.shape != (len(labels), len(names)) or not torch.isfinite(logits).all():
                raise ValueError('Invalid/nonfinite model logits')
            probs = logits.softmax(dim=1).cpu().tolist()
            pred = logits.argmax(dim=1).cpu().tolist()
            actual = labels.tolist()
            truth.extend(actual)
            predictions.extend(pred)
            rows.extend({'path': path, 'label': y, 'prediction': p,
                         **{f'p_{i}': value for i, value in enumerate(prob)}}
                        for path, y, p, prob in zip(paths, actual, pred, probs))
    return classification_metrics(truth, predictions, names), rows


def _write_metrics(out, metrics, predictions, prefix=''):
    write_json(out / (prefix + 'metrics.json'), metrics)
    for filename, rows in [('predictions.csv', predictions), ('per_class.csv', metrics['per_class'])]:
        with (out / (prefix + filename)).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def validate_calibration(source, expected):
    if (source.get('status') != 'complete' or source.get('phase') != 'calibrate'
            or source.get('final_epoch') != 30
            or type(source.get('best_epoch')) is not int or not 1 <= source['best_epoch'] <= 30
            or any(source.get(key) != value for key, value in expected.items())):
        raise ValueError('Calibration identity or completion mismatch')
    return source['best_epoch']


def validate_source_run(source, meta, exit_text):
    expected = {'phase': meta['execution']['phase'], 'model': meta['model'], 'dataset': meta['dataset'],
                'seed': meta['training']['seed'], 'training': meta['training'],
                'architecture': meta['architecture'], 'checkpoint_sha256': meta['checkpoint_sha256'],
                'manifest_sha256': meta['data']['manifest_sha256'],
                'pretrained_sha256': meta['pretrained']['sha256'],
                'final_epoch': meta['final_epoch'], 'code': meta.get('code')}
    if meta['execution']['phase'] != 'refit':
        expected['best_epoch'] = meta['best']['epoch']
    if (source.get('status') != 'complete' or exit_text != 'exit_code=0\n'
            or any(source.get(key) != value for key, value in expected.items())):
        raise ValueError('Source run completion or checkpoint identity mismatch')


def _checkpoint_metadata(checkpoint, manifest_path, manifest):
    meta = load_checkpoint_metadata(str(checkpoint))
    if not meta:
        raise ValueError('Checkpoint identity metadata missing')
    if (meta.get('dataset') != manifest['dataset'] or meta.get('class_names') != manifest['class_names']
            or meta.get('data', {}).get('manifest_sha256') != sha256_file(str(manifest_path))
            or meta.get('architecture') != architecture(meta.get('model'), len(manifest['class_names']))
            or meta.get('entry_version') != 'scene-v1'):
        raise ValueError('Checkpoint identity mismatch')
    if meta.get('checkpoint_sha256') != sha256_file(str(checkpoint)):
        raise ValueError('Checkpoint hash mismatch')
    return meta


def _restore(checkpoint, meta, device):
    model = build_model(meta['model'], meta['architecture']['num_classes']).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True), strict=True)
    return model.eval()


def run_training(cfg):
    with _run_output(cfg.output, cfg.phase) as (out, start):
        return _train(cfg, out, start)


def _train(cfg, out, start):
    if cfg.phase not in ('smoke', 'calibrate', 'refit'):
        raise ValueError('Unknown phase')
    if cfg.batch_size < 1 or cfg.workers < 0 or min(cfg.train_per_class, cfg.val_per_class) < 1:
        raise ValueError('Invalid loader/sample size')
    if cfg.random_init == bool(cfg.pretrained_path):
        raise ValueError('Choose one local --pretrained-path or explicit --random-init for smoke')
    if cfg.phase != 'smoke' and cfg.random_init:
        raise ValueError('Formal phases require pretrained weights')
    device = _device(cfg.device)
    manifest = load_manifest(cfg.manifest, cfg.data_root)
    code = _code_identity()
    if cfg.phase != 'smoke':
        if manifest['dataset'] == 'synthetic' or device.type != 'cuda' or cfg.batch_size != 32:
            raise ValueError('Formal phases require real data, CUDA and batch32')
        assert_clean_git_state(code['git'])
    if cfg.calibration and cfg.phase != 'refit':
        raise ValueError('--calibration is only for refit')
    names = manifest['class_names']
    pretrained_hash = sha256_file(cfg.pretrained_path) if cfg.pretrained_path else None
    training = training_identity(cfg)
    identity = {'model': cfg.model, 'seed': cfg.seed, 'manifest_sha256': sha256_file(cfg.manifest),
                'pretrained_sha256': pretrained_hash, 'training': training,
                'architecture': architecture(cfg.model, len(names)), 'code': code}
    epochs, source_identity = (1 if cfg.phase == 'smoke' else 30), None
    if cfg.phase == 'refit':
        if not cfg.calibration:
            raise ValueError('Refit requires --calibration directory')
        source_dir = Path(cfg.calibration)
        source = json.loads((source_dir / 'run.json').read_text())
        epochs = validate_calibration(source, identity)
        source_meta = _checkpoint_metadata(source_dir / 'best.pth', cfg.manifest, manifest)
        validate_source_run(source, source_meta, (source_dir / 'run.exit').read_text())
        source_identity = {'run_json_sha256': sha256_file(str(source_dir / 'run.json')),
                           'best_checkpoint_sha256': source_meta['checkpoint_sha256']}
    train_rows = select_records(manifest, 'pool' if cfg.phase == 'refit' else 'train',
                               per_class=cfg.train_per_class if cfg.phase == 'smoke' else None)
    val_rows = [] if cfg.phase == 'refit' else select_records(manifest, 'val',
                               per_class=cfg.val_per_class if cfg.phase == 'smoke' else None)
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg.model, len(names), cfg.pretrained_path).to(device)
    graph_capture, hook = capture_graph_once(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0)
    train_loader = _loader(cfg.data_root, train_rows, cfg, True)
    val_loader = _loader(cfg.data_root, val_rows, cfg, False) if val_rows else None
    # Separate initial model construction from the training RNG stream. Worker RNGs
    # use the loader's own generator; exact augmented pixels with workers=0 are not promised.
    set_seed(cfg.seed)
    hardware = _hardware(device)
    metadata = {'entry_version': 'scene-v1', 'model': cfg.model, 'dataset': manifest['dataset'], 'code': code,
                'class_names': names, 'architecture': identity['architecture'], 'training': training,
                'execution': {'phase': cfg.phase, 'epochs': epochs, 'workers': cfg.workers,
                    'train_per_class': cfg.train_per_class if cfg.phase == 'smoke' else None,
                    'val_per_class': cfg.val_per_class if cfg.phase == 'smoke' else None},
                'pretrained': {'path': cfg.pretrained_path, 'sha256': pretrained_hash, 'random_init': cfg.random_init},
                'data': {'manifest_sha256': identity['manifest_sha256'], 'protocol': manifest['protocol'],
                         'counts': {'train': len(train_rows), 'val': len(val_rows)},
                         'fit_scope': 'train_plus_val' if cfg.phase == 'refit' else 'internal_train'},
                'graph_topology': graph_topology_identity('sparse_hybrid', knn_k=5) if cfg.model == 'gavit' else None,
                'hardware': hardware, 'refit_source': source_identity,
                'optimizer_parameter_groups': [{'name': 'all_trainable', 'weight_decay': 1e-4,
                    'parameter_tensors': len(optimizer.param_groups[0]['params'])}],
                'dropout_modules': {name: {'type': type(module).__name__, 'p': float(module.p)}
                    for name, module in model.named_modules() if isinstance(module, torch.nn.Dropout)},
                'drop_path_modules': {name: float(module.drop_prob) for name, module in model.named_modules()
                    if hasattr(module, 'drop_prob')},
                'graph_attention_dropout': {name: float(module.dropout) for name, module in model.named_modules()
                    if type(module).__name__ == 'GATConv'}, 'command_argv': sys.argv}
    write_json(out / 'config.json', {**asdict(cfg), 'resolved': metadata})
    write_json(out / 'selected_samples.json', {'train': [r['path'] for r in train_rows],
                                              'val': [r['path'] for r in val_rows]})
    write_json(out / 'run.json', {'status': 'running', 'phase': cfg.phase, **identity})
    print(f'MODEL {cfg.model}; DATASET {manifest["dataset"]}; DEVICE {hardware["device_name"]}; '
          f'PHASE {cfg.phase}; train={len(train_rows)} val={len(val_rows)}', flush=True)
    history, best_score, best_epoch = [], -1.0, None
    check_rows = val_rows[:2] if val_rows else train_rows[:2]
    check_loader = _loader(cfg.data_root, check_rows, cfg, False)
    check_inputs = next(iter(check_loader))[0].to(device)
    selected_reference = None
    for epoch in range(1, epochs + 1):
        _sync(device)
        epoch_start = time.perf_counter()
        model.train()
        loss_sum, seen = 0.0, 0
        for batch_index, (images, labels, _paths) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            seen += len(labels)
            if epoch == 1 and batch_index == 0:
                print(f'TRAIN FIRST BATCH OK loss={float(loss.detach()):.6f}; graph={graph_capture}', flush=True)
        _sync(device)
        train_seconds = time.perf_counter() - epoch_start
        val_start = time.perf_counter()
        metrics = _predict(model, val_loader, device, names)[0] if val_loader else None
        _sync(device)
        val_seconds = time.perf_counter() - val_start
        row = {'epoch': epoch, 'train_loss': loss_sum / seen, 'lr': optimizer.param_groups[0]['lr'],
               'train_seconds': train_seconds, 'train_images_per_second': seen / train_seconds,
               'validation_seconds': val_seconds, 'validation': metrics}
        history.append(row)
        scheduler.step()
        if metrics and metrics['oa_percent'] > best_score:
            best_score, best_epoch = metrics['oa_percent'], epoch
            metadata['best'] = {'metric': 'val_OA', 'value': best_score, 'epoch': epoch}
            with torch.no_grad():
                selected_reference = model(check_inputs).detach().clone()
            _save_checkpoint(model, out / 'best.pth', metadata)
        metadata['final_epoch'] = epoch
        if cfg.phase == 'refit' and epoch == epochs:
            model.eval()
            with torch.no_grad():
                selected_reference = model(check_inputs).detach().clone()
        _save_checkpoint(model, out / 'last.pth', metadata)
        save_training_state(model, optimizer, scheduler, str(out / 'last.train_state.pth'),
                            epoch=epoch, best_metric=best_score, resume_identity=identity)
        write_json(out / 'history.json', history, replace=True)
        print(f'EPOCH {epoch}/{epochs}; loss={row["train_loss"]:.6f}; '
              f'val_OA={metrics["oa_percent"] if metrics else None}; train_s={train_seconds:.2f}', flush=True)
    if hook:
        hook.remove()
    checkpoint = out / ('last.pth' if cfg.phase == 'refit' else 'best.pth')
    final_meta = _checkpoint_metadata(checkpoint, cfg.manifest, manifest)
    final_meta['final_epoch'] = epochs
    write_checkpoint_metadata(str(checkpoint), final_meta)
    # Compare disk reconstruction against logits captured before the selected save.
    rebuilt = _restore(checkpoint, final_meta, device)
    with torch.no_grad():
        diff = float((rebuilt(check_inputs) - selected_reference).abs().max())
    if not np.isfinite(diff) or diff > 1e-5:
        raise ValueError(f'Checkpoint reload mismatch: {diff}')
    if val_loader:
        metrics, predictions = _predict(rebuilt, val_loader, device, names)
        _write_metrics(out, metrics, predictions, prefix='validation_')
    _sync(device)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {**identity, 'status': 'complete', 'phase': cfg.phase, 'dataset': manifest['dataset'],
              'counts': metadata['data']['counts'], 'best_epoch': best_epoch,
              'best_val_oa_percent': best_score if val_loader else None, 'final_epoch': epochs,
              'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256_file(str(checkpoint)),
              'reload_max_logit_diff': diff, 'graph_capture': graph_capture,
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'hardware': hardware, 'elapsed_seconds': time.perf_counter() - start,
              'main_process_peak_rss_bytes': rss if sys.platform == 'darwin' else rss * 1024,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
              'memory_scope': 'entire run including checkpoint reconstruction', 'refit_source': source_identity}
    write_json(out / 'run.json', report, replace=True)
    (out / 'run.exit').write_text('exit_code=0\n')
    print(f'SCENE {cfg.phase.upper()} COMPLETE: {out}', flush=True)
    return report


def evaluate_checkpoint(checkpoint, manifest_path, data_root, output, *, split='test',
                        device='cuda', batch_size=32, workers=2):
    with _run_output(output, 'evaluate') as (out, _start):
        if split not in ('val', 'test') or batch_size < 1 or workers < 0:
            raise ValueError('Invalid evaluation arguments')
        manifest = load_manifest(manifest_path, data_root)
        meta = _checkpoint_metadata(checkpoint, manifest_path, manifest)
        source_dir = Path(checkpoint).resolve().parent
        validate_source_run(json.loads((source_dir / 'run.json').read_text()), meta,
                            (source_dir / 'run.exit').read_text())
        code = _code_identity()
        if code['files_sha256'] != meta.get('code', {}).get('files_sha256'):
            raise ValueError('Evaluation code identity changed from checkpoint; use the recorded source version')
        phase = meta['execution']['phase']
        if phase == 'smoke' and split == 'test':
            raise ValueError('smoke checkpoint cannot evaluate test')
        if phase == 'refit' and split == 'val':
            raise ValueError('Refit already trained on val; it is not an independent validation split')
        if phase not in ('smoke', 'calibrate', 'refit'):
            raise ValueError('Unknown checkpoint phase')
        selected = select_records(manifest, split, per_class=meta['execution']['val_per_class'] if phase == 'smoke' else None)
        cfg = RunConfig(meta['model'], manifest_path, data_root, output, batch_size=batch_size,
                        workers=workers, seed=meta['training']['seed'])
        actual_device = _device(device)
        evaluation_identity = {'code': code, 'hardware': _hardware(actual_device),
                               'batch_size': batch_size, 'workers': workers, 'pin_memory': False,
                               'precision': 'float32_no_amp', 'tf32': False, 'command_argv': sys.argv}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model = _restore(checkpoint, meta, actual_device)
        metrics, predictions = _predict(model, _loader(data_root, selected, cfg, False), actual_device, manifest['class_names'])
        metrics.update(dataset=manifest['dataset'], split=split, checkpoint_sha256=meta['checkpoint_sha256'],
                       manifest_sha256=sha256_file(manifest_path), training_phase=phase,
                       fit_scope=meta['data']['fit_scope'], class_names=manifest['class_names'],
                       evaluation=evaluation_identity)
        _write_metrics(out, metrics, predictions)
        write_json(out / 'run.json', {'status': 'complete', 'split': split, 'samples': metrics['samples'],
                                     'checkpoint_sha256': meta['checkpoint_sha256'],
                                     'evaluation': evaluation_identity})
        (out / 'run.exit').write_text('exit_code=0\n')
        print(f'SCENE EVALUATION COMPLETE: {out}; OA={metrics["oa_percent"]:.4f}%', flush=True)
        return metrics
