"""AID-only disposable epoch with durable update evidence and strict reload.

Importing this module is standard-library-only. Tensor dependencies are loaded
only by actual execution; no NWPU binding or checkpoint identity is accepted.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import time

from .config import config_sha256, validate_config


PROTOCOL = {'seed': 42, 'epochs': 1, 'batch_size': 2, 'optimizer': 'Adam', 'lr': 1e-4,
            'loss': 'CrossEntropyLoss', 'augmentation': False, 'amp': False, 'tf32': False,
            'preprocessing': 'RGB_resize224_bilinear_ImageNet_normalize'}
_GROUPS = {'patch_embed', 'transformer', 'dual_attention', 'graph', 'head'}
_CHECKPOINT_SCHEMA = 'hgtnet-aid-smoke-checkpoint-v1'
_ARTIFACTS = ('first_step.json', 'optimizer_progress.json', 'smoke_epoch1.pt',
              'history.json', 'val_predictions.json', '.progress.tmp')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _binding(value):
    try:
        value = json.loads(_json(value))
        if not isinstance(value, dict) or value.get('schema') != 'hgtnet-aid-smoke-binding-v1':
            raise ValueError('Explicit AID smoke binding required')
        if value.get('input_scope') not in ('real_aid', 'cpu_fixture'):
            raise ValueError('AID input_scope must be real_aid or explicit cpu_fixture')
        cfg = validate_config(value['config'])
        if cfg['num_classes'] != 30 or cfg['seed'] != 42 or value['config_sha256'] != config_sha256(cfg):
            raise ValueError('AID smoke requires bound 30-class seed42 config')
        counts = value['counts']
        if (not isinstance(counts, dict) or set(counts) != {'train', 'val', 'test'}
                or any(type(count) is not int for count in counts.values())
                or counts['train'] < 2 or counts['val'] < 1 or counts['test'] != 0):
            raise ValueError('AID smoke requires nonempty integer train/val counts and test0')
        names = value['class_names']
        if (not isinstance(names, list) or len(names) != 30
                or any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != 30):
            raise ValueError('AID smoke requires 30 distinct class names')
        for split in ('train', 'val'):
            if split == 'train' and 'train_paths' not in value and 'train_labels' not in value:
                continue
            paths, labels = value[split + '_paths'], value[split + '_labels']
            if (not isinstance(paths, list) or not isinstance(labels, list)
                    or len(paths) != counts[split] or len(labels) != counts[split]
                    or any(not isinstance(path, str) or not path or '\\' in path or '\x00' in path
                           or path.startswith('/') or len(PurePosixPath(path).parts) < 2
                           or '..' in PurePosixPath(path).parts or PurePosixPath(path).as_posix() != path
                           for path in paths)
                    or len(set(paths)) != len(paths)
                    or any(type(label) is not int or not 0 <= label < 30 for label in labels)):
                raise ValueError('Ordered valid AID split paths and labels required')
        if 'train_paths' in value and set(value['train_paths']) & set(value['val_paths']):
            raise ValueError('AID train and validation paths overlap')
        if value['input_scope'] == 'real_aid':
            if counts != {'train': 240, 'val': 60, 'test': 0} or value['runtime']['device'] != 'cuda:0':
                raise ValueError('Real AID smoke requires fixed240/60 and cuda:0')
        elif value['runtime']['device'] != 'cpu':
            raise ValueError('Explicit CPU fixture cannot claim CUDA execution')
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('AID smoke binding fields are invalid or incomplete') from exc
    return value


def _write(path, value):
    with Path(path).open('x') as stream:
        stream.write(_json(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def _progress(path, value):
    """Atomically replace only engine-owned progress; fsync before replacement."""
    path = Path(path)
    temporary = path.parent / '.progress.tmp'
    try:
        _write(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path):
    from .aid_data import inspector
    with inspector.open_regular_readonly(path) as stream:
        return inspector.hash_stream(stream)[0]


def save_smoke_checkpoint(model, path, *, binding):
    import torch
    binding = _binding(binding)
    state = model.state_dict()
    if not state or any(not isinstance(tensor, torch.Tensor) or tensor.is_meta
                        or tensor.layout != torch.strided or not torch.isfinite(tensor).all().item()
                        for tensor in state.values()):
        raise ValueError('AID checkpoint requires actual finite dense tensor state')
    payload = {'schema': _CHECKPOINT_SCHEMA, 'binding': binding, 'protocol': dict(PROTOCOL),
               'formal_training_started': False, 'synthetic_inputs': binding['input_scope'] == 'cpu_fixture',
               'best_epoch': 1, 'final_epoch': 1,
               'state_dict': {key: tensor.detach().cpu().clone() for key, tensor in state.items()}}
    path = Path(path)
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    return {'path': str(path.absolute()), 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
            'best_epoch': 1, 'final_epoch': 1, 'synthetic_inputs': payload['synthetic_inputs']}


def reload_smoke_checkpoint(model, path, *, expected_sha256, binding):
    import torch
    from .aid_data import inspector
    binding = _binding(binding)
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('AID checkpoint requires a complete expected SHA256')
    with inspector.open_regular_readonly(path) as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('AID checkpoint SHA256 mismatch')
    try:
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('AID checkpoint cannot be loaded as tensor-only data') from exc
    keys = {'schema', 'binding', 'protocol', 'formal_training_started', 'synthetic_inputs',
            'best_epoch', 'final_epoch', 'state_dict'}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError('AID checkpoint schema fields mismatch')
    try:
        metadata_matches = (payload['schema'] == _CHECKPOINT_SCHEMA
                            and _json(payload['binding']) == _json(binding)
                            and _json(payload['protocol']) == _json(PROTOCOL)
                            and payload['formal_training_started'] is False
                            and payload['synthetic_inputs'] is (binding['input_scope'] == 'cpu_fixture')
                            and type(payload['best_epoch']) is int and payload['best_epoch'] == 1
                            and type(payload['final_epoch']) is int and payload['final_epoch'] == 1)
    except (TypeError, ValueError) as exc:
        raise ValueError('AID checkpoint metadata is not finite JSON') from exc
    if not metadata_matches:
        raise ValueError('AID checkpoint metadata/identity mismatch')
    state, target = payload['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(target):
        raise ValueError('AID checkpoint state keys mismatch')
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.is_meta or tensor.layout != torch.strided
                or tensor.shape != target[key].shape or tensor.dtype != target[key].dtype
                or not torch.isfinite(tensor).all().item()):
            raise ValueError('AID checkpoint tensor shape/type/value mismatch: ' + key)
    # Complete validation precedes the first mutation of the destination model.
    model.load_state_dict({key: tensor.to(target[key].device).clone() for key, tensor in state.items()}, strict=True)
    return {'loaded': True, 'sha256': expected_sha256, 'state_tensors': len(state)}


def _loaders(train_loader, val_loader, binding):
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        if len(loader.dataset) != binding['counts'][split] or loader.batch_size != 2 or loader.drop_last:
            raise ValueError('AID smoke loader count/batch/drop_last differs')
        if split + '_paths' in binding and hasattr(loader.dataset, 'records'):
            rows = loader.dataset.records
            if ([row['path'] for row in rows] != binding[split + '_paths']
                    or [row['label'] for row in rows] != binding[split + '_labels']):
                raise ValueError('AID dataset ordered identity differs from binding')
    sampler = train_loader.sampler
    if (type(sampler) is not RandomSampler or sampler.replacement
            or sampler.num_samples != binding['counts']['train'] or sampler.generator is None
            or str(sampler.generator.device) != 'cpu'
            or not torch.equal(sampler.generator.get_state(), torch.Generator().manual_seed(42).get_state())):
        raise ValueError('AID training requires fresh reproducible seed42 shuffle without replacement')
    if type(val_loader.sampler) is not SequentialSampler:
        raise ValueError('AID validation must preserve sequential manifest order')


def run_epoch_smoke(model, train_loader, val_loader, output, *, binding, device, model_factory, on_progress=None):
    """Run one complete AID-only epoch and verify a fresh model numerically."""
    import torch
    from torch import nn
    from .nwpu_engine import _batch, _logits, _metrics, _sync
    from .smoke import gradient_report, _group

    binding = _binding(binding)
    if str(torch.device(device)) != binding['runtime']['device']:
        raise ValueError('Actual AID engine device differs from binding')
    _loaders(train_loader, val_loader, binding)
    output = Path(output)
    for name in _ARTIFACTS:
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError(output / name)
    output.mkdir(parents=True, exist_ok=True)
    steps = seen = correct = 0
    started = time.perf_counter()
    try:
        model.to(device)
        if (any(not parameter.requires_grad or parameter.is_meta or parameter.dtype != torch.float32
                or not torch.isfinite(parameter).all().item() for parameter in model.parameters())
                or {_group(name) for name, _ in model.named_parameters()} != _GROUPS):
            raise ValueError('AID smoke requires all five trainable finite float32 parameter groups')
        if hasattr(model, 'config_sha256') and model.config_sha256 != binding['config_sha256']:
            raise ValueError('AID model config identity differs from binding')
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['lr'])
        loss_sum = 0.0
        first_gradients = None
        _sync(device)
        train_started = time.perf_counter()
        for images, labels in train_loader:
            images, labels = _batch(images, labels, 30, device)
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, images, 30)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss).item():
                raise ValueError('AID training loss is nonfinite')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            if not math.isfinite(float(norm)):
                raise ValueError('AID gradients are nonfinite')
            if steps == 0:
                first_gradients = gradient_report(model)
                if not first_gradients['passed'] or set(first_gradients['groups']) != _GROUPS:
                    raise ValueError('All five AID first-batch gradient groups must be finite/nonzero')
                probes = {}
                for name, parameter in model.named_parameters():
                    group = _group(name)
                    if group not in probes and parameter.grad is not None and parameter.grad.count_nonzero().item():
                        probes[group] = (name, parameter.detach().clone())
            optimizer.step()
            steps += 1
            seen += len(images)
            progress = {'stage': 'training', 'training_started': True, 'optimizer_steps': steps, 'samples_seen': seen}
            # Record actual updates before diagnostic/callback/validation failures.
            _progress(output / 'optimizer_progress.json', progress)
            loss_sum += float(loss.detach()) * len(images)
            correct += int((logits.argmax(1) == labels).sum())
            if steps == 1:
                first = {**progress, 'stage': 'first_optimizer_step', 'optimizer_step_succeeded': True,
                         'diagnostic_passed': False, 'loss': float(loss.detach()), 'device': str(device),
                         'gradients': first_gradients, 'updated_groups': {},
                         'elapsed_seconds': time.perf_counter() - train_started}
                _write(output / 'first_step.json', first)
                try:
                    parameters = dict(model.named_parameters())
                    updates = {group: not torch.equal(previous, parameters[name])
                               for group, (name, previous) in probes.items()}
                    first['updated_groups'] = updates
                    if set(updates) != _GROUPS or not all(updates.values()):
                        raise ValueError('AID first optimizer step did not update all five groups')
                    if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
                        raise ValueError('AID updated parameters are nonfinite')
                    first['diagnostic_passed'] = True
                finally:
                    _progress(output / 'first_step.json', first)
                    if on_progress:
                        on_progress(first)
                del probes
            elif steps % 30 == 0 and on_progress:
                on_progress(progress)
        _sync(device)
        train_seconds = time.perf_counter() - train_started
        if seen != binding['counts']['train'] or steps != math.ceil(binding['counts']['train'] / 2):
            raise ValueError('AID epoch did not consume all selected training records exactly once')
        optimizer.zero_grad(set_to_none=True)
        del optimizer, loss, logits
        model.eval()
        labels_all, predictions, val_loss, reference = [], [], 0.0, None
        _sync(device)
        val_started = time.perf_counter()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = _batch(images, labels, 30, device)
                logits = _logits(model, images, 30)
                val_loss += float(nn.functional.cross_entropy(logits, labels)) * len(images)
                labels_all.extend(labels.cpu().tolist())
                predictions.extend(logits.argmax(1).cpu().tolist())
                if reference is None:
                    reference = (images.detach().cpu(), logits.detach().cpu())
        _sync(device)
        val_seconds = time.perf_counter() - val_started
        if labels_all != binding['val_labels']:
            raise ValueError('AID validation order/labels differ from bound manifest records')
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise ValueError('AID final parameters are nonfinite')
        metrics = _metrics(labels_all, predictions, binding['class_names'])
        checkpoint = save_smoke_checkpoint(model, output / 'smoke_epoch1.pt', binding=binding)
        fresh = model_factory().to(device)
        reload_smoke_checkpoint(fresh, output / 'smoke_epoch1.pt', expected_sha256=checkpoint['sha256'], binding=binding)
        fresh.eval()
        with torch.no_grad():
            reloaded = _logits(fresh, reference[0].to(device), 30).cpu()
        error = float((reloaded - reference[1]).abs().max())
        if not math.isfinite(error) or error > 1e-6:
            raise ValueError('Independent AID checkpoint reload exceeds tolerance')
        checkpoint.update(reload_max_abs_error=error, reload_tolerance=1e-6)
        history = {'epoch': 1, 'train_loss': loss_sum / seen, 'train_accuracy_percent': 100 * correct / seen,
                   'val_loss': val_loss / len(labels_all), 'train_seconds': train_seconds,
                   'validation_seconds': val_seconds, 'train_images_per_second': seen / train_seconds,
                   'optimizer_steps': steps}
        _write(output / 'history.json', [history])
        _write(output / 'val_predictions.json', {'scope': 'smoke_validation_only', 'paths': binding['val_paths'],
                                               'labels': labels_all, 'predictions': predictions, 'metrics': metrics})
        return {'schema': 'hgtnet-aid-smoke-result-v1', 'completed': True, 'training_started': True,
                'formal_training_started': False, 'input_scope': binding['input_scope'], 'counts': binding['counts'],
                'protocol': dict(PROTOCOL), 'best_epoch': 1, 'final_epoch': 1, 'optimizer_steps': steps,
                'val_metrics': metrics, 'history': history, 'checkpoint': checkpoint,
                'first_batch_gradients': first_gradients, 'wall_seconds': time.perf_counter() - started}
    except BaseException as exc:
        exc.training_started = steps > 0
        exc.optimizer_steps = steps
        exc.samples_seen = seen
        raise
