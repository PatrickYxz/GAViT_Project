"""Synthetic CPU engineering checks, deliberately separate from dataset training."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

import torch
from torch import nn
import torchvision

from .checkpoint import load_checkpoint, save_checkpoint, source_identity
from .config import load_config
from .model import HGTNet, parameter_counts
from .pretrained import load_pretrained_file


def prepare_output(path):
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f'Output already exists: {path}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def _group(name):
    if name.startswith('patch_embed.'):
        return 'patch_embed'
    if name.startswith('stages.'):
        return 'dual_attention' if '.da.' in name else 'transformer'
    if name.startswith('graph_'):
        return 'graph'
    if name.startswith('head.'):
        return 'head'
    raise ValueError(f'Unclassified parameter: {name}')


def gradient_report(model):
    missing, nonfinite, groups = [], [], {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups.setdefault(_group(name), {'tensors': 0, 'squared_l2': 0.0})
        group['tensors'] += 1
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all().item():
            nonfinite.append(name)
        else:
            group['squared_l2'] += float(parameter.grad.detach().double().square().sum())
    for group in groups.values():
        group['l2'] = math.sqrt(group.pop('squared_l2'))
    return {'passed': bool(groups) and not missing and not nonfinite and all(g['l2'] > 0 for g in groups.values()),
            'missing': missing, 'nonfinite': nonfinite, 'groups': groups}


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _runtime(threads):
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    return {'python': sys.version, 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32',
            'cpu_threads': torch.get_num_threads(), 'deterministic_algorithms': True, 'amp': False,
            'pid': os.getpid()}


def _validate_trace(trace, batch, classes):
    expected = {'input': [batch, 3, 224, 224], 'patch': [batch, 128, 56, 56],
                'stage_outputs': [[batch, 256, 28, 28], [batch, 512, 14, 14], [batch, 512, 7, 7]],
                'graph_outputs': [[batch, 256, 7, 7], [batch, 512, 5, 5]],
                'fusion': [batch, 512], 'logits': [batch, classes]}
    if any(trace.get(key) != value for key, value in expected.items()):
        raise ValueError('Actual full-model shape trace differs from the candidate')
    if [len(g) for g in trace['graph_indices']] != [6, 2]:
        raise ValueError('Graph depth trace differs from the candidate')
    for records, nodes in zip(trace['graph_indices'], [49, 25]):
        for record in records:
            if record['indices_shape'] != [batch, nodes, 20] or record['self_edges'] != 0:
                raise ValueError('Actual kNN shape or self exclusion failed')


def run_smoke(config_path, output, *, weights=None, weights_sha256=None, cpu_threads=1):
    """Run full B2 forward/backward/update and save a fresh-process test fixture.

    Completion remains false until a *different* process executes verify_reload.
    No data loader, dataset split, epoch selection or accuracy is involved.
    """
    if (weights is None) != (weights_sha256 is None):
        raise ValueError('Supply both candidate weights and complete SHA256, or neither')
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    cfg = load_config(config_path)
    folder = prepare_output(output)
    started = time.perf_counter()
    runtime = _runtime(cpu_threads)
    sources_before = source_identity()
    torch.manual_seed(cfg['seed'])
    model = HGTNet(cfg)
    pretraining = None
    if weights is not None:
        pretraining = load_pretrained_file(model, weights, weights_sha256)
        model.pretraining_loaded = True
        if not pretraining['source_identity_verified'] or pretraining['mode'] != 'applied':
            raise ValueError('Candidate loader did not verify actual source values and identity')
        _write(folder / 'pretraining.json', pretraining)
    print('HGTNET_LOCAL_MODEL', json.dumps(parameter_counts(model)), flush=True)
    image = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(cfg['seed']))
    targets = torch.tensor([0, 1], dtype=torch.long)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    compute_started = time.perf_counter()
    logits, trace2 = model.forward_with_trace(image)
    _validate_trace(trace2, 2, cfg['num_classes'])
    loss = nn.functional.cross_entropy(logits, targets)
    if not torch.isfinite(loss).item():
        raise ValueError('Synthetic loss is nonfinite')
    loss.backward()
    gradients = gradient_report(model)
    if not gradients['passed']:
        _write(folder / 'failed_gradients.json', gradients)
        raise ValueError('Not all trainable parameters/groups have finite, nonzero group gradients')
    representatives = {}
    for name, parameter in model.named_parameters():
        group = _group(name)
        if group not in representatives and parameter.grad is not None and parameter.grad.count_nonzero().item():
            representatives[group] = (name, parameter.detach().clone())
    optimizer.step()
    updated = {}
    parameters = dict(model.named_parameters())
    for group, (name, previous) in representatives.items():
        updated[group] = {'parameter': name, 'changed': not torch.equal(previous, parameters[name]),
                          'max_abs_change': float((previous - parameters[name].detach()).abs().max())}
    if len(updated) != 5 or not all(record['changed'] for record in updated.values()):
        raise ValueError('Optimizer did not update every major architectural group')
    if not all(torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError('Updated parameters are nonfinite')
    train_seconds = time.perf_counter() - compute_started
    optimizer.zero_grad(set_to_none=True)
    del optimizer, logits, representatives
    model.eval()
    with torch.no_grad():
        expected1, trace1 = model.forward_with_trace(image[:1])
        expected2 = model(image)
    _validate_trace(trace1, 1, cfg['num_classes'])
    if source_identity() != sources_before:
        raise ValueError('Implementation changed during smoke')
    checkpoint = save_checkpoint(model, folder / 'model.pt', pretraining=pretraining)
    fixture_path = folder / 'inference_fixture.pt'
    with fixture_path.open('xb') as stream:
        torch.save({'image': image, 'expected1': expected1, 'expected2': expected2}, stream)
    record = {'schema': 'hgtnet-local-smoke-v1', 'implementation': model.implementation,
              'completed': False, 'checkpoint_saved': True, 'fresh_process_reload_pending': True,
              'scope': 'synthetic_cpu_engineering_only', 'formal_training_started': False,
              'accuracy_metrics': None, 'synthetic_samples': 2, 'seed': cfg['seed'],
              'optimizer': {'kind': 'Adam', 'lr': 1e-4, 'steps': 1},
              'config': cfg, 'config_sha256': model.config_sha256, 'runtime': runtime,
              'source_identity': sources_before, 'parameters': parameter_counts(model),
              'pretraining_loaded': model.pretraining_loaded,
              'pretraining_source': pretraining['source'] if pretraining else None,
              'loss': float(loss.detach()), 'gradients': gradients, 'updated_groups': updated,
              'batch1_trace': trace1, 'batch2_trace': trace2, 'checkpoint': checkpoint,
              'fixture_sha256': _hash(fixture_path), 'train_step_seconds': train_seconds,
              'wall_seconds': time.perf_counter() - started,
              'command': list(sys.argv)}
    _write(folder / 'training_summary.json', record)
    print('HGTNET_LOCAL_CHECKPOINT_SAVED', str(folder / 'training_summary.json'), flush=True)
    return record


def verify_reload(output):
    """Verify saved B1/B2 outputs in a process distinct from run_smoke."""
    folder = Path(output).absolute()
    if (folder / 'summary.json').exists():
        raise FileExistsError('Completed summary already exists')
    record = json.loads((folder / 'training_summary.json').read_text())
    if record['schema'] != 'hgtnet-local-smoke-v1' or record['completed'] is not False:
        raise ValueError('Expected an unfinished local engineering smoke')
    if os.getpid() == record['runtime']['pid']:
        raise ValueError('Reload must run in a fresh process')
    if source_identity() != record['source_identity']:
        raise ValueError('Implementation changed before fresh reload')
    runtime = _runtime(record['runtime']['cpu_threads'])
    model = HGTNet(record['config'])
    checkpoint = load_checkpoint(model, folder / 'model.pt', expected_sha256=record['checkpoint']['sha256'])
    fixture_path = folder / 'inference_fixture.pt'
    with fixture_path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != record['fixture_sha256']:
            raise ValueError('Inference fixture SHA256 mismatch')
        stream.seek(0)
        fixture = torch.load(stream, map_location='cpu', weights_only=True)
    if set(fixture) != {'image', 'expected1', 'expected2'}:
        raise ValueError('Unexpected inference fixture')
    model.eval()
    with torch.no_grad():
        actual1 = model(fixture['image'][:1])
        actual2 = model(fixture['image'])
    errors = {'batch1': float((actual1 - fixture['expected1']).abs().max()),
              'batch2': float((actual2 - fixture['expected2']).abs().max())}
    if not all(math.isfinite(error) and error <= 1e-6 for error in errors.values()):
        raise ValueError('Fresh reload differs from saved inference outputs')
    if model.config_sha256 != record['config_sha256']:
        raise ValueError('Reload config identity differs')
    record.update(completed=True, fresh_process_reload_pending=False,
                  reload={'runtime': runtime, 'max_abs_errors': errors, 'tolerance': 1e-6,
                          'checkpoint_mode': checkpoint['mode'],
                          'training_summary_sha256': _hash(folder / 'training_summary.json')})
    _write(folder / 'summary.json', record)
    print('HGTNET_LOCAL_COMPLETE', str(folder / 'summary.json'), flush=True)
    return record
