"""Frozen AID campaign: calibrate, fresh refit, and one committed test pass.

Public entry point: ``run_job(settings: dict) -> dict``. Imports of Torch and
the established scene runner are delayed so refusal/recovery checks stay usable
in the standard-library delivery environment. This module does not alter the
original runner, models, historic runs, or experiment tables.
"""
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
import uuid

from .aid_campaign_state import (atomic_json, commit_transaction, establish_identity,
                                  file_digest, verified_latest)


AID_MANIFEST_SHA256 = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
# Verified from original AID Swin/GAViT calibration42 run.json, 2026-09-20.
AID_WEIGHTS_SHA256 = 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
ENGINE_VERSION = 'aid-campaign-20261110-v1'


def validate_settings(settings):
    cfg = {'mode': 'formal', 'device': 'cuda:0', 'workers': 2, 'batch_size': 32,
           'cpu_threads': 8, 'resume': False, **settings}
    if cfg['mode'] not in ('formal', 'smoke') or cfg.get('variant') not in ('soft', 'fixed', 'swin'):
        raise ValueError('Unknown campaign mode or variant')
    if type(cfg.get('seed')) is not int:
        raise ValueError('Integer seed required')
    if (type(cfg['workers']) is not int or cfg['workers'] < 0
            or type(cfg['batch_size']) is not int or cfg['batch_size'] < 1
            or type(cfg['cpu_threads']) is not int or cfg['cpu_threads'] < 1):
        raise ValueError('Invalid loader/thread settings')
    if cfg['device'] not in ('cpu', 'cuda:0'):
        raise ValueError('Device must be cpu fixture or bound cuda:0')
    if cfg['mode'] == 'formal':
        if cfg['device'] != 'cuda:0':
            raise ValueError('Formal campaign requires CUDA cuda:0')
        if cfg['batch_size'] != 32 or cfg['workers'] != 2:
            raise ValueError('Changed formal batch32/workers2 recipe')
        if ((cfg['variant'] == 'soft' and cfg['seed'] not in (42, 43, 44, 45))
                or (cfg['variant'] != 'soft' and cfg['seed'] != 45)):
            raise ValueError('Campaign refuses historical runs or unregistered seeds')
    if any(key in cfg for key in ('grouping', 'architecture', 'token_refinement', 'epochs', 'lr')):
        raise ValueError('Campaign does not accept architecture or recipe overrides')
    return cfg


def validate_manifest_identity(manifest, digest, *, formal):
    if formal and (digest != AID_MANIFEST_SHA256 or manifest.get('dataset') != 'AID'
                   or len(manifest.get('class_names', [])) != 30):
        raise ValueError('Frozen AID manifest identity mismatch')
    if formal:
        counts = dict(Counter(row['split'] for row in manifest.get('records', [])))
        if counts != {'train': 4000, 'val': 1000, 'test': 5000} or manifest.get('counts') != counts:
            raise ValueError('Frozen AID actual split counts mismatch')
    elif manifest.get('dataset') not in ('AID', 'synthetic'):
        raise ValueError('Smoke requires AID or local synthetic fixture')


def _verify_artifacts(out, hashes):
    if not hashes:
        raise ValueError('Missing completion artifact hash list')
    for name, expected in hashes.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe completion artifact path')
        path = Path(out) / relative
        if not path.is_file() or file_digest(path) != expected:
            raise ValueError(f'Completion artifact hash mismatch: {path}')


def completed_summary(out, identity):
    out = Path(out)
    path = out / 'completion.json'
    if not path.exists():
        return None
    marker = out / 'COMPLETED'
    # A crash between the atomic receipt and its seal is finalized by reusing
    # the already committed phases/evaluation; no training or test is replayed.
    if not marker.is_file():
        return None
    if marker.read_text() != file_digest(path) + '\n':
        raise ValueError('Completion receipt hash mismatch or missing COMPLETED marker')
    report = json.loads(path.read_text())
    if report.get('status') != 'complete' or report.get('identity') != identity:
        raise ValueError('Completed output identity mismatch')
    _verify_artifacts(out, report.get('artifact_sha256'))
    if (out / 'run.exit').read_text() != 'exit_code=0\n':
        raise ValueError('Completed output missing successful exit')
    return {**report['summary'], 'reused': True}


def resolve_gpu_uuid(property_uuid, expected, verified=None):
    actual = str(property_uuid) if property_uuid is not None else None
    if actual is None and verified:
        if (verified.get('device') != 'cuda:0' or verified.get('visible_cuda_devices') != 1
                or verified.get('verification_method') != 'cudaDeviceGetUuid'):
            raise ValueError('GPU UUID fallback requires independent CUDA runtime measurement')
        actual = verified.get('cuda_device_uuid')
    if expected and (not isinstance(actual, str) or actual.lower() != expected.lower()):
        raise ValueError('Bound GPU UUID mismatch')
    return actual


def _source_snapshot(provided, *, formal):
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ('utils.py', 'experiment_identity.py', 'grouping_identity.py',
             'soft_assignment_diagnostics.py', 'scene_classification/aid_campaign_engine.py',
             'scene_classification/aid_campaign_state.py', 'scene_classification/runner.py',
             'scene_classification/modeling.py', 'scene_classification/data.py',
             'scene_classification/metrics.py', 'scene_classification/refinement_audit.py')]
    files += sorted((root / 'models').glob('*.py'))
    if (root / 'tools/run_aid_campaign.py').is_file():
        files.append(root / 'tools/run_aid_campaign.py')
    actual = {p.relative_to(root).as_posix(): file_digest(p) for p in files}
    if provided is None:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True, text=True)
        provided = {'base_commit': result.stdout.strip() if result.returncode == 0 else None,
                    'delivery': 'source_snapshot_without_clean_git_assertion'}
    provided = json.loads(json.dumps(provided))
    commit = provided.get('base_commit', provided.get('full_commit'))
    if formal and not (isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit)):
        raise ValueError('Formal source identity requires a prior full base_commit and file hashes')
    for name, expected in provided.get('files_sha256', {}).items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or file_digest(root / relative) != expected:
            raise ValueError(f'Provided source hash mismatch: {name}')
    return {'provided': provided, 'engine_files_sha256': actual}


def _atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def _metric_files(folder, metrics, predictions, names):
    atomic_json(folder / 'metrics.json', metrics)
    _atomic_bytes(folder / 'predictions.csv', _csv_bytes(predictions))
    _atomic_bytes(folder / 'per_class.csv', _csv_bytes(metrics['per_class']))
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['true_class', *names])
    writer.writerows([name, *row] for name, row in zip(names, metrics['confusion_matrix']))
    _atomic_bytes(folder / 'confusion_matrix.csv', stream.getvalue().encode('utf-8'))


def _torch_save(torch, value, path):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _model_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def summarize_parameter_updates(rows, required_groups):
    groups = {}
    for row in rows:
        if (not row['gradient_finite'] or not row['parameter_finite']
                or not math.isfinite(row['max_abs_update'])):
            raise ValueError(f'Nonfinite/missing first-step gradient or parameter: {row["parameter"]}')
        group = groups.setdefault(row['group'], {'parameter_tensors': 0, 'parameter_count': 0,
            'nonzero_gradient_tensors': 0, 'changed_tensors': 0, 'max_abs_update': 0.0,
            'zero_delta_parameters': []})
        group['parameter_tensors'] += 1
        group['parameter_count'] += row['numel']
        group['nonzero_gradient_tensors'] += int(row['gradient_nonzero'])
        group['changed_tensors'] += int(row['max_abs_update'] > 0)
        group['max_abs_update'] = max(group['max_abs_update'], row['max_abs_update'])
        if row['max_abs_update'] == 0:
            group['zero_delta_parameters'].append(row['parameter'])
    for name in required_groups:
        if name not in groups:
            raise ValueError(f'Missing core parameter group: {name}')
        if groups[name]['nonzero_gradient_tensors'] == 0:
            raise ValueError(f'No nonzero gradient in core group: {name}')
        if groups[name]['changed_tensors'] == 0:
            raise ValueError(f'No actual parameter update in core group: {name}')
    return groups


def _parameter_group(name):
    for prefix, group in (
        ('region_grouping.assignment_projection.', 'soft_assignment'),
        ('region_grouping.', 'grouping_attention'),
        ('graph_reasoning.input_proj.', 'graph_input_projection'),
        ('graph_reasoning.gat_layers.0.', 'gat_layer_0'),
        ('graph_reasoning.gat_layers.1.', 'gat_layer_1'),
        ('graph_reasoning.norms.', 'graph_normalization'),
        ('feedback_proj.', 'feedback_projection'),
        ('backbone.', 'backbone'), ('classifier.', 'classifier')):
        if name.startswith(prefix):
            return group
    raise ValueError(f'Unexpected trainable parameter in frozen architecture: {name}')


def capture_rng(torch, np, train_loader, val_loader):
    # NumPy's ndarray is converted to primitives/tensors for weights_only loading.
    numpy_state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'train_generator': train_loader.generator.get_state(),
            'val_generator': val_loader.generator.get_state() if val_loader is not None else None}


def restore_rng(torch, np, saved, train_loader, val_loader):
    random.setstate(saved['python'])
    numpy_state = saved['numpy']
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    torch.set_rng_state(saved['torch'].cpu())
    if saved['cuda']:
        if not torch.cuda.is_available() or len(saved['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count changed on resume')
        torch.cuda.set_rng_state_all([item.cpu() for item in saved['cuda']])
    train_loader.generator.set_state(saved['train_generator'].cpu())
    if val_loader is not None:
        val_loader.generator.set_state(saved['val_generator'].cpu())


def _retire_old_training_states(phase, current):
    # Keep current + previous recovery states. All receipts and complete epoch
    # histories remain, but obsolete optimizer tensors must not fill the disk.
    phase = Path(phase)
    folders = list((phase / 'transactions').glob('epoch-*'))
    latest = json.loads((phase / 'latest.json').read_text())
    keep = {Path(current)}
    if latest.get('previous_transaction'):
        keep.add(phase / latest['previous_transaction'])
    for folder in folders:
        path = folder / 'train_state.pth'
        if folder not in keep and path.exists():
            path.unlink()
            atomic_json(folder / 'retired.json', {'reason': 'superseded_optimizer_state',
                                                'accepted_history_preserved': True})


def _phase(settings, identity, manifest, phase_name, epochs, out, *, expected_initial_hash=None):
    import numpy as np
    import torch
    from . import runner
    from .modeling import build_model, capture_graph_once
    from .data import select_records
    from utils import set_seed

    phase = out / phase_name
    done = phase / 'completion.json'
    if done.exists():
        report = json.loads(done.read_text())
        if report.get('identity') != identity or report.get('epochs') != epochs:
            raise ValueError('Phase completion identity mismatch')
        _verify_artifacts(phase, report['artifact_sha256'])
        verified_latest(phase)
        if expected_initial_hash and report['initial_model_sha256'] != expected_initial_hash:
            raise ValueError('Calibration/refit initial state hash mismatch')
        return report

    cfg = runner.RunConfig(identity['model'], settings['manifest'], settings['data_root'], str(phase),
        phase='refit' if phase_name == 'refit' else 'calibrate', device=settings['device'],
        seed=settings['seed'], batch_size=settings['batch_size'], workers=settings['workers'],
        pretrained_path=settings.get('weights'), grouping=identity['architecture'].get('grouping'))
    smoke = settings['mode'] == 'smoke'
    train_rows = select_records(manifest, 'pool' if phase_name == 'refit' else 'train',
                               per_class=2 if smoke else None)
    val_rows = (select_records(manifest, 'val', per_class=1 if smoke else None)
                if phase_name == 'calibration' else [])
    names = manifest['class_names']
    device = torch.device(settings['device'])
    set_seed(cfg.seed)
    model = build_model(cfg.model, len(names), cfg.pretrained_path,
                        architecture_config=identity['architecture']).to(device)
    initial_hash = _model_hash(model.state_dict())
    if expected_initial_hash and initial_hash != expected_initial_hash:
        raise ValueError('Fresh refit initial state differs from calibration')
    graph, hook = capture_graph_once(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4,
                                  betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0.0)
    train_loader = runner._loader(cfg.data_root, train_rows, cfg, True)
    val_loader = runner._loader(cfg.data_root, val_rows, cfg, False) if val_rows else None
    set_seed(cfg.seed)
    check_rows = val_rows[:2] if val_rows else train_rows[:2]
    check_inputs = next(iter(runner._loader(cfg.data_root, check_rows, cfg, False)))[0].to(device)
    accepted = verified_latest(phase)
    if accepted:
        folder, receipt = accepted
        state = torch.load(folder / 'train_state.pth', map_location='cpu', weights_only=True)
        if (state.get('format_version') != 2 or state.get('identity') != identity or state.get('phase') != phase_name
                or state.get('epoch') != receipt['epoch'] or state.get('initial_model_sha256') != initial_hash
                or [row['epoch'] for row in state.get('history', [])] != list(range(1, receipt['epoch'] + 1))):
            raise ValueError('Training state identity/history mismatch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        graph.update(state.get('graph_capture', {}))
        restore_rng(torch, np, state['rng'], train_loader, val_loader)
        print(f'RESUME {phase_name}: committed epoch {state["epoch"]}; partial epoch discarded', flush=True)
    else:
        if phase.exists() and (phase / 'completion.json').exists():
            raise ValueError('Phase completion has no accepted transaction')
        state = {'format_version': 2, 'identity': identity, 'phase': phase_name, 'epoch': 0,
                 'initial_model_sha256': initial_hash, 'history': [], 'best_score': -1.0,
                 'best_epoch': None, 'best_model': None, 'best_reference': None,
                 'first_update': None, 'graph_capture': {}}

    def commit():
        state.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), rng=capture_rng(torch, np, train_loader, val_loader),
                     graph_capture=dict(graph))
        def write(folder):
            _torch_save(torch, state, folder / 'train_state.pth')
            atomic_json(folder / 'history.json', state['history'])
            atomic_json(folder / 'epoch.json', {'epoch': state['epoch'], 'identity': identity,
                        'best_epoch': state['best_epoch'], 'best_score': state['best_score']})
        folder, receipt = commit_transaction(phase, state['epoch'], write)
        atomic_json(folder / 'receipt.json', receipt)
        _retire_old_training_states(phase, folder)
        return folder

    if not accepted:
        commit()  # Epoch zero makes an interrupted first epoch replayable.
    try:
        for epoch in range(state['epoch'] + 1, epochs + 1):
            runner._sync(device)
            started = time.perf_counter()
            model.train()
            loss_sum, seen = 0.0, 0
            for batch_index, (images, labels, _paths) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                if logits.shape != (len(labels), len(names)) or not bool(torch.isfinite(logits).all()):
                    raise ValueError('Invalid/nonfinite training logits')
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
                first = state['first_update'] is None
                if first:
                    before, parameter_audit = {}, []
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad:
                            continue
                        gradient = parameter.grad.detach().cpu() if parameter.grad is not None else None
                        before[name] = parameter.detach().cpu().clone()
                        parameter_audit.append({'parameter': name, 'group': _parameter_group(name),
                            'numel': parameter.numel(),
                            'gradient_finite': gradient is not None and bool(torch.isfinite(gradient).all()),
                            'gradient_nonzero': gradient is not None and bool((gradient != 0).any())})
                optimizer.step()
                if first:
                    parameters = dict(model.named_parameters())
                    for record in parameter_audit:
                        name = record['parameter']
                        after = parameters[name].detach().cpu()
                        record.update(parameter_finite=bool(torch.isfinite(after).all()),
                                      max_abs_update=float((after - before[name]).abs().max()))
                    required_groups = ['backbone', 'classifier']
                    if cfg.model == 'gavit':
                        required_groups += ['grouping_attention', 'graph_input_projection', 'gat_layer_0',
                                            'gat_layer_1', 'graph_normalization', 'feedback_projection']
                    if settings['variant'] == 'soft':
                        required_groups.append('soft_assignment')
                    group_audit = summarize_parameter_updates(parameter_audit, required_groups)
                    if cfg.model == 'gavit' and graph.get('edges_per_image') != 80:
                        raise ValueError('Actual 48+32 graph was not observed')
                    state['first_update'] = {'loss': float(loss.detach()), 'gradient_norm': float(norm),
                        'finite': True, 'batch_size': len(labels), 'graph_capture': dict(graph),
                        'parameter_groups': group_audit, 'parameters': parameter_audit,
                        'epoch': epoch, 'batch_index': batch_index,
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                        'memory_scope': 'this invocation through first optimizer step including audit',
                        'evidence_scope': 'first observed optimizer step; CPU copies do not alter training RNG'}
                    atomic_json(phase / 'first_update.json', state['first_update'])
                    del before, parameter_audit, parameters, gradient
                    print(f'TRAIN FIRST BATCH OK {phase_name}: loss={state["first_update"]["loss"]}; '
                          f'core_groups={group_audit}; graph={graph}', flush=True)
                loss_sum += float(loss.detach()) * len(labels)
                seen += len(labels)
            runner._sync(device)
            train_seconds = time.perf_counter() - started
            validation_start = time.perf_counter()
            metrics = runner._predict(model, val_loader, device, names)[0] if val_loader else None
            runner._sync(device)
            row = {'epoch': epoch, 'train_loss': loss_sum / seen, 'lr': optimizer.param_groups[0]['lr'],
                   'train_seconds': train_seconds, 'train_images_per_second': seen / train_seconds,
                   'validation_seconds': time.perf_counter() - validation_start, 'validation': metrics}
            model.eval()
            with torch.no_grad():
                state['last_reference'] = model(check_inputs).detach().cpu().clone()
            if metrics is not None and metrics['oa_percent'] > state['best_score']:
                state.update(best_score=metrics['oa_percent'], best_epoch=epoch,
                             best_model=_clone_state(model), best_reference=state['last_reference'].clone())
            state['history'].append(row)
            state['epoch'] = epoch
            scheduler.step()
            commit()
            print(f'EPOCH {phase_name} {epoch}/{epochs}; loss={row["train_loss"]:.6f}; '
                  f'val_OA={metrics["oa_percent"] if metrics else None}; train_s={train_seconds:.2f}', flush=True)
    finally:
        if hook:
            hook.remove()

    if state['epoch'] != epochs:
        raise ValueError('Phase did not reach its final accepted epoch')
    _torch_save(torch, model.state_dict(), phase / 'final.pth')
    selected_state = state['best_model'] if val_loader is not None else _clone_state(model)
    _torch_save(torch, selected_state, phase / 'best.pth')
    atomic_json(phase / 'history.json', state['history'])
    atomic_json(phase / 'first_update.json', state['first_update'])
    atomic_json(phase / 'selected_samples.json', {'train': [r['path'] for r in train_rows],
                'val': [r['path'] for r in val_rows], 'check': [r['path'] for r in check_rows]})
    # Reconstruct from the actual exported files, rather than trusting memory.
    del optimizer, scheduler
    rebuilt = build_model(cfg.model, len(names), architecture_config=identity['architecture']).to(device).eval()
    differences = {}
    for filename, reference in [('final.pth', state['last_reference']),
            ('best.pth', state['best_reference'] if val_loader is not None else state['last_reference'])]:
        rebuilt.load_state_dict(torch.load(phase / filename, map_location=device, weights_only=True), strict=True)
        with torch.no_grad():
            difference = float((rebuilt(check_inputs).cpu() - reference).abs().max())
        if not math.isfinite(difference) or difference > 1e-5:
            raise ValueError(f'{phase_name}/{filename} checkpoint reload mismatch: {difference}')
        differences[filename] = difference
    if val_loader is not None:
        metrics, predictions = runner._predict(rebuilt, val_loader, device, names)
        _metric_files(phase, metrics, predictions, names)
    folder, receipt = verified_latest(phase)
    metadata = {'entry_version': ENGINE_VERSION, 'identity': identity, 'phase': phase_name,
                'architecture': identity['architecture'], 'training': identity['training'],
                'class_names': names, 'initial_model_sha256': initial_hash,
                'selected_epoch': state['best_epoch'] if val_loader is not None else epochs,
                'final_epoch': epochs, 'reload_max_logit_diff': differences,
                'graph_capture': graph, 'first_update': state['first_update']}
    for filename in ('best.pth', 'final.pth'):
        atomic_json(phase / filename.replace('.pth', '.meta.json'),
                    {**metadata, 'checkpoint_sha256': file_digest(phase / filename)})
    artifacts = [p for p in phase.iterdir() if p.is_file() and p.name != 'completion.json']
    artifacts += [folder / filename for filename in receipt['files_sha256']]
    report = {**metadata, 'status': 'complete', 'epochs': epochs, 'best_epoch': state['best_epoch'],
              'best_val_oa_percent': state['best_score'] if val_loader is not None else None,
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'artifact_sha256': {p.relative_to(phase).as_posix(): file_digest(p) for p in artifacts}}
    atomic_json(done, report)
    del rebuilt, model, state
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return report


def _evaluate(settings, identity, manifest, out, refit):
    import torch
    from . import runner
    from .data import select_records
    from .modeling import build_model
    phase = out / 'evaluation'
    accepted = verified_latest(phase)
    checkpoint = out / 'refit/final.pth'
    checkpoint_hash = file_digest(checkpoint)
    if accepted:
        folder, _receipt = accepted
        source = json.loads((folder / 'source.json').read_text())
        if source != {'identity': identity, 'checkpoint_sha256': checkpoint_hash, 'split': 'test'}:
            raise ValueError('Committed test evaluation source mismatch')
    else:
        cfg = runner.RunConfig(identity['model'], settings['manifest'], settings['data_root'], str(phase),
            seed=settings['seed'], batch_size=settings['batch_size'], workers=settings['workers'])
        model = build_model(identity['model'], len(manifest['class_names']),
                            architecture_config=identity['architecture']).to(settings['device']).eval()
        model.load_state_dict(torch.load(checkpoint, map_location=settings['device'], weights_only=True), strict=True)
        metrics, predictions = runner._predict(model,
            runner._loader(settings['data_root'], select_records(manifest, 'test'), cfg, False),
            torch.device(settings['device']), manifest['class_names'])
        metrics.update(dataset='AID', split='test', selected_epoch=refit['selected_epoch'],
                       checkpoint_sha256=checkpoint_hash, manifest_sha256=identity['manifest_sha256'],
                       class_names=manifest['class_names'], fit_scope='train_plus_val',
                       selection_scope='internal_validation_only', test_used_for_selection=False)
        def write(folder):
            _metric_files(folder, metrics, predictions, manifest['class_names'])
            atomic_json(folder / 'source.json', {'identity': identity,
                        'checkpoint_sha256': checkpoint_hash, 'split': 'test'})
        folder, _receipt = commit_transaction(phase, 1, write)
        del model
    for name in ('metrics.json', 'predictions.csv', 'per_class.csv', 'confusion_matrix.csv'):
        _atomic_bytes(out / name, (folder / name).read_bytes())
    return json.loads((folder / 'metrics.json').read_text())


def run_job(settings: dict) -> dict:
    """Run/recover one registered job; completed jobs are verified and reused.

    Required: variant, seed, manifest, data_root, weights, output. Smoke permits
    ``weights=None`` with a synthetic local fixture. Formal always verifies the
    original pretrained file, frozen AID manifest, batch32 and workers2.
    """
    settings = validate_settings(settings)
    for name in ('manifest', 'data_root', 'output'):
        settings[name] = str(Path(settings[name]).resolve())
    if settings.get('weights'):
        settings['weights'] = str(Path(settings['weights']).resolve())
    formal = settings['mode'] == 'formal'
    weights_hash = file_digest(settings['weights']) if settings.get('weights') else None
    expected = settings.get('expected_weights_sha256', AID_WEIGHTS_SHA256 if formal else weights_hash)
    if (formal and expected != AID_WEIGHTS_SHA256) or weights_hash != expected:
        raise ValueError('Frozen pretrained weights hash mismatch')
    source = _source_snapshot(settings.get('source_identity'), formal=formal)
    from .data import load_manifest
    manifest = load_manifest(settings['manifest'], settings['data_root'])
    manifest_hash = file_digest(settings['manifest'])
    validate_manifest_identity(manifest, manifest_hash, formal=formal)
    import torch
    from . import runner
    from .modeling import architecture
    torch.set_num_threads(settings['cpu_threads'])
    device = runner._device(settings['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hardware = runner._hardware(device)
    if device.type == 'cuda':
        if torch.cuda.device_count() != 1:
            raise ValueError('Campaign requires exactly one visible bound GPU')
        actual_uuid = getattr(torch.cuda.get_device_properties(device), 'uuid', None)
        if actual_uuid is None:
            # CUDA 12.1 / Torch 2.2 properties may omit UUID. Repeat the
            # entrypoint's libcudart measurement here rather than trust NVML.
            from tools.run_aid_campaign import _cuda_uuid
            actual_uuid = _cuda_uuid()
        hardware['gpu_uuid'] = resolve_gpu_uuid(actual_uuid, settings.get('gpu_uuid'),
                                               settings.get('verified_gpu_identity'))
    model_name = 'swin' if settings['variant'] == 'swin' else 'gavit'
    grouping = {'swin': None, 'fixed': 'attentive_spatial', 'soft': 'soft_prototype'}[settings['variant']]
    cfg = runner.RunConfig(model_name, settings['manifest'], settings['data_root'], settings['output'],
        seed=settings['seed'], batch_size=settings['batch_size'], workers=settings['workers'], grouping=grouping)
    identity = {'engine_version': ENGINE_VERSION, 'mode': settings['mode'], 'variant': settings['variant'],
                'model': model_name, 'seed': settings['seed'], 'manifest_sha256': manifest_hash,
                'pretrained_sha256': weights_hash, 'architecture': architecture(model_name,
                    len(manifest['class_names']), grouping=grouping), 'training': runner.training_identity(cfg),
                'source': source, 'class_names': manifest['class_names'],
                'paths': {key: settings.get(key) for key in ('manifest', 'data_root', 'weights', 'output')},
                'device': settings['device'], 'expected_gpu_uuid': settings.get('gpu_uuid'),
                'verified_gpu_identity': settings.get('verified_gpu_identity'),
                'cpu_threads': settings['cpu_threads'], 'runtime': hardware}
    out = Path(settings['output'])
    establish_identity(out, identity, resume=settings['resume'])
    import fcntl
    with (out / '.job.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process owns this campaign job') from exc
        reused = completed_summary(out, identity)
        if reused:
            return reused
        attempts = out / 'attempts'
        attempts.mkdir(exist_ok=True)
        attempt = attempts / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex)
        attempt.mkdir()
        started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with (attempt / 'run.log').open('x') as log:
            with redirect_stdout(runner._Tee(sys.stdout, log)), redirect_stderr(runner._Tee(sys.stderr, log)):
                atomic_json(attempt / 'start.json', {'identity': identity, 'resume': settings['resume'],
                                                    'pid': os.getpid()})
                try:
                    calibration = _phase(settings, identity, manifest, 'calibration', 30 if formal else 1, out)
                    selected_epoch = calibration['best_epoch']
                    if type(selected_epoch) is not int or not 1 <= selected_epoch <= (30 if formal else 1):
                        raise ValueError('Missing earliest strict-best calibration epoch')
                    atomic_json(out / 'selection.json', {'selected_epoch': selected_epoch,
                        'criterion': 'earliest_strict_best_internal_validation_OA',
                        'test_used_for_selection': False, 'scheduler_t_max': 30,
                        'calibration_completion_sha256': file_digest(out / 'calibration/completion.json')})
                    refit = _phase(settings, identity, manifest, 'refit', selected_epoch, out,
                                   expected_initial_hash=calibration['initial_model_sha256'])
                    if formal:
                        metrics = _evaluate(settings, identity, manifest, out, refit)
                    else:
                        metrics = json.loads((out / 'calibration/metrics.json').read_text())
                    summary = {'status': 'complete', 'variant': settings['variant'], 'seed': settings['seed'],
                        'mode': settings['mode'], 'selected_epoch': selected_epoch, 'output': str(out),
                        'oa_percent': metrics['oa_percent'], 'macro_f1_percent': metrics['macro_f1_percent'],
                        'samples': metrics['samples'], 'test_evaluated': formal, 'reused': False,
                        'initial_model_sha256': calibration['initial_model_sha256'],
                        'refit_initial_model_sha256': refit['initial_model_sha256']}
                    atomic_json(attempt / 'result.json', {**summary,
                        'attempt_elapsed_seconds': time.perf_counter() - started,
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                        'resource_scope': 'this invocation only; earlier attempts retained'})
                    _atomic_bytes(out / 'run.exit', b'exit_code=0\n')
                    artifacts = [out / 'job_identity.json', out / 'selection.json', out / 'run.exit']
                    for phase in ('calibration', 'refit'):
                        report = calibration if phase == 'calibration' else refit
                        artifacts += [out / phase / name for name in report['artifact_sha256']]
                        artifacts.append(out / phase / 'completion.json')
                    if formal:
                        folder, receipt = verified_latest(out / 'evaluation')
                        artifacts += [out / name for name in ('metrics.json', 'predictions.csv', 'per_class.csv', 'confusion_matrix.csv')]
                        artifacts += [folder / name for name in receipt['files_sha256']]
                        artifacts.append(out / 'evaluation/latest.json')
                    atomic_json(out / 'completion.json', {'status': 'complete', 'identity': identity,
                        'summary': summary, 'artifact_sha256': {
                            path.relative_to(out).as_posix(): file_digest(path) for path in artifacts}})
                    _atomic_bytes(out / 'COMPLETED', (file_digest(out / 'completion.json') + '\n').encode())
                    print(f'AID CAMPAIGN JOB COMPLETE: {summary}', flush=True)
                    return summary
                except BaseException as exc:
                    traceback.print_exc()
                    atomic_json(attempt / 'failure.json', {'status': 'failed',
                        'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.perf_counter() - started})
                    _atomic_bytes(out / 'run.exit', b'exit_code=1\n')
                    raise
