"""One disposable real-image epoch; no formal training, test or resume path."""
from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import nn
from torch.utils.data import SequentialSampler

from .config import config_sha256, validate_config
from .smoke import gradient_report, _group

PROTOCOL = {'seed': 42, 'epochs': 1, 'batch_size': 2, 'optimizer': 'Adam', 'lr': 1e-4,
            'loss': 'CrossEntropyLoss', 'augmentation': False, 'amp': False, 'tf32': False,
            'preprocessing': 'RGB_resize224_bilinear_ImageNet_normalize'}


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def _write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write('\n')


def _binding(value):
    value = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    if value.get('schema') != 'hgtnet-nwpu-smoke-binding-v1' or value.get('input_scope') not in ('real_nwpu', 'cpu_fixture'):
        raise ValueError('Explicit real NWPU or CPU fixture binding required')
    cfg = validate_config(value['config'])
    if value['config_sha256'] != config_sha256(cfg) or cfg['seed'] != 42 or cfg['num_classes'] != 45:
        raise ValueError('Candidate config identity changed')
    counts = value['counts']
    if set(counts) != {'train', 'val', 'test'} or any(type(n) is not int for n in counts.values()):
        raise ValueError('Explicit integer split counts required')
    if counts['test'] != 0 or counts['train'] < 2 or counts['val'] < 1:
        raise ValueError('Only nonempty train/validation smoke, test0, is supported')
    if len(value['class_names']) != 45 or len(set(value['class_names'])) != 45:
        raise ValueError('45 distinct class names required')
    if len(value['val_paths']) != counts['val'] or len(value['val_labels']) != counts['val']:
        raise ValueError('Ordered validation identities required')
    if value['input_scope'] == 'real_nwpu':
        if counts != {'train': 360, 'val': 90, 'test': 0} or value['runtime']['device'] != 'cuda:0':
            raise ValueError('Real NWPU smoke requires fixed360/90 on selected CUDA device')
    elif value['runtime']['device'] != 'cpu':
        raise ValueError('CPU fixture cannot claim CUDA execution')
    return value


def save_smoke_checkpoint(model, path, *, binding):
    binding = _binding(binding)
    state = model.state_dict()
    if any(v.is_meta or v.layout != torch.strided or not torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError('Checkpoint must contain actual finite dense tensors')
    payload = {'schema': 'hgtnet-real-image-smoke-checkpoint-v1', 'binding': binding,
               'protocol': PROTOCOL, 'formal_training_started': False,
               'synthetic_inputs': binding['input_scope'] == 'cpu_fixture',
               'best_epoch': 1, 'final_epoch': 1,
               'state_dict': {k: v.detach().cpu().clone() for k, v in state.items()}}
    path = Path(path)
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            torch.save(payload, stream)
    except BaseException:
        if created: path.unlink(missing_ok=True)
        raise
    return {'path': str(path.absolute()), 'sha256': sha256_file(path), 'bytes': path.stat().st_size,
            'best_epoch': 1, 'final_epoch': 1, 'synthetic_inputs': payload['synthetic_inputs']}


def reload_smoke_checkpoint(model, path, *, expected_sha256, binding):
    binding = _binding(binding)
    if not isinstance(expected_sha256, str) or not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('Complete checkpoint SHA256 required')
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Smoke checkpoint SHA256 mismatch')
    payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {'schema','binding','protocol','formal_training_started',
            'synthetic_inputs','best_epoch','final_epoch','state_dict'}:
        raise ValueError('Smoke checkpoint schema mismatch')
    if (payload['schema'] != 'hgtnet-real-image-smoke-checkpoint-v1' or payload['binding'] != binding
            or payload['protocol'] != PROTOCOL or payload['formal_training_started'] is not False
            or payload['synthetic_inputs'] is not (binding['input_scope'] == 'cpu_fixture')
            or type(payload['best_epoch']) is not int or payload['best_epoch'] != 1
            or type(payload['final_epoch']) is not int or payload['final_epoch'] != 1):
        raise ValueError('Smoke checkpoint metadata/identity mismatch')
    state, target = payload['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(target):
        raise ValueError('Smoke checkpoint keys differ')
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.is_meta or tensor.layout != torch.strided
                or tensor.shape != target[key].shape or tensor.dtype != target[key].dtype
                or not torch.isfinite(tensor).all().item()):
            raise ValueError('Smoke checkpoint tensor mismatch: ' + key)
    copied = {k: v.to(target[k].device).clone() for k, v in state.items()}
    model.load_state_dict(copied, strict=True)
    return {'loaded': True, 'sha256': expected_sha256, 'state_tensors': len(state)}


def _sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def _batch(images, labels, classes, device):
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224)
            or images.dtype != torch.float32 or not torch.isfinite(images).all().item()):
        raise ValueError('Images must be actual finite float32 RGB224 tensors')
    if (not isinstance(labels, torch.Tensor) or labels.dtype != torch.long or labels.shape != (len(images),)
            or len(images) < 1 or ((labels < 0) | (labels >= classes)).any().item()):
        raise ValueError('Labels must be one valid integer class per image')
    return images.to(device), labels.to(device)


def _logits(model, images, classes):
    result = model(images)
    if result.shape != (len(images), classes) or not torch.isfinite(result).all().item():
        raise ValueError('Model produced invalid logits')
    return result


def _metrics(labels, predictions, names):
    classes = len(names)
    cm = torch.bincount(torch.tensor(labels) * classes + torch.tensor(predictions), minlength=classes**2).reshape(classes, classes)
    per_class = []
    f1s = []
    for i, name in enumerate(names):
        tp = int(cm[i, i]); support = int(cm[i].sum()); predicted = int(cm[:, i].sum())
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        f1s.append(f1)
        per_class.append({'class': name, 'support': support, 'correct': tp,
                          'accuracy_percent': 100 * tp / support if support else 0.0, 'f1_percent': 100 * f1})
    return {'accuracy_percent': 100 * int(cm.diag().sum()) / len(labels),
            'macro_f1_percent': 100 * sum(f1s) / classes, 'per_class': per_class,
            'confusion_matrix': cm.tolist(), 'scope': 'smoke_validation_only'}


def run_epoch_smoke(model, train_loader, val_loader, output, *, binding, device, model_factory, on_progress=None):
    """Actually train all supplied samples once, validate and reload independently."""
    binding = _binding(binding)
    if str(torch.device(device)) != binding['runtime']['device']:
        raise ValueError('Actual engine device differs from binding')
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        if len(loader.dataset) != binding['counts'][split] or loader.batch_size != 2 or loader.drop_last:
            raise ValueError('Smoke loader count/batch/drop_last changed')
    if not isinstance(val_loader.sampler, SequentialSampler):
        raise ValueError('Validation must preserve manifest order')
    output = Path(output)
    for name in ['first_step.json','smoke_epoch1.pt','history.json','val_predictions.json']:
        if (output / name).exists() or (output / name).is_symlink(): raise FileExistsError(output / name)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); model.to(device); model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=PROTOCOL['lr'])
    seen = steps = correct = 0; loss_sum = 0.0; first_gradients = None
    _sync(device); train_started = time.perf_counter()
    for images, labels in train_loader:
        images, labels = _batch(images, labels, 45, device)
        optimizer.zero_grad(set_to_none=True)
        logits = _logits(model, images, 45)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss).item(): raise ValueError('Nonfinite training loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        if not math.isfinite(float(norm)): raise ValueError('Nonfinite training gradients')
        if steps == 0:
            first_gradients = gradient_report(model)
            if not first_gradients['passed']: raise ValueError('First batch gradients missing, zero or nonfinite')
            probes = {}
            for name, parameter in model.named_parameters():
                group = _group(name)
                if group not in probes and parameter.grad is not None and parameter.grad.count_nonzero().item():
                    probes[group] = (name, parameter.detach().clone())
        optimizer.step()
        steps += 1; seen += len(images); loss_sum += float(loss.detach()) * len(images)
        correct += int((logits.argmax(1) == labels).sum())
        if steps == 1:
            parameters = dict(model.named_parameters())
            updates = {group: not torch.equal(value, parameters[name]) for group, (name, value) in probes.items()}
            if set(updates) != set(first_gradients['groups']) or not all(updates.values()):
                raise ValueError('First optimizer step did not update every gradient group')
            del probes
            if not all(torch.isfinite(p).all().item() for p in model.parameters()): raise ValueError('Updated parameters are nonfinite')
            first = {'stage': 'first_optimizer_step', 'training_started': True, 'optimizer_step_succeeded': True,
                     'optimizer_steps': 1, 'samples_seen': seen, 'loss': float(loss.detach()),
                     'device': str(device), 'gradients': first_gradients,
                     'updated_groups': updates,
                     'elapsed_seconds': time.perf_counter() - train_started}
            _write(output / 'first_step.json', first)
            if on_progress: on_progress(first)
        elif steps % 30 == 0 and on_progress:
            on_progress({'stage': 'training', 'training_started': True, 'optimizer_steps': steps, 'samples_seen': seen})
    _sync(device); train_seconds = time.perf_counter() - train_started
    if seen != binding['counts']['train']: raise ValueError('Training did not consume all selected records')
    optimizer.zero_grad(set_to_none=True); del optimizer, loss, logits
    model.eval(); labels_all=[]; predictions=[]; val_loss=0.0; reference=None
    _sync(device); val_started=time.perf_counter()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = _batch(images, labels, 45, device)
            logits = _logits(model, images, 45)
            val_loss += float(nn.functional.cross_entropy(logits, labels)) * len(images)
            labels_all.extend(labels.cpu().tolist()); predictions.extend(logits.argmax(1).cpu().tolist())
            if reference is None: reference=(images.detach().cpu(), logits.detach().cpu())
    _sync(device); val_seconds=time.perf_counter()-val_started
    if labels_all != binding['val_labels']: raise ValueError('Validation order/labels differ from bound manifest records')
    if not all(torch.isfinite(p).all().item() for p in model.parameters()): raise ValueError('Final parameters nonfinite')
    metrics=_metrics(labels_all,predictions,binding['class_names'])
    checkpoint=save_smoke_checkpoint(model,output/'smoke_epoch1.pt',binding=binding)
    fresh=model_factory().to(device)
    reload_smoke_checkpoint(fresh,output/'smoke_epoch1.pt',expected_sha256=checkpoint['sha256'],binding=binding)
    fresh.eval()
    with torch.no_grad(): reloaded=_logits(fresh,reference[0].to(device),45).cpu()
    error=float((reloaded-reference[1]).abs().max())
    if not math.isfinite(error) or error>1e-6: raise ValueError('Independent smoke checkpoint reload differs')
    checkpoint.update(reload_max_abs_error=error,reload_tolerance=1e-6)
    history={'epoch':1,'train_loss':loss_sum/seen,'train_accuracy_percent':100*correct/seen,
             'val_loss':val_loss/len(labels_all),'train_seconds':train_seconds,'validation_seconds':val_seconds,
             'train_images_per_second':seen/train_seconds,'optimizer_steps':steps}
    _write(output/'history.json',[history])
    _write(output/'val_predictions.json',{'scope':'smoke_validation_only','paths':binding['val_paths'],
           'labels':labels_all,'predictions':predictions,'metrics':metrics})
    return {'schema':'hgtnet-nwpu-smoke-result-v1','completed':True,'training_started':True,
            'formal_training_started':False,'input_scope':binding['input_scope'],'counts':binding['counts'],
            'protocol':PROTOCOL,'best_epoch':1,'final_epoch':1,'optimizer_steps':steps,
            'val_metrics':metrics,'history':history,'checkpoint':checkpoint,'first_batch_gradients':first_gradients,
            'wall_seconds':time.perf_counter()-started}
