"""Four disposable FP32 optimizer updates; compute timing, no epoch or metrics."""
from __future__ import annotations

import math
import statistics
import time

import torch
from torch import nn

from .nwpu_engine import _batch, _logits, _sync
from .smoke import gradient_report, _group

OPTIMIZER = dict(name='Adam', lr=1e-5, weight_decay=5e-5,
                 betas=[0.9, 0.999], eps=1e-8, amsgrad=False,
                 foreach=None, fused=None,
                 scope='paper lr/weight_decay; betas/eps/implementation defaults explicit engineering choices')
COMPUTE_SCOPE = ('CUDA-synchronized forward, CE, backward, finite-gradient check and Adam update on '
                 'already decoded/device-resident fixed16 RGB224 inputs; excludes image IO, preprocessing, '
                 'transfer, initialization, callbacks and first-step diagnostics; no epoch/loader throughput')
MEMORY_SCOPE = ('peak from model/input device placement through four updates and parameter/gradient '
                'diagnostics, including Adam states, gradients, activations and diagnostic temporaries; '
                'one model, no checkpoint reload; no full-dataset loader or augmentation')


def _finite_parameters(model):
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def run_batch_probe(model, images, labels, *, device, batch_size=16,
                    warmup_steps=1, measured_steps=3, on_progress=None):
    """Bounded capacity check. CPU is supported only for numerical unit fixtures."""
    if (type(batch_size) is not int or batch_size != 16
            or type(warmup_steps) is not int or warmup_steps != 1
            or type(measured_steps) is not int or measured_steps != 3):
        raise ValueError('Capacity probe is fixed to batch16, warmup1, measured3')
    parameters = list(model.parameters())
    if not parameters or any(not p.requires_grad or p.dtype != torch.float32 or p.is_meta for p in parameters):
        raise ValueError('Entire actual model must be trainable float32')
    images, labels = _batch(images, labels, 45, device)
    if len(images) != batch_size:
        raise ValueError('Exactly sixteen input images required')
    model.to(device); model.train()
    if not _finite_parameters(model):
        raise ValueError('Initial model parameters nonfinite')
    optimizer = torch.optim.Adam(model.parameters(), lr=OPTIMIZER['lr'],
        weight_decay=OPTIMIZER['weight_decay'], betas=tuple(OPTIMIZER['betas']),
        eps=OPTIMIZER['eps'], amsgrad=False, foreach=None, fused=None)
    is_cuda = torch.device(device).type == 'cuda'
    _sync(device)
    if is_cuda: torch.cuda.reset_peak_memory_stats(device)
    measured = []; warmup_seconds = None; first_gradients = None; updates = None
    for index in range(warmup_steps + measured_steps):
        phase = 'warmup' if index < warmup_steps else 'measured'
        _sync(device); started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = _logits(model, images, 45)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss).item(): raise ValueError('Nonfinite probe loss')
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        if not math.isfinite(float(norm)): raise ValueError('Nonfinite probe gradients')
        if index == 0:
            first_gradients = gradient_report(model)
            if not first_gradients['passed']:
                raise ValueError('First probe gradients missing, zero or nonfinite')
            probes = {}
            for name, parameter in model.named_parameters():
                group = _group(name)
                if group not in probes and parameter.grad is not None and parameter.grad.count_nonzero().item():
                    probes[group] = (name, parameter.detach().clone())
        optimizer.step()
        _sync(device)
        elapsed = time.perf_counter() - started
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('Invalid probe timer')
        if phase == 'warmup': warmup_seconds = elapsed
        else: measured.append(elapsed)
        event = dict(stage='first_optimizer_step' if index == 0 else 'measurement',
            optimizer_step_completed=True,optimizer_step_succeeded=True,optimizer_steps=index+1,
            step_index=index+1,phase=phase,training_started=True,formal_training_started=False,
            completed=False,batch_size=16,samples_seen=(index+1)*16,elapsed_seconds=elapsed,
            loss=float(loss.detach()))
        # A post-update diagnostic may fail after Adam really changed parameters.
        # Preserve that observed step, rather than falsely reporting zero updates.
        try:
            if index == 0:
                named = dict(model.named_parameters())
                updates = {group:not torch.equal(value,named[name]) for group,(name,value) in probes.items()}
                del probes
                event.update(gradients=first_gradients,updated_groups=updates)
                if set(updates) != set(first_gradients['groups']) or not all(updates.values()):
                    raise ValueError('First probe step did not update every gradient group')
                if not _finite_parameters(model): raise ValueError('Updated probe parameters nonfinite')
            event['post_update_checks_passed'] = True
        except BaseException:
            event['post_update_checks_passed'] = False
            if on_progress: on_progress(event)
            raise
        if on_progress: on_progress(event)
    if not _finite_parameters(model): raise ValueError('Final probe parameters nonfinite')
    _sync(device)
    peak_allocated = torch.cuda.max_memory_allocated(device) if is_cuda else None
    peak_reserved = torch.cuda.max_memory_reserved(device) if is_cuda else None
    optimizer.zero_grad(set_to_none=True)
    return dict(schema='hgtnet-batch16-capacity-probe-v1',completed=True,training_started=True,
        formal_training_started=False,device=str(torch.device(device)),
        input_scope='caller_provided_tensors',batch_size=16,
        source_identity_scope='engine does not infer real-data identity from device; wrapper binds verified records',
        unique_images=16,processed_images=64,optimizer_steps=4,warmup_steps=1,measured_steps=3,
        warmup_seconds=warmup_seconds,step_seconds=measured,
        median_step_seconds=statistics.median(measured),mean_step_seconds=statistics.mean(measured),
        compute_images_per_second=16/statistics.median(measured),compute_timing_scope=COMPUTE_SCOPE,
        warmup_timing_scope='includes initial Adam-state allocation and gradient diagnostics; excluded from median',
        memory_scope=MEMORY_SCOPE,cuda_peak_allocated_bytes=peak_allocated,cuda_peak_reserved_bytes=peak_reserved,
        first_step_gradients=first_gradients,first_step_updated_groups=updates,parameters_finite_after=True,
        optimizer=dict(OPTIMIZER),gradient_clipping=False,accuracy_metrics=None,checkpoint_saved=False,
        protocol_scope='fixed16 train inputs reused; disposable capacity/compute check, no formal epoch')
