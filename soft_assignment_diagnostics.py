"""Fixed-input assignment probes; no test-set access or training RNG effects."""
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch


class SoftAssignmentProbe:
    def __init__(self, model, inputs, sample_ids, output_dir, split, *, resume=False):
        if split not in ('train', 'val', 'pool'):
            raise ValueError('Probe split must be train, val or pool; test is forbidden')
        if (getattr(model, 'grouping', None) != 'soft_prototype'
                or inputs.ndim != 4 or not 1 <= len(inputs) <= 8
                or len(inputs) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids)
                or not inputs.is_floating_point() or not torch.isfinite(inputs).all()):
            raise ValueError('Probe requires a soft model and 1..8 finite, identified images')
        self.model, self.split = model, split
        self.inputs = inputs.detach().cpu().clone()
        self.sample_ids = list(sample_ids)
        self.out = Path(output_dir)
        self.identity = {'split': split, 'sample_ids': self.sample_ids,
                         'input_sha256': hashlib.sha256(self.inputs.numpy().tobytes()).hexdigest(),
                         'input_shape': list(self.inputs.shape), 'input_dtype': str(self.inputs.dtype),
                         'input_scope': 'fixed_preprocessed_eval_tensor_no_random_augmentation',
                         'soft_assignment': dict(model.region_grouping.config)}
        self.initial_s, self.last_epoch, self.total_seconds = None, -1, 0.0
        if resume:
            if json.loads((self.out / 'probe_manifest.json').read_text()) != self.identity:
                raise ValueError('Probe resume identity mismatch')
            initial = torch.load(self.out / 'epoch_000.pt', map_location='cpu', weights_only=True)
            if initial['identity'] != self.identity or initial['epoch'] != 0:
                raise ValueError('Probe initial snapshot identity mismatch')
            self.initial_s = initial['tensors']['S'].detach().clone()
            rows = [json.loads(line) for line in (self.out / 'assignment_diagnostics.jsonl').read_text().splitlines()]
            if (not rows or [r['epoch'] for r in rows] != list(range(len(rows)))
                    or any(r['split'] != split or r['sample_ids'] != self.sample_ids for r in rows)):
                raise ValueError('Probe resume history identity/epochs mismatch')
            self.last_epoch = rows[-1]['epoch']
            self.total_seconds = sum(r['probe_seconds'] for r in rows)
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            (self.out / 'probe_manifest.json').write_text(json.dumps(self.identity, indent=2) + '\n')
            (self.out / 'assignment_diagnostics.jsonl').touch(exist_ok=False)

    def _capture(self):
        model = self.model
        device = next(model.parameters()).device
        devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
        modes = [(module, module.training) for module in model.modules()]
        previous = model.region_grouping.assignment_observer
        captured, handles = {}, []

        def observer(values):
            captured.update({k: v.detach().cpu().clone() for k, v in values.items()})

        def graph(_module, args):
            captured['edge_index'] = args[1].detach().cpu().clone()

        def feedback(_module, _args, output):
            captured['feedback'] = output.detach().cpu().clone()

        try:
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                model.eval()
                model.region_grouping.assignment_observer = observer
                handles.append(model.graph_reasoning.register_forward_pre_hook(graph))
                handles.append(model.feedback_proj.register_forward_hook(feedback))
                logits = model(self.inputs.to(device))
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite probe logits')
        finally:
            model.region_grouping.assignment_observer = previous
            for handle in handles:
                handle.remove()
            for module, training in modes:
                module.training = training
        return captured

    def record(self, epoch, *, save_tensors=False):
        if type(epoch) is not int or epoch != self.last_epoch + 1:
            raise ValueError(f'Probe epoch must be {self.last_epoch + 1}, got {epoch!r}')
        start = time.perf_counter()
        tensors = self._capture()
        s = tensors['S']
        initial = s if self.initial_s is None else self.initial_s
        if s.shape != initial.shape or not torch.isfinite(initial).all():
            raise ValueError('Invalid initial assignment snapshot')
        feedback = torch.einsum('bnk,bkd->bnd', s, tensors['feedback'])
        feedback_norm = feedback.norm(dim=-1).mean(-1)
        token_norm = tensors['token_norm'].mean(-1)
        cosine = tensors['prototypes'] @ tensors['prototypes'].transpose(1, 2)
        off_diagonal = cosine[:, ~torch.eye(16, dtype=torch.bool)]
        occupancy = torch.stack([torch.bincount(row.argmax(-1), minlength=16) for row in s])
        row = {'epoch': epoch, 'split': self.split, 'samples': len(s), 'sample_ids': self.sample_ids,
               'entropy_mean': float(-(s * s.clamp_min(1e-30).log()).sum(-1).mean()),
               'region_mass': s.sum(1).tolist(), 'argmax_occupancy': occupancy.tolist(),
               'prototype_norm_min': float(tensors['prototype_norms'].min()),
               'prototype_norm_mean': float(tensors['prototype_norms'].mean()),
               'low_norm_prototypes': int((tensors['prototype_norms'] < 1e-6).sum()),
               'prototype_cosine_off_diagonal_mean': float(off_diagonal.mean()),
               'prototype_cosine_off_diagonal_max': float(off_diagonal.max()),
               'mean_abs_delta_S': float((s - initial).abs().mean()),
               'feedback_norm': feedback_norm.tolist(), 'token_norm': token_norm.tolist(),
               'feedback_to_token_ratio': [float(f / t) if t > 0 else None for f, t in zip(feedback_norm, token_norm)],
               'zero_input_images': int((token_norm == 0).sum()),
               'probe_seconds': time.perf_counter() - start}
        line = json.dumps(row, allow_nan=False)
        if save_tensors or epoch == 0:
            path = self.out / f'epoch_{epoch:03d}.pt'
            with path.open('xb') as stream:
                torch.save({'epoch': epoch, 'identity': self.identity, 'tensors': tensors}, stream)
        with (self.out / 'assignment_diagnostics.jsonl').open('a') as stream:
            stream.write(line + '\n')
            stream.flush()
        self.initial_s = initial.clone()
        self.last_epoch = epoch
        self.total_seconds += row['probe_seconds']
        return row

    def select_snapshot(self, epoch, name):
        if name not in ('best', 'final') or type(epoch) is not int or epoch != self.last_epoch:
            raise ValueError('Snapshot selection requires current epoch and best/final label')
        self._publish_snapshot(epoch, name)

    def _publish_snapshot(self, epoch, name):
        source = self.out / f'epoch_{epoch:03d}.pt'
        saved = torch.load(source, map_location='cpu', weights_only=True)
        if saved['epoch'] != epoch or saved['identity'] != self.identity:
            raise ValueError('Probe selected snapshot identity/epoch mismatch')
        target = self.out / f'{name}.pt'
        temporary = self.out / f'{name}.pt.tmp'
        shutil.copyfile(source, temporary)
        temporary.replace(target)

    def reconcile_snapshots(self, *, best_epoch=None, final_epoch=None):
        """Repair interrupted alias publication using persisted, validated snapshots."""
        for name, epoch in (('best', best_epoch), ('final', final_epoch)):
            if epoch is None:
                continue
            if type(epoch) is not int or not 1 <= epoch <= self.last_epoch:
                raise ValueError('Snapshot recovery epoch is outside persisted history')
            if name == 'final' and epoch != self.last_epoch:
                raise ValueError('Final snapshot must match last persisted epoch')
            self._publish_snapshot(epoch, name)

    def summary(self):
        return {'split': self.split, 'samples': len(self.sample_ids), 'output': str(self.out),
                'last_epoch': self.last_epoch, 'probe_seconds': self.total_seconds}
