"""Small, observational audit for the bounded FFN training trial."""
import hashlib
import json
import math

import torch


def state_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class RefinementAudit:
    def __init__(self, module):
        self.module = module
        self.report = {'parameters': sum(p.numel() for p in module.parameters()),
                       'initial_state_sha256': state_sha256(module), 'gradient_steps': []}
        self.handle = module.register_forward_hook(self._first_forward)

    def _first_forward(self, _module, inputs, output):
        inputs = inputs[0]
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (49, 768):
            raise ValueError('FFN must run on tokens before pooling')
        error = float((output.detach() - inputs.detach()).abs().max())
        if not math.isfinite(error) or error != 0.0:
            raise ValueError('FFN initial residual must be exactly zero')
        self.report['first_forward'] = {'input_shape': list(inputs.shape), 'max_abs_residual': error}
        self.handle.remove()

    def record_gradients(self):
        """First two backwards, before optimizer.step; does not alter tensors or RNG."""
        if len(self.report['gradient_steps']) >= 2:
            return
        row = {'step': len(self.report['gradient_steps']) + 1}
        for name in ('norm', 'fc1', 'fc2'):
            grads = [p.grad for p in getattr(self.module, name).parameters()]
            if any(g is None or not torch.isfinite(g).all() for g in grads):
                raise ValueError(f'FFN missing/nonfinite gradient: {name}')
            row[name] = sum(float(g.detach().abs().sum()) for g in grads)
        if row['fc2'] <= 0 or (row['step'] == 2 and (row['fc1'] <= 0 or row['norm'] <= 0)):
            raise ValueError('FFN failed to learn in the first two steps')
        self.report['gradient_steps'].append(row)

    def finish(self, selected_module):
        self.handle.remove()
        self.report['selected_state_sha256'] = state_sha256(selected_module)
        self.report['final_state_sha256'] = state_sha256(self.module)
        return self.report
