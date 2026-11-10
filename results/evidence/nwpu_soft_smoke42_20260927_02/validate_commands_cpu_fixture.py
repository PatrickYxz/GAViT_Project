from pathlib import Path
import copy
import hashlib
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np
from PIL import Image
import torch

repo = Path('/private/tmp/gavit-soft-release-git-20260927')
sys.path.insert(0, str(repo))
from scene_classification.modeling import architecture, build_model, image_transform
from soft_assignment_diagnostics import SoftAssignmentProbe

evidence = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2/results/evidence/nwpu_soft_smoke42_20260927_02')
commands = json.loads((evidence / 'final_checks_commands.json').read_text())
root = Path(tempfile.mkdtemp(prefix='gavit-nwpu-final-checks-', dir='/private/tmp'))
run, images = root / 'run', root / 'images'
run.mkdir()
(images / 'synthetic').mkdir(parents=True)
torch.set_num_threads(2)
torch.manual_seed(42)
rng = np.random.default_rng(42)
ids = [f'synthetic/image_{i}.jpg' for i in range(8)]
for name in ids:
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(images / name)
transform = image_transform(False)
inputs = torch.stack([transform(Image.open(images / name).convert('RGB')) for name in ids])
arch = architecture('gavit', 45, grouping='soft_prototype')
model = build_model('gavit', 45, architecture_config=arch).eval()
probe = SoftAssignmentProbe(model, inputs, ids, run / 'assignment_probe', 'val')
probe.record(0, save_tensors=True)
with torch.no_grad():
    model.region_grouping.assignment_projection[1].weight.add_(torch.randn_like(model.region_grouping.assignment_projection[1].weight) * .001)
probe.record(1, save_tensors=True)
probe.select_snapshot(1, 'best')
probe.select_snapshot(1, 'final')
torch.save(model.state_dict(), run / 'best.pth')
metadata = {'status': 'complete', 'architecture': arch, 'checkpoint_sha256': hashlib.sha256((run / 'best.pth').read_bytes()).hexdigest(), 'dataset': 'synthetic_validation_fixture'}
(run / 'run.json').write_text(json.dumps(metadata))
del model, probe

def local_code(name):
    code = commands[name]['code']
    return (code.replace('/home/featurize/work/GAViT_SOFT_20260927', str(repo))
            .replace('/home/featurize/work/nwpu_soft_smoke42_20260927_02', str(run))
            .replace('/home/featurize/data/NWPU-RESISC45', str(images))
            .replace('assert torch.cuda.is_available()', 'assert not torch.cuda.is_available()')
            .replace('device=torch.device("cuda")', 'device=torch.device("cpu")')
            .replace('torch.set_num_threads(8)', 'torch.set_num_threads(2)'))

buffer = io.StringIO()
with redirect_stdout(buffer):
    exec(local_code('snapshots'), {})
    exec(local_code('gradient'), {})

# Counterexample: a non-normalized saved assignment must be rejected.
initial_path = run / 'assignment_probe/epoch_000.pt'
original = initial_path.read_bytes()
bad = torch.load(initial_path, weights_only=True)
bad['tensors']['S'] *= 2
torch.save(bad, initial_path)
try:
    exec(local_code('snapshots'), {})
except AssertionError:
    buffer.write('REJECTED_CORRUPT_SNAPSHOT\n')
else:
    raise AssertionError('Corrupt assignment was accepted')
finally:
    initial_path.write_bytes(original)

# Counterexample: severing S from all model parameters must not pass gradients.
broken = local_code('gradient').replace('(s*torch.randn_like(s)).sum().backward()', 's=s.detach().requires_grad_(); (s*torch.randn_like(s)).sum().backward()')
try:
    exec(broken, {})
except AssertionError:
    buffer.write('REJECTED_DISCONNECTED_ASSIGNMENT\n')
else:
    raise AssertionError('Detached S was accepted')

result = {'fixture': str(root), 'scope': 'Real Swin/PyG on local CPU with synthetic images and random weights; no server/CUDA verification', 'local_adaptations': 'repo/run/image paths, CUDA availability assertion, CPU device, two CPU threads', 'checks': ['snapshot_valid', 'assignment_gradient_valid', 'corrupt_snapshot_rejected', 'disconnected_assignment_rejected'], 'output': buffer.getvalue()}
(evidence / 'final_checks_local_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(buffer.getvalue())
print('LOCAL_COMMAND_CHECKS_COMPLETE', root)
