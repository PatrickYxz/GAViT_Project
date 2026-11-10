import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

evidence = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2/results/evidence/bigearth_soft_preflight_20260928')
commands = json.loads((evidence / 'launch_commands.json').read_text())
repo = Path('/private/tmp/gavit-soft-release-git-20260927')
fixture = Path(tempfile.mkdtemp(prefix='gavit-be-launch-check-', dir='/private/tmp'))
raw = fixture / 'raw'
src = fixture / 'splits'
out = fixture / 'output'
src.mkdir()
pre = fixture / 'unused-pretrained.safetensors'
pre.write_bytes(b'local CPU fixture only, not pretrained weights')
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_hashes = []
for split, size in [('train', 128), ('val', 96)]:
    with (src / (split + '.csv')).open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['patch_path'] + ['label_' + str(i) for i in range(19)])
        for i in range(size):
            patch = raw / split / (split + '_' + str(i))
            patch.mkdir(parents=True)
            yy, xx = np.indices((120, 120))
            for j, band in enumerate(('B02', 'B03', 'B04')):
                pixels = ((xx * (i + 1) + yy * (j + 7) + i * 19) % 4096).astype(np.uint16)
                Image.fromarray(pixels).save(patch / (patch.name + '_' + band + '.tif'))
            writer.writerow([str(patch)] + [int((i + k) % 5 == 0) for k in range(19)])
    source_hashes.append(digest(src / (split + '.csv')))

replacements = {
    '/home/featurize/work/GAViT_SOFT_20260927': str(repo),
    '/home/featurize/work/bigearth_soft_smoke42_20260928_01': str(out),
    '/home/featurize/data/BigEarthNet-S2': str(raw),
    '/home/featurize/work/GAViT_Project/bigearth_files/splits': str(src),
    '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors': str(pre),
    '25734fd9e8dc7fd78fb259e768ae91fe11b297e1caa50a2d57b74b0296413d8f': source_hashes[0],
    '1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45': source_hashes[1],
    'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36': digest(pre),
    '{"train":237871,"val":122342}': '{"train":128,"val":96}',
}
def adapt(code):
    for old, new in replacements.items():
        code = code.replace(old, new)
    return code

for key in ('prepare_command', 'run_command'):
    for shell in ('bash', 'zsh'):
        checked = subprocess.run([shell, '-n'], input=commands[key] + '\n', text=True, capture_output=True)
        assert checked.returncode == 0, checked.stderr
for key in ('prepare_code', 'run_code'):
    compile(commands[key], '<server-command>', 'exec')

env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
def run(code, log):
    with (fixture / log).open('w') as stream:
        result = subprocess.run([sys.executable, '-u', '-c', code], stdout=stream, stderr=subprocess.STDOUT, env=env)
    print(log, 'exit', result.returncode, flush=True)
    print((fixture / log).read_text()[-2800:], flush=True)
    return result.returncode

prepare = adapt(commands['prepare_code'])
assert run(prepare, 'prepare.log') == 0
info = json.loads((out / 'preflight.json').read_text())
assert info['status'] == 'ready' and len(info['band_sha256']) == 480
assert run(prepare, 'refuse-existing.log') != 0
saved_band = raw / next(iter(info['band_sha256']))
held = saved_band.with_suffix('.held')
saved_band.rename(held)
assert run(prepare, 'missing-band.log') != 0
assert 'Missing RGB bands' in (fixture / 'missing-band.log').read_text()
held.rename(saved_band)

training = adapt(commands['run_code'])
for old, new in {
    'assert torch.cuda.is_available(),"CUDA unavailable"': 'assert not torch.cuda.is_available()',
    'torch.set_num_threads(8)': 'torch.set_num_threads(2)',
    'torch.cuda.reset_peak_memory_stats()': 'None',
    'torch.cuda.get_device_name(0)': '"local CPU fixture"',
    'torch.cuda.synchronize()': 'None',
    'ns["DEVICE"]=="cuda"': 'ns["DEVICE"]=="cpu"',
    '.to("cuda")': '.to("cpu")',
    'torch.cuda.max_memory_allocated()': '0',
    'torch.cuda.max_memory_reserved()': '0',
}.items():
    assert old in training, old
    training = training.replace(old, new)
patch = '''import timm
import torch.utils.data
real_create = timm.create_model
def offline_create(*args, **kwargs):
    kwargs['pretrained'] = False
    return real_create(*args, **kwargs)
timm.create_model = offline_create
RealLoader = torch.utils.data.DataLoader
class CpuLoader(RealLoader):
    def __init__(self, *args, **kwargs):
        kwargs.update(num_workers=0, pin_memory=False)
        super().__init__(*args, **kwargs)
torch.utils.data.DataLoader = CpuLoader
'''
assert run(patch + '\n' + training, 'training.log') == 0
assert (out / 'run.exit').read_text() == 'exit_code=0\n'
summary = json.loads((out / 'summary.json').read_text())
assert summary['counts'] == {'train': 96, 'val': 64}
assert summary['reload_max_logit_diff'] <= 1e-5 and summary['graph']['actual_forward_checked']
result = {'scope': 'Local CPU real Swin/PyG, synthetic 120x120 uint16 TIFFs, random weights, workers0; no server or CUDA validation', 'fixture': str(fixture), 'checks': ['prepare_archive_crc_and_manifest', 'existing_output_rejected', 'missing_band_rejected', 'original_entry_96_64_train_validate_save', 'reload_actual_graph_and_exit_record'], 'summary': summary}
(evidence / 'launch_local_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print('LOCAL_BIGEARTH_COMMANDS_PASS', fixture, flush=True)
