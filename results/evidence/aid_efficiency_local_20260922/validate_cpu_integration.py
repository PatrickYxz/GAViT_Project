import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import torch

ROOT = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
RELEASE = Path('/Users/patrick/.codex/worktrees/aid-entry/GAViT_Project')
spec = importlib.util.spec_from_file_location('probe', ROOT / 'tools/benchmark_aid.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
sys.path.insert(0, str(RELEASE))
from scene_classification import runner as r
from scene_classification.modeling import build_model, architecture, capture_graph_once
from experiment_identity import write_checkpoint_metadata

checks = {}
torch.set_num_threads(2)
code = r._code_identity()
assert code['git'] == {'commit': 'deb2b45', 'dirty': False}
with tempfile.TemporaryDirectory() as td:
    work = Path(td)
    manifest_path = work / 'aid_protocol_20260920_03.json'
    manifest = {'dataset': 'AID', 'class_names': [str(i) for i in range(30)]}
    manifest_path.write_text(json.dumps(manifest))
    p.MANIFEST_SHA = p.sha256(manifest_path)
    for model in ('swin', 'gavit'):
        out = work / f'aid_{model}_refit42_20260920_01'
        out.mkdir()
        ckpt = out / 'last.pth'
        torch.manual_seed(123)
        m = build_model(model, 30).eval()
        x = torch.rand(1, 3, 224, 224)
        graph, handle = capture_graph_once(m)
        with torch.inference_mode():
            y = m(x)
        if handle:
            handle.remove()
        assert y.shape == (1, 30) and torch.isfinite(y).all()
        assert sum(t.numel() for t in m.parameters()) == p.PARAMETERS[model]
        assert model == 'swin' or graph['edges_per_image'] == 80
        torch.save(m.state_dict(), ckpt)
        p.WEIGHTS[model] = p.sha256(ckpt)
        training = r.training_identity(r.RunConfig(model, '', '', ''))
        meta = {'entry_version': 'scene-v1', 'model': model, 'dataset': 'AID',
                'class_names': manifest['class_names'], 'architecture': architecture(model, 30),
                'execution': {'phase': 'refit'}, 'training': training,
                'data': {'manifest_sha256': p.MANIFEST_SHA, 'fit_scope': 'train_plus_val'},
                'pretrained': {'sha256': p.PRETRAINED_SHA},
                'checkpoint_sha256': p.WEIGHTS[model], 'final_epoch': 21 if model == 'swin' else 20,
                'code': code}
        write_checkpoint_metadata(str(ckpt), meta)
        source = {'status': 'complete', 'phase': 'refit', 'model': model, 'dataset': 'AID',
                  'seed': 42, 'training': training, 'architecture': meta['architecture'],
                  'checkpoint_sha256': meta['checkpoint_sha256'], 'manifest_sha256': p.MANIFEST_SHA,
                  'pretrained_sha256': p.PRETRAINED_SHA, 'final_epoch': meta['final_epoch'],
                  'code': code, 'counts': {'train': 5000, 'val': 0}}
        (out / 'run.json').write_text(json.dumps(source))
        (out / 'run.exit').write_text('exit_code=0\n')
        restored = r._restore(ckpt, meta, torch.device('cpu'))
        with torch.inference_mode():
            z = restored(x)
            samples = p.measure(lambda: restored(x), lambda: None, __import__('time').perf_counter, 2, 1)
        assert torch.equal(y, z)
        assert all(t > 0 for t in samples)
        checks[model + '_cpu_restore_forward'] = {'passed': True, 'parameters': p.PARAMETERS[model],
                                                 'logit_diff': float((y-z).abs().max()), 'graph': graph}
        del m, restored, x, y, z
    args = argparse.Namespace(repo=str(RELEASE), work=str(work))
    p.preflight(args)
    checks['synthetic_matching_identity_accepted'] = True
    target = work / 'aid_swin_refit42_20260920_01/run.exit'
    target.write_text('exit_code=1\n')
    try:
        p.preflight(args)
        raise AssertionError('Failed source accepted')
    except ValueError as e:
        assert 'Source run' in str(e)
    checks['failed_source_rejected'] = True
    target.write_text('exit_code=0\n')
    ckpt = target.parent / 'last.pth'
    with ckpt.open('ab') as f:
        f.write(b'corruption')
    try:
        p.preflight(args)
        raise AssertionError('Corrupt checkpoint accepted')
    except ValueError as e:
        assert 'hash mismatch' in str(e)
    checks['corrupt_checkpoint_rejected'] = True
    manifest_path.write_text('{}')
    try:
        p.preflight(args)
        raise AssertionError('Wrong manifest accepted')
    except ValueError as e:
        assert 'manifest hash mismatch' in str(e)
    checks['wrong_manifest_rejected'] = True
assert r._code_identity() == code
checks['release_code_unchanged'] = True
checks['scope'] = 'CPU random-initialized synthetic fixtures; no CUDA timing or formal checkpoints tested'
evidence = ROOT / 'results/evidence/aid_efficiency_local_20260922'
evidence.mkdir(parents=True, exist_ok=True)
(evidence / 'integration.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
