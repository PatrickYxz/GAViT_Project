import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]


def entry(script, args, *, env=None):
    return subprocess.run([sys.executable, str(ROOT / 'tests/soft_entry_worker.py'), script, *map(str, args)],
        cwd=ROOT, text=True, capture_output=True, timeout=120,
        env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', **(env or {})})


@pytest.fixture
def splits(tmp_path):
    root = tmp_path / 'splits'
    root.mkdir()
    for part in ('train', 'val', 'test'):
        rows = []
        for i in range(2):
            patch = tmp_path / f'{part}_{i}'
            patch.mkdir()
            for b, band in enumerate(('B02', 'B03', 'B04')):
                pixels = ((np.arange(256).reshape(16, 16) * (b + 1) + i * 801) % 4096).astype('uint16')
                with rasterio.open(patch / f'{patch.name}_{band}.tif', 'w', driver='GTiff',
                                   height=16, width=16, count=1, dtype='uint16') as image:
                    image.write(pixels, 1)
            rows.append({'patch_path': str(patch), **{f'label_{k}': (i+k) % 2 for k in range(19)}})
        with (root / f'{part}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return root


@pytest.mark.parametrize('script', ['train_bigearth.py', 'test_bigearth.py', 'run_scene.py'])
def test_help_advertises_new_mode(script):
    result = entry(script, ['train', '--help'] if script == 'run_scene.py' else ['--help'])
    assert result.returncode == 0, result.stderr
    assert 'soft_prototype' in result.stdout


@pytest.mark.parametrize('model,grouping', [('swin', None), ('gavit', 'attentive_spatial'), ('gavit', 'soft_prototype')])
def test_real_bigearth_entry_train_reload_and_test(tmp_path, splits, model, grouping):
    ckpt = tmp_path / 'model.pth'
    args = ['--model', model, '--data_dir', splits, '--epochs', 1, '--batch_size', 2,
            '--seed', 42, '--run_stage', 'smoke', '--run_tag', 'soft_entry_test', '--checkpoint_path', ckpt]
    if grouping:
        args += ['--grouping', grouping, '--edge_type', 'sparse_hybrid']
    result = entry('train_bigearth.py', args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ENTRY RELOAD PASS' in result.stdout
    meta = json.loads(ckpt.with_suffix('.meta.json').read_text())
    assert meta['dataset'] == 'BigEarthNet-19'
    assert ckpt.with_suffix('.last.train_state.pth').is_file()
    evaluated = entry('test_bigearth.py', ['--model', model, '--ckpt', ckpt, '--data_dir', splits, '--batch_size', 2])
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    assert 'Test mAP' in evaluated.stdout and 'Test F1    (micro)' in evaluated.stdout
    if grouping == 'soft_prototype':
        probe = ckpt.with_suffix('.assignment_probe')
        rows = [json.loads(line) for line in (probe / 'assignment_diagnostics.jsonl').read_text().splitlines()]
        assert [row['epoch'] for row in rows] == [0, 1]
        assert rows[0]['split'] == 'val' and (probe / 'final.pt').is_file()
        assert meta['architecture']['soft_assignment']['assignment_dim'] == 128
        sidecar = ckpt.with_suffix('.meta.json')
        for key in meta['architecture']['soft_assignment']:
            # Full per-field validator is covered in identity tests; one real entry corruption suffices.
            del meta['architecture']['soft_assignment'][key]
            break
        sidecar.write_text(json.dumps(meta))
        invalid = entry('test_bigearth.py', ['--model', model, '--ckpt', ckpt, '--data_dir', tmp_path / 'missing'])
        assert invalid.returncode != 0 and 'soft_assignment' in invalid.stderr


@pytest.mark.parametrize('stop_after', [1, 2])
def test_new_mode_resume_uses_persisted_initial_probe(tmp_path, splits, stop_after):
    ckpt = tmp_path / 'resume.pth'
    args = ['--model', 'gavit', '--grouping', 'soft_prototype', '--edge_type', 'sparse_hybrid',
            '--data_dir', splits, '--epochs', 2, '--batch_size', 2, '--seed', 42,
            '--run_stage', 'smoke', '--run_tag', 'resume_probe', '--checkpoint_path', ckpt]
    stopped = entry('train_bigearth.py', args, env={'SOFT_TEST_STOP_AFTER': str(stop_after)})
    assert stopped.returncode == 7, stopped.stdout + stopped.stderr
    probe = ckpt.with_suffix('.assignment_probe')
    initial = (probe / 'epoch_000.pt').read_bytes()
    resumed = entry('train_bigearth.py', [*args, '--resume'])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (probe / 'epoch_000.pt').read_bytes() == initial
    rows = [json.loads(line) for line in (probe / 'assignment_diagnostics.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in rows] == [0, 1, 2]
    assert torch.load(probe / 'final.pt', weights_only=True)['epoch'] == 2
    meta = json.loads(ckpt.with_suffix('.meta.json').read_text())
    assert torch.load(probe / 'best.pt', weights_only=True)['epoch'] == meta['best']['epoch']


def test_legacy_nwpu_entry_rejects_new_sidecar_before_data_access(tmp_path):
    from grouping_identity import complete_grouping_architecture
    ckpt = tmp_path / 'model.pth'
    arch = complete_grouping_architecture({'num_classes': 45, 'num_regions': 16,
        'knn_k': 5, 'gat_hidden': 256, 'gat_heads': 4, 'gat_layers': 2, 'dropout': 0.1,
        'grouping': 'soft_prototype', 'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'})
    ckpt.with_suffix('.meta.json').write_text(json.dumps({'model': 'gavit', 'dataset': 'NWPU-RESISC45', 'architecture': arch}))
    result = entry('test_gavit.py', ['--ckpt', ckpt])
    assert result.returncode != 0 and 'run_scene.py' in result.stderr
