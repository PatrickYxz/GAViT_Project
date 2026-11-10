"""Real CPU lifecycle: FFN metadata, gradient audit, reload and bounded scope."""
import json
from unittest.mock import patch

import pytest
import torch

from scene_classification import runner
from scene_classification.data import prepare_manifest
from test_scene_data_metrics import fixture


def config(tmp_path, **changes):
    root = tmp_path / 'images'
    fixture(root)
    manifest = tmp_path / 'manifest.json'
    prepare_manifest(root, manifest, dataset='synthetic')
    kwargs = dict(model='swin', manifest=str(manifest), data_root=str(root),
                  output=str(tmp_path / 'run'), phase='smoke', device='cpu',
                  random_init=True, batch_size=2, workers=0, train_per_class=2,
                  val_per_class=1, token_refinement='residual_ffn_v1')
    kwargs.update(changes)
    return runner.RunConfig(**kwargs)


@pytest.mark.parametrize('name', ['swin', 'gavit'])
def test_real_ffn_training_reloads_and_records_learning_before_pool(tmp_path, name):
    torch.set_num_threads(2)
    cfg = config(tmp_path, model=name, grouping='soft_prototype' if name == 'gavit' else None)
    report = runner.run_training(cfg)
    assert report['status'] == 'complete' and report['final_epoch'] == 1
    assert 'token_refinement_audit' in report, 'Missing initialization/actual learning audit'
    audit = report['token_refinement_audit']
    assert audit['parameters'] == 297408
    assert audit['first_forward']['input_shape'][1:] == [49, 768]
    assert audit['first_forward']['max_abs_residual'] == 0
    assert len(audit['gradient_steps']) == 2
    assert audit['gradient_steps'][0]['fc2'] > 0 and audit['gradient_steps'][0]['fc1'] == 0
    assert audit['gradient_steps'][1]['fc1'] > 0 and audit['gradient_steps'][1]['norm'] > 0
    assert audit['initial_state_sha256'] != audit['selected_state_sha256']
    assert report['reload_max_logit_diff'] <= 1e-5
    out = tmp_path / 'run'
    assert (out / 'run.exit').read_text() == 'exit_code=0\n'
    assert json.loads((out / 'refinement_audit.json').read_text()) == audit
    meta = json.loads((out / 'best.meta.json').read_text())
    assert meta['architecture']['token_refinement'] == report['architecture']['token_refinement']
    assert meta['refinement_initial_state_sha256'] == audit['initial_state_sha256']
    assert set(report['artifact_sha256']) == {'best.pth', 'last.pth', 'last.train_state.pth'}
    scores = runner.evaluate_checkpoint(str(out / 'best.pth'), cfg.manifest, cfg.data_root,
        str(out / 'eval'), split='val', device='cpu', workers=0, batch_size=2)
    assert scores['samples'] == 2
    with pytest.raises(ValueError, match='FFN trial'):
        runner.evaluate_checkpoint(str(out / 'best.pth'), cfg.manifest, cfg.data_root,
            str(out / 'test'), split='test', device='cpu')
    original = (out / 'best.meta.json').read_bytes()
    meta['architecture']['token_refinement']['hidden_dim'] = 384
    (out / 'best.meta.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='refinement'):
        runner.evaluate_checkpoint(str(out / 'best.pth'), cfg.manifest, cfg.data_root,
            str(out / 'corrupt'), split='val', device='cpu')
    (out / 'best.meta.json').write_bytes(original)


def test_aid_is_rejected_even_for_smoke(tmp_path):
    cfg = config(tmp_path)
    manifest = json.loads(open(cfg.manifest).read())
    manifest['dataset'] = 'AID'
    # Boundary already fully exercises real manifest verification in lifecycle tests.
    with patch.object(runner, 'load_manifest', return_value=manifest):
        with pytest.raises(ValueError, match='FFN trial'):
            runner.run_training(cfg)
    assert not (tmp_path / 'run' / 'best.pth').exists()


def test_changed_nwpu_protocol_is_rejected_before_training(tmp_path):
    cfg = config(tmp_path)
    manifest = json.loads(open(cfg.manifest).read())
    manifest['dataset'] = 'NWPU-RESISC45'
    with patch.object(runner, 'load_manifest', return_value=manifest):
        with pytest.raises(ValueError, match='frozen NWPU'):
            runner.run_training(cfg)
    assert not (tmp_path / 'run' / 'best.pth').exists()
