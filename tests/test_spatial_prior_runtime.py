"""Exercise explicit scene identity and a bounded candidate smoke lifecycle."""
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from scene_classification import runner
from scene_classification.data import prepare_manifest
from scene_classification.modeling import architecture, build_model
from test_scene_data_metrics import fixture
from test_soft_gavit import TokenBackbone


def cfg(tmp_path, **changes):
    settings = dict(model='gavit', grouping='soft_prototype', spatial_prior='anchor_gaussian_v1',
        phase='smoke', seed=42, manifest='missing', data_root='missing', output=str(tmp_path / 'out'),
        device='cpu', random_init=True, batch_size=2, workers=0, train_per_class=1, val_per_class=1)
    settings.update(changes)
    return runner.RunConfig(**settings)


def test_cli_help_and_architecture_guard():
    result = subprocess.run([sys.executable, 'run_scene.py', 'train', '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--spatial-prior {anchor_gaussian_v1}' in result.stdout
    for model, grouping, ffn in [('swin', None, None), ('gavit', None, None),
                                ('gavit', 'soft_prototype', 'residual_ffn_v1')]:
        with pytest.raises(ValueError, match='spatial|prior'):
            architecture(model, 45, grouping=grouping, token_refinement=ffn, spatial_prior='anchor_gaussian_v1')
    arch = architecture('gavit', 45, grouping='soft_prototype', spatial_prior='anchor_gaussian_v1')
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        model = build_model('gavit', 45, architecture_config=arch)
    assert model.region_grouping.config == arch['soft_assignment']


@pytest.mark.parametrize('changes', [dict(seed=43), dict(phase='refit'), dict(model='swin'),
    dict(grouping='attentive_spatial'), dict(token_refinement='residual_ffn_v1'),
    dict(phase='calibrate', workers=0), dict(spatial_prior='unknown')])
def test_candidate_scope_rejected_before_loading_data(tmp_path, changes):
    with patch.object(runner, 'load_manifest') as load:
        with pytest.raises(ValueError, match='[Pp]rior|spatial'):
            runner.run_training(cfg(tmp_path, **changes))
        load.assert_not_called()
    assert (tmp_path / 'out' / 'run.exit').read_text() == 'exit_code=1\n'


def test_non_nwpu_candidate_rejected(tmp_path):
    with patch.object(runner, 'load_manifest', return_value={'dataset': 'AID'}):
        with pytest.raises(ValueError, match='NWPU'):
            runner.run_training(cfg(tmp_path))


def test_wrong_nwpu_identity_rejected_before_model(tmp_path):
    weights = tmp_path / 'weights'
    weights.write_bytes(b'wrong pretrained')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{}')
    with patch.object(runner, 'load_manifest', return_value={'dataset': 'NWPU-RESISC45',
            'class_names': ['a'], 'counts': {'train': 5040, 'val': 1260, 'test': 25200}}), \
            patch.object(runner, 'build_model') as build:
        with pytest.raises(ValueError, match='frozen NWPU'):
            runner.run_training(cfg(tmp_path, random_init=False, pretrained_path=str(weights), manifest=str(manifest)))
        build.assert_not_called()


def test_candidate_smoke_checkpoint_probe_and_validation(tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / 'images'
    fixture(data)
    manifest = tmp_path / 'manifest.json'
    prepare_manifest(data, manifest, dataset='synthetic')
    settings = cfg(tmp_path, data_root=str(data), manifest=str(manifest))
    # Real grouping/GAT/training/serialization; only the expensive Swin is a small fixture.
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        report = runner.run_training(settings)
        out = Path(settings.output)
        assert report['status'] == 'complete' and report['counts'] == {'train': 2, 'val': 2}
        assert report['graph_capture']['edges_per_image'] == 80
        assert report['reload_max_logit_diff'] <= 1e-5
        arch = report['architecture']
        assert arch['soft_assignment']['spatial_prior'] == 'anchor_gaussian_v1'
        assert 'token_refinement' not in arch
        probe = json.loads((out / 'assignment_probe' / 'probe_manifest.json').read_text())
        assert probe['soft_assignment'] == arch['soft_assignment']
        assert (out / 'run.exit').read_text() == 'exit_code=0\n'
        state = torch.load(out / 'best.pth', map_location='cpu', weights_only=True)
        assert 'region_grouping.spatial_log_prior' in state
        result = runner.evaluate_checkpoint(str(out / 'best.pth'), str(manifest), str(data),
            str(out / 'eval'), split='val', device='cpu', batch_size=2, workers=0)
        assert result['samples'] == 2
        with pytest.raises(ValueError, match='[Pp]rior|spatial'):
            runner.evaluate_checkpoint(str(out / 'best.pth'), str(manifest), str(data),
                str(out / 'test_rejected'), split='test', device='cpu')
    with pytest.raises(FileExistsError):
        runner.run_training(settings)
