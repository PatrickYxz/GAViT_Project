import copy
import importlib
import json
from unittest.mock import patch
import pytest
import torch
from test_soft_gavit import make_model


def probe(*args, **kwargs):
    try:
        cls = importlib.import_module('soft_assignment_diagnostics').SoftAssignmentProbe
    except (ImportError, AttributeError) as exc:
        pytest.fail(f'Probe missing: {exc}')
    return cls(*args, **kwargs)


def test_probe_records_recomputable_stats_and_restores_modes_rng(tmp_path):
    model = make_model()
    model.train()
    model.classifier.eval()
    images = torch.randn(2, 3, 16, 16)
    before_rng = torch.get_rng_state().clone()
    before_modes = [m.training for m in model.modules()]
    p = probe(model, images, ['a', 'b'], tmp_path / 'diag', 'val')
    row = p.record(0, save_tensors=True)
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_modes == [m.training for m in model.modules()]
    assert model.region_grouping.assignment_observer is None
    assert row['samples'] == 2 and row['mean_abs_delta_S'] == 0
    saved = torch.load(tmp_path / 'diag' / 'epoch_000.pt', weights_only=True)
    tensors = saved['tensors']
    assert all(not t.requires_grad and t.grad_fn is None and t.device.type == 'cpu' for t in tensors.values())
    s = tensors['S']
    expected_entropy = float(-(s * s.clamp_min(1e-30).log()).sum(-1).mean())
    assert row['entropy_mean'] == pytest.approx(expected_entropy)
    assert row['region_mass'] == s.sum(1).tolist()
    assert tensors['edge_index'].shape == (2, 160)
    assert p.record(1)['mean_abs_delta_S'] == 0
    with torch.no_grad():
        model.region_grouping.assignment_projection[1].weight.add_(torch.randn_like(model.region_grouping.assignment_projection[1].weight) * .1)
    assert p.record(2, save_tensors=True)['mean_abs_delta_S'] > 0
    p.select_snapshot(2, 'best')
    assert torch.load(tmp_path / 'diag' / 'best.pt', weights_only=True)['epoch'] == 2


def test_record_failure_restores_hooks_modes_and_rng(tmp_path):
    model = make_model()
    model.classifier.eval()
    p = probe(model, torch.randn(1, 3, 16, 16), ['a'], tmp_path / 'diag', 'val')
    rng = torch.get_rng_state().clone()
    modes = [m.training for m in model.modules()]
    hooks = (len(model.graph_reasoning._forward_pre_hooks), len(model.feedback_proj._forward_hooks))
    def fail(_x):
        torch.rand(4)
        raise RuntimeError('probe deliberately failed')
    with patch.object(model, 'forward', fail), pytest.raises(RuntimeError, match='deliberately'):
        p.record(0)
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == [m.training for m in model.modules()]
    assert model.region_grouping.assignment_observer is None
    assert hooks == (len(model.graph_reasoning._forward_pre_hooks), len(model.feedback_proj._forward_hooks))
    assert p.record(0)['epoch'] == 0


def test_inserting_probe_does_not_change_training_update(tmp_path):
    torch.manual_seed(44)
    plain = make_model()
    inspected = copy.deepcopy(plain)
    x = torch.randn(2, 3, 16, 16)
    y = torch.tensor([0, 1])
    p = probe(inspected, x, ['a', 'b'], tmp_path / 'diag', 'val')
    for model in (plain, inspected):
        torch.manual_seed(66)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        if model is inspected:
            p.record(0)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        optimizer.step()
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(value, inspected.state_dict()[key], atol=0, rtol=0)


def test_resume_preserves_initial_assignment_and_rejects_bad_identity(tmp_path):
    model = make_model()
    x = torch.randn(1, 3, 16, 16)
    out = tmp_path / 'diag'
    p = probe(model, x, ['a'], out, 'val')
    p.record(0)
    p.record(1)
    with pytest.raises(FileExistsError):
        probe(model, x, ['a'], out, 'val')
    restored = probe(model, x, ['a'], out, 'val', resume=True)
    assert restored.last_epoch == 1
    assert restored.record(2)['mean_abs_delta_S'] == 0
    with pytest.raises(ValueError):
        restored.record(2)
    with pytest.raises(ValueError, match='identity'):
        probe(model, x + 1, ['a'], out, 'val', resume=True)
    (out / 'epoch_000.pt').unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        probe(model, x, ['a'], out, 'val', resume=True)


def test_test_split_invalid_inputs_and_legacy_models_are_rejected(tmp_path):
    model = make_model()
    x = torch.randn(1, 3, 16, 16)
    for split in ('test', 'anything'):
        with pytest.raises(ValueError, match='split'):
            probe(model, x, ['a'], tmp_path / split, split)
    with pytest.raises(ValueError):
        probe(model, x, [], tmp_path / 'wrong-count', 'val')
    with pytest.raises(ValueError):
        probe(make_model('attentive_spatial'), x, ['a'], tmp_path / 'old', 'val')


@pytest.mark.parametrize('stale_aliases', [False, True])
def test_resume_reconciles_best_and_final_without_new_forward(tmp_path, stale_aliases):
    model = make_model()
    x = torch.randn(1, 3, 16, 16)
    out = tmp_path / 'diag'
    p = probe(model, x, ['a'], out, 'val')
    p.record(0)
    if stale_aliases:
        p.select_snapshot(0, 'best')
        p.select_snapshot(0, 'final')
    p.record(1, save_tensors=True)  # Best epoch; interruption skips alias selection.
    p.record(2, save_tensors=True)  # No improvement; final alias selection also skipped.
    initial = (out / 'epoch_000.pt').read_bytes()
    history = (out / 'assignment_diagnostics.jsonl').read_bytes()
    restored = probe(model, x, ['a'], out, 'val', resume=True)
    with patch.object(model, 'forward', side_effect=AssertionError('No new forward')):
        restored.reconcile_snapshots(best_epoch=1, final_epoch=2)
    assert torch.load(out / 'best.pt', weights_only=True)['epoch'] == 1
    assert torch.load(out / 'final.pt', weights_only=True)['epoch'] == 2
    assert (out / 'epoch_000.pt').read_bytes() == initial
    assert (out / 'assignment_diagnostics.jsonl').read_bytes() == history
    (out / 'epoch_001.pt').unlink()
    with pytest.raises(FileNotFoundError):
        restored.reconcile_snapshots(best_epoch=1)
    invalid = torch.load(out / 'epoch_002.pt', weights_only=True)
    invalid['identity']['split'] = 'train'
    torch.save(invalid, out / 'epoch_002.pt')
    with pytest.raises(ValueError, match='identity'):
        restored.reconcile_snapshots(final_epoch=2)
