"""Soft membership must be differentiable, local and independent per image."""
import importlib
import pytest
import torch
from models.region_grouping import AttentiveSpatialGrouping


def layer(**kwargs):
    try:
        cls = importlib.import_module('models.soft_region_grouping').SoftPrototypeGrouping
    except (ImportError, AttributeError) as exc:
        pytest.fail(f'SoftPrototypeGrouping missing: {exc}')
    return cls(**kwargs)


def test_normalization_locality_and_assignment_only_gradients():
    torch.manual_seed(42)
    m = layer()
    x = torch.randn(2, 49, 768, requires_grad=True)
    log_s, w0, norms, prototypes = m._assignment(x, m.attn(x))
    r, w = m._pool(x, m.attn(x), log_s)
    s = log_s.exp()
    torch.testing.assert_close(s.sum(-1), torch.ones(2, 49))
    torch.testing.assert_close(w.sum(1), torch.ones(2, 16))
    torch.testing.assert_close(w0.sum(1), torch.ones(2, 16))
    assert not s.masked_select(~m.candidate_mask[None]).any()
    assert m.candidate_mask[0].nonzero().flatten().tolist() == [0, 1, 4, 5]
    assert set(m.membership.sum(0).tolist()) == {1., 2., 4.}
    assert r.shape == (2, 16, 768)
    cost = torch.randn_like(s)
    gradients = torch.autograd.grad((s * cost).sum(),
        (x, m.assignment_projection[1].weight, m.attn[0].weight))
    for grad in gradients:
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_batch_isolation_repeatability_and_content_dependence():
    torch.manual_seed(7)
    m = layer().eval()
    x = torch.randn(2, 49, 768)
    expected_r, expected_s = m(x)
    repeated_r, repeated_s = m(x)
    assert torch.equal(repeated_s, expected_s)
    changed = x.clone()
    changed[1] *= -2
    r, s = m(changed)
    torch.testing.assert_close(r[0], expected_r[0], atol=0, rtol=0)
    torch.testing.assert_close(s[0], expected_s[0], atol=0, rtol=0)
    changed = x.clone()
    changed[0, 2] = x[0, 0]  # Candidate region 1 changes, query token 0 does not.
    assert (m(changed)[1][0, 0] - expected_s[0, 0]).abs().max() > 1e-6


def test_onehot_membership_reproduces_original_pooling():
    torch.manual_seed(10)
    old = AttentiveSpatialGrouping()
    torch.manual_seed(10)
    soft = layer()
    x = torch.randn(2, 49, 768)
    log_s = torch.where(soft.membership.bool(), 0., -torch.inf)[None].expand(2, -1, -1)
    actual, _ = soft._pool(x, soft.attn(x), log_s)
    torch.testing.assert_close(actual, old(x)[0], atol=1e-6, rtol=1e-6)


def test_shared_attention_rng_and_projection_reproducibility():
    torch.manual_seed(42)
    old = AttentiveSpatialGrouping()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    soft = layer()
    assert torch.equal(old_rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, soft.state_dict()[key])
    torch.manual_seed(42)
    assert torch.equal(soft.assignment_projection[1].weight, layer().assignment_projection[1].weight)
    torch.manual_seed(43)
    assert not torch.equal(soft.assignment_projection[1].weight, layer().assignment_projection[1].weight)


@pytest.mark.parametrize('kind', ['zero', 'identical', 'opposite'])
def test_degenerate_features_remain_finite(kind):
    m = layer()
    v = torch.randn(1, 1, 768)
    x = v.expand(1, 49, -1).clone()
    if kind == 'zero':
        x.zero_()
    if kind == 'opposite':
        x[:, ::2] *= -1
    x.requires_grad_()
    r, s = m(x)
    (r.square().mean() + s.square().mean()).backward()
    assert torch.isfinite(r).all() and torch.isfinite(s).all()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


def test_invalid_shapes_types_and_nonfinite_data_fail_clearly():
    for kwargs in ({'num_regions': 9}, {'grid_size': 8}, {'feat_dim': 8}):
        with pytest.raises(ValueError):
            layer(**kwargs)
    m = layer()
    for x in (torch.zeros(0, 49, 768), torch.zeros(1, 48, 768), torch.zeros(1, 49, 8),
              torch.zeros(49, 768), torch.ones(1, 49, 768, dtype=torch.long),
              torch.full((1, 49, 768), float('nan')), torch.full((1, 49, 768), float('inf'))):
        with pytest.raises(ValueError):
            m(x)


def test_observer_receives_detached_current_tensors_without_retention():
    m = layer()
    captured = []
    m.assignment_observer = captured.append
    m(torch.randn(1, 49, 768, requires_grad=True))
    assert len(captured) == 1
    assert {'S', 'W', 'W0', 'prototype_norms', 'prototypes', 'token_norm'} <= captured[0].keys()
    assert all(not x.requires_grad and x.grad_fn is None for x in captured[0].values())
    m.assignment_observer = None
    m(torch.randn(1, 49, 768))
    assert len(captured) == 1
