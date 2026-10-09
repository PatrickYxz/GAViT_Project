"""The new candidate changes membership logits only; legacy remains explicit."""
from copy import deepcopy
import math

import pytest
import torch

from grouping_identity import soft_assignment_identity, validate_soft_assignment
from models.soft_region_grouping import SoftPrototypeGrouping


def candidate():
    return soft_assignment_identity('anchor_gaussian_v1')


def test_versioned_identity_is_explicit_strict_and_preserves_legacy():
    legacy = soft_assignment_identity()
    assert legacy['assignment_version'] == 'local_prototype_cosine_v1'
    assert not any(k.startswith('spatial_prior') for k in legacy)
    new = candidate()
    assert new['assignment_version'] == 'local_prototype_cosine_spatial_v1'
    assert validate_soft_assignment(new) == new
    assert all(new[k] == v for k, v in legacy.items() if k != 'assignment_version')
    for key in new:
        bad = deepcopy(new)
        del bad[key]
        with pytest.raises(ValueError):
            validate_soft_assignment(bad)
    for value in (True, 0.0, 1.0, float('nan'), '0.5'):
        with pytest.raises(ValueError):
            validate_soft_assignment({**new, 'spatial_prior_strength': value})
    with pytest.raises(ValueError):
        validate_soft_assignment({**new, 'assignment_version': legacy['assignment_version']})
    with pytest.raises(ValueError):
        soft_assignment_identity('unknown')


def test_prior_geometry_and_equal_content_odds():
    m = SoftPrototypeGrouping(soft_assignment=candidate()).eval()
    # All semantic similarities equal: posterior odds are independently calculable.
    x = torch.ones(1, 49, 768)
    _, s = m(x)
    # Token (0,0), centers (.5,.5) and (.5,2.5): squared distance difference 6.
    assert float(s[0, 0, 1] / s[0, 0, 0]) == pytest.approx(math.exp(-0.5 * 6 / 1.75**2), rel=2e-6)
    # Uneven last partition center is (6,6), NOT ideal 4x4 center (5.625,5.625).
    assert float(m.spatial_log_prior[48, 15]) == 0.0
    assert float(m.spatial_log_prior[48, 14]) == pytest.approx(-0.5 * 1.5**2 / 1.75**2)
    assert float(m.spatial_log_prior[0, 0]) == pytest.approx(-0.5 * 0.5 / 1.75**2)
    assert not s.masked_select(~m.candidate_mask[None]).any()


def test_identical_parameters_rng_and_only_logit_prior_changes():
    torch.manual_seed(42)
    old = SoftPrototypeGrouping()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = SoftPrototypeGrouping(soft_assignment=candidate())
    assert torch.equal(rng, torch.get_rng_state())
    assert dict(old.named_parameters()).keys() == dict(new.named_parameters()).keys()
    for name, value in old.named_parameters():
        assert torch.equal(value, dict(new.named_parameters())[name])
    x = torch.randn(2, 49, 768, requires_grad=True)
    old_log, *old_rest = old._assignment(x, old.attn(x))
    log_s, *new_rest = new._assignment(x, new.attn(x))
    for a, b in zip(old_rest, new_rest):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    expected = (old_log + new.spatial_log_prior).log_softmax(-1)
    torch.testing.assert_close(log_s, expected, atol=2e-6, rtol=2e-6)
    s = log_s.exp()
    regions, w = new._pool(x, new.attn(x), log_s)
    torch.testing.assert_close(s.sum(-1), torch.ones(2, 49))
    torch.testing.assert_close(w.sum(1), torch.ones(2, 16))
    # Pooling is still softmax(log S + original attention) over tokens.
    torch.testing.assert_close(w, (log_s + new.attn(x)).softmax(1), atol=0, rtol=0)
    grads = torch.autograd.grad((s * torch.randn_like(s)).sum(),
        (x, new.assignment_projection[1].weight, new.attn[0].weight))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert torch.isfinite(regions).all()


def test_candidate_checkpoint_cannot_silently_load_as_legacy():
    new = SoftPrototypeGrouping(soft_assignment=candidate()).eval()
    old = SoftPrototypeGrouping().eval()
    with pytest.raises(RuntimeError, match='spatial_log_prior'):
        old.load_state_dict(new.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match='spatial_log_prior'):
        new.load_state_dict(old.state_dict(), strict=True)
    restored = SoftPrototypeGrouping(soft_assignment=candidate()).eval()
    restored.load_state_dict(new.state_dict(), strict=True)
    x = torch.randn(1, 49, 768)
    for a, b in zip(new(x), restored(x)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    legacy_explicit = SoftPrototypeGrouping(soft_assignment=soft_assignment_identity()).eval()
    legacy_explicit.load_state_dict(old.state_dict(), strict=True)
    assert 'spatial_log_prior' not in old.state_dict()
    for a, b in zip(old(x), legacy_explicit(x)):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('kind', ['zero', 'identical', 'opposite'])
def test_candidate_degenerate_inputs_and_batch_independence(kind):
    m = SoftPrototypeGrouping(soft_assignment=candidate()).eval()
    x = torch.randn(2, 1, 768).expand(2, 49, 768).clone()
    if kind == 'zero':
        x.zero_()
    elif kind == 'opposite':
        x[:, ::2] *= -1
    x.requires_grad_()
    r, s = m(x)
    (r.square().mean() + s.square().mean()).backward()
    assert torch.isfinite(x.grad).all()
    for a, b in zip(m(x[:1]), (r[:1], s[:1])):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
