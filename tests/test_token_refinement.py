"""Protect placement, initialization fairness and checkpoint identity of the FFN trial."""
import copy
import importlib
from unittest.mock import patch

import pytest
import torch

from scene_classification.modeling import architecture, build_model
from test_soft_gavit import TokenBackbone


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def candidate(name):
    # Missing implementation must produce an assertion failure, not a broken import.
    try:
        return architecture(name, 45, grouping='soft_prototype' if name == 'gavit' else None,
                            token_refinement='residual_ffn_v1')
    except TypeError as exc:
        pytest.fail(f'FFN architecture selection is not implemented: {exc}')


def construct(name, arch):
    with patch('models.gavit.SwinBackbone', TokenBackbone), \
            patch('scene_classification.modeling.SwinBackbone', TokenBackbone):
        return build_model(name, 45, architecture_config=arch)


@pytest.mark.parametrize('name', ['swin', 'gavit'])
def test_identity_start_preserves_existing_parameters_rng_logits_and_old_loads(name):
    new_arch = candidate(name)
    old_arch = architecture(name, 45, grouping='soft_prototype' if name == 'gavit' else None)
    assert 'token_refinement' not in old_arch
    torch.manual_seed(42)
    old = construct(name, old_arch).eval()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = construct(name, new_arch).eval()
    assert torch.equal(rng, torch.get_rng_state())
    old_state = old.state_dict()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old_state.items())
    assert all(k.startswith('token_refinement.') for k in new.state_dict().keys() - old_state.keys())
    images = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(old(images), new(images), atol=0, rtol=0)
    restored_old = construct(name, old_arch).eval()
    restored_old.load_state_dict(old_state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored_old(images), old(images), atol=0, rtol=0)
    with pytest.raises(RuntimeError):
        restored_old.load_state_dict(new.state_dict(), strict=True)


def test_both_models_get_identical_ffn_despite_different_construction_rng():
    torch.manual_seed(1)
    a = construct('swin', candidate('swin')).token_refinement
    torch.manual_seed(99)
    b = construct('gavit', candidate('gavit')).token_refinement
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    assert sum(p.numel() for p in a.parameters()) == 297408


@pytest.mark.parametrize('name', ['swin', 'gavit'])
def test_ffn_is_nonlinear_before_pool_and_reloads_after_learning(name):
    arch = candidate(name)
    model = construct(name, arch).eval()
    ffn = model.token_refinement
    images = torch.randn(2, 3, 16, 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(images), torch.tensor([1, 2])).backward()
        assert torch.isfinite(ffn.fc2.weight.grad).all() and ffn.fc2.weight.grad.abs().sum() > 0
        for p in (ffn.fc1.weight, ffn.norm.weight):
            assert torch.isfinite(p.grad).all()
            assert bool(p.grad.abs().sum() > 0) == (step == 1)
        optimizer.step()
    captured = {}

    def backbone(_module, _args, output):
        captured['tokens'] = output

    def grouping(_module, _args, output):
        captured['membership'] = output[1]

    def feedback(_module, _args, output):
        captured['feedback'] = output

    def capture(_module, args, result):
        captured['input'], captured['output'] = args[0], result

    handle = ffn.register_forward_hook(capture)
    handles = [model.backbone.register_forward_hook(backbone)]
    if name == 'gavit':
        handles.extend([model.region_grouping.register_forward_hook(grouping),
                        model.feedback_proj.register_forward_hook(feedback)])
    with torch.no_grad():
        logits = model(images)
    handle.remove()
    for hook in handles:
        hook.remove()
    assert captured['input'].shape == (2, 49, 768)
    expected_input = captured['tokens']
    if name == 'gavit':
        expected_input = expected_input + torch.einsum('bnk,bkd->bnd', captured['membership'], captured['feedback'])
    torch.testing.assert_close(captured['input'], expected_input, atol=0, rtol=0)
    with torch.no_grad():
        expected = model.classifier(captured['output'].mean(1))
        wrong = model.classifier(ffn(captured['input'].mean(1)))
    torch.testing.assert_close(logits, expected, atol=0, rtol=0)
    assert not torch.allclose(logits, wrong, atol=1e-7, rtol=1e-7)
    rebuilt = construct(name, arch).eval()
    rebuilt.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt(images), logits, atol=0, rtol=0)


def test_malformed_or_unapproved_refinement_cannot_silently_reconstruct():
    arch = candidate('swin')
    for value in (None, {}, 'residual_ffn_v1', {**arch['token_refinement'], 'hidden_dim': 384},
                  {**arch['token_refinement'], 'init_seed': True}):
        bad = copy.deepcopy(arch)
        bad['token_refinement'] = value
        with pytest.raises(ValueError, match='refinement'):
            build_model('swin', 45, architecture_config=bad)
    with pytest.raises(ValueError):
        architecture('gavit', 45, grouping='attentive_spatial', token_refinement='residual_ffn_v1')


def test_scope_rejects_refit_other_datasets_and_extra_seeds_before_model_allocation(tmp_path):
    candidate('swin')
    runner = importlib.import_module('scene_classification.runner')
    for changes in ({'phase': 'refit'}, {'seed': 43}, {'model': 'gavit', 'grouping': None}):
        kwargs = dict(model='swin', manifest='missing.json', data_root='missing',
                      output=str(tmp_path / str(len(list(tmp_path.iterdir())))),
                      token_refinement='residual_ffn_v1', random_init=True, device='cpu')
        kwargs.update(changes)
        with pytest.raises(ValueError, match='FFN trial'):
            runner.run_training(runner.RunConfig(**kwargs))
