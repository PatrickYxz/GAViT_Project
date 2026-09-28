import copy
from unittest.mock import patch
import pytest
import torch
from torch.nn import functional as F

from grouping_identity import complete_grouping_architecture
from models.gavit import GAViT
from scene_classification.modeling import capture_graph_once
from utils import load_checkpoint_metadata, resolve_arch_config, save_checkpoint


class TokenBackbone(torch.nn.Module):
    hidden_dim = 768

    def __init__(self, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(3, 768)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (7, 7))
        return self.proj(x.flatten(2).transpose(1, 2))


def make_model(grouping='soft_prototype', **kwargs):
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        return GAViT(pretrained=False, grouping=grouping, edge_type='sparse_hybrid',
                     gat_hidden=4, gat_heads=2, num_classes=19, **kwargs)


def test_shared_initialization_and_rng_match_old_model():
    torch.manual_seed(42)
    old = make_model('attentive_spatial')
    rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    new = make_model()
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in old.state_dict().items():
        assert torch.equal(value, new.state_dict()[key]), key
    assert sum(p.numel() for p in new.parameters()) - sum(p.numel() for p in old.parameters()) == 99840


def test_onehot_soft_model_matches_old_regions_edges_feedback_and_logits():
    torch.manual_seed(42)
    old = make_model('attentive_spatial').eval()
    torch.manual_seed(42)
    new = make_model().eval()
    x = torch.randn(2, 3, 16, 16)
    helper = new.region_grouping._assignment

    def onehot(tokens, scores):
        _, w0, norms, prototypes = helper(tokens, scores)
        log_s = torch.where(new.region_grouping.membership.bool(), 0., -torch.inf)
        return log_s[None].expand(len(tokens), -1, -1), w0, norms, prototypes

    observed = [{}, {}]
    handles = []
    for model, box in zip((old, new), observed):
        def grouping_hook(_module, _args, output, box=box):
            box['regions'] = output[0].detach()
        def graph_hook(_module, args, box=box):
            box['edges'] = args[1].detach()
        def feedback_hook(_module, _args, output, box=box):
            box['feedback'] = output.detach()
        handles += [model.region_grouping.register_forward_hook(grouping_hook),
                    model.graph_reasoning.register_forward_pre_hook(graph_hook),
                    model.feedback_proj.register_forward_hook(feedback_hook)]
    try:
        with patch.object(new.region_grouping, '_assignment', onehot), torch.no_grad():
            expected, actual = old(x), new(x)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        for key in ('regions', 'edges', 'feedback'):
            torch.testing.assert_close(observed[0][key], observed[1][key], atol=1e-6, rtol=1e-6)
    finally:
        for handle in handles:
            handle.remove()


@pytest.mark.parametrize('loss_name', ['ce', 'bce'])
def test_actual_graph_forward_backward_and_assignment_only_backbone_gradient(loss_name):
    torch.manual_seed(21)
    model = make_model()
    images = torch.randn(2, 3, 16, 16)
    capture, handle = capture_graph_once(model)
    try:
        logits = model(images)
    finally:
        handle.remove()
    assert capture['edges_per_image'] == 80 and capture['actual_forward_checked']
    loss = F.cross_entropy(logits, torch.tensor([1, 2])) if loss_name == 'ce' else F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits))
    loss.backward()
    for module in (model.region_grouping.assignment_projection, model.graph_reasoning, model.backbone):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    model.zero_grad(set_to_none=True)
    tokens = model.backbone(images)
    s = model.region_grouping(tokens)[1]
    (s * torch.randn_like(s)).sum().backward()
    assert model.backbone.proj.weight.grad.abs().sum() > 0


def test_new_checkpoint_reconstructs_exact_architecture(tmp_path):
    arch = complete_grouping_architecture(dict(num_classes=19, num_regions=16, knn_k=5,
        gat_hidden=4, gat_heads=2, gat_layers=2, dropout=.1, grouping='soft_prototype',
        edge_type='sparse_hybrid', integration='token_feedback'))
    x = torch.randn(2, 3, 16, 16)
    with patch('models.gavit.SwinBackbone', TokenBackbone):
        model = GAViT(**arch, pretrained=False).eval()
        path = str(tmp_path / 'model.pth')
        save_checkpoint(model, path, {'model': 'gavit', 'dataset': 'synthetic', 'architecture': arch})
        defaults = {k: v for k, v in arch.items() if k != 'soft_assignment'}
        resolved, _ = resolve_arch_config({}, load_checkpoint_metadata(path), defaults)
        restored = GAViT(**resolved, pretrained=False).eval()
        restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x), atol=1e-6, rtol=1e-6)


def test_bad_combinations_and_broken_membership_are_rejected():
    for kwargs in ({'edge_type': 'knn'}, {'integration': 'fusion'}, {'num_regions': 9}):
        config = dict(grouping='soft_prototype', edge_type='sparse_hybrid', pretrained=False)
        config.update(kwargs)
        with pytest.raises(ValueError):
            GAViT(**config)
    model = make_model().eval()
    x = torch.randn(1, 3, 16, 16)
    regions, membership = model.region_grouping(model.backbone(x))
    for bad in (membership.long(), membership[:, :, 0], membership * 2,
                torch.full_like(membership, float('nan')), membership.roll(8, -1)):
        with patch.object(model.region_grouping, 'forward', return_value=(regions, bad)):
            with pytest.raises(ValueError, match='membership'):
                model(x)
