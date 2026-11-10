"""Behavioral contracts for the independently implemented HGTNet visual path."""

import math
import unittest

import torch
from torch import nn
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock

from external_baselines.hgtnet.attention import (
    DualAttention,
    PatchEmbedding,
    TransformerStage,
)


class PatchEmbeddingTests(unittest.TestCase):
    def test_patch_projection_then_channel_layernorm_matches_hand_fixture(self):
        module = PatchEmbedding(in_channels=1, embed_dim=2, patch_size=2)
        with torch.no_grad():
            module.projection.weight[0].fill_(1.0)
            module.projection.weight[1].fill_(2.0)
            module.projection.bias.zero_()
        result = module(torch.arange(16.0).reshape(1, 1, 4, 4))
        self.assertEqual(tuple(result.shape), (1, 2, 2, 2))
        torch.testing.assert_close(
            result,
            torch.tensor([[[[-1.0, -1.0], [-1.0, -1.0]], [[1.0, 1.0], [1.0, 1.0]]]]),
            atol=2e-6,
            rtol=0,
        )
        self.assertEqual(module.norm.eps, 1e-5)

    def test_non_square_patch_embedding_has_finite_gradients(self):
        module = PatchEmbedding(in_channels=3, embed_dim=8, patch_size=2).double()
        inputs = torch.randn(2, 3, 6, 10, dtype=torch.float64, requires_grad=True)
        outputs = module(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 8, 3, 5))
        (outputs * torch.randn_like(outputs)).sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_patch_embedding_rejects_malformed_inputs_and_implicit_cropping(self):
        module = PatchEmbedding(embed_dim=8)
        bad_inputs = [
            torch.randn(3, 8, 8),
            torch.randn(1, 4, 8, 8),
            torch.ones(1, 3, 8, 8, dtype=torch.int64),
            torch.ones(1, 3, 8, 8, dtype=torch.float64),
            torch.full((1, 3, 8, 8), math.nan),
            torch.empty(0, 3, 8, 8),
            torch.empty(1, 3, 0, 8),
            torch.randn(1, 3, 3, 8),
            torch.randn(1, 3, 9, 8),
        ]
        for inputs in bad_inputs:
            with self.subTest(shape=tuple(inputs.shape), dtype=inputs.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    module(inputs)

    def test_patch_embedding_rejects_invalid_configuration(self):
        for kwargs in [
            {"in_channels": 0}, {"embed_dim": -1}, {"patch_size": 0},
            {"patch_size": 2.5}, {"embed_dim": True},
            {"norm_eps": 0}, {"norm_eps": math.nan},
        ]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    PatchEmbedding(**kwargs)


class DualAttentionTests(unittest.TestCase):
    def test_zero_gates_give_exact_parallel_identity(self):
        module = DualAttention(4)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
        inputs = torch.randn(2, 4, 3, 5)
        self.assertTrue(torch.equal(module(inputs), inputs))

    def test_avg_and_max_pool_use_the_same_mlp_before_one_sigmoid(self):
        module = DualAttention(2, reduction=2)
        linear = [layer for layer in module.channel_mlp.modules() if isinstance(layer, nn.Linear)]
        self.assertEqual(len(linear), 2)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            linear[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
            linear[1].weight.copy_(torch.tensor([[1.0], [2.0]]))
        inputs = torch.tensor([[[[1.0, 3.0]], [[2.0, 4.0]]]])
        # Channel 0 mean=2 and max=3 pass through the shared 1->(1,2) MLP.
        expected_gates = torch.tensor([1.4933071491, 1.4999546021]).reshape(1, 2, 1, 1)
        torch.testing.assert_close(module(inputs), inputs * expected_gates)

    def test_spatial_branch_uses_three_dilated_convolutions_without_intermediate_relu(self):
        module = DualAttention(2)
        convolutions = [layer for layer in module.spatial.modules() if isinstance(layer, nn.Conv2d)]
        self.assertEqual(len(convolutions), 3)
        self.assertEqual([(layer.in_channels, layer.out_channels) for layer in convolutions], [(2, 1), (1, 1), (1, 1)])
        self.assertEqual([layer.dilation for layer in convolutions], [(1, 1), (2, 2), (3, 3)])
        self.assertEqual([layer.padding for layer in convolutions], [(1, 1), (2, 2), (3, 3)])
        self.assertTrue(all(layer.kernel_size == (3, 3) for layer in convolutions))
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.zero_()
            convolutions[0].weight[0, 0, 1, 1] = -1.0
            convolutions[1].weight[0, 0, 1, 1] = 1.0
            convolutions[2].weight[0, 0, 1, 1] = 1.0
        inputs = torch.tensor([[[[1.0, 3.0]], [[3.0, 5.0]]]])
        # Spatial means are 2 and 4; the negative response must survive both layers.
        expected_gates = torch.tensor([0.6192029220, 0.5179862100]).reshape(1, 1, 1, 2)
        torch.testing.assert_close(module(inputs), inputs * expected_gates)

    def test_non_square_dual_attention_broadcasts_and_backpropagates(self):
        module = DualAttention(8, reduction=4).double()
        inputs = torch.randn(2, 8, 5, 9, dtype=torch.float64, requires_grad=True)
        result = module(inputs)
        self.assertEqual(tuple(result.shape), (2, 8, 5, 9))
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        self.assertGreater(inputs.grad.abs().sum().item(), 0)
        for parameter in module.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

    def test_dual_attention_rejects_invalid_inputs(self):
        module = DualAttention(4)
        for inputs in [
            torch.randn(1, 5, 3, 7), torch.randn(4, 3, 7),
            torch.ones(1, 4, 3, 7, dtype=torch.int64),
            torch.ones(1, 4, 3, 7, dtype=torch.float64),
            torch.full((1, 4, 3, 7), math.inf), torch.empty(1, 4, 0, 7),
        ]:
            with self.subTest(shape=tuple(inputs.shape), dtype=inputs.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    module(inputs)

    def test_dual_attention_rejects_invalid_configuration(self):
        for channels, reduction in [(0, 16), (4, 0), (4, -1), (4, 1.5), (True, 2)]:
            with self.subTest(channels=channels, reduction=reduction):
                with self.assertRaises((TypeError, ValueError)):
                    DualAttention(channels, reduction)


class TransformerStageTests(unittest.TestCase):
    def test_one_stb_runs_two_real_attention_blocks_with_alternating_shifts(self):
        module = TransformerStage(8, 1, 2, output_channels=16, downsample_kind="patch_merge")
        self.assertIsInstance(module.blocks, nn.Sequential)
        self.assertEqual(len(module.blocks), 2)
        self.assertTrue(all(isinstance(block, SwinTransformerBlock) for block in module.blocks))
        self.assertEqual([block.attn.shift_size for block in module.blocks], [[0, 0], [3, 3]])
        seen_shapes = []
        handles = [block.register_forward_hook(lambda _module, _args, output: seen_shapes.append(tuple(output.shape))) for block in module.blocks]
        try:
            result = module(torch.randn(1, 8, 14, 28))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen_shapes, [(1, 14, 28, 8), (1, 14, 28, 8)])
        self.assertEqual(tuple(result.shape), (1, 16, 7, 14))

    def test_nine_stbs_run_eighteen_blocks_and_alternate_shift_behavior(self):
        module = TransformerStage(8, 9, 2, output_channels=8, downsample_kind="conv2_same_channels")
        self.assertEqual(len(module.blocks), 18)
        self.assertEqual([block.attn.shift_size for block in module.blocks], [[0, 0], [3, 3]] * 9)
        inputs = torch.randn(1, 14, 14, 8)
        shifted = module.blocks[1]
        regular = module.blocks[0]
        regular.load_state_dict(shifted.state_dict())
        shifted.eval()
        regular.eval()
        self.assertFalse(torch.allclose(shifted(inputs), regular(inputs)))
        self.assertEqual(tuple(module(torch.randn(1, 8, 14, 14)).shape), (1, 8, 7, 7))

    def test_stage_shapes_follow_three_stage_candidate(self):
        # Small channel widths retain all actual spatial sizes and the 1/1/9 STB counts.
        stage0 = TransformerStage(8, 1, 2, output_channels=16, downsample_kind="patch_merge")
        stage1 = TransformerStage(16, 1, 4, output_channels=32, downsample_kind="patch_merge")
        stage2 = TransformerStage(32, 9, 4, output_channels=32, downsample_kind="conv2_same_channels")
        inputs = torch.randn(1, 8, 56, 56)
        for stage, expected in [(stage0, (1, 16, 28, 28)), (stage1, (1, 32, 14, 14)), (stage2, (1, 32, 7, 7))]:
            inputs = stage(inputs)
            self.assertEqual(tuple(inputs.shape), expected)

    def test_patch_merge_preserves_candidate_pretraining_parameter_namespace(self):
        module = TransformerStage(8, 1, 2, output_channels=16, downsample_kind="patch_merge")
        self.assertIsInstance(module.downsample, PatchMerging)
        self.assertEqual(tuple(module.state_dict()["downsample.reduction.weight"].shape), (16, 32))
        self.assertEqual(tuple(module.state_dict()["downsample.norm.weight"].shape), (32,))
        self.assertIn("blocks.1.attn.qkv.weight", module.state_dict())

    def test_third_stage_uses_biasless_conv2_stride2_then_channel_layernorm(self):
        module = TransformerStage(8, 1, 2, output_channels=8, downsample_kind="conv2_same_channels", norm_eps=1e-4)
        conv = module.downsample.projection
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual((conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride), (8, 8, (2, 2), (2, 2)))
        self.assertIsNone(conv.bias)
        self.assertEqual(module.downsample.norm.normalized_shape, (8,))
        self.assertEqual(module.downsample.norm.eps, 1e-4)
        self.assertEqual(tuple(module(torch.randn(2, 8, 14, 14)).shape), (2, 8, 7, 7))

    def test_stage_non_square_finite_gradients_reach_da_attention_and_downsample(self):
        module = TransformerStage(8, 1, 2, output_channels=16, downsample_kind="patch_merge")
        inputs = torch.randn(2, 8, 10, 18, requires_grad=True)
        outputs = module(inputs)
        self.assertEqual(tuple(outputs.shape), (2, 16, 5, 9))
        (outputs * torch.randn_like(outputs)).sum().backward()
        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all().item(), name)
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        for parameter in [module.blocks[0].attn.qkv.weight, module.blocks[1].attn.qkv.weight, module.downsample.reduction.weight]:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_stage_rejects_incompatible_channels_or_invalid_configuration(self):
        for kwargs in [
            {"channels": 0}, {"channels": 7}, {"stbs": 0}, {"stbs": 1.5},
            {"heads": 0}, {"heads": 3}, {"window_size": 0},
            {"output_channels": 8}, {"output_channels": True},
            {"downsample_kind": "identity"},
            {"downsample_kind": "conv2_same_channels", "output_channels": 16},
            {"mlp_ratio": 0}, {"mlp_ratio": math.nan}, {"dropout": -0.1},
            {"attention_dropout": 1.1}, {"stochastic_depth_prob": math.nan},
            {"norm_eps": -1},
        ]:
            settings = dict(channels=8, stbs=1, heads=2, output_channels=16, downsample_kind="patch_merge")
            settings.update(kwargs)
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    TransformerStage(**settings)

    def test_stage_rejects_malformed_inputs_instead_of_reinterpreting_layout(self):
        module = TransformerStage(8, 1, 2, output_channels=16, downsample_kind="patch_merge")
        for inputs in [
            torch.randn(1, 14, 14, 8), torch.randn(8, 14, 14),
            torch.ones(1, 8, 14, 14, dtype=torch.int64),
            torch.ones(1, 8, 14, 14, dtype=torch.float64),
            torch.full((1, 8, 14, 14), math.nan), torch.empty(0, 8, 14, 14),
        ]:
            with self.subTest(shape=tuple(inputs.shape), dtype=inputs.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    module(inputs)


if __name__ == "__main__":
    unittest.main()
