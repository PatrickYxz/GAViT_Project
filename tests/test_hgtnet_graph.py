"""Real tensor checks for the independent HGTNet graph implementation."""

import importlib
import unittest

import torch
from torch import nn


def _graph():
    try:
        return importlib.import_module("external_baselines.hgtnet.graph")
    except ModuleNotFoundError as error:
        if error.name.startswith("external_baselines"):
            raise AssertionError("HGTNet graph primitives are not implemented") from error
        raise


class HGTNetKNNTests(unittest.TestCase):
    def test_knn_returns_ordered_euclidean_indices_without_self(self):
        nodes = torch.tensor([[[0.0], [1.0], [3.0], [10.0]]], requires_grad=True)
        neighbours = _graph().knn_indices(nodes, 2)
        torch.testing.assert_close(neighbours, torch.tensor([[[1, 2], [0, 2], [1, 0], [2, 1]]]))
        self.assertEqual(neighbours.dtype, torch.long)
        self.assertEqual(neighbours.device, nodes.device)
        self.assertFalse(neighbours.requires_grad)
        self.assertIsNone(neighbours.grad_fn)
        self.assertIsNone(nodes.grad)

    def test_knn_breaks_equal_distance_ties_by_original_index(self):
        nodes = torch.tensor([[[0.0], [-1.0], [1.0], [0.0]]])
        actual = _graph().knn_indices(nodes, 3)
        expected = torch.tensor([[[3, 1, 2], [0, 3, 2], [0, 3, 1], [0, 1, 2]]])
        torch.testing.assert_close(actual, expected)

    def test_knn_self_policy_is_explicit_even_for_duplicate_nodes(self):
        nodes = torch.ones(1, 3, 2)
        torch.testing.assert_close(
            _graph().knn_indices(nodes, 2), torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        )
        torch.testing.assert_close(
            _graph().knn_indices(nodes, 3, exclude_self=False),
            torch.tensor([[[0, 1, 2], [0, 1, 2], [0, 1, 2]]]),
        )

    def test_knn_keeps_samples_independent(self):
        nodes = torch.tensor([[[0.0], [1.0], [10.0]], [[0.0], [10.0], [1.0]]])
        torch.testing.assert_close(
            _graph().knn_indices(nodes, 1), torch.tensor([[[1], [0], [1]], [[2], [2], [0]]])
        )

    def test_knn_handles_finite_float32_distances_that_overflow_float32(self):
        nodes = torch.tensor([[[0.0], [1e38], [-1e38], [2e38]]])
        self.assertTrue(torch.isfinite(nodes).all())
        torch.testing.assert_close(
            _graph().knn_indices(nodes, 1), torch.tensor([[[1], [0], [0], [1]]])
        )

    def test_knn_accepts_real_float_dtypes(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                nodes = torch.tensor([[[0.0], [1.0], [3.0]]], dtype=dtype)
                torch.testing.assert_close(
                    _graph().knn_indices(nodes, 1), torch.tensor([[[1], [0], [1]]])
                )

    def test_knn_rejects_small_graph_instead_of_clipping_k(self):
        for count, k in ((16, 20), (20, 20), (1, 1)):
            with self.subTest(count=count, k=k), self.assertRaises(ValueError):
                _graph().knn_indices(torch.zeros(1, count, 2), k)

    def test_knn_rejects_invalid_k_and_self_policy(self):
        for k in (0, -1, True, 1.5, "2"):
            with self.subTest(k=k), self.assertRaises((TypeError, ValueError)):
                _graph().knn_indices(torch.zeros(1, 4, 2), k)
        for policy in (0, 1, "true", None):
            with self.subTest(policy=policy), self.assertRaises(TypeError):
                _graph().knn_indices(torch.zeros(1, 4, 2), 1, exclude_self=policy)

    def test_knn_rejects_invalid_nodes_and_nonfinite_features(self):
        invalid = [
            [[0.0], [1.0]], torch.ones(4, 2), torch.ones(1, 2, 3, 4),
            torch.empty(0, 4, 2), torch.empty(1, 0, 2), torch.empty(1, 4, 0),
            torch.ones(1, 4, 2, dtype=torch.long), torch.ones(1, 4, 2, dtype=torch.bool),
            torch.ones(1, 4, 2, dtype=torch.complex64),
            torch.tensor([[[float("nan")], [1.0]]]),
            torch.tensor([[[float("inf")], [1.0]]]),
        ]
        for index, nodes in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                _graph().knn_indices(nodes, 1)


class HGTRelativeMessageTests(unittest.TestCase):
    def test_relative_max_and_mean_use_neighbour_minus_query(self):
        nodes = torch.tensor([[[5.0], [6.0], [20.0], [8.0]]])
        neighbours = torch.tensor([[[1, 3], [0, 3], [1, 3], [0, 1]]])
        torch.testing.assert_close(
            _graph().relative_aggregate(nodes, neighbours), torch.tensor([[[3.0], [2.0], [-12.0], [-2.0]]])
        )
        torch.testing.assert_close(
            _graph().relative_aggregate(nodes, neighbours, "mean"), torch.tensor([[[2.0], [0.5], [-13.0], [-2.5]]])
        )

    def test_relative_aggregate_keeps_batches_separate_and_neighbour_order_irrelevant(self):
        nodes = torch.tensor([[[5.0], [6.0], [8.0]], [[100.0], [300.0], [500.0]]])
        neighbours = torch.tensor([[[1, 2], [0, 2], [0, 1]], [[2, 1], [0, 2], [1, 0]]])
        for reduction, expected in (
            ("max", [[[3.0], [2.0], [-2.0]], [[400.0], [200.0], [-200.0]]]),
            ("mean", [[[2.0], [0.5], [-2.5]], [[300.0], [0.0], [-300.0]]]),
        ):
            with self.subTest(reduction=reduction):
                actual = _graph().relative_aggregate(nodes, neighbours, reduction)
                torch.testing.assert_close(actual, torch.tensor(expected))
                torch.testing.assert_close(
                    actual, _graph().relative_aggregate(nodes, neighbours.flip(-1), reduction)
                )

    def test_relative_aggregate_preserves_node_permutation(self):
        nodes = torch.tensor([[[0.0, 1.0], [2.0, 3.0], [5.0, 9.0], [11.0, 21.0]]])
        neighbours = torch.tensor([[[1, 2], [0, 2], [1, 3], [0, 2]]])
        permutation = torch.tensor([2, 0, 3, 1])
        inverse = torch.argsort(permutation)
        permuted_neighbours = inverse[neighbours[:, permutation]]
        for reduction in ("max", "mean"):
            with self.subTest(reduction=reduction):
                original = _graph().relative_aggregate(nodes, neighbours, reduction)
                permuted = _graph().relative_aggregate(nodes[:, permutation], permuted_neighbours, reduction)
                torch.testing.assert_close(permuted, original[:, permutation])

    def test_relative_aggregate_has_correct_message_gradients(self):
        nodes = torch.tensor([[[5.0], [6.0], [8.0]]], requires_grad=True)
        neighbours = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        _graph().relative_aggregate(nodes, neighbours, "mean")[0, 0].sum().backward()
        torch.testing.assert_close(nodes.grad, torch.tensor([[[-1.0], [0.5], [0.5]]]))

    def test_relative_aggregate_rejects_invalid_indices_and_reduction(self):
        nodes = torch.ones(2, 3, 2)
        invalid = [
            torch.zeros(2, 3, 1, dtype=torch.float32),
            torch.zeros(2, 3, 1, dtype=torch.int32),
            torch.zeros(2, 3, 1, dtype=torch.bool),
            torch.zeros(3, 1, dtype=torch.long),
            torch.zeros(1, 3, 1, dtype=torch.long),
            torch.zeros(2, 2, 1, dtype=torch.long),
            torch.empty(2, 3, 0, dtype=torch.long),
            torch.full((2, 3, 1), -1, dtype=torch.long),
            torch.full((2, 3, 1), 3, dtype=torch.long),
        ]
        for index, neighbours in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises((TypeError, ValueError)):
                _graph().relative_aggregate(nodes, neighbours)
        with self.assertRaises(ValueError):
            _graph().relative_aggregate(nodes, torch.zeros(2, 3, 1, dtype=torch.long), "sum")
        with self.assertRaises(ValueError):
            _graph().relative_aggregate(nodes * float("nan"), torch.zeros(2, 3, 1, dtype=torch.long))

    def test_mrconv_concatenates_center_before_relative_message(self):
        layer = _graph().MRConv(1, 1)
        self.assertIsInstance(layer.projection, nn.Linear)
        with torch.no_grad():
            layer.projection.weight.copy_(torch.tensor([[10.0, 1.0]]))
            layer.projection.bias.zero_()
        nodes = torch.tensor([[[5.0], [6.0], [8.0]]])
        neighbours = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        torch.testing.assert_close(layer(nodes, neighbours), torch.tensor([[[53.0], [62.0], [78.0]]]))

    def test_mrconv_mean_and_gradients_reach_projection_and_nodes(self):
        torch.manual_seed(31)
        layer = _graph().MRConv(2, 3, reduction="mean")
        nodes = torch.tensor([[[0.0, 1.0], [2.0, 4.0], [5.0, 8.0]]], requires_grad=True)
        neighbours = torch.tensor([[[1, 2], [0, 2], [0, 1]]])
        result = layer(nodes, neighbours)
        self.assertEqual(tuple(result.shape), (1, 3, 3))
        result.square().mean().backward()
        for value in (nodes, layer.projection.weight, layer.projection.bias):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.0)

    def test_mrconv_rejects_wrong_channels_dtype_and_constructor(self):
        for arguments in ((0, 2), (2, 0), (True, 2), (2, 1.5)):
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                _graph().MRConv(*arguments)
        with self.assertRaises(ValueError):
            _graph().MRConv(2, 3, reduction="sum")
        layer = _graph().MRConv(2, 3)
        neighbours = torch.zeros(1, 3, 1, dtype=torch.long)
        for nodes in (torch.ones(1, 3, 1), torch.ones(1, 3, 2, dtype=torch.float64)):
            with self.subTest(dtype=nodes.dtype, shape=nodes.shape), self.assertRaises((TypeError, ValueError)):
                layer(nodes, neighbours)


class HGTGraphBlockTests(unittest.TestCase):
    def test_graphblock_builds_actual_graph_from_current_projected_features(self):
        block = _graph().GraphBlock(2, 1)
        with torch.no_grad():
            block.in_projection.weight.copy_(torch.eye(2))
            block.in_projection.bias.copy_(torch.tensor([-0.05, 0.0]))
        nodes = torch.tensor([[[0.0, 0.0], [0.1, 0.0], [-10.0, 0.0], [1.0, 0.0]]])
        output, actual = block(nodes, return_neighbors=True)
        self.assertEqual(tuple(output.shape), (1, 4, 2))
        torch.testing.assert_close(actual, torch.tensor([[[2], [3], [0], [1]]]))
        self.assertEqual(_graph().knn_indices(nodes, 1)[0, 0, 0].item(), 1)
        with torch.no_grad():
            block.in_projection.bias.copy_(torch.tensor([0.05, 0.0]))
        _, updated = block(nodes, return_neighbors=True)
        self.assertEqual(updated[0, 0, 0].item(), 1)

    def test_graphblock_default_returns_tensor_and_two_residuals_preserve_identity(self):
        block = _graph().GraphBlock(4, 2)
        with torch.no_grad():
            block.out_projection.weight.zero_()
            block.out_projection.bias.zero_()
            block.ffn[-1].weight.zero_()
            block.ffn[-1].bias.zero_()
        nodes = torch.randn(2, 5, 4)
        output = block(nodes)
        self.assertIsInstance(output, torch.Tensor)
        torch.testing.assert_close(output, nodes, rtol=0, atol=0)

    def test_graphblock_is_permutation_equivariant_without_distance_ties(self):
        torch.manual_seed(32)
        block = _graph().GraphBlock(4, 2).eval()
        nodes = torch.randn(2, 7, 4)
        permutation = torch.tensor([2, 0, 5, 1, 6, 4, 3])
        torch.testing.assert_close(block(nodes[:, permutation]), block(nodes)[:, permutation])

    def test_graphblock_has_real_finite_gradients_and_optimizer_update(self):
        torch.manual_seed(33)
        block = _graph().GraphBlock(4, 2)
        nodes = torch.randn(2, 9, 4, requires_grad=True)
        before = block.relative_conv.projection.weight.detach().clone()
        optimizer = torch.optim.SGD(block.parameters(), lr=0.05)
        block(nodes).square().mean().backward()
        for name, parameter in block.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertGreater(nodes.grad.abs().sum().item(), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(before, block.relative_conv.projection.weight))

    def test_graphblock_mean_probe_and_explicit_self_policy(self):
        block = _graph().GraphBlock(3, 4, operator="relative_mean", exclude_self=False)
        nodes = torch.randn(1, 4, 3)
        output, neighbours = block(nodes, return_neighbors=True)
        self.assertEqual(tuple(output.shape), (1, 4, 3))
        self.assertEqual(tuple(neighbours.shape), (1, 4, 4))
        self.assertEqual(block.relative_conv.reduction, "mean")

    def test_graphblock_rejects_invalid_arguments_inputs_and_small_k20_graph(self):
        cases = [
            {"channels": 0, "k": 1}, {"channels": 3, "k": 0},
            {"channels": 3, "k": True}, {"channels": 3, "k": 1, "operator": "gcn"},
            {"channels": 3, "k": 1, "ffn_ratio": 0},
            {"channels": 3, "k": 1, "ffn_ratio": 1.5},
            {"channels": 3, "k": 1, "norm_eps": 0.0},
            {"channels": 3, "k": 1, "norm_eps": float("nan")},
            {"channels": 3, "k": 1, "exclude_self": 1},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                _graph().GraphBlock(**arguments)
        with self.assertRaises(ValueError):
            _graph().GraphBlock(3, 20)(torch.ones(1, 16, 3))
        block = _graph().GraphBlock(3, 1)
        invalid = [torch.ones(1, 4, 2), torch.ones(1, 4, 3, dtype=torch.float64), torch.ones(1, 4, 3) * float("inf")]
        for nodes in invalid:
            with self.subTest(shape=nodes.shape, dtype=nodes.dtype), self.assertRaises((TypeError, ValueError)):
                block(nodes)


class HGTDepthwiseProjectionTests(unittest.TestCase):
    def test_depthwise_projection_matches_hand_spatial_and_channel_result(self):
        layer = _graph().DepthwiseProjection(2, 2, padding=0)
        with torch.no_grad():
            layer.depthwise.weight.zero_()
            layer.depthwise.weight[:, 0, 1, 1] = 1.0
            layer.pointwise.weight.zero_()
            layer.pointwise.weight[0, 0, 0, 0] = 1.0
            layer.pointwise.weight[1, 1, 0, 0] = 1.0
        image = torch.stack((torch.ones(1, 3, 3), torch.full((1, 3, 3), 3.0)), dim=1)
        expected = torch.tensor([[[[-1.0 / (1.0 + 1e-5) ** 0.5]], [[1.0 / (1.0 + 1e-5) ** 0.5]]]])
        torch.testing.assert_close(layer(image), expected)

    def test_depthwise_projection_transitions_noncontiguous_map_and_backpropagates(self):
        torch.manual_seed(34)
        layer = _graph().DepthwiseProjection(4, 8, padding=0)
        image = torch.randn(2, 4, 7, 9).transpose(-1, -2).requires_grad_()
        self.assertFalse(image.is_contiguous())
        output = layer(image)
        self.assertEqual(tuple(output.shape), (2, 8, 7, 5))
        output.square().mean().backward()
        for name, parameter in layer.named_parameters():
            with self.subTest(name=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertEqual(layer.depthwise.groups, 4)
        self.assertIsNone(layer.depthwise.bias)
        self.assertIsNone(layer.pointwise.bias)

    def test_depthwise_projection_supports_stride_and_preserves_padding_shape(self):
        self.assertEqual(tuple(_graph().DepthwiseProjection(4, 6)(torch.randn(1, 4, 7, 7)).shape), (1, 6, 7, 7))
        self.assertEqual(tuple(_graph().DepthwiseProjection(4, 6, stride=2)(torch.randn(1, 4, 7, 7)).shape), (1, 6, 4, 4))

    def test_depthwise_projection_rejects_invalid_constructor_and_inputs(self):
        cases = [
            {"in_channels": 0, "out_channels": 2},
            {"in_channels": 2, "out_channels": True},
            {"in_channels": 2, "out_channels": 3, "stride": 0},
            {"in_channels": 2, "out_channels": 3, "padding": -1},
            {"in_channels": 2, "out_channels": 3, "norm_eps": float("inf")},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                _graph().DepthwiseProjection(**arguments)
        layer = _graph().DepthwiseProjection(2, 3, padding=0)
        invalid = [
            torch.ones(1, 2, 3), torch.ones(1, 3, 3, 3), torch.ones(1, 2, 2, 2),
            torch.ones(1, 2, 3, 3, dtype=torch.long), torch.ones(1, 2, 3, 3, dtype=torch.float64),
            torch.ones(1, 2, 3, 3) * float("nan"), torch.empty(0, 2, 3, 3),
        ]
        for image in invalid:
            with self.subTest(shape=image.shape, dtype=image.dtype), self.assertRaises((TypeError, ValueError)):
                layer(image)


if __name__ == "__main__":
    unittest.main()
