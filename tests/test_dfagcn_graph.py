"""Real-tensor DFAGCN graph checks using native unittest and hand-derived fixtures."""
import unittest
import importlib
import math
import torch

def _graph():
    try:
        return importlib.import_module('external_baselines.dfagcn.graph')
    except ModuleNotFoundError as error:
        if error.name.startswith('external_baselines'):
            raise AssertionError('DFAGCN graph primitives are not implemented') from error
        raise

class DFAGCNGraphTests(unittest.TestCase):

    def test_knn_uses_symmetric_binary_union_without_self_edges(self):
        for metric, nodes in [('euclidean', [[[0.0], [1.0], [3.0], [10.0]]]), ('cosine', [[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]]])]:
            with self.subTest(metric=metric, nodes=nodes):
                adjacency = _graph().knn_union_adjacency(torch.tensor(nodes), k=1, metric=metric)
                expected = torch.tensor([[[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]], dtype=torch.bool)
                assert adjacency.dtype == torch.bool
                torch.testing.assert_close(adjacency, expected)

    def test_explicit_neighbor_metrics_can_produce_different_topologies(self):
        nodes = torch.tensor([[[1.0, 0.0], [10.0, 0.0], [1.0, 1.0], [10.0, 1.0]]])
        euclidean = _graph().knn_union_adjacency(nodes, k=1, metric='euclidean')
        cosine = _graph().knn_union_adjacency(nodes, k=1, metric='cosine')
        torch.testing.assert_close(euclidean, torch.tensor([[[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]], dtype=torch.bool))
        torch.testing.assert_close(cosine, torch.tensor([[[0, 1, 0, 1], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]]], dtype=torch.bool))

    def test_duplicate_features_break_neighbor_ties_by_node_index(self):
        for metric in ['euclidean', 'cosine']:
            with self.subTest(metric=metric):
                adjacency = _graph().knn_union_adjacency(torch.ones(1, 4, 2), k=1, metric=metric)
                torch.testing.assert_close(adjacency, torch.tensor([[[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]], dtype=torch.bool))

    def test_knn_keeps_samples_independent_and_returns_detached_topology(self):
        nodes = torch.tensor([[[0.0], [1.0], [3.0], [10.0]], [[0.0], [10.0], [1.0], [3.0]]], requires_grad=True)
        adjacency = _graph().knn_union_adjacency(nodes, k=1, metric='euclidean')
        expected = torch.tensor([[[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]], [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]], dtype=torch.bool)
        torch.testing.assert_close(adjacency, expected)
        assert not adjacency.requires_grad
        assert adjacency.grad_fn is None
        assert nodes.grad is None

    def test_default_k_ten_requires_more_than_ten_nodes(self):
        adjacency = _graph().knn_union_adjacency(torch.arange(11.0).view(1, 11, 1), metric='euclidean')
        torch.testing.assert_close(adjacency, ~torch.eye(11, dtype=torch.bool).unsqueeze(0))

    def test_default_k_ten_rejects_undersized_graphs(self):
        for node_count in [1, 2, 9, 10]:
            with self.subTest(node_count=node_count):
                with self.assertRaises(ValueError):
                    _graph().knn_union_adjacency(torch.ones(1, node_count, 2), metric='euclidean')

    def test_knn_rejects_invalid_k(self):
        for k in [0, -1, 4, 10, True, 1.5]:
            with self.subTest(k=k):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().knn_union_adjacency(torch.ones(1, 4, 2), k=k, metric='euclidean')

    def test_knn_requires_explicit_supported_metric(self):
        with self.assertRaises(TypeError):
            _graph().knn_union_adjacency(torch.ones(1, 4, 2), k=1)
        with self.assertRaises(ValueError):
            _graph().knn_union_adjacency(torch.ones(1, 4, 2), k=1, metric='manhattan')

    def test_knn_rejects_invalid_node_features(self):
        for nodes in [torch.ones(4, 2), torch.ones(1, 2, 3, 4), torch.empty(0, 4, 2), torch.empty(1, 0, 2), torch.empty(1, 4, 0), torch.tensor([[[float('nan')], [1.0]]]), torch.tensor([[[float('inf')], [1.0]]]), torch.ones(1, 4, 2, dtype=torch.long), torch.ones(1, 4, 2, dtype=torch.complex64)]:
            with self.subTest(nodes=nodes):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().knn_union_adjacency(nodes, k=1, metric='euclidean')

    def test_cosine_knn_rejects_zero_norm_nodes(self):
        with self.assertRaises(ValueError):
            _graph().knn_union_adjacency(torch.tensor([[[0.0, 0.0], [1.0, 0.0]]]), k=1, metric='cosine')

    def test_cosine_knn_rejects_overflowed_norms_of_finite_features(self):
        nodes = torch.tensor([[[1e38, 0.0], [0.0, 1e38], [1e38, 1e38]]])
        self.assertTrue(torch.isfinite(nodes).all())
        with self.assertRaises(ValueError):
            _graph().knn_union_adjacency(nodes, k=1, metric='cosine')

    def test_knn_accepts_real_floating_feature_dtypes(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                adjacency = _graph().knn_union_adjacency(torch.tensor([[[0.0], [1.0], [3.0]]], dtype=dtype), k=1, metric='euclidean')
                torch.testing.assert_close(adjacency, torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.bool))

    def test_normalization_matches_hand_computed_three_node_path(self):
        adjacency = torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.bool)
        normalized = _graph().normalize_adjacency(adjacency, dtype=torch.float64)
        edge = 1.0 / math.sqrt(6.0)
        expected = torch.tensor([[[0.5, edge, 0.0], [edge, 1.0 / 3.0, edge], [0.0, edge, 0.5]]], dtype=torch.float64)
        torch.testing.assert_close(normalized, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(adjacency, torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.bool))

    def test_normalization_adds_each_self_loop_exactly_once(self):
        adjacency = torch.tensor([[[1, 1, 0], [1, 0, 1], [0, 1, 1]]], dtype=torch.float64)
        edge = 1.0 / math.sqrt(6.0)
        expected = torch.tensor([[[0.5, edge, 0.0], [edge, 1.0 / 3.0, edge], [0.0, edge, 0.5]]], dtype=torch.float64)
        torch.testing.assert_close(_graph().normalize_adjacency(adjacency), expected, rtol=1e-14, atol=1e-14)
        assert adjacency[0, 1, 1].item() == 0.0

    def test_normalization_dtype_contract(self):
        for dtype in [torch.bool, torch.int64, torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                adjacency = torch.eye(3, dtype=dtype).unsqueeze(0)
                normalized = _graph().normalize_adjacency(adjacency)
                assert normalized.dtype == (dtype if dtype.is_floating_point else torch.float32)
                torch.testing.assert_close(normalized, torch.eye(3, dtype=normalized.dtype).unsqueeze(0))
                assert _graph().normalize_adjacency(adjacency, dtype=torch.float64).dtype == torch.float64

    def test_normalization_handles_independent_graphs_and_isolated_nodes(self):
        adjacency = torch.tensor([[[0, 1], [1, 0]], [[0, 0], [0, 0]]], dtype=torch.bool)
        expected = torch.tensor([[[0.5, 0.5], [0.5, 0.5]], [[1.0, 0.0], [0.0, 1.0]]])
        torch.testing.assert_close(_graph().normalize_adjacency(adjacency), expected)

    def test_normalization_rejects_invalid_adjacency(self):
        for adjacency in [torch.ones(3, 3), torch.ones(1, 2, 3), torch.empty(0, 2, 2), torch.empty(1, 0, 0), torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]), torch.tensor([[[0.0, 0.5], [0.5, 0.0]]]), torch.tensor([[[0.0, -1.0], [-1.0, 0.0]]]), torch.tensor([[[0.0, float('nan')], [float('nan'), 0.0]]]), torch.tensor([[[0.0, float('inf')], [float('inf'), 0.0]]]), torch.ones(1, 2, 2, dtype=torch.complex64)]:
            with self.subTest(adjacency=adjacency):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().normalize_adjacency(adjacency)

    def test_normalization_rejects_non_real_floating_output_dtype(self):
        for dtype in [torch.bool, torch.int64, torch.complex64]:
            with self.subTest(dtype=dtype):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().normalize_adjacency(torch.eye(2).unsqueeze(0), dtype=dtype)

    def test_gcn_propagation_matches_hand_computed_path_without_implicit_activation(self):
        edge = 1.0 / math.sqrt(6.0)
        normalized = torch.tensor([[[0.5, edge, 0.0], [edge, 1.0 / 3.0, edge], [0.0, edge, 0.5]]], dtype=torch.float64)
        nodes = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]], dtype=torch.float64)
        layer = _graph().GCNLayer(2, 1).double()
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[2.0], [-1.0]], dtype=torch.float64))
        expected = torch.tensor([[[1.0 - 2.0 * edge], [7.0 * edge - 2.0 / 3.0], [2.5 - 2.0 * edge]]], dtype=torch.float64)
        torch.testing.assert_close(layer(nodes, normalized), expected, rtol=1e-14, atol=1e-14)
        assert layer.bias is None
        with torch.no_grad():
            layer.weight.neg_()
        torch.testing.assert_close(layer(nodes, normalized), -expected, rtol=1e-14, atol=1e-14)

    def test_optional_bias_is_added_after_graph_propagation(self):
        normalized = _graph().normalize_adjacency(torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.bool))
        layer = _graph().GCNLayer(1, 2, bias=True)
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.copy_(torch.tensor([2.0, -3.0]))
        torch.testing.assert_close(layer(torch.ones(1, 3, 1), normalized), torch.tensor([[[2.0, -3.0]] * 3]))

    def test_mean_readout_and_gcn_keep_real_node_weight_and_bias_gradients(self):
        nodes = torch.tensor([[[1.0], [2.0], [4.0]]], dtype=torch.float64, requires_grad=True)
        normalized = torch.tensor([[[0.5, 0.5, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float64)
        layer = _graph().GCNLayer(1, 1, bias=True).double()
        with torch.no_grad():
            layer.weight.fill_(2.0)
            layer.bias.fill_(1.0)
        _graph().mean_readout(layer(nodes, normalized)).sum().backward()
        torch.testing.assert_close(nodes.grad, torch.full_like(nodes, 2.0 / 3.0))
        torch.testing.assert_close(layer.weight.grad, torch.tensor([[7.0 / 3.0]], dtype=torch.float64))
        torch.testing.assert_close(layer.bias.grad, torch.tensor([1.0], dtype=torch.float64))

    def test_gcn_is_node_permutation_equivariant_and_mean_readout_is_invariant(self):
        nodes = torch.tensor([[[1.0, 2.0], [3.0, -2.0], [5.0, 4.0]]])
        normalized = _graph().normalize_adjacency(torch.tensor([[[0, 1, 0], [1, 0, 1], [0, 1, 0]]], dtype=torch.bool))
        layer = _graph().GCNLayer(2, 2)
        permutation = torch.tensor([2, 0, 1])
        original = layer(nodes, normalized)
        permuted = layer(nodes[:, permutation], normalized[:, permutation][:, :, permutation])
        torch.testing.assert_close(permuted, original[:, permutation])
        torch.testing.assert_close(_graph().mean_readout(permuted), _graph().mean_readout(original))

    def test_mean_readout_averages_each_samples_nodes_independently(self):
        nodes = torch.tensor([[[1.0, 4.0], [3.0, -2.0]], [[5.0, 0.0], [-1.0, 2.0]]])
        torch.testing.assert_close(_graph().mean_readout(nodes), torch.tensor([[2.0, 1.0], [2.0, 1.0]]))

    def test_gcn_rejects_invalid_feature_dimensions(self):
        for dimensions in [(0, 2), (2, 0), (-1, 2), (True, 2), (1.5, 2)]:
            with self.subTest(dimensions=dimensions):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().GCNLayer(*dimensions)

    def test_gcn_rejects_incompatible_or_invalid_inputs(self):
        for nodes, adjacency in [(torch.ones(3, 2), torch.eye(3).unsqueeze(0)), (torch.ones(1, 3, 3), torch.eye(3).unsqueeze(0)), (torch.ones(1, 3, 2), torch.eye(2).unsqueeze(0)), (torch.ones(2, 3, 2), torch.eye(3).unsqueeze(0)), (torch.empty(0, 3, 2), torch.empty(0, 3, 3)), (torch.empty(1, 0, 2), torch.empty(1, 0, 0)), (torch.full((1, 3, 2), float('nan')), torch.eye(3).unsqueeze(0)), (torch.ones(1, 3, 2), torch.full((1, 3, 3), float('inf'))), (torch.ones(1, 3, 2), torch.eye(3, dtype=torch.bool).unsqueeze(0))]:
            with self.subTest(nodes=nodes, adjacency=adjacency):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().GCNLayer(2, 2)(nodes, adjacency)

    def test_mean_readout_rejects_empty_malformed_or_nonfinite_features(self):
        for nodes in [torch.ones(3, 2), torch.empty(0, 3, 2), torch.empty(1, 0, 2), torch.empty(1, 3, 0), torch.full((1, 3, 2), float('nan')), torch.full((1, 3, 2), float('inf')), torch.ones(1, 3, 2, dtype=torch.long)]:
            with self.subTest(nodes=nodes):
                with self.assertRaises((TypeError, ValueError)):
                    _graph().mean_readout(nodes)

if __name__ == "__main__":
    unittest.main()
