import unittest

import torch

from models.graph_construction import (
    build_knn_graph,
    build_sparse_hybrid_graph,
)


class BuildKnnGraphTests(unittest.TestCase):
    def test_each_query_node_receives_messages_from_its_nearest_neighbor(self):
        # Directed 1-NN relationships by cosine similarity:
        # query 0 selects 1, query 1 selects 0, query 2 selects 1.
        features = torch.tensor(
            [[[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]]],
            dtype=torch.float32,
        )

        edge_index, edge_weight, batch = build_knn_graph(features, k=1)

        self.assertEqual(
            set(map(tuple, edge_index.t().tolist())),
            {(1, 0), (0, 1), (1, 2)},
        )
        self.assertEqual(
            torch.bincount(edge_index[1], minlength=3).tolist(),
            [1, 1, 1],
        )
        self.assertEqual(edge_weight.shape, (3,))
        self.assertEqual(batch.tolist(), [0, 0, 0])


class BuildSparseHybridGraphTests(unittest.TestCase):
    def setUp(self):
        identity = torch.eye(16, dtype=torch.float32)
        self.features = torch.stack([identity, identity], dim=0)

    def test_builds_budgeted_unique_edges_without_crossing_batch_graphs(self):
        edge_index, edge_weight, batch = build_sparse_hybrid_graph(self.features)

        self.assertEqual(edge_index.shape, (2, 160))
        self.assertEqual(edge_weight.shape, (160,))
        self.assertEqual(batch.tolist(), [0] * 16 + [1] * 16)

        expected_spatial_indegree = [
            2, 3, 3, 2,
            3, 4, 4, 3,
            3, 4, 4, 3,
            2, 3, 3, 2,
        ]
        for batch_index in range(2):
            start = batch_index * 80
            local_edges = edge_index[:, start:start + 80] - batch_index * 16
            spatial_edges = local_edges[:, :48]
            feature_edges = local_edges[:, 48:]

            pairs = list(map(tuple, local_edges.t().tolist()))
            self.assertEqual(len(set(pairs)), 80)
            self.assertTrue(all(source != target for source, target in pairs))
            self.assertTrue(
                all(0 <= source < 16 and 0 <= target < 16 for source, target in pairs)
            )
            self.assertEqual(
                torch.bincount(spatial_edges[1], minlength=16).tolist(),
                expected_spatial_indegree,
            )
            for source, target in spatial_edges.t().tolist():
                source_row, source_col = divmod(source, 4)
                target_row, target_col = divmod(target, 4)
                self.assertEqual(
                    abs(source_row - target_row) + abs(source_col - target_col),
                    1,
                )
            self.assertEqual(
                torch.bincount(feature_edges[1], minlength=16).tolist(),
                [2] * 16,
            )

    def test_feature_edges_exclude_spatial_neighbors_and_break_ties_by_index(self):
        edge_index, _, _ = build_sparse_hybrid_graph(self.features[:1])
        feature_pairs = set(map(tuple, edge_index[:, 48:].t().tolist()))

        self.assertTrue({(2, 0), (3, 0)}.issubset(feature_pairs))
        self.assertTrue({(0, 5), (2, 5)}.issubset(feature_pairs))
        self.assertFalse({(1, 0), (4, 0), (1, 5), (4, 5), (6, 5), (9, 5)} & feature_pairs)

    def test_feature_edges_select_highest_cosine_non_spatial_candidates(self):
        features = torch.full((1, 16, 2), -1.0, dtype=torch.float32)
        features[0, 0] = torch.tensor([1.0, 0.0])
        features[0, 7] = torch.tensor([1.0, 0.0])
        features[0, 8] = torch.tensor([0.8, 0.6])

        edge_index, _, _ = build_sparse_hybrid_graph(features)
        query_zero_sources = edge_index[0, 48:][edge_index[1, 48:] == 0]

        self.assertEqual(query_zero_sources.tolist(), [7, 8])

    def test_rejects_non_batched_feature_tensor(self):
        with self.assertRaisesRegex(ValueError, "ndim=2"):
            build_sparse_hybrid_graph(torch.eye(16))

    def test_rejects_region_count_other_than_sixteen(self):
        with self.assertRaisesRegex(ValueError, "num_regions=9"):
            build_sparse_hybrid_graph(torch.randn(1, 9, 4))

    def test_rejects_feature_budget_other_than_two(self):
        with self.assertRaisesRegex(ValueError, "feature_k=3"):
            build_sparse_hybrid_graph(self.features, feature_k=3)


if __name__ == "__main__":
    unittest.main()
