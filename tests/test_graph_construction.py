import unittest

import torch

from models.graph_construction import build_knn_graph


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


if __name__ == "__main__":
    unittest.main()
