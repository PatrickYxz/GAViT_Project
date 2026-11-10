"""A small operator-chain fixture, not the unspecified full DFAGCN model."""
import importlib
import unittest

import torch


class ComponentChainTests(unittest.TestCase):
    def test_three_graph_branches_fuse_with_fc_and_backpropagate_independently(self):
        graph = importlib.import_module("external_baselines.dfagcn.graph")
        fusion = importlib.import_module("external_baselines.dfagcn.fusion")
        generator = torch.Generator().manual_seed(42)
        # Twelve nodes/four channels are a unit fixture, not a paper pooling choice.
        nodes = [torch.randn(2, 12, 4, generator=generator, requires_grad=True) for _ in range(3)]
        layers = [graph.GCNLayer(4, 512) for _ in range(3)]
        outputs = []
        for values, layer in zip(nodes, layers):
            adjacency = graph.knn_union_adjacency(values, metric="euclidean")
            assert not adjacency.requires_grad
            outputs.append(graph.mean_readout(layer(values, graph.normalize_adjacency(adjacency))))
        fc = torch.randn(2, 4096, generator=generator, requires_grad=True)
        output = fusion.weighted_concat(*outputs, fc, **fusion.paper_coefficients("NWPU"))
        assert output.shape == (2, 5632)
        loss = output.square().mean()
        loss.backward()
        for values, layer in zip(nodes, layers):
            assert values.grad is not None and torch.isfinite(values.grad).all()
            assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
            assert layer.weight.grad.abs().sum() > 0
        assert fc.grad is not None and fc.grad.abs().sum() > 0
