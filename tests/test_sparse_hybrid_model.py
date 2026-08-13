import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gavit import GAViT
from utils import load_checkpoint_metadata, resolve_arch_config, save_checkpoint


class TinyBackbone(nn.Module):
    hidden_dim = 8

    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Linear(3, self.hidden_dim)

    def forward(self, x):
        pooled_rgb = x.mean(dim=(-2, -1))
        token = self.proj(pooled_rgb).unsqueeze(1)
        return token.expand(-1, 49, -1)


class SparseHybridModelTests(unittest.TestCase):
    architecture = {
        "num_classes": 19,
        "num_regions": 16,
        "knn_k": 5,
        "gat_hidden": 4,
        "gat_heads": 2,
        "gat_layers": 2,
        "dropout": 0.1,
        "grouping": "attentive_spatial",
        "edge_type": "sparse_hybrid",
        "integration": "token_feedback",
    }

    def test_forward_and_backward_are_finite(self):
        with patch("models.gavit.SwinBackbone", TinyBackbone):
            model = GAViT(
                pretrained=False,
                **self.architecture,
            )

        images = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 2, (2, 19), dtype=torch.float32)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        self.assertEqual(logits.shape, (2, 19))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(loss))
        graph_gradients = [
            parameter.grad
            for parameter in model.graph_reasoning.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(graph_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in graph_gradients))

    def test_metadata_roundtrip_rebuilds_same_sparse_hybrid_logits(self):
        images = torch.randn(2, 3, 8, 8)
        with patch("models.gavit.SwinBackbone", TinyBackbone):
            model = GAViT(pretrained=False, **self.architecture)
            model.eval()
            with torch.no_grad():
                expected_logits = model(images)

            with tempfile.TemporaryDirectory() as temporary_directory:
                checkpoint = os.path.join(temporary_directory, "model.pth")
                save_checkpoint(
                    model,
                    checkpoint,
                    {
                        "model": "gavit",
                        "dataset": "unit-test",
                        "architecture": dict(self.architecture),
                        "training": {"seed": 42},
                        "best": {"metric": "val_mAP", "value": 0.0, "epoch": 0},
                    },
                )
                metadata = load_checkpoint_metadata(checkpoint)
                rebuilt_architecture, source = resolve_arch_config(
                    {}, metadata, dict(self.architecture)
                )
                rebuilt = GAViT(pretrained=False, **rebuilt_architecture)
                rebuilt.load_state_dict(torch.load(checkpoint, weights_only=True))
                rebuilt.eval()
                with torch.no_grad():
                    actual_logits = rebuilt(images)

        self.assertEqual(source, "metadata")
        self.assertTrue(torch.allclose(expected_logits, actual_logits, atol=1e-6))

    def test_rejects_non_sixteen_region_sparse_hybrid_model(self):
        with patch("models.gavit.SwinBackbone", TinyBackbone):
            with self.assertRaisesRegex(ValueError, "num_regions=9"):
                GAViT(
                    num_regions=9,
                    edge_type="sparse_hybrid",
                    pretrained=False,
                )


if __name__ == "__main__":
    unittest.main()
