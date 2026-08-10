"""Checkpoint identity / round-trip tests (research diary 2026-08-05, P0).

Covers:
- GAViT rejects unknown grouping / edge_type / integration instead of
  silently building a different architecture;
- save_checkpoint writes a sidecar that lets evaluation code rebuild the
  exact same model and reproduce its logits;
- CLI overrides conflicting with checkpoint metadata are rejected;
- missing metadata falls back to CLI values.
"""

import os
import tempfile
import unittest

import torch

from models.gavit import GAViT
import utils
from utils import (load_checkpoint_metadata, resolve_arch_config,
                   save_checkpoint, set_seed)

ARCH = dict(
    num_classes=7,
    num_regions=4,
    knn_k=2,
    gat_hidden=16,
    gat_heads=2,
    gat_layers=2,
    dropout=0.1,
    grouping="attentive_spatial",
    edge_type="knn",
    integration="token_feedback",
)


def require_helper(name):
    try:
        return getattr(utils, name)
    except AttributeError as exc:
        raise AssertionError(f"utils.{name} is not implemented") from exc


class ConfigValidationTests(unittest.TestCase):
    def test_unknown_grouping_rejected(self):
        with self.assertRaises(ValueError):
            GAViT(grouping="attentive_spatiall", pretrained=False)

    def test_unknown_edge_type_rejected(self):
        with self.assertRaises(ValueError):
            GAViT(edge_type="knn2", pretrained=False)

    def test_unknown_integration_rejected(self):
        with self.assertRaises(ValueError):
            # "graph_only" was historically accepted by the test CLI but
            # never implemented by the model class.
            GAViT(integration="graph_only", pretrained=False)


class CheckpointRoundTripTests(unittest.TestCase):
    def test_metadata_roundtrip_reproduces_logits(self):
        set_seed(0)
        model = GAViT(pretrained=False, **ARCH)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits_ref = model(x)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "model.pth")
            save_checkpoint(model, ckpt, {
                "model": "gavit",
                "dataset": "unit-test",
                "architecture": dict(ARCH),
                "training": {"seed": 0},
                "best": {"metric": "val_acc", "value": 0.0, "epoch": 0},
            })

            meta = load_checkpoint_metadata(ckpt)
            self.assertIsNotNone(meta)
            self.assertEqual(meta["architecture"]["num_regions"], 4)
            self.assertEqual(meta["training"]["seed"], 0)

            cfg, source = resolve_arch_config({}, meta, dict(ARCH))
            self.assertEqual(source, "metadata")
            rebuilt = GAViT(pretrained=False, **cfg)
            rebuilt.load_state_dict(torch.load(ckpt, weights_only=True))
            rebuilt.eval()
            with torch.no_grad():
                logits_new = rebuilt(x)

        self.assertTrue(torch.allclose(logits_ref, logits_new, atol=1e-6))

    def test_cli_conflict_with_metadata_rejected(self):
        meta = {"architecture": dict(ARCH)}
        with self.assertRaises(ValueError):
            resolve_arch_config({"num_regions": 9}, meta, dict(ARCH))

    def test_cli_fallback_without_metadata(self):
        overrides = {k: None for k in ARCH}
        overrides["num_regions"] = 9
        cfg, source = resolve_arch_config(overrides, None, dict(ARCH))
        self.assertEqual(source, "cli")
        self.assertEqual(cfg["num_regions"], 9)
        self.assertEqual(cfg["knn_k"], ARCH["knn_k"])


class TrainingStateTests(unittest.TestCase):
    def test_last_state_restores_model_optimizer_scheduler_and_epoch(self):
        save_training_state = require_helper("save_training_state")
        load_training_state = require_helper("load_training_state")

        set_seed(7)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        optimizer.zero_grad()
        loss = model(torch.ones(2, 3)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        expected_model = {
            key: value.detach().clone() for key, value in model.state_dict().items()
        }
        expected_scheduler_epoch = scheduler.last_epoch
        expected_lr = optimizer.param_groups[0]["lr"]

        with tempfile.TemporaryDirectory() as tmp:
            state_path = os.path.join(tmp, "model.last.train_state.pth")
            save_training_state(
                model,
                optimizer,
                scheduler,
                state_path,
                epoch=3,
                best_metric=71.2,
            )

            restored_model = torch.nn.Linear(3, 2)
            restored_optimizer = torch.optim.AdamW(
                restored_model.parameters(), lr=9e-4
            )
            restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                restored_optimizer, T_max=10
            )
            start_epoch, best_metric = load_training_state(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                state_path,
                map_location="cpu",
            )

        for key, expected in expected_model.items():
            self.assertTrue(
                torch.equal(restored_model.state_dict()[key], expected), key
            )
        self.assertTrue(restored_optimizer.state)
        self.assertEqual(restored_scheduler.last_epoch, expected_scheduler_epoch)
        self.assertAlmostEqual(
            restored_optimizer.param_groups[0]["lr"], expected_lr
        )
        self.assertEqual(start_epoch, 4)
        self.assertEqual(best_metric, 71.2)

    def test_last_state_missing_file_is_fatal(self):
        load_training_state = require_helper("load_training_state")
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10
        )

        with self.assertRaises(FileNotFoundError):
            load_training_state(
                model,
                optimizer,
                scheduler,
                "/definitely/missing/train-state.pth",
                map_location="cpu",
            )


if __name__ == "__main__":
    unittest.main()
