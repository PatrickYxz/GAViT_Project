"""Pretraining mapping boundaries; small fixtures are not real ImageNet models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

try:
    from external_baselines.hgtnet.pretrained import (
        apply_pretrained_state, audit_pretrained_state, load_pretrained_file,
    )
except ImportError:
    apply_pretrained_state = audit_pretrained_state = load_pretrained_file = None


class MappingFixture(nn.Module):
    """Actual CPU parameters under target prefixes, with deliberately tiny dimensions."""
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.projection = nn.Linear(2, 2)
        self.patch_embed.norm = nn.LayerNorm(2)
        self.stages = nn.ModuleList()
        for index in range(3):
            stage = nn.Module()
            block = nn.Module()
            block.norm1 = nn.LayerNorm(2)
            block.register_buffer("relative_position_index", torch.tensor([0, 1]))
            stage.blocks = nn.Sequential(block)
            stage.da = nn.Linear(2, 2)
            stage.downsample = nn.Module()
            if index < 2:
                stage.downsample.norm = nn.LayerNorm(2)
                stage.downsample.reduction = nn.Linear(2, 2, bias=False)
            else:
                stage.downsample.projection = nn.Conv2d(2, 2, 2, 2)
                stage.downsample.norm = nn.LayerNorm(2)
            self.stages.append(stage)
        self.graph_transition1 = nn.Linear(2, 2)
        self.graph_blocks1 = nn.Sequential(nn.Linear(2, 2))
        self.graph_transition2 = nn.Linear(2, 2)
        self.graph_blocks2 = nn.Sequential(nn.Linear(2, 2))
        self.head = nn.Linear(2, 3)


def source_fixture(model):
    """Explicit key table independent of production mapper, with distinct copied values."""
    pairs = [
        ("features.0.0.weight", "patch_embed.projection.weight"),
        ("features.0.0.bias", "patch_embed.projection.bias"),
        ("features.0.2.weight", "patch_embed.norm.weight"),
        ("features.0.2.bias", "patch_embed.norm.bias"),
    ]
    for stage, feature in [(0, 1), (1, 3), (2, 5)]:
        for suffix in ["norm1.weight", "norm1.bias", "relative_position_index"]:
            pairs.append((f"features.{feature}.0.{suffix}", f"stages.{stage}.blocks.0.{suffix}"))
    for stage, feature in [(0, 2), (1, 4)]:
        for suffix in ["norm.weight", "norm.bias", "reduction.weight"]:
            pairs.append((f"features.{feature}.{suffix}", f"stages.{stage}.downsample.{suffix}"))
    target = model.state_dict()
    result = {}
    for index, (source_key, target_key) in enumerate(pairs, start=1):
        result[source_key] = torch.full_like(target[target_key], index)
    result["features.6.norm.weight"] = torch.ones(2)
    result["features.7.0.norm1.weight"] = torch.ones(2)
    result["norm.weight"] = torch.ones(2)
    result["head.weight"] = torch.ones(3, 2)
    return result, pairs


class PretrainedMappingTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(callable(audit_pretrained_state), "pretrained audit API must be implemented")
        self.model = MappingFixture()
        self.source, self.pairs = source_fixture(self.model)

    def unchanged_after_error(self, source):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        with self.assertRaises(ValueError):
            apply_pretrained_state(self.model, source)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_audit_is_read_only_and_reports_exact_mapping_and_new_parameters(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        report = audit_pretrained_state(self.model, self.source)
        self.assertEqual({(item["source"], item["target"]) for item in report["compatible"]}, set(self.pairs))
        self.assertNotIn("loaded", report)
        self.assertTrue(report["can_apply"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["unexpected"], [])
        self.assertIn("stages.2.downsample.projection.weight", report["intentionally_initialized"])
        self.assertIn("stages.2.downsample.norm.weight", report["intentionally_initialized"])
        self.assertIn("head.weight", report["intentionally_initialized"])
        self.assertEqual(set(report["source_intentionally_skipped"]), {
            "features.6.norm.weight", "features.7.0.norm1.weight", "norm.weight", "head.weight"})
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)
        json.dumps(report, allow_nan=False)

    def test_apply_copies_real_tensors_and_preserves_all_new_modules(self):
        before = {key: value.clone() for key, value in self.model.state_dict().items()}
        report = apply_pretrained_state(self.model, self.source)
        self.assertEqual(len(report["loaded"]), 19)
        self.assertEqual(report["coverage"]["loaded_tensors"], 19)
        self.assertEqual(report["coverage"]["loaded_numel"], 44)
        after = self.model.state_dict()
        for source_key, target_key in self.pairs:
            self.assertTrue(torch.equal(after[target_key], self.source[source_key]), target_key)
        for key in report["intentionally_initialized"]:
            self.assertTrue(torch.equal(after[key], before[key]), key)
        self.assertGreater(report["coverage"]["loaded_target_numel_fraction"], 0)
        self.assertLess(report["coverage"]["loaded_target_numel_fraction"], 1)

    def test_missing_pretrained_target_is_reported_and_blocks_all_mutation(self):
        del self.source["features.3.0.norm1.weight"]
        report = audit_pretrained_state(self.model, self.source)
        self.assertEqual(report["missing"], ["stages.1.blocks.0.norm1.weight"])
        self.unchanged_after_error(self.source)

    def test_unexpected_source_blocks_all_mutation(self):
        self.source["features.3.0.typo"] = torch.ones(2)
        self.assertEqual(audit_pretrained_state(self.model, self.source)["unexpected"], ["features.3.0.typo"])
        self.unchanged_after_error(self.source)

    def test_typo_in_skipped_source_namespace_is_still_unexpected(self):
        self.source["features.6.not_a_real_parameter"] = torch.ones(2)
        report = audit_pretrained_state(self.model, self.source)
        self.assertEqual(report["unexpected"], ["features.6.not_a_real_parameter"])
        self.unchanged_after_error(self.source)

    def test_unknown_target_is_not_mislabeled_as_intentionally_initialized(self):
        self.model.unknown_module = nn.Linear(2, 2)
        report = audit_pretrained_state(self.model, self.source)
        self.assertEqual(report["missing"], ["unknown_module.bias", "unknown_module.weight"])
        self.unchanged_after_error(self.source)

    def test_shape_and_dtype_mismatch_prevent_partial_loading(self):
        self.source["features.0.0.weight"] = torch.ones(3, 2)
        self.source["features.5.0.norm1.bias"] = torch.ones(2, dtype=torch.float64)
        report = audit_pretrained_state(self.model, self.source)
        self.assertEqual(report["shape_mismatch"][0]["target"], "patch_embed.projection.weight")
        self.assertEqual(report["dtype_mismatch"][0]["target"], "stages.2.blocks.0.norm1.bias")
        self.unchanged_after_error(self.source)

    def test_nonfinite_loaded_or_intentionally_skipped_values_block_loading(self):
        for source_key in ["features.5.0.norm1.bias", "head.weight"]:
            with self.subTest(source_key=source_key):
                source = {key: value.clone() for key, value in self.source.items()}
                source[source_key].flatten()[0] = float("inf")
                report = audit_pretrained_state(self.model, source)
                self.assertEqual([item["source"] for item in report["nonfinite"]], [source_key])
                self.unchanged_after_error(source)

    def test_schema_only_and_meta_are_never_reported_as_successful_loading(self):
        report = audit_pretrained_state(self.model, self.source, validate_values=False)
        self.assertEqual(report["mode"], "schema_only")
        self.assertFalse(report["can_apply"])
        self.assertNotIn("loaded", report)
        with torch.device("meta"):
            meta_model = MappingFixture()
        meta_source = {key: value.to("meta") for key, value in self.source.items()}
        report = audit_pretrained_state(meta_model, meta_source)
        self.assertEqual(report["mode"], "schema_only")
        self.assertFalse(report["can_apply"])
        self.assertEqual(len(report["compatible"]), 19)
        with self.assertRaises(ValueError):
            apply_pretrained_state(meta_model, meta_source)
        self.unchanged_after_error(meta_source)

    def test_sparse_and_non_tensor_state_are_rejected_before_mutation(self):
        self.source["features.0.0.weight"] = torch.ones(2, 2).to_sparse()
        self.assertFalse(audit_pretrained_state(self.model, self.source)["can_apply"])
        self.unchanged_after_error(self.source)
        self.source["features.0.0.weight"] = "not a tensor"
        with self.assertRaises((TypeError, ValueError)):
            audit_pretrained_state(self.model, self.source)

    def test_file_loader_requires_complete_official_hash_and_checks_bytes_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pth"
            torch.save(self.source, path)
            before = {key: value.clone() for key, value in self.model.state_dict().items()}
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            for expected in ["68c6b09e", "g" * 64, actual_sha, "68c6b09e" + "0" * 56]:
                with self.subTest(expected=expected), self.assertRaises(ValueError):
                    load_pretrained_file(self.model, path, expected)
            for key, value in self.model.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]), key)


if __name__ == "__main__":
    unittest.main()
