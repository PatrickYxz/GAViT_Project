"""Torch-free tests for experiment identity and artifact metadata."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


def load_identity_module():
    """Import the production module while reporting missing behavior as RED."""
    try:
        return importlib.import_module("experiment_identity")
    except (ImportError, AttributeError) as exc:
        raise AssertionError(
            "experiment_identity helpers are not implemented yet"
        ) from exc


def require_helper(module, name):
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AssertionError(f"experiment_identity.{name} is not implemented") from exc


class CheckpointIdentityTests(unittest.TestCase):
    def test_sparse_hybrid_topology_identity_is_explicit(self):
        identity = load_identity_module()

        self.assertEqual(
            identity.graph_topology_identity("sparse_hybrid", knn_k=5),
            {
                "name": "sparse_hybrid_4n_top2",
                "spatial_connectivity": 4,
                "feature_k": 2,
                "spatial_directed_edges": 48,
                "feature_directed_edges": 32,
                "total_directed_edges": 80,
                "message_direction": "neighbor_to_query",
                "cosine_role": "topology_only",
            },
        )

    def test_knn_topology_identity_uses_configured_neighbor_count(self):
        identity = load_identity_module()

        self.assertEqual(
            identity.graph_topology_identity("knn", knn_k=5),
            {
                "name": "corrected_knn",
                "feature_k": 5,
                "message_direction": "neighbor_to_query",
                "cosine_role": "topology_only",
            },
        )

    def test_resume_identity_binds_graph_topology_snapshot(self):
        identity = load_identity_module()
        metadata = {
            "model": "gavit",
            "graph_topology": {"name": "sparse_hybrid_4n_top2"},
        }

        self.assertEqual(
            identity.resume_identity_for(metadata),
            metadata,
        )

    def test_checkpoint_identity_accepts_exact_match(self):
        identity = load_identity_module()
        metadata = {
            "model": "gavit",
            "dataset": "NWPU-RESISC45",
            "architecture": {"num_classes": 45},
        }

        identity.validate_checkpoint_identity(
            metadata,
            expected_model="gavit",
            expected_dataset="NWPU-RESISC45",
            expected_num_classes=45,
        )

    def test_checkpoint_identity_rejects_wrong_dataset(self):
        identity = load_identity_module()
        metadata = {
            "model": "gavit",
            "dataset": "BigEarthNet-19",
            "architecture": {"num_classes": 19},
        }

        with self.assertRaisesRegex(ValueError, "dataset"):
            identity.validate_checkpoint_identity(
                metadata,
                expected_model="gavit",
                expected_dataset="NWPU-RESISC45",
                expected_num_classes=45,
            )

    def test_checkpoint_identity_rejects_wrong_model(self):
        identity = load_identity_module()
        metadata = {
            "model": "swin",
            "dataset": "NWPU-RESISC45",
            "architecture": {"num_classes": 45},
        }

        with self.assertRaisesRegex(ValueError, "model"):
            identity.validate_checkpoint_identity(
                metadata,
                expected_model="gavit",
                expected_dataset="NWPU-RESISC45",
                expected_num_classes=45,
            )

    def test_checkpoint_identity_rejects_wrong_class_count(self):
        identity = load_identity_module()
        metadata = {
            "model": "gavit",
            "dataset": "NWPU-RESISC45",
            "architecture": {"num_classes": 19},
        }

        with self.assertRaisesRegex(ValueError, "num_classes"):
            identity.validate_checkpoint_identity(
                metadata,
                expected_model="gavit",
                expected_dataset="NWPU-RESISC45",
                expected_num_classes=45,
            )

    def test_complete_architecture_rejects_missing_key(self):
        identity = load_identity_module()
        metadata = {"architecture": {"num_regions": 16}}

        with self.assertRaisesRegex(ValueError, "knn_k"):
            identity.validate_complete_architecture(
                metadata, ("num_regions", "knn_k")
            )


class MetadataFileTests(unittest.TestCase):
    def test_metadata_path_replaces_checkpoint_suffix(self):
        identity = load_identity_module()
        self.assertEqual(
            identity.metadata_path_for("checkpoints/model.pth"),
            "checkpoints/model.meta.json",
        )

    def test_metadata_roundtrip_adds_checkpoint_path(self):
        identity = load_identity_module()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model.pth")
            identity.write_checkpoint_metadata(
                checkpoint,
                {
                    "model": "gavit",
                    "dataset": "unit-test",
                    "architecture": {"num_classes": 3},
                },
            )

            loaded = identity.load_checkpoint_metadata(checkpoint)

        self.assertEqual(loaded["checkpoint_path"], checkpoint)
        self.assertEqual(loaded["model"], "gavit")
        self.assertIn("saved_at", loaded)

    def test_metadata_file_is_valid_json(self):
        identity = load_identity_module()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = os.path.join(tmp, "model.pth")
            identity.write_checkpoint_metadata(
                checkpoint,
                {"model": "swin", "dataset": "unit-test"},
            )
            with open(identity.metadata_path_for(checkpoint)) as handle:
                raw = json.load(handle)

        self.assertEqual(raw["model"], "swin")


class GitStateTests(unittest.TestCase):
    def test_git_state_distinguishes_clean_and_dirty_tree(self):
        identity = load_identity_module()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "init", "-q"], cwd=tmp, check=True)
            subprocess.run(
                ["git", "config", "user.name", "GAViT Test"],
                cwd=tmp,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "gavit-test@example.invalid"],
                cwd=tmp,
                check=True,
            )
            tracked = Path(tmp, "tracked.txt")
            tracked.write_text("clean\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=tmp, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "initial"],
                cwd=tmp,
                check=True,
            )

            clean = identity.get_git_state(tmp)
            tracked.write_text("dirty\n")
            dirty = identity.get_git_state(tmp)

        self.assertFalse(clean["dirty"])
        self.assertTrue(dirty["dirty"])
        self.assertEqual(clean["commit"], dirty["commit"])
        self.assertTrue(clean["commit"])

    def test_clean_git_requirement_rejects_dirty_or_unknown_state(self):
        identity = load_identity_module()
        assert_clean = require_helper(identity, "assert_clean_git_state")
        assert_clean({"commit": "abc1234", "dirty": False})

        invalid_states = (
            {"commit": "abc1234", "dirty": True},
            {"commit": None, "dirty": None},
        )
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(RuntimeError):
                    assert_clean(state)


class ArtifactPathTests(unittest.TestCase):
    def test_run_tag_accepts_safe_slug(self):
        identity = load_identity_module()
        validate_run_tag = require_helper(identity, "validate_run_tag")
        self.assertEqual(
            validate_run_tag("corrected_knn.seed42"),
            "corrected_knn.seed42",
        )

    def test_run_tag_rejects_path_characters_and_spaces(self):
        identity = load_identity_module()
        validate_run_tag = require_helper(identity, "validate_run_tag")
        for value in ("../formal", "a/b", "", "contains space"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_run_tag(value)

    def test_run_stage_accepts_only_declared_stages(self):
        identity = load_identity_module()
        validate_run_stage = require_helper(identity, "validate_run_stage")
        for stage in ("smoke", "proxy", "formal"):
            with self.subTest(stage=stage):
                self.assertEqual(validate_run_stage(stage), stage)
        with self.assertRaises(ValueError):
            validate_run_stage("experiment")

    def test_training_state_path_is_separate_from_best_checkpoint(self):
        identity = load_identity_module()
        training_state_path_for = require_helper(
            identity, "training_state_path_for"
        )
        self.assertEqual(
            training_state_path_for("checkpoints/model.pth"),
            "checkpoints/model.last.train_state.pth",
        )

    def test_fresh_output_accepts_unused_paths(self):
        identity = load_identity_module()
        assert_fresh = require_helper(identity, "assert_fresh_output_paths")
        with tempfile.TemporaryDirectory() as tmp:
            assert_fresh(
                [os.path.join(tmp, "model.pth"), os.path.join(tmp, "model.json")]
            )

    def test_fresh_output_rejects_every_existing_artifact(self):
        identity = load_identity_module()
        assert_fresh = require_helper(identity, "assert_fresh_output_paths")
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp, "model.pth")
            second = Path(tmp, "model.meta.json")
            first.touch()
            second.touch()

            with self.assertRaisesRegex(FileExistsError, "model.pth") as error:
                assert_fresh([str(first), str(second)])

        self.assertIn("model.meta.json", str(error.exception))


if __name__ == "__main__":
    unittest.main()
