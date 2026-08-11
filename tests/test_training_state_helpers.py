"""Torch-free contract tests for the training-state helpers in ``utils``.

The local development environment intentionally has no PyTorch.  These tests
load the real ``utils.py`` module with a small serialization-compatible torch
boundary so the checkpoint lifecycle can still follow a local RED/GREEN cycle.
The existing ``test_checkpoint_roundtrip`` suite remains the real-PyTorch
integration test that must run on Featurize.
"""

import importlib.util
import inspect
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class SerializationTorch(types.ModuleType):
    """Minimal stand-in for torch's save/load boundary used by ``utils``."""

    Tensor = object
    nn = types.SimpleNamespace(Module=object)

    def __init__(self):
        super().__init__("torch")

    @staticmethod
    def save(payload, path):
        with open(path, "wb") as handle:
            pickle.dump(payload, handle)

    @staticmethod
    def load(path, *, map_location=None, weights_only=False):
        del map_location, weights_only
        with open(path, "rb") as handle:
            return pickle.load(handle)


class StatefulObject:
    def __init__(self, state):
        self._state = state
        self.loaded_state = None

    def state_dict(self):
        return self._state

    def load_state_dict(self, state):
        self.loaded_state = state


def load_utils_without_real_torch():
    fake_torch = SerializationTorch()
    module_name = "utils_training_state_contract"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "utils.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        spec.loader.exec_module(module)
    return module


class TrainingStateHelperContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.utils = load_utils_without_real_torch()

    def test_round_trip_restores_all_states_and_returns_next_epoch(self):
        self.assertTrue(
            hasattr(self.utils, "save_training_state"),
            "utils.save_training_state is not implemented",
        )
        self.assertTrue(
            hasattr(self.utils, "load_training_state"),
            "utils.load_training_state is not implemented",
        )

        model = StatefulObject({"weight": [1.0, 2.0]})
        optimizer = StatefulObject({"step": 9, "lr": 0.0002})
        scheduler = StatefulObject({"last_epoch": 3})

        with tempfile.TemporaryDirectory() as tmp:
            state_path = str(Path(tmp) / "run.last.train_state.pth")
            self.utils.save_training_state(
                model,
                optimizer,
                scheduler,
                state_path,
                epoch=3,
                best_metric=71.2,
            )

            restored_model = StatefulObject({"weight": [-1.0]})
            restored_optimizer = StatefulObject({"step": 0, "lr": 0.9})
            restored_scheduler = StatefulObject({"last_epoch": 0})
            start_epoch, best_metric = self.utils.load_training_state(
                restored_model,
                restored_optimizer,
                restored_scheduler,
                state_path,
                map_location="cpu",
            )

        self.assertEqual(restored_model.loaded_state, {"weight": [1.0, 2.0]})
        self.assertEqual(
            restored_optimizer.loaded_state,
            {"step": 9, "lr": 0.0002},
        )
        self.assertEqual(restored_scheduler.loaded_state, {"last_epoch": 3})
        self.assertEqual(start_epoch, 4)
        self.assertEqual(best_metric, 71.2)

    def test_missing_training_state_is_fatal(self):
        self.assertTrue(
            hasattr(self.utils, "load_training_state"),
            "utils.load_training_state is not implemented",
        )

        stateful = StatefulObject({})
        with self.assertRaises(FileNotFoundError):
            self.utils.load_training_state(
                stateful,
                stateful,
                stateful,
                "/definitely/missing/run.last.train_state.pth",
                map_location="cpu",
            )

    def test_validated_resume_requires_checkpoint_metadata(self):
        self.assertTrue(
            hasattr(self.utils, "load_validated_training_state"),
            "utils.load_validated_training_state is not implemented",
        )

        stateful = StatefulObject({})
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            with self.assertRaisesRegex(FileNotFoundError, "metadata"):
                self.utils.load_validated_training_state(
                    stateful,
                    stateful,
                    stateful,
                    ckpt_path,
                    map_location="cpu",
                    expected_model="gavit",
                    expected_dataset="BigEarthNet-19",
                    expected_num_classes=19,
                    expected_architecture={
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                )

    def test_validated_resume_requires_last_training_state(self):
        self.assertTrue(
            hasattr(self.utils, "load_validated_training_state"),
            "utils.load_validated_training_state is not implemented",
        )

        stateful = StatefulObject({})
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            self.utils.write_checkpoint_metadata(
                ckpt_path,
                {
                    "model": "gavit",
                    "dataset": "BigEarthNet-19",
                    "architecture": {
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                },
            )
            with self.assertRaisesRegex(FileNotFoundError, "Training state"):
                self.utils.load_validated_training_state(
                    stateful,
                    stateful,
                    stateful,
                    ckpt_path,
                    map_location="cpu",
                    expected_model="gavit",
                    expected_dataset="BigEarthNet-19",
                    expected_num_classes=19,
                    expected_architecture={
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                )

    def test_validated_resume_rejects_architecture_mismatch(self):
        self.assertTrue(
            hasattr(self.utils, "load_validated_training_state"),
            "utils.load_validated_training_state is not implemented",
        )

        stateful = StatefulObject({})
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            self.utils.write_checkpoint_metadata(
                ckpt_path,
                {
                    "model": "gavit",
                    "dataset": "BigEarthNet-19",
                    "architecture": {
                        "num_classes": 19,
                        "edge_type": "hybrid",
                    },
                },
            )
            with self.assertRaisesRegex(ValueError, "edge_type"):
                self.utils.load_validated_training_state(
                    stateful,
                    stateful,
                    stateful,
                    ckpt_path,
                    map_location="cpu",
                    expected_model="gavit",
                    expected_dataset="BigEarthNet-19",
                    expected_num_classes=19,
                    expected_architecture={
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                )

    def test_validated_resume_uses_last_state_without_best_checkpoint(self):
        self.assertTrue(
            hasattr(self.utils, "load_validated_training_state"),
            "utils.load_validated_training_state is not implemented",
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            state_path = str(Path(tmp) / "model.last.train_state.pth")
            metadata = {
                "model": "gavit",
                "dataset": "BigEarthNet-19",
                "architecture": {
                    "num_classes": 19,
                    "edge_type": "knn",
                },
            }
            self.utils.write_checkpoint_metadata(ckpt_path, metadata)
            self.assertIn(
                "resume_identity",
                inspect.signature(self.utils.save_training_state).parameters,
                "training states are not bound to experiment identity",
            )
            self.utils.save_training_state(
                StatefulObject({"weight": [4.0]}),
                StatefulObject({"step": 8}),
                StatefulObject({"last_epoch": 4}),
                state_path,
                epoch=4,
                best_metric=73.5,
                resume_identity=metadata,
            )

            model = StatefulObject({})
            optimizer = StatefulObject({})
            scheduler = StatefulObject({})
            start_epoch, best_metric = self.utils.load_validated_training_state(
                model,
                optimizer,
                scheduler,
                ckpt_path,
                map_location="cpu",
                expected_model="gavit",
                expected_dataset="BigEarthNet-19",
                expected_num_classes=19,
                expected_architecture={
                    "num_classes": 19,
                    "edge_type": "knn",
                },
            )

        self.assertEqual(model.loaded_state, {"weight": [4.0]})
        self.assertEqual(optimizer.loaded_state, {"step": 8})
        self.assertEqual(scheduler.loaded_state, {"last_epoch": 4})
        self.assertEqual(start_epoch, 5)
        self.assertEqual(best_metric, 73.5)

    def test_validated_resume_rejects_state_from_another_experiment(self):
        self.assertIn(
            "resume_identity",
            inspect.signature(self.utils.save_training_state).parameters,
            "training states are not bound to experiment identity",
        )

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            state_path = str(Path(tmp) / "model.last.train_state.pth")
            metadata = {
                "model": "gavit",
                "dataset": "BigEarthNet-19",
                "architecture": {
                    "num_classes": 19,
                    "edge_type": "knn",
                },
            }
            self.utils.write_checkpoint_metadata(ckpt_path, metadata)
            self.utils.save_training_state(
                StatefulObject({"weight": [4.0]}),
                StatefulObject({"step": 8}),
                StatefulObject({"last_epoch": 4}),
                state_path,
                epoch=4,
                best_metric=73.5,
                resume_identity={
                    "model": "gavit",
                    "dataset": "BigEarthNet-19",
                    "architecture": {
                        "num_classes": 19,
                        "edge_type": "hybrid",
                    },
                },
            )

            stateful = StatefulObject({})
            with self.assertRaisesRegex(ValueError, "identity"):
                self.utils.load_validated_training_state(
                    stateful,
                    stateful,
                    stateful,
                    ckpt_path,
                    map_location="cpu",
                    expected_model="gavit",
                    expected_dataset="BigEarthNet-19",
                    expected_num_classes=19,
                    expected_architecture={
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                )

    def test_failed_state_write_preserves_previous_resume_point(self):
        stateful = StatefulObject({"value": "old"})
        with tempfile.TemporaryDirectory() as tmp:
            state_path = str(Path(tmp) / "model.last.train_state.pth")
            self.utils.save_training_state(
                stateful,
                stateful,
                stateful,
                state_path,
                epoch=1,
                best_metric=60.0,
            )
            original_bytes = Path(state_path).read_bytes()
            original_save = self.utils.torch.save

            def fail_after_partial_write(payload, path):
                del payload
                Path(path).write_bytes(b"partial")
                raise OSError("simulated interrupted write")

            self.utils.torch.save = fail_after_partial_write
            try:
                with self.assertRaisesRegex(OSError, "interrupted"):
                    self.utils.save_training_state(
                        StatefulObject({"value": "new"}),
                        stateful,
                        stateful,
                        state_path,
                        epoch=2,
                        best_metric=61.0,
                    )
            finally:
                self.utils.torch.save = original_save

            self.assertEqual(Path(state_path).read_bytes(), original_bytes)
            self.assertFalse(Path(state_path + ".tmp").exists())

    def test_validated_resume_rejects_best_checkpoint_hash_mismatch(self):
        metadata = {
            "model": "gavit",
            "dataset": "BigEarthNet-19",
            "architecture": {
                "num_classes": 19,
                "edge_type": "knn",
            },
        }
        model = StatefulObject({"weight": [1.0]})
        optimizer = StatefulObject({"step": 1})
        scheduler = StatefulObject({"last_epoch": 1})

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            self.utils.save_epoch_artifacts(
                model,
                optimizer,
                scheduler,
                ckpt_path,
                metadata,
                epoch=1,
                metric_name="val_mAP",
                metric_value=80.0,
                best_metric=0.0,
            )
            Path(ckpt_path).write_bytes(b"different best model")

            stateful = StatefulObject({})
            with self.assertRaisesRegex(ValueError, "hash"):
                self.utils.load_validated_training_state(
                    stateful,
                    stateful,
                    stateful,
                    ckpt_path,
                    map_location="cpu",
                    expected_model="gavit",
                    expected_dataset="BigEarthNet-19",
                    expected_num_classes=19,
                    expected_architecture={
                        "num_classes": 19,
                        "edge_type": "knn",
                    },
                )

    def test_non_best_epoch_updates_last_state_without_replacing_best_model(self):
        self.assertTrue(
            hasattr(self.utils, "save_epoch_artifacts"),
            "utils.save_epoch_artifacts is not implemented",
        )

        model = StatefulObject({"weight": [1.0]})
        optimizer = StatefulObject({"step": 1})
        scheduler = StatefulObject({"last_epoch": 1})
        metadata = {
            "model": "gavit",
            "dataset": "BigEarthNet-19",
            "architecture": {"num_classes": 19},
        }

        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = str(Path(tmp) / "model.pth")
            best_metric, improved = self.utils.save_epoch_artifacts(
                model,
                optimizer,
                scheduler,
                ckpt_path,
                metadata,
                epoch=1,
                metric_name="val_mAP",
                metric_value=80.0,
                best_metric=0.0,
            )
            self.assertTrue(improved)

            model._state = {"weight": [2.0]}
            optimizer._state = {"step": 2}
            scheduler._state = {"last_epoch": 2}
            best_metric, improved = self.utils.save_epoch_artifacts(
                model,
                optimizer,
                scheduler,
                ckpt_path,
                metadata,
                epoch=2,
                metric_name="val_mAP",
                metric_value=70.0,
                best_metric=best_metric,
            )

            best_payload = self.utils.torch.load(
                ckpt_path,
                map_location="cpu",
                weights_only=True,
            )
            last_payload = self.utils.torch.load(
                str(Path(tmp) / "model.last.train_state.pth"),
                map_location="cpu",
                weights_only=True,
            )

        self.assertFalse(improved)
        self.assertEqual(best_metric, 80.0)
        self.assertEqual(best_payload, {"weight": [1.0]})
        self.assertEqual(last_payload["model"], {"weight": [2.0]})
        self.assertEqual(last_payload["optimizer"], {"step": 2})
        self.assertEqual(last_payload["scheduler"], {"last_epoch": 2})
        self.assertEqual(last_payload["epoch"], 2)
        self.assertEqual(last_payload["best_metric"], 80.0)


if __name__ == "__main__":
    unittest.main()
