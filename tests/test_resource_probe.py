"""Resource preflight tests: protect datasets, budgets and experiment identity."""

import csv
from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def module():
    try:
        return importlib.import_module("tools.resource_probe")
    except ImportError as exc:
        raise AssertionError("resource probe is not implemented") from exc


def write_split(root, name, rows):
    with (root / f"{name}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patch_path", *[f"label_{i}" for i in range(19)]])
        writer.writerows(rows)


class SplitTests(unittest.TestCase):
    def test_sampling_is_repeatable_and_never_copies_test_split(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("train", "val"):
                write_split(root, name, [[f"/{name}/{i}", *([1] * 19)] for i in range(16)])
            # Invalid test content must not be opened by the smoke sampler.
            (root / "test.csv").write_text("not a dataset")
            original = (root / "train.csv").read_bytes()
            a = probe.prepare_smoke_split(root, root / "a", train_count=8, val_count=4)
            b = probe.prepare_smoke_split(root, root / "b", train_count=8, val_count=4)
            self.assertEqual(a["splits"]["train"]["sample_sha256"], b["splits"]["train"]["sample_sha256"])
            self.assertEqual(a["splits"]["val"]["rows"], 4)
            self.assertFalse((root / "a" / "test.csv").exists())
            self.assertEqual((root / "train.csv").read_bytes(), original)

    def test_existing_output_is_never_overwritten(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing"
            output.mkdir()
            (output / "keep").write_text("user artifact")
            with self.assertRaises(FileExistsError):
                probe.prepare_smoke_split(root, output)
            self.assertEqual((output / "keep").read_text(), "user artifact")

    def test_malformed_and_duplicate_rows_are_rejected(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rows in ([["/a", *([2] * 19)]], [["", *([0] * 19)]], [["/a", *([0] * 19)]] * 2):
                write_split(root, "train", rows)
                with self.assertRaises(ValueError):
                    probe.read_rows(root / "train.csv")

    def test_scan_finds_missing_band_not_just_existing_patch_directory(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            patch = root / "patch"
            patch.mkdir()
            (patch / "patch_B02.tif").touch()
            (patch / "patch_B03.tif").touch()
            write_split(root, "train", [[str(patch), *([1] * 19)]])
            report = probe.scan_dataset(root, {"train": 1})
            self.assertEqual(report["train"]["missing_bands"], 1)
            self.assertFalse(report["train"]["complete"])
            with self.assertRaises(ValueError):
                probe.scan_dataset(root, {"train": 2})


class CommandTests(unittest.TestCase):
    def test_formal_pair_preserves_protocol_and_unique_outputs(self):
        probe = module()
        swin = probe.training_args(Path("/repo"), Path("/run"), "swin", "formal", 44)
        gavit = probe.training_args(Path("/repo"), Path("/run"), "gavit", "formal", 44)
        for command in (swin, gavit):
            self.assertEqual(command[command.index("--epochs") + 1], "30")
            self.assertEqual(command[command.index("--batch_size") + 1], "32")
            self.assertEqual(command[command.index("--seed") + 1], "44")
        self.assertEqual(gavit[gavit.index("--edge_type") + 1], "sparse_hybrid")
        self.assertNotEqual(swin[-1], gavit[-1])
        with self.assertRaises(ValueError):
            probe.training_args(Path("/repo"), Path("/run"), "gavit", "formal", 42)

    def test_plan_cli_never_creates_output_or_imports_cuda(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "not-created"
            result = subprocess.run([sys.executable, probe.__file__, "plan", "--gpu", "3080", "--stage", "formal", "--seed", "44", "--output-dir", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertEqual(len(plan["runs"]), 2)
            self.assertFalse(plan["executed"])
            self.assertFalse(output.exists())
            self.assertIn("test_bigearth.py", plan["runs"][0]["evaluation_command"])
            self.assertIn("swin.pth", plan["runs"][0]["evaluation_command"])

    def test_formal_gate_rejects_incomplete_or_mismatched_model_reports(self):
        probe = module()
        identity = {"commit": "frozen", "gpu": "3080"}
        valid = {"status": "passed", "identity": identity, "models": {
            model: {"status": "passed", "stage": "smoke", "seed": 42,
                    "model": model, "identity": identity, "artifacts_verified": True}
            for model in ("swin", "gavit")}}
        with tempfile.TemporaryDirectory() as tmp:
            filename = Path(tmp) / "smoke_report.json"
            filename.write_text(json.dumps(valid))
            self.assertEqual(probe.validate_smoke_gate(Path(tmp), identity)["status"], "passed")
            for key, value in (("stage", "formal"), ("seed", 45), ("identity", {}), ("artifacts_verified", False)):
                broken = deepcopy(valid)
                broken["models"]["gavit"][key] = value
                filename.write_text(json.dumps(broken))
                with self.assertRaises(RuntimeError):
                    probe.validate_smoke_gate(Path(tmp), identity)

    def test_smoke_requires_full_preflight_records_not_just_sample_csv(self):
        probe = module()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((RuntimeError, FileNotFoundError)):
                probe.validate_smoke_inputs(Path(tmp), {})


if __name__ == "__main__":
    unittest.main()
