"""Local launch gates only; no claim about server data or CUDA training."""
import csv
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


class Seed45Tests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "tools/run_bigearth_seed45.py"
        spec = importlib.util.spec_from_file_location("seed45", path)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)

    def test_full_file_gate_and_frozen_split(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            patch = p / "patch"
            patch.mkdir()
            for band in ("B02", "B03", "B04"):
                (patch / f"patch_{band}.tif").write_bytes(b"fixture")
            csv_path = p / "train.csv"
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["patch_path"] + [f"label_{i}" for i in range(19)])
                w.writerow([str(patch)] + [1] * 19)
            sha = self.tool.digest(csv_path)
            result = self.tool.scan_split(csv_path, 1, sha, set())
            self.assertEqual(result["count"], 1)
            with self.assertRaises(RuntimeError):
                self.tool.scan_split(csv_path, 1, "bad", set())
            with self.assertRaises(RuntimeError):
                self.tool.scan_split(csv_path, 1, sha, {str(patch)})
            (patch / "patch.jpg").write_bytes(b"different input")
            with self.assertRaises(RuntimeError):
                self.tool.scan_split(csv_path, 1, sha, set())
            (patch / "patch.jpg").unlink()
            (patch / "patch_B04.tif").unlink()
            with self.assertRaises(RuntimeError):
                self.tool.scan_split(csv_path, 1, sha, set())

    def test_failure_stops_queue_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            events = p / "events"
            script = p / "fake.py"
            script.write_text("import pathlib,sys\np=pathlib.Path(sys.argv[1]); p.open('a').write(sys.argv[2]); print('child output',flush=True); sys.exit(int(sys.argv[3]))\n")
            jobs = [("first", [sys.executable, str(script), str(events), "a", "0"]),
                    ("second", [sys.executable, str(script), str(events), "b", "7"]),
                    ("third", [sys.executable, str(script), str(events), "c", "0"])]
            with self.assertRaises(subprocess.CalledProcessError):
                self.tool.run_jobs(p, jobs, lambda: None)
            self.assertEqual(events.read_text(), "ab")
            self.assertEqual((p / "second/run.exit").read_text(), "exit_code=7\n")
            self.assertFalse((p / "third").exists())
            with self.assertRaises(FileExistsError):
                self.tool.run_jobs(p, jobs, lambda: None)

    def test_recipe_and_test_use_best_weight(self):
        root = Path("/out")
        for model in ("swin", "gavit"):
            argv = self.tool.entry_args(root, f"{model}_train")
            values = dict(zip(argv[1::2], argv[2::2]))
            self.assertEqual(values["--seed"], "45")
            self.assertEqual(values["--epochs"], "30")
            self.assertEqual(values["--batch_size"], "32")
            self.assertEqual(values["--run_stage"], "formal")
            self.assertEqual(values["--edge_type"], "sparse_hybrid")
            test = self.tool.entry_args(root, f"{model}_test")
            test_values = dict(zip(test[1::2], test[2::2]))
            self.assertEqual(test_values["--ckpt"], values["--checkpoint_path"])
            self.assertNotIn("--resume", argv)

    def test_worker_records_exact_metrics_without_changing_save_return(self):
        import torch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "swin_smoke").mkdir()
            self.tool.REPO = root
            self.tool.identity = lambda: None
            self.tool.output = lambda argv: "fixture GPU identity"
            (root / "train_bigearth.py").write_text('''import json,sys,hashlib,torch
from pathlib import Path
from types import SimpleNamespace
from utils import save_epoch_artifacts
CKPT_PATH = sys.argv[sys.argv.index("--checkpoint_path")+1]
TRAIN_STATE_PATH = str(Path(CKPT_PATH).with_suffix(".last.train_state.pth"))
model = torch.nn.Linear(3,19)
val_set = [(torch.ones(3), torch.zeros(19))]
avg_loss, val_f1 = 0.123456789, 67.8912345
result = save_epoch_artifacts(model, SimpleNamespace(param_groups=[{"lr":0.0}]), None, CKPT_PATH, {}, epoch=1, metric_name="val_mAP", metric_value=72.12345678, best_metric=0.0)
assert result == (72.12345678, True), result
''')

            def original(model, optimizer, scheduler, ckpt, metadata, **kw):
                torch.save(model.state_dict(), ckpt)
                path = Path(ckpt)
                path.with_suffix(".meta.json").write_text(json.dumps({"best": {"epoch": 1, "value": kw["metric_value"]}, "checkpoint_sha256": self.tool.digest(path)}))
                torch.save({"epoch": 1}, path.with_suffix(".last.train_state.pth"))
                return kw["metric_value"], True

            original_to = torch.Tensor.to
            def cpu_to(tensor, device, *args, **kwargs):
                return original_to(tensor, "cpu" if device == "cuda" else device, *args, **kwargs)

            old_cwd, old_argv, old_path = os.getcwd(), sys.argv, sys.path[:]
            try:
                with patch.dict(sys.modules, {"utils": types.SimpleNamespace(save_epoch_artifacts=original)}), patch.multiple(torch.cuda, is_available=lambda: True, device_count=lambda: 1, get_device_name=lambda n: "CPU fixture", reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 0, max_memory_reserved=lambda: 0), patch.object(torch.Tensor, "to", cpu_to), patch.object(importlib.metadata, "version", return_value="fixture"), redirect_stdout(io.StringIO()):
                    self.tool.worker(root, "swin_smoke")
            finally:
                os.chdir(old_cwd)
                sys.argv, sys.path = old_argv, old_path
            report = json.loads((root / "swin_smoke/run.json").read_text())
            history = json.loads((root / "swin_smoke/history.json").read_text())
            self.assertEqual(history[0]["val_mAP_percent"], 72.12345678)
            self.assertEqual(history[0]["train_loss"], 0.123456789)
            self.assertEqual(report["reload_max_logit_diff"], 0)
            self.assertEqual(report["status"], "complete")


if __name__ == "__main__":
    unittest.main()
