"""Test measurements independently of CUDA using real generators and proc fixtures."""

import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def module():
    try:
        return importlib.import_module("tools.profile_training")
    except ImportError as exc:
        raise AssertionError("training profiler is not implemented") from exc


class TimingTests(unittest.TestCase):
    def test_timing_includes_consumer_work_and_excludes_warmup(self):
        profile = module()
        stats, clock = {}, [0.0]
        seen = []
        for value in profile.timed_batches(["first", "second", "third"], stats, "train", lambda: clock[0], lambda: None, warmup=1):
            seen.append(value)
            clock[0] += 2.0
        self.assertEqual(seen, ["first", "second", "third"])
        self.assertEqual(stats["train"]["batches"], 3)
        self.assertEqual(stats["train"]["measured_batches"], 2)
        self.assertEqual(stats["train"]["measured_seconds"], 4.0)

    def test_loading_time_is_included_and_errors_propagate(self):
        profile = module()
        stats, clock = {}, [0.0]
        def source():
            clock[0] += 3.0
            yield "one"
            raise RuntimeError("missing band")
        iterator = profile.timed_batches(source(), stats, "val", lambda: clock[0], lambda: None, warmup=0)
        self.assertEqual(next(iterator), "one")
        clock[0] += 2.0
        with self.assertRaisesRegex(RuntimeError, "missing band"):
            next(iterator)
        self.assertEqual(stats["val"]["measured_seconds"], 5.0)


class MemoryTests(unittest.TestCase):
    def test_process_family_includes_loader_children_and_pss(self):
        profile = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meminfo").write_text("MemAvailable: 9000 kB\n")
            for pid, rss, pss, children in ((10, 200, 150, "11"), (11, 100, 75, "")):
                proc = root / str(pid)
                (proc / "task" / str(pid)).mkdir(parents=True)
                (proc / "status").write_text(f"VmRSS: {rss} kB\n")
                (proc / "smaps_rollup").write_text(f"Pss: {pss} kB\n")
                (proc / "task" / str(pid) / "children").write_text(children)
            sample = profile.read_process_memory(10, root)
            self.assertEqual(sample["rss_sum_kib"], 300)
            self.assertEqual(sample["pss_sum_kib"], 225)
            self.assertEqual(sample["host_available_kib"], 9000)
            self.assertEqual(sample["pids"], [10, 11])

    def test_missing_memory_measurements_are_null_not_zero(self):
        profile = module()
        with tempfile.TemporaryDirectory() as tmp:
            sample = profile.read_process_memory(1234, Path(tmp))
            self.assertIsNone(sample["rss_sum_kib"])
            self.assertIsNone(sample["pss_sum_kib"])
            self.assertIsNone(sample["host_available_kib"])

    def test_report_collision_preserves_old_report(self):
        profile = module()
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.json"
            profile.write_json_new(report, {"status": "failed"})
            with self.assertRaises(FileExistsError):
                profile.write_json_new(report, {"status": "passed"})
            self.assertIn("failed", report.read_text())

    def test_tee_persists_training_output_and_preserves_console(self):
        profile = module()
        console, logfile = io.StringIO(), io.StringIO()
        tee = profile.Tee(console, logfile)
        self.assertEqual(tee.write("epoch 1\n"), 8)
        tee.flush()
        self.assertEqual(console.getvalue(), "epoch 1\n")
        self.assertEqual(logfile.getvalue(), "epoch 1\n")

    def test_preflight_failure_has_nonzero_exit_and_retained_diagnostics(self):
        profile = module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "failed-run"
            result = subprocess.run([sys.executable, profile.__file__, "--repo", str(root / "missing-repo"),
                "--output-dir", str(output), "--gpu", "3080", "--hourly-rate", "0.93",
                "--stage", "formal", "--model", "swin", "--seed", "44"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            report = json.loads((output / "profile.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertFalse(report["artifacts_verified"])
            self.assertIn("Traceback", (output / "training.log").read_text())


if __name__ == "__main__":
    unittest.main()
