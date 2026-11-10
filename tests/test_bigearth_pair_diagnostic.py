"""Behavior tests for paired validation analysis; no GPU dependencies."""

import csv
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys


def core():
    try:
        return importlib.import_module("tools.bigearth_pair_core")
    except ModuleNotFoundError as exc:
        raise AssertionError("paired diagnostic core has not been implemented") from exc


def sample(index, labels):
    return {"row_index": index, "patch_path": f"/data/patch{index}", "labels": labels}


class SplitAndSelectionTests(unittest.TestCase):
    def test_rejects_invalid_binary_labels_and_duplicate_paths(self):
        for content in [
            "patch_path,label_0,label_1\n/a,1,NaN\n",
            "patch_path,label_0,label_1\n/a,1,0.2\n",
            "patch_path,label_0,label_1\n/a,1,0\n/a,0,1\n",
            "patch_path,label_0,label_1\n,1,0\n",
        ]:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                file = Path(tmp) / "val.csv"
                file.write_text(content)
                with self.assertRaises(ValueError):
                    list(core().iter_split(file, 2))

    def test_keeps_csv_row_identity_and_accepts_binary_float_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "val.csv"
            file.write_text("patch_path,label_0,label_1\n/a,1.0,0.0\n/b,0,1\n")
            self.assertEqual(list(core().iter_split(file, 2)), [
                {"row_index": 0, "patch_path": "/a", "labels": [1, 0]},
                {"row_index": 1, "patch_path": "/b", "labels": [0, 1]},
            ])

    def test_focus_supplements_do_not_change_uniform_sample(self):
        records = [sample(i, [int(i >= 8), int(i % 2 == 0)]) for i in range(10)]
        first = core().select_samples(records, 4, 42, [0], 0)
        enriched = core().select_samples(records, 4, 42, [0], 2)
        self.assertEqual([r["row_index"] for r in first], [0, 1, 4, 9])
        self.assertEqual([r["row_index"] for r in enriched if r["cohort"] == "uniform"], [0, 1, 4, 9])
        self.assertEqual([r["row_index"] for r in enriched if r["cohort"] == "focus_supplement"], [8])
        self.assertEqual(enriched, core().select_samples(records, 4, 42, [0], 2))
        self.assertTrue(all("cohort" not in r for r in records))

    def test_label_statistics_use_multilabel_cooccurrence(self):
        result = core().label_statistics([sample(0, [1, 1]), sample(1, [1, 0]), sample(2, [0, 0])], 2, [0])
        self.assertEqual(result, {"n": 3, "positive": [2, 1], "focus_cooccurrence": {0: [2, 1]}})


class MetricsAndCasesTests(unittest.TestCase):
    def test_average_precision_groups_tied_scores(self):
        # At .9: precision=1, recall=.5. At .8: precision=2/3, recall=1.
        m, curve = core().binary_metrics([1, 0, 1, 0], [.9, .8, .8, .1])
        self.assertAlmostEqual(m["ap"], 5 / 6)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 0, 1))
        self.assertAlmostEqual(m["f1"], .8)
        self.assertEqual([p["threshold"] for p in curve], [.9, .8, .1])
        tied, _ = core().binary_metrics([1, 1, 0, 0], [.9, .8, .8, .1])
        self.assertAlmostEqual(tied["ap"], m["ap"])

    def test_missing_positive_or_negative_is_not_rank_evidence(self):
        for labels, reason in [([0, 0], "no_positive"), ([1, 1], "no_negative")]:
            metrics, curve = core().binary_metrics(labels, [.2, .8])
            self.assertIsNone(metrics["ap"])
            self.assertEqual(metrics["rank_status"], reason)
            self.assertEqual(curve, [])

    def test_case_direction_and_threshold_include_true_negatives(self):
        self.assertEqual(core().case_bucket(1, .5, .49), "swin_only_correct")
        self.assertEqual(core().case_bucket(0, .5, .49), "gavit_only_correct")
        self.assertEqual(core().case_bucket(0, .1, .2), "both_correct")
        self.assertEqual(core().case_bucket(1, .1, .2), "both_wrong")

    def test_rejects_reordered_missing_or_nonfinite_predictions(self):
        manifest = core().select_samples([sample(0, [1]), sample(1, [0])], 2, 42, [], 0)
        invalid = [
            [{"row_index": 1, "scores": [.9]}, {"row_index": 0, "scores": [.1]}],
            [{"row_index": 0, "scores": [.9]}],
            [{"row_index": 0, "scores": [float("nan")]}, {"row_index": 1, "scores": [.1]}],
            [{"row_index": 0, "scores": [1.1]}, {"row_index": 1, "scores": [.1]}],
            [{"row_index": 0, "scores": [.8, .1]}, {"row_index": 1, "scores": [.1]}],
        ]
        for predictions in invalid:
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                core().validate_predictions(manifest, predictions, 1)

    def test_report_excludes_supplement_from_ap_and_preserves_all_cases(self):
        manifest = [dict(sample(i, [y]), cohort=cohort) for i, y, cohort in [
            (0, 1, "uniform"), (1, 0, "uniform"), (2, 1, "focus_supplement")]]
        sw = [{"row_index": i, "scores": [p]} for i, p in enumerate([.9, .1, .1])]
        ga = [{"row_index": i, "scores": [p]} for i, p in enumerate([.4, .6, .9])]
        with tempfile.TemporaryDirectory() as tmp:
            summary = core().write_analysis(Path(tmp), manifest, sw, ga, ["Coastal wetlands"], [0], 2)
            self.assertEqual(summary["uniform_n"], 2)
            self.assertEqual(summary["supplement_n"], 1)
            with (Path(tmp) / "per_class.csv").open() as f:
                row = next(csv.DictReader(f))
            self.assertEqual(float(row["swin_ap_percent"]), 100)
            self.assertEqual(float(row["gavit_ap_percent"]), 50)
            with (Path(tmp) / "cases.csv").open() as f:
                cases = list(csv.DictReader(f))
            self.assertEqual(len(cases), 3)
            self.assertEqual({r["row_index"] for r in cases}, {"0", "1", "2"})
            with (Path(tmp) / "case_counts.csv").open() as f:
                counts = list(csv.DictReader(f))
            self.assertEqual(len(counts), 8)  # four outcomes, each split by true label
            self.assertEqual(sum(int(r["available"]) for r in counts), 3)
            self.assertTrue((Path(tmp) / "pr_curves.csv").exists())
            json.loads((Path(tmp) / "summary.json").read_text())


class CommandTests(unittest.TestCase):
    script = Path(__file__).resolve().parents[1] / "tools/diagnose_bigearth_pair.py"

    def fixture(self, root):
        repo = root / "repo"
        (repo / "models").mkdir(parents=True)
        (repo / "models/bigearth_dataset.py").write_text("CLASSES_19 = " + repr([f"Class {i}" for i in range(19)]))
        splits = repo / "splits"
        splits.mkdir()
        for name in ("train", "val", "test"):
            with (splits / f"{name}.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["patch_path"] + [f"label_{i}" for i in range(19)])
                for i in range(10):
                    writer.writerow([f"/data/{name}_{i}"] + [int(i % 2 == 0)] * 19)
        return repo, splits

    def run_cli(self, *args):
        self.assertTrue(self.script.exists(), "paired diagnostic command is missing")
        return subprocess.run([sys.executable, str(self.script), *map(str, args)], capture_output=True, text=True)

    def test_prepare_only_reads_validation_without_gpu_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, splits = self.fixture(Path(tmp))
            output = Path(tmp) / "out"
            args = ["--repo", repo, "--data-dir", splits, "--out", output, "--samples", 4,
                    "--min-focus-support", 0, "--prepare-only"]
            before = (splits / "val.csv").read_bytes()
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual([r["row_index"] for r in manifest], [0, 1, 4, 9])
            self.assertTrue(all("/val_" in r["patch_path"] for r in manifest))
            self.assertEqual(json.loads((output / "run.json").read_text())["status"], "prepared_only")
            self.assertFalse((output / "summary.json").exists())
            sentinel = (output / "manifest.json").read_bytes()
            repeat = self.run_cli(*args)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual((output / "manifest.json").read_bytes(), sentinel)
            self.assertEqual((splits / "val.csv").read_bytes(), before)

    def test_count_mismatch_stops_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, splits = self.fixture(Path(tmp))
            output = Path(tmp) / "out"
            result = self.run_cli("--repo", repo, "--data-dir", splits, "--out", output,
                                  "--expected-counts", "10,11,10", "--prepare-only")
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(json.loads((output / "failure.json").read_text())["status"], "failed")
            self.assertFalse((output / "summary.json").exists())

    def test_gallery_escapes_labels_and_keeps_both_prediction_directions(self):
        try:
            cli = importlib.import_module("tools.diagnose_bigearth_pair")
        except ModuleNotFoundError as exc:
            self.fail(f"paired diagnostic command is missing: {exc}")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            records = [dict(sample(i, [1]), cohort="uniform") for i in range(2)]
            sw = [{"row_index": 0, "scores": [.9]}, {"row_index": 1, "scores": [.1]}]
            ga = [{"row_index": 0, "scores": [.1]}, {"row_index": 1, "scores": [.9]}]
            core().write_analysis(output, records, sw, ga, ["Wetland <script>"], [0])
            cli.write_gallery(output)
            content = (output / "cases.html").read_text()
            self.assertNotIn("<script>", content)
            self.assertIn("&lt;script&gt;", content)
            self.assertIn("Swin正确 / GAViT错误", content)
            self.assertIn("GAViT正确 / Swin错误", content)

    def test_selected_images_are_checked_before_gpu_work(self):
        cli = importlib.import_module("tools.diagnose_bigearth_pair")
        self.assertTrue(hasattr(cli, "validate_image_paths"), "selected-file precheck is missing")
        with tempfile.TemporaryDirectory() as tmp:
            patch = Path(tmp) / "patch1"
            patch.mkdir()
            for band in ("B04", "B03"):
                (patch / f"patch1_{band}.tif").touch()
            records = [{"patch_path": str(patch)}]
            with self.assertRaises(FileNotFoundError):
                cli.validate_image_paths(records)
            (patch / "patch1_B02.tif").touch()
            cli.validate_image_paths(records)

    def test_failed_archive_does_not_mark_run_complete(self):
        cli = importlib.import_module("tools.diagnose_bigearth_pair")
        self.assertTrue(hasattr(cli, "finish_run"), "failure-safe finalization is missing")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "results.zip").write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                cli.finish_run(out, {"status": "complete"})
            self.assertFalse((out / "run.json").exists())
            self.assertEqual((out / "results.zip").read_bytes(), b"existing")

    def test_archive_contains_completed_metadata_and_relative_files(self):
        import zipfile
        cli = importlib.import_module("tools.diagnose_bigearth_pair")
        self.assertTrue(hasattr(cli, "finish_run"), "failure-safe finalization is missing")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "report.md").write_text("report")
            cli.finish_run(out, {"status": "complete"})
            with zipfile.ZipFile(out / "results.zip") as archive:
                self.assertEqual(sorted(archive.namelist()), ["report.md", "run.json"])
                self.assertEqual(json.loads(archive.read("run.json"))["status"], "complete")
            self.assertEqual(json.loads((out / "run.json").read_text())["status"], "complete")

    def test_launcher_worker_records_nonzero_exit_on_preflight_failure(self):
        launcher = self.script.parent.parent / "jobs/run_bigearth_pair_seed44.sh"
        self.assertTrue(launcher.exists(), "seed44 launcher is missing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo, _ = self.fixture(root)
            (repo / "bigearth_files").mkdir()
            (repo / "splits").rename(repo / "bigearth_files/splits")
            out = root / "run"
            out.mkdir()
            env = dict(os.environ, GAVIT_PAIR_REPO=str(repo), GAVIT_PAIR_WORK=str(root), GAVIT_PAIR_PYTHON=sys.executable)
            result = subprocess.run(["bash", str(launcher), "--worker", str(out)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((out / "diagnostic.exit").read_text(), f"diagnostic_exit={result.returncode}\n")
            self.assertTrue((out / "diagnosis/failure.json").exists())
            self.assertFalse((out / "diagnosis/run.json").exists())


if __name__ == "__main__":
    unittest.main()
