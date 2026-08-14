import csv
import json
import os
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from baselines.bigearth.prepare_proxy_split import (
    EXPECTED_HEADER,
    ProxySplitError,
    evaluate_candidate,
    prepare_proxy_split,
    read_split_csv,
    sample_indices,
    select_proxy_candidate,
    sha256_file,
    summarize_indices,
)


def make_labels(primary: int, *, all_other_positive: bool = True):
    values = [primary]
    values.extend([1 if all_other_positive else 0] * 18)
    return values


def write_split(path: Path, label_rows, *, header=EXPECTED_HEADER):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, labels in enumerate(label_rows):
            writer.writerow([f"/data/patch_{index:03d}", *labels])


class ReadSplitCsvTests(unittest.TestCase):
    def test_reads_binary_schema_and_compacts_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            write_split(path, [make_labels(0), make_labels(1)])

            data = read_split_csv(path)

            self.assertEqual(data.size, 2)
            self.assertEqual(data.labels[0], bytes(make_labels(0)))
            self.assertEqual(data.positive_counts[0], 1)
            self.assertEqual(data.sha256, sha256_file(path))

    def test_rejects_reordered_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.csv"
            header = list(EXPECTED_HEADER)
            header[1], header[2] = header[2], header[1]
            write_split(path, [make_labels(1)], header=header)

            with self.assertRaisesRegex(ProxySplitError, "Unexpected CSV header"):
                read_split_csv(path)

    def test_rejects_empty_path_nonbinary_label_and_empty_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            empty_path = root / "empty_path.csv"
            with empty_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(EXPECTED_HEADER)
                writer.writerow(["", *make_labels(1)])
            with self.assertRaisesRegex(ProxySplitError, "patch_path must be non-empty"):
                read_split_csv(empty_path)

            nonbinary = root / "nonbinary.csv"
            write_split(nonbinary, [[2, *([1] * 18)]])
            with self.assertRaisesRegex(ProxySplitError, "must be binary"):
                read_split_csv(nonbinary)

            no_rows = root / "no_rows.csv"
            write_split(no_rows, [])
            with self.assertRaisesRegex(ProxySplitError, "has no data rows"):
                read_split_csv(no_rows)


class CandidateSelectionTests(unittest.TestCase):
    def _load_pair(self, root: Path, rows=40):
        train_path = root / "train.csv"
        val_path = root / "val.csv"
        labels = [make_labels(index % 2) for index in range(rows)]
        write_split(train_path, labels)
        write_split(val_path, list(reversed(labels)))
        return read_split_csv(train_path), read_split_csv(val_path)

    def test_sampling_is_deterministic_sorted_and_without_replacement(self):
        first = sample_indices(100, 10, 42)
        second = sample_indices(100, 10, 42)

        self.assertEqual(first, second)
        self.assertEqual(first, tuple(sorted(first)))
        self.assertEqual(len(set(first)), 10)

    def test_selects_minimum_score_then_lower_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = self._load_pair(Path(tmp))
            evaluated = [
                evaluate_candidate(train, val, fraction=0.25, seed=seed)
                for seed in range(42, 52)
            ]
            valid = [candidate for candidate in evaluated if candidate is not None]

            selected = select_proxy_candidate(
                train,
                val,
                fraction=0.25,
                seed_start=42,
                candidate_count=10,
            )

            expected = min(valid, key=lambda candidate: (candidate.score, candidate.seed))
            self.assertEqual(selected.seed, expected.seed)
            self.assertEqual(selected.score, expected.score)
            self.assertEqual(len(selected.train_indices), 10)
            self.assertEqual(len(selected.val_indices), 10)

    def test_equal_scores_choose_the_lower_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = [make_labels(1) for _ in range(20)]
            write_split(root / "train.csv", labels)
            write_split(root / "val.csv", labels)
            train = read_split_csv(root / "train.csv")
            val = read_split_csv(root / "val.csv")

            selected = select_proxy_candidate(
                train,
                val,
                fraction=0.1,
                seed_start=42,
                candidate_count=5,
            )

            self.assertEqual(selected.score, 0.0)
            self.assertEqual(selected.seed, 42)

    def test_missing_positive_class_rejects_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, _ = self._load_pair(Path(tmp))
            indices = tuple(index for index in range(10) if index % 2 == 0)

            self.assertIsNone(summarize_indices(train, indices))

    def test_all_invalid_candidates_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = [make_labels(1 if index == 0 else 0) for index in range(20)]
            write_split(root / "train.csv", labels)
            write_split(root / "val.csv", labels)
            train = read_split_csv(root / "train.csv")
            val = read_split_csv(root / "val.csv")
            bad_seed = next(
                seed
                for seed in range(1000)
                if 0 not in random.Random(seed).sample(range(20), 2)
            )

            with self.assertRaisesRegex(ProxySplitError, "All proxy candidates"):
                select_proxy_candidate(
                    train,
                    val,
                    fraction=0.1,
                    seed_start=bad_seed,
                    candidate_count=1,
                )

    def test_rejects_invalid_fraction_and_candidate_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            train, val = self._load_pair(Path(tmp))
            for fraction in (0.0, 1.0, float("nan")):
                with self.subTest(fraction=fraction):
                    with self.assertRaises(ProxySplitError):
                        evaluate_candidate(train, val, fraction=fraction, seed=42)
            with self.assertRaisesRegex(ProxySplitError, "candidate_count"):
                select_proxy_candidate(train, val, candidate_count=0)


class PrepareProxySplitTests(unittest.TestCase):
    def _write_inputs(self, root: Path, rows=20):
        input_dir = root / "input"
        input_dir.mkdir()
        labels = [make_labels(index % 2) for index in range(rows)]
        write_split(input_dir / "train.csv", labels)
        write_split(input_dir / "val.csv", list(reversed(labels)))
        return input_dir

    def test_writes_proxy_csvs_and_auditable_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_inputs(root)
            output_dir = root / "proxy"

            report = prepare_proxy_split(
                input_dir,
                output_dir,
                fraction=0.1,
                seed_start=42,
                candidate_count=100,
            )

            with (output_dir / "train.csv").open(encoding="utf-8", newline="") as handle:
                train_rows = list(csv.reader(handle))
            with (output_dir / "val.csv").open(encoding="utf-8", newline="") as handle:
                val_rows = list(csv.reader(handle))
            stored_report = json.loads((output_dir / "prevalence.json").read_text("utf-8"))

            self.assertEqual(len(train_rows), 3)
            self.assertEqual(len(val_rows), 3)
            self.assertEqual(tuple(train_rows[0]), EXPECTED_HEADER)
            self.assertEqual(stored_report, report)
            self.assertEqual(report["splits"]["train"]["proxy_rows"], 2)
            self.assertEqual(
                report["splits"]["train"]["output_sha256"],
                sha256_file(output_dir / "train.csv"),
            )
            self.assertTrue(
                all(
                    details["proxy_positive_count"] > 0
                    for details in report["splits"]["train"]["labels"].values()
                )
            )

    def test_refuses_to_overwrite_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_inputs(root)
            output_dir = root / "proxy"
            output_dir.mkdir()
            sentinel = output_dir / "train.csv"
            sentinel.write_text("keep me", encoding="utf-8")

            with self.assertRaisesRegex(ProxySplitError, "refusing to overwrite"):
                prepare_proxy_split(input_dir, output_dir, fraction=0.25)
            self.assertEqual(sentinel.read_text("utf-8"), "keep me")

    def test_publish_failure_rolls_back_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = self._write_inputs(root)
            output_dir = root / "proxy"
            real_replace = os.replace
            calls = 0

            def fail_on_second_publish(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("controlled publish failure")
                return real_replace(source, destination)

            with mock.patch(
                "baselines.bigearth.prepare_proxy_split.os.replace",
                side_effect=fail_on_second_publish,
            ):
                with self.assertRaisesRegex(OSError, "controlled publish failure"):
                    prepare_proxy_split(input_dir, output_dir, fraction=0.25)

            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
