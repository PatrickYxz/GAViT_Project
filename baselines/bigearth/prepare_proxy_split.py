"""Build a fixed, prevalence-matched BigEarthNet proxy split.

The proxy is selected from the existing formal train/validation CSVs.  For
each candidate seed, the script uniformly samples the requested fraction from
both splits, rejects candidates that lose a positive class, and minimizes the
maximum class-prevalence deviation across train and validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


NUM_CLASSES = 19
LABEL_COLUMNS = tuple(f"label_{index}" for index in range(NUM_CLASSES))
EXPECTED_HEADER = ("patch_path", *LABEL_COLUMNS)
REPORT_SCHEMA_VERSION = 1
ALGORITHM_NAME = "uniform_random_candidate_search_minimax_prevalence"


class ProxySplitError(ValueError):
    """Raised when proxy inputs or selection results violate the contract."""


@dataclass(frozen=True)
class SplitData:
    source_path: Path
    paths: tuple[str, ...]
    labels: tuple[bytes, ...]
    sha256: str
    positive_counts: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.paths)

    @property
    def prevalences(self) -> tuple[float, ...]:
        return tuple(count / self.size for count in self.positive_counts)


@dataclass(frozen=True)
class SplitMetrics:
    positive_counts: tuple[int, ...]
    prevalences: tuple[float, ...]
    absolute_deviations: tuple[float, ...]
    max_deviation: float


@dataclass(frozen=True)
class ProxyCandidate:
    seed: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    train_metrics: SplitMetrics
    val_metrics: SplitMetrics
    score: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_binary_label(value: str, *, path: Path, line_number: int, column: str) -> int:
    normalized = value.strip()
    if normalized in {"0", "0.0"}:
        return 0
    if normalized in {"1", "1.0"}:
        return 1
    raise ProxySplitError(
        f"{path}:{line_number}: {column} must be binary, got {value!r}"
    )


def _positive_counts(labels: Iterable[bytes]) -> tuple[int, ...]:
    counts = [0] * NUM_CLASSES
    for label_row in labels:
        for class_index, value in enumerate(label_row):
            counts[class_index] += value
    return tuple(counts)


def read_split_csv(path: str | Path) -> SplitData:
    source_path = Path(path)
    if not source_path.is_file():
        raise ProxySplitError(f"Split CSV not found: {source_path}")

    paths: list[str] = []
    labels: list[bytes] = []
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = tuple(next(reader))
        except StopIteration as exc:
            raise ProxySplitError(f"Split CSV is empty: {source_path}") from exc
        if header != EXPECTED_HEADER:
            raise ProxySplitError(
                f"Unexpected CSV header in {source_path}: expected "
                f"{EXPECTED_HEADER!r}, got {header!r}"
            )

        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_HEADER):
                raise ProxySplitError(
                    f"{source_path}:{line_number}: expected {len(EXPECTED_HEADER)} "
                    f"columns, got {len(row)}"
                )
            patch_path = row[0].strip()
            if not patch_path:
                raise ProxySplitError(
                    f"{source_path}:{line_number}: patch_path must be non-empty"
                )
            label_values = bytes(
                _parse_binary_label(
                    value,
                    path=source_path,
                    line_number=line_number,
                    column=LABEL_COLUMNS[index],
                )
                for index, value in enumerate(row[1:])
            )
            paths.append(patch_path)
            labels.append(label_values)

    if not paths:
        raise ProxySplitError(f"Split CSV has no data rows: {source_path}")

    frozen_labels = tuple(labels)
    return SplitData(
        source_path=source_path,
        paths=tuple(paths),
        labels=frozen_labels,
        sha256=sha256_file(source_path),
        positive_counts=_positive_counts(frozen_labels),
    )


def sample_indices(row_count: int, sample_count: int, seed: int) -> tuple[int, ...]:
    if not 0 < sample_count <= row_count:
        raise ProxySplitError(
            f"sample_count must be in [1, {row_count}], got {sample_count}"
        )
    return tuple(sorted(random.Random(seed).sample(range(row_count), sample_count)))


def summarize_indices(data: SplitData, indices: Sequence[int]) -> SplitMetrics | None:
    if not indices:
        raise ProxySplitError("Proxy indices must be non-empty")
    if any(index < 0 or index >= data.size for index in indices):
        raise ProxySplitError("Proxy index is outside the source split")
    if len(set(indices)) != len(indices):
        raise ProxySplitError("Proxy indices must be unique")

    counts = _positive_counts(data.labels[index] for index in indices)
    if any(count == 0 for count in counts):
        return None
    prevalences = tuple(count / len(indices) for count in counts)
    deviations = tuple(
        abs(proxy - full)
        for proxy, full in zip(prevalences, data.prevalences)
    )
    return SplitMetrics(
        positive_counts=counts,
        prevalences=prevalences,
        absolute_deviations=deviations,
        max_deviation=max(deviations),
    )


def evaluate_candidate(
    train: SplitData,
    val: SplitData,
    *,
    fraction: float,
    seed: int,
) -> ProxyCandidate | None:
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ProxySplitError(f"fraction must be finite and in (0, 1), got {fraction}")

    train_count = math.floor(train.size * fraction)
    val_count = math.floor(val.size * fraction)
    if train_count < 1 or val_count < 1:
        raise ProxySplitError(
            "fraction produces an empty proxy split: "
            f"train={train_count}, val={val_count}"
        )

    train_indices = sample_indices(train.size, train_count, seed)
    val_indices = sample_indices(val.size, val_count, seed)
    train_metrics = summarize_indices(train, train_indices)
    val_metrics = summarize_indices(val, val_indices)
    if train_metrics is None or val_metrics is None:
        return None

    return ProxyCandidate(
        seed=seed,
        train_indices=train_indices,
        val_indices=val_indices,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        score=max(train_metrics.max_deviation, val_metrics.max_deviation),
    )


def select_proxy_candidate(
    train: SplitData,
    val: SplitData,
    *,
    fraction: float = 0.1,
    seed_start: int = 42,
    candidate_count: int = 100,
) -> ProxyCandidate:
    if candidate_count <= 0:
        raise ProxySplitError(
            f"candidate_count must be positive, got {candidate_count}"
        )

    best: ProxyCandidate | None = None
    for seed in range(seed_start, seed_start + candidate_count):
        candidate = evaluate_candidate(train, val, fraction=fraction, seed=seed)
        if candidate is None:
            continue
        if best is None or (candidate.score, candidate.seed) < (best.score, best.seed):
            best = candidate

    if best is None:
        raise ProxySplitError(
            "All proxy candidates were rejected because at least one class "
            "had no positive sample in train or validation"
        )
    return best


def _write_proxy_csv(
    path: Path,
    data: SplitData,
    indices: Sequence[int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXPECTED_HEADER)
        for index in indices:
            writer.writerow(
                [data.paths[index], *("1.0" if value else "0.0" for value in data.labels[index])]
            )


def _split_report(
    data: SplitData,
    metrics: SplitMetrics,
    *,
    proxy_rows: int,
    output_sha256: str,
) -> dict:
    labels = {}
    for index, column in enumerate(LABEL_COLUMNS):
        labels[column] = {
            "full_positive_count": data.positive_counts[index],
            "proxy_positive_count": metrics.positive_counts[index],
            "full_prevalence": data.prevalences[index],
            "proxy_prevalence": metrics.prevalences[index],
            "absolute_deviation": metrics.absolute_deviations[index],
        }
    return {
        "source_path": str(data.source_path.resolve()),
        "source_sha256": data.sha256,
        "output_sha256": output_sha256,
        "source_rows": data.size,
        "proxy_rows": proxy_rows,
        "max_absolute_deviation": metrics.max_deviation,
        "labels": labels,
    }


def _new_temp_path(output_dir: Path, artifact_name: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{artifact_name}.", suffix=".tmp", dir=output_dir
    )
    os.close(descriptor)
    return Path(raw_path)


def prepare_proxy_split(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fraction: float = 0.1,
    seed_start: int = 42,
    candidate_count: int = 100,
) -> dict:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    targets = {
        "train": output_path / "train.csv",
        "val": output_path / "val.csv",
        "report": output_path / "prevalence.json",
    }

    if output_path.exists():
        if not output_path.is_dir():
            raise ProxySplitError(f"Output path is not a directory: {output_path}")
        existing = list(output_path.iterdir())
        if existing:
            raise ProxySplitError(
                f"Output directory must be empty; refusing to overwrite: {output_path}"
            )

    train = read_split_csv(input_path / "train.csv")
    val = read_split_csv(input_path / "val.csv")
    selection = select_proxy_candidate(
        train,
        val,
        fraction=fraction,
        seed_start=seed_start,
        candidate_count=candidate_count,
    )

    created_output_dir = not output_path.exists()
    output_path.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    published_paths: list[Path] = []
    try:
        train_temp = _new_temp_path(output_path, "train.csv")
        val_temp = _new_temp_path(output_path, "val.csv")
        report_temp = _new_temp_path(output_path, "prevalence.json")
        temporary_paths.extend((train_temp, val_temp, report_temp))

        _write_proxy_csv(train_temp, train, selection.train_indices)
        _write_proxy_csv(val_temp, val, selection.val_indices)
        train_hash = sha256_file(train_temp)
        val_hash = sha256_file(val_temp)

        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "algorithm": ALGORITHM_NAME,
            "fraction": fraction,
            "seed_start": seed_start,
            "candidate_count": candidate_count,
            "seed_end": seed_start + candidate_count - 1,
            "selected_seed": selection.seed,
            "selected_score": selection.score,
            "max_absolute_deviation": selection.score,
            "splits": {
                "train": _split_report(
                    train,
                    selection.train_metrics,
                    proxy_rows=len(selection.train_indices),
                    output_sha256=train_hash,
                ),
                "val": _split_report(
                    val,
                    selection.val_metrics,
                    proxy_rows=len(selection.val_indices),
                    output_sha256=val_hash,
                ),
            },
        }
        with report_temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

        for temp_path, target_path in (
            (train_temp, targets["train"]),
            (val_temp, targets["val"]),
            (report_temp, targets["report"]),
        ):
            os.replace(temp_path, target_path)
            published_paths.append(target_path)
        return report
    except Exception:
        for path in published_paths:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        if created_output_dir:
            try:
                output_path.rmdir()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a fixed prevalence-matched BigEarthNet proxy split"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--candidate_count", type=int, default=100)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = prepare_proxy_split(
        args.input_dir,
        args.output_dir,
        fraction=args.fraction,
        seed_start=args.seed_start,
        candidate_count=args.candidate_count,
    )
    print(f"Selected candidate seed: {report['selected_seed']}")
    print(f"Maximum prevalence deviation: {report['selected_score']:.8f}")
    print(f"Train rows: {report['splits']['train']['proxy_rows']}")
    print(f"Val rows: {report['splits']['val']['proxy_rows']}")
    print(f"Written: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
