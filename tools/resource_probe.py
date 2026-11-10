"""Portable BigEarthNet resource smoke. Plans never launch paid training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import re
import shlex
import subprocess
import sys
import time

TRAINING_COMMIT = "bb9f1cff463c996bc5f72695af538898c91490eb"
COUNTS = {"train": 237871, "val": 122342, "test": 119825}
HEADER = ["patch_path", *[f"label_{i}" for i in range(19)]]
RATES = {"3080": 0.93, "3060": 0.69}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_rows(path):
    rows, seen = [], set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != HEADER:
            raise ValueError(f"Unexpected split header: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != 20 or not row[0] or row[0] in seen:
                raise ValueError(f"Empty/duplicate path or invalid row: {path}:{line}")
            if any(value not in {"0", "1", "0.0", "1.0"} for value in row[1:]):
                raise ValueError(f"Non-binary label: {path}:{line}")
            seen.add(row[0])
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def scan_dataset(source, expected_counts=COUNTS):
    """Check every referenced RGB band; directory existence is insufficient."""
    source, reports, all_paths = Path(source), {}, set()
    for name, expected in expected_counts.items():
        csv_path = source / f"{name}.csv"
        rows = read_rows(csv_path)
        if len(rows) != expected:
            raise ValueError(f"{name}: expected {expected} rows, observed {len(rows)}")
        missing, examples = 0, []
        for row in rows:
            if row[0] in all_paths:
                raise ValueError(f"Overlapping splits: {row[0]}")
            all_paths.add(row[0])
            patch = Path(row[0])
            if not patch.is_absolute():
                raise ValueError(f"Absolute patch path required: {patch}")
            for band in ("B02", "B03", "B04"):
                filename = patch / f"{patch.name}_{band}.tif"
                if not filename.is_file():
                    missing += 1
                    if len(examples) < 5:
                        examples.append(str(filename))
        reports[name] = {"rows": len(rows), "sha256": sha256(csv_path),
                         "missing_bands": missing, "examples": examples,
                         "complete": missing == 0}
    return reports


def prepare_smoke_split(source, output, train_count=4096, val_count=1024):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    selected, manifest = {}, {"sampling_seed": 42, "splits": {}}
    for name, count in (("train", train_count), ("val", val_count)):
        rows = read_rows(source / f"{name}.csv")
        if count <= 0 or count > len(rows):
            raise ValueError(f"Invalid {name} sample count: {count}")
        indices = sorted(random.Random(42).sample(range(len(rows)), count))
        selected[name] = [rows[index] for index in indices]
        manifest["splits"][name] = {"rows": count, "source_sha256": sha256(source / f"{name}.csv")}
    if {row[0] for row in selected["train"]} & {row[0] for row in selected["val"]}:
        raise ValueError("Overlapping train/val samples")
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in selected.items():
        filename = output / f"{name}.csv"
        with filename.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)
        manifest["splits"][name]["sample_sha256"] = sha256(filename)
    write_json_new(output / "manifest.json", manifest)
    return manifest


def training_args(repo, output, model, stage, seed):
    if model not in ("swin", "gavit") or stage not in ("smoke", "formal"):
        raise ValueError("Unsupported model or stage")
    if (stage == "smoke" and seed != 42) or (stage == "formal" and seed not in (44, 45)):
        raise ValueError("Smoke uses seed42; formal plans support only seed44/45")
    repo, output = Path(repo), Path(output)
    data = output.parent / "splits" if stage == "smoke" else repo / "bigearth_files/splits"
    argv = ["--model", model, "--data_dir", str(data), "--epochs", "1" if stage == "smoke" else "30",
            "--batch_size", "32", "--lr", "3e-4", "--seed", str(seed),
            "--run_stage", stage, "--run_tag", f"resource_{stage}_{model}_seed{seed}",
            "--pretrained_path", str(repo / "bigearth_files/model.safetensors")]
    if model == "gavit":
        argv += ["--num_regions", "16", "--grouping", "attentive_spatial", "--edge_type", "sparse_hybrid",
                 "--knn_k", "5", "--gat_hidden", "256", "--gat_heads", "4", "--gat_layers", "2",
                 "--integration", "token_feedback", "--dropout", "0.1"]
    return argv + ["--checkpoint_path", str(output / f"{model}.pth")]


def source_identity(repo, gpu):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if commit != TRAINING_COMMIT or git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Require clean training commit bb9f1cf; keep tools and outputs outside the checkout")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"], text=True)
    devices = list(csv.reader(raw.strip().splitlines(), skipinitialspace=True))
    if len(devices) != 1 or len(devices[0]) != 3:
        raise RuntimeError("Probe requires one visible physical GPU")
    name, memory, driver = devices[0]
    memory = float(memory)
    low, high = (9500, 11000) if gpu == "3080" else (11500, 13000)
    if not name.endswith(f"RTX {gpu}") or not low <= memory <= high:
        raise RuntimeError(f"Expected RTX {gpu} with requested capacity; observed {name}, {memory} MiB")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "torch-geometric", "timm", "numpy", "scikit-learn")}
    return {"commit": commit, "gpu": {"name": name, "memory_mib": memory, "driver": driver},
            "python": sys.version, "versions": versions,
            "data_sha256": {name: sha256(repo / "bigearth_files/splits" / f"{name}.csv") for name in COUNTS},
            "pretrained_sha256": sha256(repo / "bigearth_files/model.safetensors")}


def check_idle_gpu():
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        raise RuntimeError("GPU already has a compute process; finish that job before benchmarking")


def validate_smoke_gate(smoke_dir, identity):
    smoke_dir = Path(smoke_dir)
    report = json.loads((smoke_dir / "smoke_report.json").read_text())
    if report.get("status") != "passed" or report.get("identity") != identity:
        raise RuntimeError("Smoke gate failed or training/data/GPU/software identity changed")
    for model in ("swin", "gavit"):
        item = report.get("models", {}).get(model, {})
        if (item.get("status") != "passed" or not item.get("artifacts_verified")
                or item.get("stage") != "smoke" or item.get("seed") != 42
                or item.get("model") != model or item.get("identity") != identity):
            raise RuntimeError(f"No successful {model} smoke")
    return report


def validate_smoke_inputs(root, identity):
    root = Path(root)
    if json.loads((root / "identity.json").read_text()) != identity:
        raise RuntimeError("Preflight identity changed")
    data = json.loads((root / "data_check.json").read_text())
    if set(data) != set(COUNTS) or not all(
        item.get("complete") and item.get("rows") == COUNTS[name]
        and item.get("sha256") == identity["data_sha256"][name]
        for name, item in data.items()
    ):
        raise RuntimeError("No complete full-dataset preflight")
    tests = (root / "tests.log").read_text()
    if (not re.search(r"^OK\s*$", tests, re.MULTILINE) or "skipped" in tests.lower()
            or not re.search(r"Ran [1-9][0-9]* tests?", tests)):
        raise RuntimeError("No successful full test gate")
    manifest = json.loads((root / "splits/manifest.json").read_text())
    if manifest.get("sampling_seed") != 42 or set(manifest.get("splits", {})) != {"train", "val"}:
        raise RuntimeError("Invalid smoke manifest")
    for name, count in (("train", 4096), ("val", 1024)):
        record = manifest["splits"][name]
        if (record.get("rows") != count or record["source_sha256"] != identity["data_sha256"][name]
                or record["sample_sha256"] != sha256(root / "splits" / f"{name}.csv")):
            raise RuntimeError("Smoke split identity changed")
    return manifest


def make_plan(args):
    repo, output = Path(args.repo).resolve(), Path(args.output_dir).resolve()
    runs = []
    for model in ("swin", "gavit"):
        destination = output / model
        command = [sys.executable, str(Path(__file__).with_name("profile_training.py")),
                   "--repo", str(repo), "--output-dir", str(destination), "--gpu", args.gpu,
                   "--hourly-rate", str(args.hourly_rate), "--stage", args.stage, "--model", model, "--seed", str(args.seed)]
        if args.smoke_dir:
            command += ["--smoke-dir", str(Path(args.smoke_dir).resolve())]
        evaluation = [sys.executable, str(repo / "test_bigearth.py"), "--model", model,
                      "--data_dir", str(repo / "bigearth_files/splits"), "--ckpt", str(destination / f"{model}.pth"),
                      "--batch_size", "32"]
        runs.append({"model": model, "trainer_argv": training_args(repo, destination, model, args.stage, args.seed),
                     "command": shlex.join(command),
                     "evaluation_command": (f"(set -C; exec {shlex.join(evaluation)} > {shlex.quote(str(destination / 'test.log'))} 2>&1)"
                                            if args.stage == "formal" else None)})
    return {"executed": False, "stage": args.stage, "seed": args.seed, "gpu": args.gpu,
            "expected_training_commit": TRAINING_COMMIT,
            "gate": "Requires passed smoke report on the same device/environment" if args.stage == "formal" else "Use the smoke subcommand for preflight and split preparation",
            "runs": runs}


def execute_smoke(args):
    repo, output = Path(args.repo).resolve(), Path(args.output_dir).resolve()
    if output == repo or repo in output.parents or repo in Path(__file__).resolve().parents:
        raise ValueError("Copy tools and choose outputs outside the training checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "models": {}, "started_at": time.time()}
    try:
        check_idle_gpu()
        identity = source_identity(repo, args.gpu)
        report["identity"] = identity
        write_json_new(output / "identity.json", identity)
        print("Checking all train/val/test RGB band paths...", flush=True)
        scan = scan_dataset(repo / "bigearth_files/splits")
        write_json_new(output / "data_check.json", scan)
        if not all(item["complete"] for item in scan.values()):
            raise RuntimeError("Dataset has missing bands; inspect data_check.json")
        print("Running the existing full training test suite...", flush=True)
        with (output / "tests.log").open("x") as handle:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
        tests = (output / "tests.log").read_text()
        if result.returncode or "skipped" in tests.lower() or not re.search(r"Ran [1-9][0-9]* tests?", tests):
            raise RuntimeError("Full tensor test gate failed or skipped tests; inspect tests.log")
        report["split"] = prepare_smoke_split(repo / "bigearth_files/splits", output / "splits")
        write_json_new(output / "plan.json", make_plan(args))
        for run in make_plan(args)["runs"]:
            model = run["model"]
            print(f"Running {model} smoke; log: {output / (model + '.log')}", flush=True)
            with (output / f"{model}.log").open("x") as handle:
                result = subprocess.run(shlex.split(run["command"]), cwd=repo, stdout=handle, stderr=subprocess.STDOUT)
            summary_path = output / model / "profile.json"
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "failed"}
            report["models"][model] = summary
            if result.returncode or summary.get("status") != "passed":
                raise RuntimeError(f"{model} smoke failed; artifacts retained")
        report["status"] = "passed"
        print("Both resource smokes passed. No formal training has been started.", flush=True)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["wall_seconds_including_preflight"] = time.time() - report["started_at"]
        write_json_new(output / "smoke_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "smoke"))
    parser.add_argument("--repo", default="/home/featurize/work/GAViT_Project")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", choices=("3080", "3060"), required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--smoke-dir", help="Passed smoke output directory; required to execute formal profile commands")
    args = parser.parse_args()
    args.seed = args.seed if args.seed is not None else (42 if args.stage == "smoke" else {"3080": 44, "3060": 45}[args.gpu])
    args.hourly_rate = args.hourly_rate if args.hourly_rate is not None else RATES[args.gpu]
    if not math.isfinite(args.hourly_rate) or args.hourly_rate <= 0:
        parser.error("hourly-rate must be positive and finite")
    plan = make_plan(args)
    if args.action == "plan":
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        if args.stage != "smoke" or args.seed != 42:
            parser.error("smoke execution is limited to stage=smoke, seed=42")
        execute_smoke(args)


if __name__ == "__main__":
    main()
