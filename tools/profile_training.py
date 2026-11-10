"""Profile the unchanged trainer; formal execution requires a matched smoke gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading
import time
import traceback

if __package__:
    from .resource_probe import (COUNTS, check_idle_gpu, scan_dataset, sha256,
                                 source_identity, training_args, validate_smoke_gate, validate_smoke_inputs,
                                 write_json_new)
else:
    from resource_probe import (COUNTS, check_idle_gpu, scan_dataset, sha256,
                                source_identity, training_args, validate_smoke_gate, validate_smoke_inputs,
                                write_json_new)


def timed_batches(iterable, stats, phase, clock, synchronize, warmup=10):
    """Time loading plus consumer work without changing yielded batches."""
    record = stats.setdefault(phase, {"batches": 0, "measured_batches": 0,
                                      "measured_seconds": 0.0, "total_seconds": 0.0})
    iterator, index = iter(iterable), 0
    while True:
        synchronize()
        started = clock()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        yield batch
        synchronize()
        elapsed = clock() - started
        record["batches"] += 1
        record["total_seconds"] += elapsed
        if index >= warmup:
            record["measured_batches"] += 1
            record["measured_seconds"] += elapsed
        index += 1


def read_kib(path, key):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(key + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def read_process_memory(root_pid, proc_root=Path("/proc")):
    proc_root, pending, pids = Path(proc_root), [root_pid], set()
    while pending:
        pid = pending.pop()
        if pid in pids:
            continue
        pids.add(pid)
        try:
            children = (proc_root / str(pid) / "task" / str(pid) / "children").read_text()
            pending.extend(int(item) for item in children.split())
        except (OSError, ValueError):
            pass
    rss = [read_kib(proc_root / str(pid) / "status", "VmRSS") for pid in pids]
    pss = [read_kib(proc_root / str(pid) / "smaps_rollup", "Pss") for pid in pids]
    return {"pids": sorted(pids),
            "rss_sum_kib": sum(rss) if rss and all(value is not None for value in rss) else None,
            "pss_sum_kib": sum(pss) if pss and all(value is not None for value in pss) else None,
            "host_available_kib": read_kib(proc_root / "meminfo", "MemAvailable")}


class Tee:
    def __init__(self, console, logfile):
        self.console, self.logfile = console, logfile

    def write(self, value):
        self.console.write(value)
        return self.logfile.write(value)

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def isatty(self):
        return False


class Sampler:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.peaks = {key: None for key in ("rss_sum_kib", "pss_sum_kib", "gpu_process_mib")}
        self.min_available_kib = None
        self.errors = []

    def sample(self):
        sample = read_process_memory(os.getpid())
        sample["timestamp"] = time.time()
        sample["gpu_process_mib"] = None
        try:
            raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], text=True, timeout=5)
            used = [float(row[1]) for row in csv.reader(raw.splitlines(), skipinitialspace=True)
                    if len(row) == 2 and int(row[0]) in sample["pids"]]
            sample["gpu_process_mib"] = sum(used) if used else None
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            self.errors = list(dict.fromkeys([*self.errors, str(exc)]))[:5]
        for key in self.peaks:
            value = sample[key]
            if value is not None:
                self.peaks[key] = max(self.peaks[key] or 0, value)
        available = sample["host_available_kib"]
        if available is not None:
            self.min_available_kib = available if self.min_available_kib is None else min(self.min_available_kib, available)
        return sample

    def run(self):
        try:
            with self.filename.open("x") as handle:
                while not self.stop_event.is_set():
                    handle.write(json.dumps(self.sample()) + "\n")
                    handle.flush()
                    self.stop_event.wait(1.0)
        except Exception as exc:
            self.errors.append(f"sampler: {exc}")

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=7)


def check_artifacts(torch, checkpoint, model, seed, epochs):
    metadata = json.loads(checkpoint.with_suffix(".meta.json").read_text())
    state = torch.load(checkpoint.with_suffix(".last.train_state.pth"), map_location="cpu", weights_only=True)
    if state.get("epoch") != epochs or metadata.get("model") != model:
        raise RuntimeError("Incomplete training state or wrong model")
    if metadata.get("training", {}).get("seed") != seed:
        raise RuntimeError("Checkpoint seed mismatch")
    if metadata.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Best checkpoint hash mismatch")
    if not math.isfinite(float(state["best_metric"])):
        raise RuntimeError("Non-finite best metric")
    return {"epoch": state["epoch"], "best": metadata.get("best"), "checkpoint_sha256": metadata["checkpoint_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu", choices=("3080", "3060"), required=True)
    parser.add_argument("--hourly-rate", type=float, required=True)
    parser.add_argument("--stage", choices=("smoke", "formal"), required=True)
    parser.add_argument("--model", choices=("swin", "gavit"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke-dir")
    args = parser.parse_args()
    if not math.isfinite(args.hourly_rate) or args.hourly_rate <= 0:
        parser.error("hourly-rate must be positive and finite")
    repo, output = Path(args.repo).resolve(), Path(args.output_dir).resolve()
    argv = training_args(repo, output, args.model, args.stage, args.seed)
    if repo == output or repo in output.parents or repo in Path(__file__).resolve().parents:
        parser.error("Tools and output must be outside the unchanged training checkout")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "stage": args.stage, "model": args.model, "seed": args.seed,
              "gpu_candidate": args.gpu, "hourly_rate": args.hourly_rate, "trainer_argv": argv,
              "artifacts_verified": False, "batch_timings": {}}
    started, sampler, torch, original_iter = time.perf_counter(), None, None, None
    original_cwd, original_argv, original_path = Path.cwd(), sys.argv[:], sys.path[:]
    original_stdout, original_stderr = sys.stdout, sys.stderr
    log = (output / "training.log").open("x", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    try:
        check_idle_gpu()
        identity = source_identity(repo, args.gpu)
        report["identity"] = identity
        report["tool_sha256"] = {name: sha256(Path(__file__).with_name(name)) for name in ("resource_probe.py", "profile_training.py")}
        if args.stage == "formal":
            if not args.smoke_dir:
                raise RuntimeError("Formal execution requires --smoke-dir from a successful paired smoke")
            validate_smoke_gate(args.smoke_dir, identity)
            scan = scan_dataset(repo / "bigearth_files/splits")
            write_json_new(output / "data_check.json", scan)
            if not all(item["complete"] for item in scan.values()):
                raise RuntimeError("Fresh full-data scan found missing bands")
        else:
            report["smoke_split"] = validate_smoke_inputs(output.parent, identity)
        os.chdir(repo)
        sys.path.insert(0, str(repo))
        import torch
        from torch.utils.data import DataLoader, RandomSampler
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device is required; no CPU fallback")
        report["cuda"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)}
        torch.cuda.reset_peak_memory_stats()
        original_iter = DataLoader.__iter__
        def measured_iter(loader):
            phase = "train" if isinstance(loader.sampler, RandomSampler) else "val"
            return timed_batches(original_iter(loader), report["batch_timings"], phase,
                                 time.perf_counter, torch.cuda.synchronize)
        DataLoader.__iter__ = measured_iter
        sampler = Sampler(output / "resource_samples.jsonl")
        sampler.thread.start()
        sys.argv = [str(repo / "train_bigearth.py"), *argv]
        training_started = time.perf_counter()
        runpy.run_path(sys.argv[0], run_name="__main__")
        torch.cuda.synchronize()
        report["training_wall_seconds"] = time.perf_counter() - training_started
        report["precision_settings"] = {
            "default_dtype": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
        report["artifacts"] = check_artifacts(torch, output / f"{args.model}.pth", args.model, args.seed, 1 if args.stage == "smoke" else 30)
        report["artifacts_verified"] = True
        for phase, count in (("train", COUNTS["train"]), ("val", COUNTS["val"])):
            record = report["batch_timings"].get(phase, {})
            if record.get("measured_batches", 0) <= 0 or record.get("measured_seconds", 0) <= 0:
                raise RuntimeError(f"No warmed {phase} throughput")
            record["mean_batch_seconds"] = record["measured_seconds"] / record["measured_batches"]
            record["estimated_full_phase_seconds"] = math.ceil(count / 32) * record["mean_batch_seconds"]
        report["estimated_30_epoch_loop_hours"] = 30 * sum(item["estimated_full_phase_seconds"] for item in report["batch_timings"].values()) / 3600
        report["estimated_loop_cost"] = report["estimated_30_epoch_loop_hours"] * args.hourly_rate
        report["estimate_excludes"] = "startup, full validation metric calculation, checkpoint IO, extraction, idle time; smoke RAM is not a full-dataset RAM peak"
        report["status"] = "passed"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if original_iter is not None:
            torch.utils.data.DataLoader.__iter__ = original_iter
        if sampler is not None:
            sampler.stop()
            report["sampled_peaks"] = sampler.peaks
            report["minimum_host_available_kib"] = sampler.min_available_kib
            report["sampling_errors"] = sampler.errors
            if any(sampler.peaks[key] is None for key in ("rss_sum_kib", "gpu_process_mib")):
                report["status"] = "failed"
                report.setdefault("error", "Required resource measurements unavailable; keep artifacts and inspect sampling_errors")
        if torch is not None and torch.cuda.is_available():
            report["peak_cuda_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            report["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
        report["wall_seconds"] = time.perf_counter() - started
        report["run_cost_estimate"] = report["wall_seconds"] / 3600 * args.hourly_rate
        os.chdir(original_cwd)
        sys.argv, sys.path = original_argv, original_path
        write_json_new(output / "profile.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        log.close()
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
