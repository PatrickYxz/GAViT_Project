"""External orchestration of frozen BigEarthNet entries; no training source edits."""
import csv
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import resource
import runpy
import subprocess
import sys
import time
import traceback

REPO = Path("/home/featurize/work/GAViT_AID_20260920")
SPLITS = Path("/home/featurize/work/GAViT_Project/bigearth_files/splits")
PRETRAINED = SPLITS.parent / "model.safetensors"
COMMIT = "deb2b457e86b6fa2feccb5a305e099b6d9ec6dfe"
PRETRAINED_SHA = "f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36"
FROZEN = {
    "train": (237871, "25734fd9e8dc7fd78fb259e768ae91fe11b297e1caa50a2d57b74b0296413d8f"),
    "val": (122342, "1c6627cb48e33588538c4e411e4f1b94560827f27d59fb7aeb0ea2b5d30c2a45"),
    "test": (119825, "cea165a6443fc5ee44de0a3c55f53ce70505c4e25aba9ca8b52d76634df77204"),
}
STAGES = ("swin_smoke", "gavit_smoke", "swin_train", "swin_test", "gavit_train", "gavit_test")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def identity():
    require(output(["git", "-C", str(REPO), "rev-parse", "HEAD"]) == COMMIT, "Wrong code commit")
    require(not output(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=all"]), "Repository is dirty; preserve changes and stop")
    require(digest(PRETRAINED) == PRETRAINED_SHA, "Pretrained hash mismatch")
    for split, (_, sha) in FROZEN.items():
        require(digest(SPLITS / f"{split}.csv") == sha, f"Changed split: {split}")


def idle_gpu():
    require(not output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]), "Another GPU process is running")


def scan_split(path, count, sha, seen):
    require(digest(path) == sha, f"Changed split: {path}")
    examples = []
    started = time.monotonic()
    with path.open(newline="") as f:
        rows = csv.reader(f)
        require(next(rows) == ["patch_path"] + [f"label_{i}" for i in range(19)], "Invalid CSV header")
        n = 0
        for row in rows:
            require(len(row) == 20 and row[0] not in seen, "Duplicate path or invalid row")
            require(all(x in ("0", "1", "0.0", "1.0") for x in row[1:]), "Non-binary labels")
            seen.add(row[0])
            p = Path(row[0])
            require(p.is_absolute(), "Expected absolute image path")
            files = set(os.listdir(p))
            require(all((p / f"{p.name}_{b}.tif").is_file() for b in ("B02", "B03", "B04")), f"Missing RGB bands: {p}")
            require(not any(p.name + ext in files for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG")), f"Merged image overrides original TIF: {p}")
            if n < 96:
                examples.append(row)
            n += 1
            if n % 10000 == 0:
                print(f"RGB CHECK {path.stem}: {n}/{count}; {time.monotonic()-started:.1f}s", flush=True)
    require(n == count, f"Wrong sample count: {path}: {n}")
    return {"count": n, "sha256": sha, "examples": examples}


def entry_args(root, stage):
    model, kind = stage.split("_")
    smoke = kind == "smoke"
    ckpt = root / (stage if kind != "test" else f"{model}_train") / f"{model}.pth"
    data = root / "smoke_splits" if smoke else SPLITS
    if kind == "test":
        return [str(REPO / "test_bigearth.py"), "--model", model, "--data_dir", str(data), "--ckpt", str(ckpt), "--batch_size", "32", "--seed", "45"]
    return [str(REPO / "train_bigearth.py"), "--model", model, "--data_dir", str(data), "--epochs", "1" if smoke else "30", "--batch_size", "32", "--lr", "0.0003", "--seed", "45", "--num_regions", "16", "--grouping", "attentive_spatial", "--edge_type", "sparse_hybrid", "--knn_k", "5", "--gat_hidden", "256", "--gat_heads", "4", "--gat_layers", "2", "--integration", "token_feedback", "--dropout", "0.1", "--run_stage", "smoke" if smoke else "formal", "--run_tag", root.name + "_" + stage, "--pretrained_path", str(PRETRAINED), "--checkpoint_path", str(ckpt)]


def worker(root, stage):
    import torch
    import numpy as np
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "One CUDA GPU required; no CPU fallback")
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    identity()
    hardware = {"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "threads": torch.get_num_threads(), "python": sys.version, "packages": {n: importlib.metadata.version(n) for n in ("torch", "torchvision", "timm", "torch-geometric", "numpy", "scikit-learn", "pillow")}, "gpu_identity": output(["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"])}
    precision = {"dtype": str(torch.get_default_dtype()), "matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32, "amp": False}
    first = root / "swin_smoke/run.json"
    if stage != "swin_smoke":
        reference = json.loads(first.read_text())
        require(reference["status"] == "complete" and reference["hardware"] == hardware and reference["precision"] == precision, "Environment changed since smoke; stop before training")
    print("DEVICE", hardware, "PRECISION", precision, flush=True)
    report = {"stage": stage, "model": stage.split("_")[0], "seed": 45, "status": "running", "commit": COMMIT, "pretrained_sha256": PRETRAINED_SHA, "splits": FROZEN, "hardware": hardware, "precision": precision, "argv": entry_args(root, stage), "launcher_sha256": digest(__file__)}
    dest = root / stage
    save(dest / "run.json", report)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if not stage.endswith("_test"):
        import utils
        original = utils.save_epoch_artifacts
        history = []
        previous = [started]

        def record(*args, **kwargs):
            caller = inspect.currentframe().f_back.f_globals
            result = original(*args, **kwargs)
            now = time.monotonic()
            row = {"epoch": kwargs["epoch"], "train_loss": float(caller["avg_loss"]), "val_mAP_percent": float(kwargs["metric_value"]), "val_macro_f1_percent": float(caller["val_f1"]), "next_epoch_lr": float(args[1].param_groups[0]["lr"]), "epoch_wall_seconds": now - previous[0]}
            require(all(math.isfinite(v) for v in row.values()), "Non-finite training result")
            history.append(row)
            save(dest / "history.json", history)
            previous[0] = now
            print("EPOCH RESOURCE", json.dumps(row), "first epoch includes initialization" if len(history) == 1 else "", flush=True)
            return result

        utils.save_epoch_artifacts = record
    sys.argv = report["argv"]
    ns = runpy.run_path(sys.argv[0], run_name="__main__")
    if stage.endswith("_test"):
        require(ns["probs_np"].shape == (119825, 19) and np.isfinite(ns["probs_np"]).all(), "Incomplete/non-finite test")
        require(len(set(ns["test_patches"])) == 119825 and len(ns["ap_per_class"]) == 19, "Wrong test identity/classes")
        ckpt = Path(sys.argv[sys.argv.index("--ckpt") + 1])
        meta = json.loads(ckpt.with_suffix(".meta.json").read_text())
        require(digest(ckpt) == meta["checkpoint_sha256"] and meta["training"]["seed"] == 45, "Test checkpoint mismatch")
        metrics = {"samples": 119825, "mAP_percent": ns["mAP"], "macro_f1_percent": ns["macro_f1"], "micro_f1_percent": ns["micro_f1"], "f1_threshold": 0.5, "checkpoint_sha256": digest(ckpt), "per_class_ap_percent": dict(zip(ns["CLASSES_19"], [float(x)*100 for x in ns["ap_per_class"]]))}
        save(dest / "metrics.json", metrics)
        np.savez_compressed(dest / "predictions.npz", paths=np.asarray(ns["test_patches"]), labels=ns["labels_np"].astype(np.uint8), probabilities=ns["probs_np"])
        report["metrics"] = metrics
    else:
        epochs = 1 if stage.endswith("_smoke") else 30
        require([r["epoch"] for r in history] == list(range(1, epochs + 1)), "Incomplete epoch history")
        ckpt = Path(ns["CKPT_PATH"])
        meta = json.loads(ckpt.with_suffix(".meta.json").read_text())
        require(digest(ckpt) == meta["checkpoint_sha256"], "Saved best hash mismatch")
        best_row = max(history, key=lambda row: row["val_mAP_percent"])
        require(meta["best"]["epoch"] == best_row["epoch"] and abs(meta["best"]["value"] - best_row["val_mAP_percent"]) <= 0.000051, "Best selection disagrees with validation history")
        state = torch.load(ns["TRAIN_STATE_PATH"], map_location="cpu", weights_only=True)
        require(state["epoch"] == epochs, "Incomplete last training state")
        del state
        if epochs == 1:
            model = ns["model"].eval()
            inputs = ns["val_set"][0][0].unsqueeze(0).to("cuda")
            with torch.no_grad():
                before = model(inputs)
                model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
                diff = float((model(inputs) - before).abs().max())
            require(diff < 1e-5, "Smoke checkpoint reload mismatch")
            report["reload_max_logit_diff"] = diff
        report.update(final_epoch=epochs, best=meta["best"], checkpoint=str(ckpt), checkpoint_sha256=digest(ckpt), parameters=sum(p.numel() for p in ns["model"].parameters()), metadata=meta)
    report.update(status="complete", elapsed_seconds=time.monotonic()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(), main_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, resource_scope="entire stage including checkpoint I/O and saved-state validation; RSS excludes worker processes")
    save(dest / "run.json", report)
    print("STAGE COMPLETE", stage, flush=True)


def run_jobs(root, jobs, before_job):
    for stage, argv in jobs:
        dest = root / stage
        dest.mkdir()
        before_job()
        save(dest / "command.json", argv)
        print("START", stage, flush=True)
        with (dest / "run.log").open("wb") as log:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (dest / "run.pid").write_text(str(proc.pid) + "\n")
            while True:
                chunk = proc.stdout.read1(65536)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            rc = proc.wait()
            proc.stdout.close()
        (dest / "run.exit").write_text(f"exit_code={rc}\n")
        require((dest / "run.log").is_file(), "Missing stage log")
        if rc:
            raise subprocess.CalledProcessError(rc, argv)


def main(root):
    import fcntl
    with (root.parent / ".bigearth_seed45.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity()
        idle_gpu()
        smoke = root / "smoke_splits"
        smoke.mkdir()
        seen, scans = set(), {}
        for name, (count, sha) in FROZEN.items():
            result = scan_split(SPLITS / f"{name}.csv", count, sha, seen)
            if name != "test":
                with (smoke / f"{name}.csv").open("w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["patch_path"] + [f"label_{i}" for i in range(19)])
                    w.writerows(result["examples"][:96 if name == "train" else 64])
            result.pop("examples")
            scans[name] = result
        save(root / "preflight.json", {"splits": scans, "commit": COMMIT, "pretrained_sha256": PRETRAINED_SHA, "status": "pass", "scope": "all referenced RGB files exist, frozen CSV bytes, original TIF path; smoke decodes selected train/val only"})
        jobs = [(s, [sys.executable, "-u", str(Path(__file__).resolve()), str(root), s]) for s in STAGES]
        run_jobs(root, jobs, idle_gpu)
        reports = {s: json.loads((root / s / "run.json").read_text()) for s in STAGES}
        for r in reports.values():
            require(r["status"] == "complete" and r["hardware"] == reports[STAGES[0]]["hardware"] and r["precision"] == reports[STAGES[0]]["precision"], "Incomplete stages or different paired environment")
        save(root / "summary.json", {"status": "complete", "dataset": "BigEarthNet-19", "seed": 45, "stages": reports})
        print("BIGEARTH SEED45 COMPLETE", str(root), flush=True)


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    if len(sys.argv) == 3:
        worker(root, sys.argv[2])
    else:
        rc = 1
        try:
            main(root)
            rc = 0
        except BaseException:
            traceback.print_exc()
        finally:
            (root / "run.exit").write_text(f"exit_code={rc}\n")
        sys.exit(rc)
