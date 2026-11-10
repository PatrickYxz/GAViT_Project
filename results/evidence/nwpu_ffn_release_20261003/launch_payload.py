import collections, csv, datetime, hashlib, io, json, os, pathlib, shlex, subprocess, sys
repo = pathlib.Path("/home/featurize/work/GAViT_FFN_20261003")
root = pathlib.Path("/home/featurize/work/nwpu_ffn42_20261003_01")
manifest = pathlib.Path("/home/featurize/work/nwpu_protocol_20260927_02.json")
weights = pathlib.Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors")
data = pathlib.Path("/home/featurize/data/NWPU-RESISC45")
release = "058f1d8b79d0be538ec358eb742d2cd3992921d4"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert not root.exists(), "Output already exists: inspect it; do not overwrite or retry"
head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip()
assert head == release and not dirty, "Release commit or clean worktree mismatch"
assert digest(manifest) == "ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e", "Manifest SHA mismatch"
assert digest(weights) == "f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36", "Pretrained SHA mismatch"
m = json.loads(manifest.read_text())
assert m["dataset"] == "NWPU-RESISC45" and len(m["class_names"]) == 45 and len(m["records"]) == 31500, "Dataset identity mismatch"
assert m["counts"] == dict(collections.Counter(r["split"] for r in m["records"])) == {"train": 5040, "val": 1260, "test": 25200}, "Split count mismatch"
assert len({r["path"] for r in m["records"]}) == 31500, "Duplicate manifest paths"
missing = [r["path"] for r in m["records"] if not (data / r["path"]).is_file() or not (data / r["path"]).resolve().is_relative_to(data.resolve())]
assert not missing, "Missing or invalid images: " + repr(missing[:10])
gpu_text = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
gpu_rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(gpu_text))]
assert len(gpu_rows) == 1 and gpu_rows[0][0] == "0" and gpu_rows[0][1].replace(" ", "").endswith("RTX2080Ti"), "Expected exactly one RTX 2080 Ti GPU"
gpu_apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"], text=True).strip()
assert not gpu_apps and int(gpu_rows[0][4]) - int(gpu_rows[0][5]) >= 8192 and int(gpu_rows[0][6]) <= 5, "GPU is not idle; preserve current jobs"
assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0", gpu_rows[0][2]), "Unexpected CUDA_VISIBLE_DEVICES"
env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0")
common = [sys.executable, "-u", str(repo / "run_scene.py"), "train", "--token-refinement", "residual_ffn_v1", "--phase", "smoke", "--seed", "42", "--train-per-class", "8", "--val-per-class", "2", "--manifest", str(manifest), "--data-root", str(data), "--pretrained-path", str(weights), "--device", "cuda", "--batch-size", "32", "--workers", "2"]
cmd1 = common + ["--model", "swin", "--output", str(root / "swin_ffn_smoke")]
cmd2 = common + ["--model", "gavit", "--grouping", "soft_prototype", "--output", str(root / "soft_ffn_smoke")]
sequence = shlex.join(cmd1) + " && " + shlex.join(cmd2)
launch = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "release_commit": release, "commands": [cmd1, cmd2], "cwd": str(repo), "manifest_sha256": digest(manifest), "pretrained_sha256": digest(weights), "manifest_counts": m["counts"], "image_paths_checked": 31500, "gpu_query_columns": ["index", "name", "uuid", "driver_version", "memory.total_MiB", "memory.used_MiB", "utilization.gpu_percent"], "gpu_snapshot": gpu_rows, "compute_apps": gpu_apps, "environment_overrides": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")}, "scope": "two sequential one-epoch smoke runs only", "state": "claimed_before_spawn"}
root.mkdir(exist_ok=False)
log = (root / "queue.log").open("x", buffering=1)
(root / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
process = subprocess.Popen(["bash", "-c", sequence], cwd=str(repo), env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
(root / "launch.pid").write_text(str(process.pid) + "\n")
launch.update(pid=process.pid, state="spawned")
(root / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
log.close()
print(json.dumps({"spawned_pid": process.pid, "root": str(root), "tail": "tail -n 80 " + str(root / "queue.log"), "notice": "PID is not training success; inspect GPU and TRAIN FIRST BATCH OK"}, indent=2))
