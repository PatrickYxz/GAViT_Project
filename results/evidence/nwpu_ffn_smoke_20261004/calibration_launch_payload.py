import datetime, hashlib, json, os, pathlib, shlex, subprocess, sys
p = pathlib.Path("/home/featurize/work/nwpu_ffn42_20261003_01")
repo = pathlib.Path("/home/featurize/work/GAViT_FFN_20261003")
release = "058f1d8b79d0be538ec358eb742d2cd3992921d4"
query = lambda args: subprocess.check_output(args, text=True).strip()
assert query(["git", "-C", str(repo), "rev-parse", "HEAD"]) == release and not query(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]), "Expected clean FFN release"
assert p.is_dir() and all(not (p / n).exists() for n in ("swin_ffn", "soft_ffn", "calibration.log", "calibration_launch.json", "calibration.pid")), "Existing/missing output: inspect; do not overwrite or retry"
smokes = {n: json.loads((p / n / "run.json").read_text()) for n in ("swin_ffn_smoke", "soft_ffn_smoke")}
assert all((p / n / "run.exit").read_text().strip() == "exit_code=0" and r["status"] == "complete" and r["phase"] == "smoke" and r["seed"] == 42 and r["counts"] == {"train": 360, "val": 90} and r["final_epoch"] == 1 and r["code"]["git"] == {"commit": "058f1d8", "dirty": False} and hashlib.sha256((p / n / "best.pth").read_bytes()).hexdigest() == r["checkpoint_sha256"] for n, r in smokes.items()), "Smoke completion/checkpoint mismatch"
manifest = pathlib.Path("/home/featurize/work/nwpu_protocol_20260927_02.json")
data = pathlib.Path("/home/featurize/data/NWPU-RESISC45")
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e", "Frozen manifest mismatch"
assert hashlib.sha256(pathlib.Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors").read_bytes()).hexdigest() == "f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36", "Frozen pretrained weights mismatch"
assert all((data / r["path"]).is_file() for r in json.loads(manifest.read_text())["records"]), "Missing NWPU images: restore before launch"
gpu = query(["nvidia-smi", "--query-gpu=index,name,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
g = [x.strip() for x in gpu.split(",")]
assert len(gpu.splitlines()) == 1 and len(g) == 5 and g[0] == "0" and g[1] == "NVIDIA GeForce RTX 2080 Ti" and int(g[3]) >= 8192 and int(g[4]) <= 5 and not query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]), "Expected idle RTX 2080 Ti with >=8 GiB free"
assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0", g[2]), "Unexpected CUDA_VISIBLE_DEVICES"
common = [sys.executable, "-u", str(repo / "run_scene.py"), "train", "--token-refinement", "residual_ffn_v1", "--phase", "calibrate", "--seed", "42", "--manifest", str(manifest), "--data-root", str(data), "--pretrained-path", "/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors", "--device", "cuda", "--batch-size", "32", "--workers", "2"]
commands = [common + ["--model", "swin", "--output", str(p / "swin_ffn")], common + ["--model", "gavit", "--grouping", "soft_prototype", "--output", str(p / "soft_ffn")]]
env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0")
launch = {"release_commit": release, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commands": commands, "gpu_index_name_uuid_free_MiB_util_percent": g, "environment_overrides": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")}, "scope": "two sequential NWPU seed42 calibrations, 30 epochs each; no refit/test", "state": "claimed_before_spawn"}
claim = p / "calibration_launch.json"
c = claim.open("x")
c.write(json.dumps(launch, indent=2) + "\n")
c.close()
f = (p / "calibration.log").open("x")
q = subprocess.Popen(["bash", "-c", " && ".join(shlex.join(c) for c in commands)], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
f.close()
(p / "calibration.pid").write_text(str(q.pid) + "\n")
launch.update(pid=q.pid, state="spawned")
claim.write_text(json.dumps(launch, indent=2) + "\n")
print("PID", q.pid, "LOG", p / "calibration.log", "Check TRAIN FIRST BATCH OK and nvidia-smi; PID alone is not training success")
