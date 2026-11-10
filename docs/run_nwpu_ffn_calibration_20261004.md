# NWPU FFN 双模型校准：2026-10-04 交付

**2026-10-05收尾更新：训练与约定缓存读取均已完成，COMPACT_CLOSEOUT_PASS及最终指标已归档。候选停止后续训练，无需重复执行以下历史命令或回传同批摘要；[最终结果](../results/nwpu_ffn_calibration42_comparison.md)。**

**2026-10-05：两组30轮校准已报告完成，FFN未通过冻结OA门槛。停止该候选追加训练；[结果表](../results/nwpu_ffn_calibration42_comparison.md)已归档，仅读已有缓存补齐记录。以下启动命令/待启动措辞保留为历史，不重复执行。**

两组360/90、一轮GPU短测的用户回传已核对，结果见[短测验收](../results/nwpu_ffn_smoke_validation.md)。以下只新增已批准的两组 seed42、各30轮校准，train5040/val1260；复用旧Swin/soft校准对照。研究问题、配方和继续条件见[冻结协议](run_nwpu_ffn_trial_20261003.md)。当前校准尚未收到启动证据。

## 1. 启动一次

复制整行。先核验已有短测完成与best文件SHA、发布commit/干净代码、冻结输入SHA、原图存在及2080Ti空闲，然后在后台依次执行Swin+FFN、soft+FFN；第一项失败停止，不重试。使用独立calibration日志与claim，不改已有smoke成果。输出存在时先检查，不能删除后无理由重跑；没有新同步/下载工具包步骤。

```bash
python -c 'import datetime, hashlib, json, os, pathlib, shlex, subprocess, sys; p = pathlib.Path("/home/featurize/work/nwpu_ffn42_20261003_01"); repo = pathlib.Path("/home/featurize/work/GAViT_FFN_20261003"); release = "058f1d8b79d0be538ec358eb742d2cd3992921d4"; query = lambda args: subprocess.check_output(args, text=True).strip(); assert query(["git", "-C", str(repo), "rev-parse", "HEAD"]) == release and not query(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"]), "Expected clean FFN release"; assert p.is_dir() and all(not (p / n).exists() for n in ("swin_ffn", "soft_ffn", "calibration.log", "calibration_launch.json", "calibration.pid")), "Existing/missing output: inspect; do not overwrite or retry"; smokes = {n: json.loads((p / n / "run.json").read_text()) for n in ("swin_ffn_smoke", "soft_ffn_smoke")}; assert all((p / n / "run.exit").read_text().strip() == "exit_code=0" and r["status"] == "complete" and r["phase"] == "smoke" and r["seed"] == 42 and r["counts"] == {"train": 360, "val": 90} and r["final_epoch"] == 1 and r["code"]["git"] == {"commit": "058f1d8", "dirty": False} and hashlib.sha256((p / n / "best.pth").read_bytes()).hexdigest() == r["checkpoint_sha256"] for n, r in smokes.items()), "Smoke completion/checkpoint mismatch"; manifest = pathlib.Path("/home/featurize/work/nwpu_protocol_20260927_02.json"); data = pathlib.Path("/home/featurize/data/NWPU-RESISC45"); assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e", "Frozen manifest mismatch"; assert hashlib.sha256(pathlib.Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors").read_bytes()).hexdigest() == "f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36", "Frozen pretrained weights mismatch"; assert all((data / r["path"]).is_file() for r in json.loads(manifest.read_text())["records"]), "Missing NWPU images: restore before launch"; gpu = query(["nvidia-smi", "--query-gpu=index,name,uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits"]); g = [x.strip() for x in gpu.split(",")]; assert len(gpu.splitlines()) == 1 and len(g) == 5 and g[0] == "0" and g[1] == "NVIDIA GeForce RTX 2080 Ti" and int(g[3]) >= 8192 and int(g[4]) <= 5 and not query(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]), "Expected idle RTX 2080 Ti with >=8 GiB free"; assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None, "0", g[2]), "Unexpected CUDA_VISIBLE_DEVICES"; common = [sys.executable, "-u", str(repo / "run_scene.py"), "train", "--token-refinement", "residual_ffn_v1", "--phase", "calibrate", "--seed", "42", "--manifest", str(manifest), "--data-root", str(data), "--pretrained-path", "/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors", "--device", "cuda", "--batch-size", "32", "--workers", "2"]; commands = [common + ["--model", "swin", "--output", str(p / "swin_ffn")], common + ["--model", "gavit", "--grouping", "soft_prototype", "--output", str(p / "soft_ffn")]]; env = dict(os.environ, OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="0"); launch = {"release_commit": release, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "commands": commands, "gpu_index_name_uuid_free_MiB_util_percent": g, "environment_overrides": {k: env[k] for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")}, "scope": "two sequential NWPU seed42 calibrations, 30 epochs each; no refit/test", "state": "claimed_before_spawn"}; claim = p / "calibration_launch.json"; c = claim.open("x"); c.write(json.dumps(launch, indent=2) + "\n"); c.close(); f = (p / "calibration.log").open("x"); q = subprocess.Popen(["bash", "-c", " && ".join(shlex.join(c) for c in commands)], cwd=repo, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True); f.close(); (p / "calibration.pid").write_text(str(q.pid) + "\n"); launch.update(pid=q.pid, state="spawned"); claim.write_text(json.dumps(launch, indent=2) + "\n"); print("PID", q.pid, "LOG", p / "calibration.log", "Check TRAIN FIRST BATCH OK and nvidia-smi; PID alone is not training success")'
```

PID仅表示派生了进程。启动包装器记录GPU UUID/命令/线程环境，训练入口记录配置、代码/数据身份、资源及每阶段run.exit；不生成总queue.exit。

## 2. 查看启动与进度

```bash
tail -n 35 /home/featurize/work/nwpu_ffn42_20261003_01/calibration.log
```

```bash
nvidia-smi
```

需看到正确模型与设备、`TRAIN FIRST BATCH OK`和GPU活动。持续观察可用以下命令，Ctrl+C只退出tail：

```bash
tail -f /home/featurize/work/nwpu_ffn42_20261003_01/calibration.log
```

先按1–1.5小时预留，参考旧同卡两组校准共47分58秒，包含余量；不是新FFN实测时长或费用。短测109/116秒包含全数据验证和存盘，不能直接乘30。第一个完整正式epoch及存盘速度出来后修正预算。

## 3. 两项结束后只回传精简摘要

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/featurize/work/nwpu_ffn42_20261003_01"); names=("swin_ffn","soft_ffn"); r={n:json.loads((p/n/"run.json").read_text()) if (p/n/"run.json").is_file() else {"status":"not_complete"} for n in names}; [print(n,json.dumps({"exit":(p/n/"run.exit").read_text().strip() if (p/n/"run.exit").is_file() else "missing",**{k:v.get(k) for k in ("status","error","phase","seed","counts","best_epoch","best_val_oa_percent","final_epoch","reload_max_logit_diff","elapsed_seconds","cuda_peak_allocated_bytes","checkpoint_sha256")}},ensure_ascii=False)) for n,v in r.items()]'
```

每模型一行，不展开混淆矩阵。两组exit_code=0、status=complete、phase=calibrate、counts5040/1260且final_epoch30才有完成摘要；missing/not_complete不等于进程正在运行，要结合日志/进程。正式验收还需同身份、30行曲线、最高OA首次最佳轮及best/last各自SHA、同checkpoint指标和配对预测。best_epoch不要求等于30，FFN selected/final哈希可以不同，不照搬smoke的相等条件。soft探针包含epoch0到30诊断，PT只在improved或final保存。

本批不访问test，不追加seed或BigEarth，不根据结果改主指标/挑轮/重跑。soft+FFN要同时严格超过旧Swin94.2063492063492%、旧soft93.73015873015873%及新Swin+FFN才讨论后续；单seed验证收益不是稳定性证据。

本地命令语法及11项控制流检查通过（含首失败停止、占用输出/忙GPU/缺图/错误checkpoint拒绝），独立只读审阅无可行动问题；没有从本地访问Featurize或启动训练。
