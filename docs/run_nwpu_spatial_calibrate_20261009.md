# NWPU 空间先验：一次 seed42 完整校准

2026-10-10 已全部收尾：best val OA92.380952%、macro-F1 92.342474%、best24/final30、exit0，身份/缓存/配对/曲线均已回传，候选关闭。本页启动和[缓存收尾命令](read_nwpu_spatial_closeout_20261010.txt)均为历史记录，**不要重复运行，无需进一步补件**；当前[结果表](../results/nwpu_spatial_prior_trial.md)。

2026-10-09：一轮真实 GPU 短测训练/保存/重载完成，360 train / 90 val，退出0；耗时98.58秒，峰值allocated3.662GiB、reserved4.039GiB。原始回传及算术核对在[证据目录](../results/evidence/nwpu_spatial_smoke_20261009/server_return.json)。短测不提供分类提升证据。

## 目的与固定边界

只检验加入固定空间距离项是否改善旧soft的完整val分类。唯一主要变量仍为S中的anchor_gaussian_v1（强度0.5）；attention、图/GAT/feedback及训练配方保持。使用已发布67fe84d、seed42、5040train/1260val、30epochs、batch32/workers2、8threads、AdamW3e-4/wd1e-4、cosine30、FP32/无AMP/双TF32关闭，从原ImageNet权重重新开始，FFN关闭。

旧soft为1181/1260（93.730159%）；Swin为1187/1260（94.206349%）。若不超过旧soft就关闭候选；仅超过旧soft不视作超过baseline。若超过Swin，仍需macro-F1、配对、逐类与曲线判断；单seed不证明稳定。结果进入[候选表](../results/nwpu_spatial_prior_trial.md)，不自动扩大到test、refit、其他seed或BigEarthNet。

## 复制运行

下一整行先在CPU检查短测缓存S/W、样本/输入字节、发布源码与固定清单/预训练，检查单张2080Ti空闲、原图路径和新输出；全部通过才启动一个后台校准。不重复GPU诊断或短测。正式入口仍执行全图文件/像素验证。命令独占创建日志，输出由入口独占创建，旧结果不会覆盖；失败先看具体信息，不删目录重试。

```bash
python -c 'import csv,io,json,hashlib,os,subprocess,sys; from pathlib import Path; p=Path("/home/featurize/work/GAViT_SPATIAL_20261008"); s=Path("/home/featurize/work/nwpu_spatial42_smoke_20261008_01"); o=Path("/home/featurize/work/nwpu_spatial42_calibrate_20261009_01"); l=Path(str(o)+".launch.log"); d=Path("/home/featurize/data/NWPU-RESISC45"); m=Path("/home/featurize/work/nwpu_protocol_20260927_02.json"); pre=Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors"); g=lambda *a:subprocess.check_output(["git","-C",str(p),*a],text=True).strip(); sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest(); assert g("rev-parse","HEAD")=="67fe84d431685cf2c7788edfae21fa4e9eb040a4" and not g("status","--porcelain","--untracked-files=all"),"Wrong or modified source"; assert not o.exists() and not l.exists(),"Output/log exists; inspect before retrying"; r=json.loads((s/"run.json").read_text()); assert (s/"run.exit").read_text().strip()=="exit_code=0" and r["status"]=="complete" and r["phase"]=="smoke" and r["seed"]==42 and r["counts"]=={"train":360,"val":90} and r["final_epoch"]==1,"Smoke identity mismatch"; assert all(sha(p/k)==v for k,v in r["code"]["files_sha256"].items()),"Smoke/source mismatch"; assert sha(m)==r["manifest_sha256"]=="ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e" and sha(pre)==r["pretrained_sha256"]=="f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36","Input identity mismatch"; assert all((d/v["path"]).is_file() for v in json.loads(m.read_text())["records"]),"NWPU images missing; restore ZIP first"; sys.dont_write_bytecode=True; sys.path.insert(0,str(p)); import torch; from grouping_identity import soft_assignment_identity; from models.soft_region_grouping import SoftPrototypeGrouping; from scene_classification.runner import SceneDataset; torch.set_num_threads(8); q=s/"assignment_probe"; ident=json.loads((q/"probe_manifest.json").read_text()); ids=json.loads((s/"selected_samples.json").read_text())["val"][:8]; assert ident["split"]=="val" and len(ids)==len(set(ids))==8 and ident["sample_ids"]==ids and ident["soft_assignment"]==r["architecture"]["soft_assignment"]==soft_assignment_identity("anchor_gaussian_v1") and "token_refinement" not in r["architecture"],"Probe identity mismatch"; data=SceneDataset(d,[{"path":n,"label":0} for n in ids],False); x=torch.stack([data[i][0] for i in range(8)]); assert list(x.shape)==ident["input_shape"] and str(x.dtype)==ident["input_dtype"] and hashlib.sha256(x.numpy().tobytes()).hexdigest()==ident["input_sha256"],"Probe pixels mismatch"; names=("epoch_000.pt","epoch_001.pt","best.pt","final.pt"); a=[torch.load(q/n,map_location="cpu",weights_only=True) for n in names]; assert [v["epoch"] for v in a]==[0,1,1,1] and all(v["identity"]==ident for v in a),"Snapshot identity mismatch"; mask=SoftPrototypeGrouping(soft_assignment=ident["soft_assignment"]).candidate_mask; assert all(tuple(v["tensors"][k].shape)==(8,49,16) and bool(torch.isfinite(v["tensors"][k]).all()) and bool((v["tensors"][k]>=0).all()) and int(torch.count_nonzero(v["tensors"][k][:,~mask]))==0 for v in a for k in ("S","W")),"Invalid S/W"; err={"S_row":max(float((v["tensors"]["S"].sum(-1)-1).abs().max()) for v in a),"W_column":max(float((v["tensors"]["W"].sum(1)-1).abs().max()) for v in a)}; assert max(err.values())<1e-5 and len({sha(q/n) for n in names[1:]})==1,"Snapshot normalization/alias mismatch"; rows=[json.loads(v) for v in (q/"assignment_diagnostics.jsonl").read_text().splitlines()]; assert [v["epoch"] for v in rows]==[0,1] and all(v["split"]=="val" and v["sample_ids"]==ids for v in rows),"Probe history mismatch"; delta=float((a[1]["tensors"]["S"]-a[0]["tensors"]["S"]).abs().mean()); assert abs(delta-rows[1]["mean_abs_delta_S"])<1e-6 and rows[0]["mean_abs_delta_S"]==0,"Probe delta mismatch"; assert os.environ.get("CUDA_VISIBLE_DEVICES") in (None,"0"),"Unexpected GPU selection"; gpu=list(csv.reader(io.StringIO(subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.free,utilization.gpu","--format=csv,noheader,nounits"],text=True)))); assert len(gpu)==1 and "2080 Ti" in gpu[0][0] and int(gpu[0][1])>=8192 and int(gpu[0][2])<=5 and not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy or changed; inspect before running"; print("SMOKE_CACHE_PASS",json.dumps({**err,"delta_S":delta,"input_sha256":ident["input_sha256"]}),flush=True); cmd=[sys.executable,"-u",str(p/"run_scene.py"),"train","--model","gavit","--grouping","soft_prototype","--spatial-prior","anchor_gaussian_v1","--phase","calibrate","--seed","42","--manifest",str(m),"--data-root",str(d),"--output",str(o),"--pretrained-path",str(pre),"--device","cuda","--batch-size","32","--workers","2"]; f=l.open("x"); f.write("SMOKE_CACHE_PASS "+json.dumps({**err,"delta_S":delta,"input_sha256":ident["input_sha256"]})+"\n"); f.flush(); child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","PYTHONDONTWRITEBYTECODE":"1","CUDA_VISIBLE_DEVICES":"0"}); f.close(); print("PID:",child.pid,"OUTPUT:",o); print("PROGRESS: tail -f",l)'
```

日志显示SMOKE_CACHE_PASS和PID只表示预检通过/已发起进程。还需下面两条检查中有实际Python进程及GPU占用，日志显示MODEL gavit / calibrate、train=5040 val=1260、TRAIN FIRST BATCH OK才算真正启动：

```bash
pgrep -af "[r]un_scene.py train"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

```bash
tail -f /home/featurize/work/nwpu_spatial42_calibrate_20261009_01.launch.log
```

Ctrl+C只退出tail，不会停后台任务；归还/重启实例会中断任务。先预留约一小时（同型号旧完整soft约25分钟；本次小样本吞吐机械外推约62分钟且含启动开销，不能当稳态），完整首轮后再更新估算。

完成标记SCENE CALIBRATE COMPLETE及run.exit内容为`exit_code=0`；结束后仅回传短摘要，不展开混淆矩阵：

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/featurize/work/nwpu_spatial42_calibrate_20261009_01"); r=json.loads((p/"run.json").read_text()); print(json.dumps({"exit":(p/"run.exit").read_text().strip(),**{k:r.get(k) for k in ("status","error","phase","seed","counts","best_epoch","best_val_oa_percent","final_epoch","reload_max_logit_diff","elapsed_seconds","cuda_peak_allocated_bytes","checkpoint_sha256","assignment_probe")}},indent=2))'
```

初始交付时校准尚未运行；现在完成摘要和缓存收尾输出均已收到，停止该候选，本页命令只保留作历史记录。
