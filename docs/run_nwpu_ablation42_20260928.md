# NWPU 三模型 seed42 校准 — 2026-09-28

**09-28完成回传：** 本页队列三阶段均exit0，Swin/fixed/soft最佳val OA94.206349/93.015873/93.730159%，总4346.87秒。现有结果应复用，不再次执行下方启动命令。下一步为[已有产物只读诊断](analyze_nwpu_calibration42_20260928.md)，[新协议结果表](../results/nwpu_seed42_calibration_comparison.md)已记录；以下启动步骤保留作实验命令依据。

## 研究问题与边界

双数据集soft工程检查已通过，进入既定E05第一批效果验证：同一NWPU清单、ImageNet预训练与配方下，软分配GAViT相对固定4×4 GAViT及Swin-T是否改善验证表现，新增成本如何。只比较三种架构，先seed42，各30轮；主指标val OA，辅指标macro-F1、每类指标、分配诊断及耗时/显存。目标是论文NWPU分组消融表的首seed候选记录及曲线/图稿，不以一个seed宣称稳定收益。

既有历史70/15/15成绩不能复用为新协议baseline。准备步骤搜索work中nwpu命名目录及其下一层的同manifest/seed42/完成calibrate结果，若发现则停止并先审计复用，避免重复付费。更便宜的本地合成测试、NWPU360/90和BigEarth96/64 GPU工程路径已完成；现有短测权重不能回答完整train/val收益问题。

固定代码ed480e60fea03ed786fdba70c144de5e815c34f5、工作树干净；manifest ce0cfef...9840e，5040train/1260val/25200test，同类重复组不跨split。各模型训练与验证不使用test作预测或选择；load_manifest会检查全部图片完整性，不等于测试集推理。完整清单与seed保持不变，不重新prepare。预训练SHA f7e9953f...6a9d36，输入224、CE、batch32、workers2、AdamW3e-4/wd1e-4、Cosine30、FP32无AMP、两项TF32 false，30轮校准；各按最佳val OA选轮，平分取先达到者（runner严格大于才更新）。refit/test不在本队列内。

## 执行顺序

用户返回NWPU哨兵图缺失，先校验持久ZIP再以-n恢复，保留已存在文件；后续入口逐图解码/哈希预检会拒绝内容错误。无需重新下载或重训工程smoke。下面都是给用户复制执行的命令，当前尚无这批服务器启动结果。

```bash
printf '%s\n' 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  /home/featurize/work/NWPU-RESISC45_a826b44.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/NWPU-RESISC45_a826b44.zip -d /home/featurize/data
```

恢复完成后准备独立队列目录，检查31500个路径、冻结身份、是否已有可复用结果、GPU空闲。路径存在检查不替代每个训练入口的全量图像哈希检查。

```bash
python -u -c 'import json,hashlib,subprocess; from pathlib import Path; repo=Path("/home/featurize/work/GAViT_SOFT_20260927"); out=Path("/home/featurize/work/nwpu_ablation42_20260928_01"); data=Path("/home/featurize/data/NWPU-RESISC45"); manifest=Path("/home/featurize/work/nwpu_protocol_20260927_02.json"); pre=Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors"); h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip(); assert commit=="ed480e60fea03ed786fdba70c144de5e815c34f5","Wrong code"; assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(),"Dirty code"; assert h(manifest)=="ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e","Manifest changed"; assert h(pre)=="f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36","Pretrained changed"; m=json.loads(manifest.read_text()); assert m["dataset"]=="NWPU-RESISC45" and m["counts"]=={"train":5040,"val":1260,"test":25200} and len(m["records"])==31500; missing=[r["path"] for r in m["records"] if not (data/r["path"]).is_file()]; assert not missing,missing[:5]; work=Path("/home/featurize/work"); prior=[str(p.parent) for p in sorted(set(work.glob("nwpu*/run.json"))|set(work.glob("nwpu*/*/run.json"))) if (r:=json.loads(p.read_text())).get("dataset")=="NWPU-RESISC45" and r.get("phase")=="calibrate" and r.get("seed")==42 and r.get("status")=="complete" and r.get("manifest_sha256")==h(manifest)]; assert not prior,("Existing calibrations: inspect before repeating",prior); assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy"; out.mkdir(exist_ok=False); info={"status":"ready","commit":commit,"manifest_sha256":h(manifest),"pretrained_sha256":h(pre),"seed":42,"counts":m["counts"],"stages":["swin","fixed","soft"],"epochs_each":30,"primary_metric":"val_OA","scope":"calibrate only; no refit or test inference"}; (out/"queue_preflight.json").write_text(json.dumps(info,indent=2)+"\n"); print("NWPU_QUEUE_READY",out,flush=True)'
```

仅出现NWPU_QUEUE_READY后启动。队列顺序Swin→固定4×4→软分配；每项都调用既有run_scene原入口，必须进程exit0且run.json完整、30轮/数量/清单身份正确，才执行下一项。三个模型串行，独立输出目录swin/fixed/soft；任何错误保留目录/日志，queue.started防止重复跑。

```bash
nohup python -u -c 'import os,sys,json,subprocess,atexit,time,hashlib; from pathlib import Path; repo=Path("/home/featurize/work/GAViT_SOFT_20260927"); out=Path("/home/featurize/work/nwpu_ablation42_20260928_01"); info=json.loads((out/"queue_preflight.json").read_text()); assert info["status"]=="ready"; assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==info["commit"]; assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(); assert hashlib.sha256(Path("/home/featurize/work/nwpu_protocol_20260927_02.json").read_bytes()).hexdigest()==info["manifest_sha256"]; assert hashlib.sha256(Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors").read_bytes()).hexdigest()==info["pretrained_sha256"]; (out/"queue.started").open("x").close(); status={"exit_code":1}; atexit.register(lambda:(out/"queue.exit").write_text("exit_code="+str(status["exit_code"])+"\n")); os.chdir(repo); start=time.monotonic(); stages=[("swin","swin",None),("fixed","gavit","attentive_spatial"),("soft","gavit","soft_prototype")]; base=[sys.executable,"-u",str(repo/"run_scene.py"),"train","--phase","calibrate","--seed","42","--manifest","/home/featurize/work/nwpu_protocol_20260927_02.json","--data-root","/home/featurize/data/NWPU-RESISC45","--pretrained-path","/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors","--device","cuda","--batch-size","32","--workers","2"]; results=[]; check=lambda name:(out/name/"run.exit").read_text().strip()=="exit_code=0" and (r:=json.loads((out/name/"run.json").read_text()))["status"]=="complete" and r["phase"]=="calibrate" and r["seed"]==42 and r["final_epoch"]==30 and r["counts"]=={"train":5040,"val":1260} and r["manifest_sha256"]==info["manifest_sha256"] and r["pretrained_sha256"]==info["pretrained_sha256"]; stage=lambda s:(print("START",s[0],flush=True),rc:=subprocess.run(base+["--model",s[1],"--output",str(out/s[0])]+(["--grouping",s[2]] if s[2] else [])).returncode,results.append({"stage":s[0],"process_exit":rc}),rc==0 and check(s[0]))[-1]; ok=all(stage(s) for s in stages); status["exit_code"]=0 if ok else 1; (out/"queue_summary.json").write_text(json.dumps({"status":"complete" if ok else "failed","elapsed_seconds":time.monotonic()-start,"stages":results},indent=2)+"\n"); print("NWPU_CALIBRATION_QUEUE_COMPLETE" if ok else "NWPU_CALIBRATION_QUEUE_FAILED",out,flush=True); sys.exit(status["exit_code"])' >> /home/featurize/work/nwpu_ablation42_20260928_01.launch.log 2>&1 < /dev/null &
```

启动需看到实际GPU进程/显存、MODEL及TRAIN FIRST BATCH OK；PID本身不等于成功。

```bash
nvidia-smi
tail -f /home/featurize/work/nwpu_ablation42_20260928_01.launch.log
```

全部成功标志NWPU_CALIBRATION_QUEUE_COMPLETE，并有queue.exit exit_code=0及每项run.exit=0。Ctrl+C仅退出tail，然后回传：

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/featurize/work/nwpu_ablation42_20260928_01"); print("QUEUE_EXIT",(p/"queue.exit").read_text().strip()); print((p/"queue_summary.json").read_text()); rows={name:json.loads((p/name/"run.json").read_text()) for name in ("swin","fixed","soft")}; keys=("status","model","architecture","seed","counts","best_epoch","best_val_oa_percent","final_epoch","checkpoint_sha256","reload_max_logit_diff","graph_capture","elapsed_seconds","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","manifest_sha256","pretrained_sha256"); print(json.dumps({name:{k:r[k] for k in keys} for name,r in rows.items()},indent=2))'
```

中途失败时保留launch日志及已完成模型，不无理由重训成功阶段。此队列不提供自动续训；意外中断后先按产物确定下一步，不删除目录以复用名称。

## 成本与本地验证范围

队列明确为3×30轮，包含全图预检、验证、频繁checkpoint保存和soft探针。NWPU先前360图短测纯训练111.25图/秒，线性缩放soft训练部分约5040/111.25×30=22.65分钟，但小样本吞吐/不同模型/验证与持久盘IO不同，不作为正式总耗时或收费报价。用每模型首轮实际train/val时间更新估计；保持用户现有实例，不主动租赁/换卡。

本地验证仅覆盖Python/Bash/zsh语法、三阶段参数/顺序、首阶段或中间阶段返回非零后停止、全局退出文件；测试工厂模拟子训练产物，未启动30轮或GPU。首次测试工厂误拦截subprocess.check_output内部Git调用，修正测试mock边界后成功，交付队列代码未为此改变。既有release与入口工程验证用于模型行为，不把模拟调度验证称模型效果。记录见[队列控制流验证](../results/evidence/nwpu_calibration42_20260928/local_validation.json)。
