# NWPU seed42 三模型正式拟合与测试

**本批已收尾：** 训练、测试与缓存配对全部回传并核对，无需再次执行下方命令。[收尾结论](../results/nwpu_seed42_closeout.md)。

当前仅执行[缓存配对收尾](analyze_nwpu_test42_20260929.md)，无需重新运行本页GPU队列。

**09-29正式refit/test已完成：** 六阶段均exit0，test OA为Swin93.932540%、fixed94.103175%、soft93.869048%。[正式测试结果](../results/nwpu_seed42_test_comparison.md)已归档；下文原启动步骤/建议保留为历史，不重复训练。

2026-09-29用户确认继续，沿用E05已冻结的单候选实验。研究问题：校准阶段soft相对fixed的改善是否延续到独立test，同时报告与Swin的差距。三种架构是主要变量，所有结果进入同协议NWPU分组消融表，不因测试分数调整温度、轮数或seed。

此前三模型30轮校准与缓存分析已完成，不再跑smoke或校准。现有校准模型仅拟合5040张，不能作为6300张完整训练池拟合的替代。正式拟合从原ImageNet预训练重新初始化，按各自验证选轮规则执行29/29/28轮，Cosine T_max保持30；train+val6300，val_OA=None正常。全部拟合完成后才串行测试三个last.pth，各25200张。没有checkpoint续训或测试选轮。

## 冻结身份与预算

- 代码ed480e60fea03ed786fdba70c144de5e815c34f5，服务器`/home/featurize/work/GAViT_SOFT_20260927`，保持干净。
- 校准来源`/home/featurize/work/nwpu_ablation42_20260928_01/{swin,fixed,soft}`，best轮29/29/28，预检固定三个best.pth SHA及run.json哈希。来源完整身份由原入口再次校验。
- Manifest `/home/featurize/work/nwpu_protocol_20260927_02.json`，SHA ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e，5040/1260/25200；6300只表示正式拟合池，测试25200保持独立。
- 预训练`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`，SHA f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36。RGB224增强/归一化、CE、AdamW3e-4/wd1e-4、batch32/workers2、seed42、FP32无AMP/双TF32-off、K16/80边与soft版本全部保持校准身份。
- 新结果根目录`/home/featurize/work/nwpu_refit_test42_20260929_01`，六个子目录`{swin,fixed,soft}_{refit,test}`。禁止覆盖；发现已完成同清单seed42的refit时先审计复用。
- 同RTX2080Ti预计1–1.5小时，纯训练按实测缩放约56.87分钟，其余为校验、存盘、重建和测试；不是时间保证，不含解压/实例准备。实际费用待租价与墙钟回传。GPU型号变化时先停止核对，不沿用原报价。

## 1. 准备（不启动训练）

检查代码/输入/校准身份、31500文件存在、同协议既有refit和GPU占用，之后才创建独立输出目录。文件存在不替代每个入口的解码/哈希核验。如果报DATA_MISSING，先从持久化原ZIP恢复，不重新prepare清单或改模型。

```bash
python -u -c 'import json,hashlib,subprocess; from pathlib import Path; repo=Path("/home/featurize/work/GAViT_SOFT_20260927"); out=Path("/home/featurize/work/nwpu_refit_test42_20260929_01"); source=Path("/home/featurize/work/nwpu_ablation42_20260928_01"); manifest=Path("/home/featurize/work/nwpu_protocol_20260927_02.json"); pre=Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors"); data=Path("/home/featurize/data/NWPU-RESISC45"); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); commit="ed480e60fea03ed786fdba70c144de5e815c34f5"; mh="ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e"; ph="f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36"; names=("swin","fixed","soft"); assert not out.exists(),"Output exists; inspect it before another launch"; assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==commit,"Wrong source commit"; assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(),"Dirty source tree"; assert sha(manifest)==mh and sha(pre)==ph,"Input hash changed"; m=json.loads(manifest.read_text()); assert m["dataset"]=="NWPU-RESISC45" and m["counts"]=={"train":5040,"val":1260,"test":25200} and len(m["records"])==31500; missing=[v["path"] for v in m["records"] if not (data/v["path"]).is_file()]; assert not missing,("DATA_MISSING",missing[:5]); assert (source/"queue.exit").read_text().strip()=="exit_code=0"; r={n:json.loads((source/n/"run.json").read_text()) for n in names}; assert [r[n]["best_epoch"] for n in names]==[29,29,28]; expected=["bbfd4e472739126276e79014f944f911d7b0abbc6f863bb291d799845b9cd0ea","1766fe7c330eb37fb350f0348bcd255bafc451b793411a927548ee005a140f67","674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12"]; assert all(r[n]["status"]=="complete" and r[n]["phase"]=="calibrate" and r[n]["seed"]==42 and r[n]["final_epoch"]==30 and r[n]["manifest_sha256"]==mh and r[n]["pretrained_sha256"]==ph and (source/n/"run.exit").read_text().strip()=="exit_code=0" and sha(source/n/"best.pth")==r[n]["checkpoint_sha256"]==expected[i] for i,n in enumerate(names)),"Calibration changed"; assert all(r[n][k]==r["swin"][k] for n in names for k in ("training","code","hardware")),"Calibration identities differ"; prior=[str(f.parent) for f in set(out.parent.glob("nwpu*/run.json"))|set(out.parent.glob("nwpu*/*/run.json")) if (v:=json.loads(f.read_text())).get("status")=="complete" and v.get("phase")=="refit" and v.get("seed")==42 and v.get("manifest_sha256")==mh]; assert not prior,("Existing refits: inspect for reuse",prior); gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip(); assert gpu==r["swin"]["hardware"]["device_name"],("GPU differs from calibration",gpu); assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy"; out.mkdir(); info={"status":"ready","commit":commit,"manifest_sha256":mh,"pretrained_sha256":ph,"calibration_run_sha256":{n:sha(source/n/"run.json") for n in names},"selected_epochs":{n:r[n]["best_epoch"] for n in names},"gpu":gpu,"scope":"seed42 refit 6300 images; test 25200 images; no calibration rerun"}; (out/"queue_preflight.json").write_text(json.dumps(info,indent=2)+"\n"); print("NWPU_REFIT_TEST_READY",out,info["selected_epochs"],flush=True)'
```

只有NWPU_REFIT_TEST_READY后执行下一步。缺图时才使用：

```bash
printf '%s\n' 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53  /home/featurize/work/NWPU-RESISC45_a826b44.zip' | sha256sum -c - && unzip -n -q /home/featurize/work/NWPU-RESISC45_a826b44.zip -d /home/featurize/data
```

## 2. 启动六阶段串行队列

Swin refit → fixed refit → soft refit → Swin test → fixed test → soft test。每项只有process exit0、run.exit0及产物身份检查通过才进入下一项；首次启动标记防止重复启动。若中途失败，保留已完成产物和launch日志，先诊断复用，不能删除目录重跑整批。

```bash
nohup python -u -c 'import json,hashlib,subprocess; from pathlib import Path; repo=Path("/home/featurize/work/GAViT_SOFT_20260927"); out=Path("/home/featurize/work/nwpu_refit_test42_20260929_01"); source=Path("/home/featurize/work/nwpu_ablation42_20260928_01"); manifest=Path("/home/featurize/work/nwpu_protocol_20260927_02.json"); pre=Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors"); data=Path("/home/featurize/data/NWPU-RESISC45"); sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); commit="ed480e60fea03ed786fdba70c144de5e815c34f5"; mh="ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e"; ph="f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36"; names=("swin","fixed","soft"); import os,sys,time,atexit; info=json.loads((out/"queue_preflight.json").read_text()); assert info["status"]=="ready" and info["commit"]==commit; assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==commit and not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(); assert sha(manifest)==mh and sha(pre)==ph and all(sha(source/n/"run.json")==info["calibration_run_sha256"][n] for n in names),"Frozen inputs changed"; assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy"; (out/"queue.started").open("x").close(); status={"exit_code":1}; atexit.register(lambda:(out/"queue.exit").write_text("exit_code="+str(status["exit_code"])+"\n")); os.chdir(repo); start=time.monotonic(); cal={n:json.loads((source/n/"run.json").read_text()) for n in names}; common=["--manifest",str(manifest),"--data-root",str(data),"--device","cuda","--batch-size","32","--workers","2"]; entry=[sys.executable,"-u",str(repo/"run_scene.py")]; stages=[(n,"refit",entry+["train","--phase","refit","--seed","42","--model","swin" if n=="swin" else "gavit","--calibration",str(source/n),"--pretrained-path",str(pre),"--output",str(out/(n+"_refit"))]+common+([] if n=="swin" else ["--grouping","attentive_spatial" if n=="fixed" else "soft_prototype"])) for n in names]+[(n,"test",entry+["evaluate","--checkpoint",str(out/(n+"_refit")/"last.pth"),"--split","test","--output",str(out/(n+"_test"))]+common) for n in names]; (out/"commands.json").write_text(json.dumps(stages,indent=2)+"\n"); read=lambda n,k,f:json.loads((out/(n+"_"+k)/f).read_text()); check_refit=lambda n,r:r["phase"]=="refit" and r["counts"]=={"train":6300,"val":0} and r["final_epoch"]==info["selected_epochs"][n] and r["best_epoch"] is None and r["best_val_oa_percent"] is None and r["reload_max_logit_diff"]<=1e-5 and r["refit_source"]=={"run_json_sha256":info["calibration_run_sha256"][n],"best_checkpoint_sha256":cal[n]["checkpoint_sha256"]} and all(r[k]==cal[n][k] for k in ("model","seed","training","architecture","code","hardware","manifest_sha256","pretrained_sha256","graph_capture")) and sha(out/(n+"_refit")/"last.pth")==r["checkpoint_sha256"]; check_test=lambda n,r:(m:=read(n,"test","metrics.json"))["samples"]==r["samples"]==25200 and m["split"]==r["split"]=="test" and m["training_phase"]=="refit" and m["fit_scope"]=="train_plus_val" and m["manifest_sha256"]==mh and m["checkpoint_sha256"]==r["checkpoint_sha256"]==read(n,"refit","run.json")["checkpoint_sha256"] and r["evaluation"]["code"]==cal[n]["code"] and r["evaluation"]["hardware"]==cal[n]["hardware"] and all((out/(n+"_test")/f).is_file() for f in ("predictions.csv","per_class.csv")); check=lambda n,k:(out/(n+"_"+k)/"run.exit").read_text().strip()=="exit_code=0" and (r:=read(n,k,"run.json"))["status"]=="complete" and (check_refit(n,r) if k=="refit" else check_test(n,r)); results=[]; stage=lambda s:(print("START",s[0],s[1],flush=True),rc:=subprocess.run(s[2]).returncode,passed:=rc==0 and check(s[0],s[1]),results.append({"model":s[0],"stage":s[1],"process_exit":rc,"checks_passed":passed}),print("STAGE_COMPLETE" if passed else "STAGE_FAILED",s[0],s[1],flush=True),passed)[-1]; ok=all(stage(s) for s in stages); summary={"status":"complete" if ok else "failed","elapsed_seconds":time.monotonic()-start,"stages":results,"tests":{n:{k:read(n,"test","metrics.json")[k] for k in ("samples","oa_percent","macro_f1_percent","training_phase","fit_scope","manifest_sha256","checkpoint_sha256")} for n in names} if ok else {}}; (out/"queue_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); status["exit_code"]=0 if ok else 1; print("NWPU_REFIT_TEST_COMPLETE" if ok else "NWPU_REFIT_TEST_FAILED",out,flush=True); sys.exit(status["exit_code"])' >> /home/featurize/work/nwpu_refit_test42_20260929_01.launch.log 2>&1 < /dev/null &
```

## 3. 观察启动与结束

```bash
nvidia-smi
tail -f /home/featurize/work/nwpu_refit_test42_20260929_01.launch.log
```

PID不是成功依据。首项需要MODEL swin、PHASE refit、train=6300 val=0、TRAIN FIRST BATCH OK和EPOCH 1/29。另两项应为29/28轮，实际图48+32边通过。Ctrl+C只退出tail；不要重启实例，中断后台任务可能无法留下queue.exit。

完成要求NWPU_REFIT_TEST_COMPLETE、queue.exit=0及六项run.exit=0，保存OA、macro-F1、逐类准确率与支持、混淆矩阵、predictions.csv、模型/来源配置和资源记录。随后回传：

```bash
python -c 'import json; from pathlib import Path; p=Path("/home/featurize/work/nwpu_refit_test42_20260929_01"); print("QUEUE_EXIT",(p/"queue.exit").read_text().strip()); print((p/"queue_summary.json").read_text()); names=("swin","fixed","soft"); r={n:json.loads((p/(n+"_refit")/"run.json").read_text()) for n in names}; keys=("status","phase","model","architecture","seed","counts","final_epoch","checkpoint_sha256","refit_source","reload_max_logit_diff","graph_capture","elapsed_seconds","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes"); print(json.dumps({n:{k:r[n][k] for k in keys} for n in names},indent=2)); print("STAGE_EXITS",{n+"_"+k:(p/(n+"_"+k)/"run.exit").read_text().strip() for n in names for k in ("refit","test")})'
```

当前仅交付命令，尚无服务器实际启动或新测试指标。模型实现、Git历史与旧输出保持。本地命令/控制流验证记录在results/evidence/nwpu_refit_test42_20260929/local_validation.json；模拟调度通过不代表真实CUDA完成。
