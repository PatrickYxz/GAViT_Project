# AID seed44 完整顺序执行

09-22完成更新：六阶段已回传complete/exit0，Swin/GAViT test OA96.84%/96.08%，差−0.76 pp。见[seed44核验](../results/aid_seed44_comparison.md)和[三seed主表](../results/aid_multiseed_summary.md)。本页命令仅作历史记录，不重新启动；后续使用[只读分析](analyze_aid_three_seeds.md)。

2026-09-21。用户在回传seed43结果后要求继续，沿用此前“一次拿齐命令、顺序执行、统一回传”方式。研究问题：完成预定42/43/44训练seed重复，检验固定划分下的收益幅度与训练方差；当前42/43的OA差+0.60/+0.30 pp不代替最终三seed结论。不因这些测试结果修改配置或挑seed。

主要变量仅训练seed43→44。冻结manifest `aid_protocol_20260920_03.json` SHA30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22，split_seed42/val_seed4242不变；Git deb2b45干净，预训练仍为model.safetensors SHAf7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36。模型、增强、AdamW、batch32/workers2、float32和T_max30沿用[冻结配方](run_aid_seed43.md)。seed44每模型校准30轮4000/1000，然后按自身最佳轮从相同ImageNet预训练重训5000pool，测试5000张。更便宜的已有checkpoint分析不能回答第三次训练随机性；已有smoke和两seed全流程不重复。

输出进入[AID多seed表](../results/aid_multiseed_summary.md)，无论结果正负均保留。参考seed43四阶段训练合计80.71分钟，另有两次测试及数据扫描；实际44轮数、耗时、GPU、显存以新报告为准，不把该参考当承诺或推理速度，费用未知。

## 一次启动

前提：seed43任务已经结束，原图仍在`/home/featurize/data/AID/AID`，同一实例保持运行。队列顺序为Swin44校准、GAViT44校准、Swin44 refit、GAViT44 refit、Swin44 test、GAViT44 test。每一步都校验数据与来源，失败停止；不覆盖旧输出。六个输出目录均为`/home/featurize/work/aid_{model}_{phase}44_20260921_01`。本地只准备命令，不代表服务器已经运行。

只复制代码内容，每块一整行；只启动一次。当前若在tail状态，先Ctrl+C退出查看，不会停止后台任务。

```bash
printf '%s\n' 'echo STARTING_AID44' 'test -f /home/featurize/data/AID/AID/Airport/airport_1.jpg || { echo AID_DATA_MISSING_RESTORE_ZIP; exit 1; }' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase calibrate --seed 44 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_swin_calibrate44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase calibrate --seed 44 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_gavit_calibrate44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase refit --seed 44 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_swin_refit44_20260921_01 --calibration /home/featurize/work/aid_swin_calibrate44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase refit --seed 44 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_gavit_refit44_20260921_01 --calibration /home/featurize/work/aid_gavit_calibrate44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_swin_refit44_20260921_01/last.pth --split test --output /home/featurize/work/aid_swin_test44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_gavit_refit44_20260921_01/last.pth --split test --output /home/featurize/work/aid_gavit_test44_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'echo AID44_COMPLETE' | nohup bash -e >> /home/featurize/work/aid44_batch_20260921_01.launch.log 2>&1 &
```

```bash
tail -f /home/featurize/work/aid44_batch_20260921_01.launch.log
```

看到实际MODEL/DEVICE、TRAIN FIRST BATCH OK及首轮才算已开始训练。各阶段SCENE COMPLETE只表示该阶段成功；最终AID44_COMPLETE才表示六步依次成功。队列是当前实例上的一次nohup任务，不支持实例停止后自动恢复。

## 完成后统一回传

看到AID44_COMPLETE后Ctrl+C，再执行下方读取命令。它要求六阶段exit0/complete、训练身份一致，核对两模型5000张同路径同标签配对及正确总数，再输出六阶段摘要、两模型OA/F1/逐类表和四格配对计数。不上传文件、不重新推理、不修改服务器文件。

```bash
python -c 'import csv,json,pathlib,sys; from collections import Counter; b=pathlib.Path(sys.argv[1]); names=["aid_swin_calibrate44_20260921_01","aid_gavit_calibrate44_20260921_01","aid_swin_refit44_20260921_01","aid_gavit_refit44_20260921_01","aid_swin_test44_20260921_01","aid_gavit_test44_20260921_01"]; paths=[b/n for n in names]; assert all((p/"run.exit").read_text().strip()=="exit_code=0" for p in paths),"A stage failed"; r=[json.loads((p/"run.json").read_text()) for p in paths]; assert all(x["status"]=="complete" for x in r); common={k:r[0][k] for k in ("training","code","hardware")}; assert all(x[k]==common[k] for x in r[:4] for k in common),"Training identity differs"; assert all(x["evaluation"][k]==common[k] for x in r[4:] for k in ("code","hardware")),"Evaluation identity differs"; m=[json.loads((p/"metrics.json").read_text()) for p in paths[4:]]; assert all(x["samples"]==5000 and x["split"]=="test" and x["training_phase"]=="refit" for x in m); A=list(csv.DictReader((paths[4]/"predictions.csv").open())); B=list(csv.DictReader((paths[5]/"predictions.csv").open())); a={x["path"]:x for x in A}; g={x["path"]:x for x in B}; assert len(A)==len(a)==len(B)==len(g)==5000 and a.keys()==g.keys(); assert all(a[p]["label"]==g[p]["label"] for p in a); c=Counter((a[p]["prediction"]==a[p]["label"],g[p]["prediction"]==g[p]["label"]) for p in a); assert abs((c[True,True]+c[True,False])/50-m[0]["oa_percent"])<1e-8 and abs((c[True,True]+c[False,True])/50-m[1]["oa_percent"])<1e-8; keys=("model","phase","seed","status","architecture","checkpoint","dataset","counts","final_epoch","best_epoch","best_val_oa_percent","checkpoint_sha256","manifest_sha256","pretrained_sha256","reload_max_logit_diff","graph_capture","refit_source","elapsed_seconds","total_parameters","trainable_parameters","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","main_process_peak_rss_bytes","samples","split"); mkeys=("samples","oa_percent","macro_f1_percent","dataset","split","training_phase","fit_scope","manifest_sha256","checkpoint_sha256","per_class"); print(json.dumps({"common_identity":common,"runs":[dict(directory=p.name,exit_code=0,**{k:x[k] for k in keys if k in x}) for p,x in zip(paths,r)],"tests":[{k:x[k] for k in mkeys} for x in m],"paired_counts":{"samples":5000,"both_correct":c[True,True],"gavit_corrects":c[False,True],"gavit_breaks":c[True,False],"both_wrong":c[False,False],"net_correct":c[False,True]-c[True,False]}},indent=2))' /home/featurize/work
```

## 出错时

先回传末80行；保留现有目录，不盲目重复启动。若明确输出AID_DATA_MISSING_RESTORE_ZIP，需先从work原ZIP恢复到`/home/featurize/data`，随后确认尚无阶段输出再继续；不重建manifest。

```bash
tail -n 80 /home/featurize/work/aid44_batch_20260921_01.launch.log
```

本地验证范围：shell/Python语法、六阶段顺序、失败停止、缺图拦截及合成5000行配对汇总；无真实GPU训练或服务器直接检查。验证结果见[记录](../results/evidence/aid44_batch_preparation_20260921/validation.json)。
