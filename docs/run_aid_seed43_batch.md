# AID seed43 后续任务一次顺序执行

09-21完成更新：六阶段已回传complete/exit0及两模型5000test指标，核验见[结果](../results/aid_seed43_comparison.md)。本页为历史执行命令，不再次启动；后续使用[seed44队列](run_aid_seed44_batch.md)。

2026-09-21。用户明确要求把后续命令一起提供、连续运行后统一回传。本次授权范围为seed43剩余五步，不包括seed44。沿用[冻结协议](run_aid_seed43.md)，不修改正式代码、不上传工具包。

前提：上一条Swin43校准`aid_swin_calibrate43_20260921_02`已经启动，或者已经成功完成。队列不会重新启动它。当前Swin若未启动，先运行此前命令；若处于tail查看状态，Ctrl+C仅退出日志查看，再执行下方队列一次。

队列最多等当前Swin43退出文件120分钟。确认exit_code=0后，串行执行GAViT43校准、Swin43 refit、GAViT43 refit、Swin43 test、GAViT43 test。各自refit由本次seed43校准报告自动选轮数，T_max仍30。bash -e会在任一阶段非零退出时停止；阶段入口检查数据、权重、配置/来源并拒绝覆盖已有输出目录。重复启动不会覆盖旧结果，但会因已有目录失败，因此只启动一次。

## 启动队列

每个代码块均为一整行。printf把固定命令交给nohup bash执行，无嵌套bash -c、无上传文件。各Python阶段的stdin单独设为/dev/null，避免读取队列脚本输入；输出既保留于各自run.log，也进入总launch日志。

```bash
printf '%s\n' 'echo WAITING_FOR_SWIN43_CALIBRATION' 'for aid43_tick in {1..360}; do test -f /home/featurize/work/aid_swin_calibrate43_20260921_02/run.exit && break; sleep 20; done' 'grep -qx exit_code=0 /home/featurize/work/aid_swin_calibrate43_20260921_02/run.exit' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase calibrate --seed 43 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_gavit_calibrate43_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase refit --seed 43 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_swin_refit43_20260921_01 --calibration /home/featurize/work/aid_swin_calibrate43_20260921_02 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model gavit --phase refit --seed 43 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --output /home/featurize/work/aid_gavit_refit43_20260921_01 --calibration /home/featurize/work/aid_gavit_calibrate43_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_swin_refit43_20260921_01/last.pth --split test --output /home/featurize/work/aid_swin_test43_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py evaluate --checkpoint /home/featurize/work/aid_gavit_refit43_20260921_01/last.pth --split test --output /home/featurize/work/aid_gavit_test43_20260921_01 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --device cuda --batch-size 32 --workers 2 < /dev/null' 'echo AID43_COMPLETE' | nohup bash -e >> /home/featurize/work/aid43_followup_20260921_01.launch.log 2>&1 &
```

## 查看进度

下面同时显示当前Swin校准与后续队列日志。初期队列可能只有WAITING_FOR_SWIN43_CALIBRATION，后续每一步都有自己的SCENE ... COMPLETE。只有最后AID43_COMPLETE才表示余下五步均成功返回。

```bash
tail -f /home/featurize/work/aid_swin_calibrate43_20260921_02.launch.log /home/featurize/work/aid43_followup_20260921_01.launch.log
```

## 全部完成后统一回传

Ctrl+C退出tail，运行下方摘要命令。它读取全部六个阶段的退出码、报告以及两份测试metrics/predictions：要求全部complete/exit0、共享训练身份一致、评估代码/硬件一致，核查5000张路径和标签配对并重算OA。输出共享身份一次、每阶段best/final/权重/资源信息、两模型OA/F1与30类指标、配对纠错计数。它不修改任何服务器文件。

```bash
python -c 'import csv,json,pathlib,sys; from collections import Counter; b=pathlib.Path(sys.argv[1]); names=["aid_swin_calibrate43_20260921_02","aid_gavit_calibrate43_20260921_01","aid_swin_refit43_20260921_01","aid_gavit_refit43_20260921_01","aid_swin_test43_20260921_01","aid_gavit_test43_20260921_01"]; paths=[b/n for n in names]; assert all((p/"run.exit").read_text().strip()=="exit_code=0" for p in paths),"A stage failed"; r=[json.loads((p/"run.json").read_text()) for p in paths]; assert all(x["status"]=="complete" for x in r); common={k:r[0][k] for k in ("training","code","hardware")}; assert all(x[k]==common[k] for x in r[:4] for k in common),"Training identity differs"; assert all(x["evaluation"][k]==common[k] for x in r[4:] for k in ("code","hardware")),"Evaluation identity differs"; m=[json.loads((p/"metrics.json").read_text()) for p in paths[4:]]; assert all(x["samples"]==5000 and x["split"]=="test" and x["training_phase"]=="refit" for x in m); A=list(csv.DictReader((paths[4]/"predictions.csv").open())); B=list(csv.DictReader((paths[5]/"predictions.csv").open())); a={x["path"]:x for x in A}; g={x["path"]:x for x in B}; assert len(A)==len(a)==len(B)==len(g)==5000 and a.keys()==g.keys(); assert all(a[p]["label"]==g[p]["label"] for p in a); c=Counter((a[p]["prediction"]==a[p]["label"],g[p]["prediction"]==g[p]["label"]) for p in a); assert abs((c[True,True]+c[True,False])/50-m[0]["oa_percent"])<1e-8 and abs((c[True,True]+c[False,True])/50-m[1]["oa_percent"])<1e-8; keys=("model","phase","seed","status","counts","final_epoch","best_epoch","best_val_oa_percent","checkpoint_sha256","manifest_sha256","pretrained_sha256","reload_max_logit_diff","graph_capture","refit_source","elapsed_seconds","total_parameters","trainable_parameters","cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","main_process_peak_rss_bytes","samples","split"); mkeys=("samples","oa_percent","macro_f1_percent","dataset","split","training_phase","fit_scope","manifest_sha256","checkpoint_sha256","per_class"); print(json.dumps({"common_identity":common,"runs":[dict(directory=p.name,exit_code=0,**{k:x[k] for k in keys if k in x}) for p,x in zip(paths,r)],"tests":[{k:x[k] for k in mkeys} for x in m],"paired_counts":{"samples":5000,"both_correct":c[True,True],"gavit_corrects":c[False,True],"gavit_breaks":c[True,False],"both_wrong":c[False,False],"net_correct":c[False,True]-c[True,False]}},indent=2))' /home/featurize/work
```

如果未出现AID43_COMPLETE或看到Traceback，保留现有结果，先回传总日志末80行和当前Swin日志末30行；不要直接重启整个队列。实例停止/重建会中断队列，没有自动恢复功能，未完成阶段需先查日志再用新输出身份继续。等待上限不代表任务必然在两小时内完成，训练总时间另按本次实际轮数/硬件记录。

```bash
tail -n 80 /home/featurize/work/aid43_followup_20260921_01.launch.log
```

本地仅验证shell/Python语法及模拟阶段顺序、失败停止、当前Swin失败/超时拒绝、汇总读取与配对计数。没有执行真实CUDA训练或确认服务器队列已启动；所有正式结果仍待用户回传。
