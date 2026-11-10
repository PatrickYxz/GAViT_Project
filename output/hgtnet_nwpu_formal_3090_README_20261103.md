# HGTNet NWPU seed42正式运行包

本地实现与验证完成后交付；真实3090正式训练仍由用户启动。使用[hgtnet_nwpu_formal_3090_20261103_v1.pyz](hgtnet_nwpu_formal_3090_20261103_v1.pyz)，上传到/home/featurize/work/，已有Swin-B/manifest/ZIP沿用。原smoke和batch16短测已通过，无需重复运行。

## 启动

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1.pyz inspect
```

ready=true时，执行输出中的run_command整行。它绑定当前空闲3090的UUID，不沿用旧卡常量。默认新output为/home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/runs/nwpu_seed42_01；scratch /environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1.pyz run --gpu-uuid GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1 --target-gpu 3090 --manifest /home/featurize/work/nwpu_protocol_20260927_02.json --data-root /home/featurize/data/NWPU-RESISC45 --archive /home/featurize/work/NWPU-RESISC45_a826b44.zip --weights /home/featurize/work/swin_b-68c6b09e.pth --config /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/configs/hgtnet_independent_v0.json --recipe /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/configs/hgtnet_nwpu_formal_v1.json --output /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1/runs/nwpu_seed42_01 --scratch-root /tmp/hgtnet_nwpu_formal_3090_20261103_v1_rtx3090_seed42_01
HGTNET_CUDA_DEVICE_PROBE {"available": true, "cuda": "12.1", "device_count": 1, "name": "NVIDIA GeForce RTX 3090", "torch": "2.2.2+cu121", "total_memory_bytes": 25295323136}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "device": "cuda:0", "epoch": 1, "formal_training_started": true, "gradients": {"groups": {"dual_attention": {"l2": 0.616735534653243, "tensors": 30}, "graph": {"l2": 9.499868945932883, "tensors": 120}, "head": {"l2": 7.592782613081288, "tensors": 2}, "patch_embed": {"l2": 9.173436827453662, "tensors": 4}, "transformer": {"l2": 14.879148673010235, "tensors": 295}}, "missing": [], "nonfinite": [], "passed": true}, "loss": 3.9306094646453857, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 1, "phase": "calibration", "pid": 2661, "post_update_checks_passed": true, "samples_seen": 16, "stage": "first_optimizer_step", "training_started": true, "updated_groups": {"dual_attention": true, "graph": true, "head": true, "patch_embed": true, "transformer": true}}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 30, "phase": "calibration", "pid": 2661, "samples_seen": 480, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 60, "phase": "calibration", "pid": 2661, "samples_seen": 960, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 90, "phase": "calibration", "pid": 2661, "samples_seen": 1440, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 120, "phase": "calibration", "pid": 2661, "samples_seen": 1920, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 150, "phase": "calibration", "pid": 2661, "samples_seen": 2400, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 180, "phase": "calibration", "pid": 2661, "samples_seen": 2880, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 210, "phase": "calibration", "pid": 2661, "samples_seen": 3360, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 240, "phase": "calibration", "pid": 2661, "samples_seen": 3840, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 270, "phase": "calibration", "pid": 2661, "samples_seen": 4320, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 300, "phase": "calibration", "pid": 2661, "samples_seen": 4800, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 1, "epoch_optimizer_steps": 315, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 315, "phase": "calibration", "pid": 2661, "post_update_checks_passed": true, "samples_seen": 5040, "stage": "training_epoch_complete", "train_seconds": 250.58360762799998, "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "best_val_accuracy_percent": 77.3015873015873, "completed": false, "epoch": 1, "epoch_wall_seconds": 326.696851827, "estimate_scope": "running completed-epoch mean for current phase only; recovered last commit may use precommit time excluding snapshot IO; excludes other phases/startup/final test/billing; not guaranteed", "estimated_remaining_phase_seconds": 32342.988330873, "formal_training_started": true, "observed_epoch_wall_mean_seconds": 326.696851827, "optimizer_steps": 315, "phase": "calibration", "pid": 2661, "selected_epoch": 1, "stage": "epoch_complete", "train_accuracy_percent": 47.916666666666664, "train_images_per_second": 20.113047488254114, "train_loss": 2.1766647955727954, "train_seconds": 250.58360762799998, "training_started": true, "val_accuracy_percent": 77.3015873015873, "val_macro_f1_percent": 76.81321440157961, "validation_seconds": 8.641767578000099}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 345, "phase": "calibration", "pid": 2661, "samples_seen": 480, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 375, "phase": "calibration", "pid": 2661, "samples_seen": 960, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 405, "phase": "calibration", "pid": 2661, "samples_seen": 1440, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 435, "phase": "calibration", "pid": 2661, "samples_seen": 1920, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 465, "phase": "calibration", "pid": 2661, "samples_seen": 2400, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 495, "phase": "calibration", "pid": 2661, "samples_seen": 2880, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 525, "phase": "calibration", "pid": 2661, "samples_seen": 3360, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 555, "phase": "calibration", "pid": 2661, "samples_seen": 3840, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 585, "phase": "calibration", "pid": 2661, "samples_seen": 4320, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 615, "phase": "calibration", "pid": 2661, "samples_seen": 4800, "stage": "training", "training_started": true}
HGTNET_FORMAL_PROGRESS {"attempt": 1, "completed": false, "epoch": 2, "epoch_optimizer_steps": 315, "formal_training_started": true, "optimizer_step_completed": true, "optimizer_step_succeeded": true, "optimizer_steps": 630, "phase": "calibration", "pid": 2661, "post_update_checks_passed": true, "samples_seen": 5040, "stage": "training_epoch_complete", "train_seconds": 236.35908696399997, "training_started": true}

为/tmp/hgtnet_nwpu_formal_3090_20261103_v1_rtx3090_seed42_01。

长任务可在生成的run_command前加`nohup `，末尾加` >> /home/featurize/work/hgtnet_formal_3090_20261103_v1.launch.log 2>&1 &`，整行执行。脚本自身还会将真实日志写到output/attempts/0001/run.log。另开终端检查：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1.pyz status
```

```sh
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader
```

首批成功后progress会出现first_optimizer_step，含真实梯度/参数更新证据；同时确认GPU进程和实际日志。随后让任务持续运行。训练每30步及每轮更新进度，轮末给出train/val指标、每轮实际耗时与当前阶段剩余时间估算。最终HGTNET_NWPU_FORMAL_COMPLETE后，回传status；完整逐图结果在output/test.json。

## 固定协议与披露

- 校准：train5040/val1260、100轮、batch16、不drop_last，主指标validation正确数/OA。严格更大才更新最佳，平手保留更早epoch。
- Fresh refit：从完全相同初值SHA重新建模，用pool6300训练所选最佳epoch数。独立新Adam，不承袭smoke/calibration学得状态。模型保存并独立新模型数值回读后，再评估25200张test一次；中断最终推理可在相同固定模型上恢复执行，不用于调参。
- Adam lr1e-5、weight_decay5e-5、100epoch/batch16/CE为原文已确认设置。betas(.9,.999)、eps1e-8、constantLR/无warmup/scheduler、无label smoothing/实际梯度裁剪为事先固定的独立工程选择。
- 训练/重训：RGB→Resize256×256 bilinear→RandomCrop224→HFlip(.5)→ImageNet normalize；评估：Resize256→CenterCrop224→同normalize。原文只确认crop/flip，精确参数/插值/归一化次序是公开工程选择。
- 保持完整74619858参数独立HGTNet/原config/Swin-B候选初值，seed42、FP32、无AMP/TF32、CPU8线程、workers0。原作者精确架构/预训练仍未确认，本结果属于独立完整方法，不能归因单个图模块或声称官方精确复现。
- 每epoch按42+epoch重设Torch种子和独立shuffle generator，以相同随机序列/顺序重启。GPU位级恢复等价未测；CPU小夹具等价不代表所有CUDA算子严格确定。

## 预算与工件

真实batch16短测：allocated4.4071GiB/reserved4.8574GiB、计算段79.4622图/秒；全probe10.059秒不代表正式epoch。按三步中位缩放，最多100轮校准+100轮refit约3.97小时纯更新计算，另有读图、增强、验证、checkpoint、阶段切换和test开销；它不是正式ETA或租卡上限，refit轮数还取决于选轮。以实际epoch_complete的phase估算为准。

保守磁盘门：work6GiB/scratch2GiB，同卷相加。work保存小包、binding/recipe/source身份、每attempt日志/runtime/命令、历史、最佳/最终calibration模型、final_refit模型、回读记录、逐图test/混淆/逐类指标和summary。两个交替epoch恢复槽、两个best槽控制checkpoint增长；原图只恢复pool6300到本地scratch，test图在最终refit/回读后才恢复/解码。

## 中断恢复

已有完整epoch快照时，保留原输出目录及小包，用同一版本执行：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_nwpu_formal_3090_20261103_v1.pyz inspect --resume
```

ready=true后执行新生成的resume命令；同软件的另一3090可用当前UUID恢复。需要binding/source/recipe/manifest/初始权重完全匹配，回滚到最近完整epoch；中断epoch会重做。每attempt保留之前的摘要/退出/进度及新日志。源码改变会拒绝兼容恢复，不能服务器手改训练配方。

换实例后scratch可以重新从固定ZIP恢复；残缺恢复目录被保留，并在新attempt目录重建。若首轮尚未完成，没有可恢复epoch，保留失败工件、用新的output/scratch开始；程序不会把PID或残缺checkpoint当作成功。

summary.wall_seconds与显存为当前attempt范围；过去attempt记录保留，不能将最后一次wall当全部账单。status completed+exit0+正确marker才接受完整实验完成。实际费用需租卡单价/账单，不在本地估算中编造。

[配方原件](../configs/hgtnet_nwpu_formal_v1.json)、[实现计划](../docs/superpowers/plans/2026-11-02-hgtnet-formal-seed42.md)、[本地完整证据](../results/evidence/hgtnet_formal_prepare_20261102/README.md)。包身份和最终验证见随包manifest及证据，不需要上传本地CPU夹具或大型checkpoint。

## 最终交付身份

v1包83604B，SHA256：b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1。

bundle ID：9a673b432dd8dea6b8cb175a559515c2092d953f8fd4016362dd325726c35710。

canonical recipe SHA：ce72e1822d55632b6dd9b02cdb4dc32da7e3e77eedca815b83981ff559c1af85。

最终386项相关测试、完整模型CPU人工JPEG流程与checkpoint独立回读误差0、独立审查和145项身份/保护核验通过。CPU夹具与本机stdlib预检不等于正式GPU训练。未交付的无v1后缀本地候选保留作审查证据，请使用本页链接的v1包。代码Git基线058f1d8b79d0be538ec358eb742d2cd3992921d4，未提交增量由包内逐文件SHA绑定。
