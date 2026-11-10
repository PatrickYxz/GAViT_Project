# HGTNet AID seed42正式运行包（RTX3060）

这是原AID30类协议上的独立HGTNet补充实验。真实3060已通过一轮240/60的保存/重载检查，以及batch16四步容量检查（allocated4.407GiB、reserved4.855GiB）。新正式包完成64项本地范围测试和独立审查；实际正式任务尚待以下命令启动。

## 上传与启动

只把同目录`hgtnet_aid_formal_3060_20261105_v1.pyz`上传到Featurize `/home/featurize/work/`，保留原文件名。无需再上传数据或初始权重；旧batch16包及输出继续保留，新入口只读其status作为同物理GPU容量证明。

复制这一整行：先显示完整预检，再由run内部重复门禁、通过后开始正式训练。

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz inspect && /environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz run
```

预检检查原pool5000张图片及初始化完整SHA、固定模型/配方/源码、当前空闲3060及实际batch16证明、至少6GiB可用work磁盘与6GiB可用主机RAM。test图片在refit与独立重载通过前不读取。若预检有blockers，run会拒绝训练；贴完整输出即可。

启动后把`HGTNET_AID_CUDA_DEVICE_PROBE`和`HGTNET_AID_FORMAL_PROGRESS`中`stage=first_optimizer_step`的行贴回；首步应是calibration、device cuda:0、451个梯度张量的五组检查/代表更新均通过。仅有PID或预检ready不能当已成功启动。第1轮`epoch_complete`也可贴回，用实际读图/训练/验证/保存耗时更新阶段预算。

这是前台长任务，保持终端与实例运行。日志、checkpoint和恢复快照持续保存在work。完整流程结束才输出`HGTNET_AID_FORMAL_COMPLETE`。

## 固定正式流程

- 原manifest：`/home/featurize/work/aid_protocol_20260920_03.json`，SHA `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`；数据根`/home/featurize/data/AID/AID`。保留原不等类计数和Industrial同像素重复组。
- 4000train+1000val跑100轮calibration，按验证correct严格增加选最佳、相同选最早；从与calibration完全相同初态fresh初始化，5000pool训练到selected epoch（1–100），严格独立重载误差≤1e-6后，对5000test只评估一次。
- batch/eval16、seed42、Adam lr1e-5/WD5e-5、CE、无scheduler/AMP/TF32/实际梯度裁剪；epoch seed42+epoch。train/pool Resize256 bilinear→RandomCrop224/flip0.5，eval CenterCrop224，ImageNet归一化。
- 原Swin-B候选SHA `68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`；不使用已训练smoke/NWPU权重。模型总/可训练74612163参数、451张量。

最多100轮calibration加最多100轮fresh refit；训练更新数25000+313×selected_epoch。已有0.469秒固定16图计算时延不是完整loader吞吐、正式总时长或费用。首轮日志给当前phase的平均epoch wall与剩余phase估计，包含验证/保存IO；未涵盖另一个phase、最终test及账单。恢复后wall/显存范围属于当前attempt，不能直接当累计账单或全部attempt峰值。

## 读取状态

在另一个终端或完成后运行：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz status
```

运行中应为incomplete，但实际更新后training_started/formal_training_started为true。最终须state complete、exit0、completion_evidence_valid/source_identity_matches true，并返回test Accuracy/macro-F1、最佳轮/最终轮、checkpoint/reload、耗时/显存和test SHA。状态检查核对记录与文件SHA，不反序列化checkpoint或重跑GPU。

## 中断与恢复

仅在原任务已退出且有有效完整epoch snapshot时使用：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz inspect --resume
```

预检通过后：

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1.pyz resume
```

不会默默从头训练；首个epoch未提交时没有自动恢复点。每次尝试保留原summary/progress/exit和已发生但被回滚的部分更新；全局步数标识当前接受轨迹。完整test packet会验证复用，不再次测试；test已started却没有完整有效packet时，明确拒绝自动重测，贴回状态再分析。

强制杀进程或实例重启可能留下独占`.lock`。程序不自动删除它。若恢复提示锁存在，先运行上面的只读`inspect --resume`，再用下面两条只读命令查看锁归属与实际GPU进程，贴回输出核对；不要直接删除锁或重新run。

```sh
cat /home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01.lock
```

```sh
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader
```

核对锁内PID已退出、没有同一输出的活动写入、run_identity/binding及完整snapshot归属后，才能处理该任务留下的死锁文件并显式resume。换实例后须先恢复原数据，新的物理GPU也必须有对应有效batch16容量证明。

## 固定产物与身份

输出：`/home/featurize/work/hgtnet_aid_formal_3060_20261105_v1/runs/aid_seed42_01`。

保存binding、source_identity、runtime、initialization、history、两phase首步、best/last验证预测、best_calibration.pt、final_calibration.pt、final_refit.pt及SHA receipt、checkpoint_reload、test_transaction、有序5000条test预测/30类指标/混淆矩阵、summary/run.exit/completion.marker。`attempts/0001/`等保存每次完整命令、preflight、runtime、容量证明、pretraining和run.log；`recovery/slot0.pt`/`slot1.pt`为交替完整epoch恢复点。

包85287B；SHA `ef2bc84e0df446b7d2bdb5ac2097b21456e66a97c381d8cf5d16e87a3bac5c4b`；bundle ID `31e712e70be7495ea8f64cdd14fe6d0e1c2ba158a30babd0918f918ee8ad0d1f`。22个源码/配置、23个payload、25个ZIP成员，无权重大文件或安装器。

Recipe canonical SHA `b4b007a30f3e96c68e777b1ce7115779c80fa21f104c65cd74009260a115584d`。Git基础提交`058f1d8b79d0be538ec358eb742d2cd3992921d4`，新代码是未提交的source-only冻结快照，实际代码由完整22项文件哈希与包SHA标识，不声称该Git提交包含新增代码。

这是各自披露配方下的单seed独立实现补充比较；不替代统一训练协议、多seed稳定性或单模块因果结论。正式指标待实际GPU完成与产物验收后填表。
