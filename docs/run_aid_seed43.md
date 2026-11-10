# AID seed43 配对重复实验

09-21完成更新：Swin `_02`及其余五阶段已回传complete/exit0，校准best25/29、refit25/29；test OA96.86%/97.16%，差+0.30 pp。详见[结果核验](../results/aid_seed43_comparison.md)。下方命令作为历史执行记录保留，不再次运行；下一步[seed44](run_aid_seed44_batch.md)。

09-21连续执行更新：用户明确要求一次拿到后续命令，完成后统一回传。见[seed43顺序队列](run_aid_seed43_batch.md)：等待下面Swin43 `_02`成功后，自动完成GAViT校准、双方refit及双方test，任一阶段失败停止。替代此前每步人工回传再继续的安排；当前仍未取得服务器启动或完成证据。

09-21恢复更新：首次`aid_swin_calibrate43_20260921_01`在预检读取airport_1.jpg时缺文件并exit1，未进入训练；保留该失败目录。用户已执行原ZIP的unzip -n -q并返回提示符，无错误输出；完整恢复仍由重试入口的10000张哈希检查确认。下方命令使用新目录`_02`，其余身份不变，尚待用户执行。

2026-09-21。沿用已定42/43/44三个训练seed路线。本轮目的：测量同一固定划分下Swin与固定4×4 GAViT的训练随机性，验证seed42正向差异是否可重复。seed42的30类与配对分析已完成，OA+0.60 pp来自93次纠正/63次新增错误，净收益集中于Commercial/Park；不据这些test类别变化调整模型或配方。

## 冻结条件与成本依据

- 主要变化只有训练seed42→43，数据划分不变：`aid_protocol_20260920_03.json`，SHA256 `30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`。外层split_seed42/内部val_seed4242保留；不重新prepare，不把三训练seed描述成三个随机划分。
- Git deb2b45且干净；ImageNet预训练SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。相同Swin-T与既定K16 attentive_spatial/sparse_hybrid/token_feedback GAViT，dropout0.1、GAT256/4heads/2layers、48+32图边。架构已冻结，不新增候选。
- 两模型均calibrate30轮、4000train/1000val，按各自本次val OA最高选epoch，同分最早；随后各自在5000训练池从相同ImageNet重新初始化并refit，Cosine T_max仍30，使用last.pth测试5000张。不能把seed42选出的21/20轮直接指定给seed43，也不能续训seed42权重。
- AdamW lr0.0003/wd0.0001、CE、batch32/workers2、float32、TF32关闭、pin_memory=false，共享RGB224增强和评估变换；除seed外配置保持一致。实际GPU/软件由首batch及报告核对，目标仍RTX2080Ti。
- 更便宜的检查已完成：相同代码/数据的两模型smoke、完整校准/refit/test及已有预测分析均通过；这些不能测量独立训练seed方差，所以安排预先选定的重复运行，不重复工程smoke或seed42。
- 输出进入固定split的多seedOA/macro-F1均值±样本标准差、配对差值及资源记录。42/43/44全部保留，无论指标正负；不依据test换seed或调参。本次按用户要求提供seed43剩余五步的连续队列，不包含seed44。
- seed42校准全程Swin约21分21秒、GAViT约22分47秒；refit约14分53秒/13分40秒，合计训练阶段约72.68分钟，另有测试/数据恢复时间。seed43重训轮数未选定，不能承诺同耗时；首轮记录训练/验证耗时后估余下30轮校准，refit预算确定后再补成本。租价未提供，不推算费用。

## 当前：Swin43校准

使用新的独立输出目录；入口拒绝覆盖已有目录，并校验全部数据与预训练文件。实例数据若仍在，无需解压；若入口报缺失，先处理数据恢复，不重建划分。

```bash
nohup python -u /home/featurize/work/GAViT_AID_20260920/run_scene.py train --model swin --phase calibrate --seed 43 --manifest /home/featurize/work/aid_protocol_20260920_03.json --data-root /home/featurize/data/AID/AID --output /home/featurize/work/aid_swin_calibrate43_20260921_02 --pretrained-path /home/featurize/work/GAViT_Project/bigearth_files/model.safetensors --device cuda --batch-size 32 --workers 2 >> /home/featurize/work/aid_swin_calibrate43_20260921_02.launch.log 2>&1 < /dev/null &
```

```bash
tail -f /home/featurize/work/aid_swin_calibrate43_20260921_02.launch.log
```

预检后要求实际GPU正常、MODEL swin / PHASE calibrate / train=4000 val=1000、TRAIN FIRST BATCH OK及EPOCH1/30；PID不能单独证明启动成功。需要核查时另行运行pgrep -af run_scene.py及nvidia-smi；Ctrl+C退出tail不会停止后台训练。

出现EPOCH30/30与SCENE CALIBRATE COMPLETE后读取：

```bash
cat /home/featurize/work/aid_swin_calibrate43_20260921_02/run.exit /home/featurize/work/aid_swin_calibrate43_20260921_02/run.json
```

核对exit0、status complete、seed43、final_epoch30、best_epoch及val OA、配对身份、权重保存重载、资源和日志。后续队列依次执行GAViT43校准、两模型各自refit/test，依据退出码和入口的来源校验推进，最后统一回传全部报告；不根据验证片段或test临时改变正式流程。当前尚未代为启动服务器任务。
