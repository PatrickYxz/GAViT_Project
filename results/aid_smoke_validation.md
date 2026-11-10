# AID双模型GPU工程验收（E00）

2026-09-20更新。只记录1 epoch / train240 / val60的smoke；表中OA不用于论文正式性能比较。两模型E00工程验收均通过；正式calibrate/refit/test尚未启动。

| 模型 | GPU | seed | 状态 | train / val | 最佳 / 最终轮 | smoke val OA % | 训练轮耗时 s | 全程耗时 s | CUDA峰值allocated / reserved GiB | 重载logit最大差 |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| Swin | RTX 2080 Ti | 42 | complete / exit0 | 240 / 60 | 1 / 1 | 41.666666666666664 | 3.89 | 114.53324206199977 | 3.5582 / 3.8301 | 0 |
| GAViT | RTX 2080 Ti | 42 | complete / exit0 | 240 / 60 | 1 / 1 | 60.0 | 3.19 | 157.37608910299969 | 3.6549 / 3.8770 | 4.76837158203125e-07 |

Swin完整数据复核耗时101.5秒；训练loss3.098898。按日志已四舍五入训练时长估计约61.70张/秒，仅用于工程资源观察，不是推理benchmark，不外推正式训练总耗时。显存覆盖整个运行及checkpoint重建；主进程RSS峰值1504694272字节，参数量27543960，全部可训练。真实费用未提供。

共享配方：AdamW lr0.0003 / wd0.0001，CosineAnnealingLR T_max30，CE，batch32，workers2，float32，TF32关闭，pin_memory=False；224×224双线性输入、H/V翻转各0.5、ColorJitter(0.2/0.2/0.2/0)、ImageNet归一化。完整环境/配方见[run.json](evidence/aid_swin_smoke_20260920_01/run.json)。

身份记录：

- Git deb2b45 / dirty=false；本地逐项复核17个源码哈希均一致。
- manifest SHA256：`30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22`。
- 预训练文件SHA256：`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。
- 最佳checkpoint：`/home/featurize/work/aid_swin_smoke_20260920_01/best.pth`，SHA256 `73efc23a14992873b2f3ab2cd4dac608d68708be1fe7fad5962a1966657cfc0e`。
- 服务器已回传TRAIN FIRST BATCH OK、EPOCH 1/1、SCENE SMOKE COMPLETE、status complete和exit0；reload_max_logit_diff=0。Swin的graph_capture为空符合无图模块实现。
- [原始回传](evidence/aid_swin_smoke_20260920_01/server_output.txt)保留；本地JSON由用户输出提取，未直接下载服务器checkpoint/metadata或完整指标文件。因此不补填尚未回传的macro-F1、逐类指标或混淆矩阵。

GAViT现已通过：实际前向actual_forward_checked=true，48空间边+32特征边=80；保存重载最大差4.76837158203125e-07≤1e-5。固定attentive_spatial K16 / sparse_hybrid / token_feedback，GAT hidden256、4heads、2layers、dropout0.1。训练loss3.010718、数据扫描140.5秒、训练3.19秒（约75.24张/秒，仅小样本训练循环），主进程RSS1845030912字节、参数31375385。单次短测计时不说明GAViT更快，OA也不足以判断模型优劣。

GAViT [run.json](evidence/aid_gavit_smoke_20260920_01/run.json)和[原始回传](evidence/aid_gavit_smoke_20260920_01/server_output.txt)已归档。checkpoint为`/home/featurize/work/aid_gavit_smoke_20260920_01/best.pth`，SHA256 `c3ca53bb504196cff398fa76854e33d3592a48a7733bc5385cf72040fba82dae`。本地逐项核对两模型manifest、预训练、seed、全部training、hardware、code及counts一致；GAViT的17个源码哈希也与发布版本一致。

E00完成，下一步执行[E01 seed42正式实验](../docs/run_aid_formal_seed42.md)，先Swin校准，首轮核查全量耗时与资源；尚未实际启动。
