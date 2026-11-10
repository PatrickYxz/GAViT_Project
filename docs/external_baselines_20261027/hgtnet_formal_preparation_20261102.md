# HGTNet正式seed42实验准备

## 本次准备的依据

3090真实NWPU360/90工程检查已由服务器短摘要回传验收：exit0、180steps、checkpoint回读误差0，配置/manifest/候选权重身份相符。结果记录见[工程报告](../../results/hgtnet_nwpu_smoke_3090_20261102.md)。本文件记录正式协议准备及剩余验证步骤，本轮未执行新GPU任务。

## 论文问题与变量

问题：在现有固定NWPU20%训练池/80%测试样本身份下，公开披露假设的HGTNet独立完整方法，相比现有Swin/GAViT/DFAGCN完整方法，是否有竞争力且额外成本合理？

首个正式运行只用训练seed42，接受正向、持平或负向结果；不因test表现改shape/算子/seed。主要方法变量是HGTNet完整recipe，不能将结果差异归因单独图模块。不同骨干/候选预训练/优化/增强/预算及2080Ti/3080/3090硬件需披露；单seed不宣称稳定性或同卡效率优势。

## 原文已确认的正式设置

来源为既有[主文审计第5节](hgtnet_audit.md)，不是由当前smoke分数推选：

- 100 epochs、batch16、Adam（不是AdamW）、lr1e-5、weight_decay5e-5、cross entropy。
- RGB224，原文提到crop/flip；精确增强参数、归一化顺序、scheduler/warmup及Adam betas/epsilon等未确认。
- 作者partial ImageNet来源未核验。当前Swin-B候选/mapping、DA/graph/第三stage/fusion均沿用既有独立架构身份，不称作者精确实现。

smoke的batch2、Adam1e-4、无增强和单轮只作工程输入，不能直接扩大epochs当正式配方。

## 固定数据与评估路径

继续使用manifest ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e及原class/record身份：

1. calibration：train5040、val1260，最多100epochs；验证选择规则和并列规则须在读取test前固定。
2. fresh refit：从相同已披露初始/预训练状态重新建模，pool6300训练选定epoch预算，不续训smoke或calibration学得状态。
3. final test：25200张只在完整refit后评估，保存OA、macro-F1、45类accuracy/完整混淆/逐图预测、best/final及配置/源码/权重/runtime/设备身份。

batch16且不drop_last时，校准每epoch315个更新、验证79批；refit每epoch394个更新；test1575批。最坏100轮校准+100轮refit为70900个optimizer steps，另有验证/test/IO开销。这是迭代次数，不是已测耗时或费用。

正式训练前另冻结并披露：betas/epsilon、是否scheduler/warmup、crop/flip具体参数、eval resize/normalize、label smoothing/clipping、FP32/AMP/TF32、val主指标/并列处理、refit epoch转移、checkpoint保存/恢复与新路径。原文没给的字段使用明确独立工程选择，不能假称论文默认；不从最终test倒推选择。

## 最便宜的后续动作

现有history已回传并经8项算术检查：train40.840893秒、validation1.984714秒、8.814695图/秒、180steps，与360训练图一致。数据已归档，无需再次读取或重跑；原只读命令留作来源：

```sh
cat /home/featurize/work/hgtnet_nwpu_smoke_3090_20261102_fix1/runs/nwpu_smoke_seed42_01/history.json
```

58.2575秒为全smoke流程时间、1.43/1.57GiB为batch2整个allocator峰值，不能直接线性外推batch16或100轮预算。相同batch2/noaug/吞吐不变时5040/6300图训练段约571.77/714.72秒，仅是有条件算术，不是正式预算。

已制作[batch16容量包与单行命令](../../output/hgtnet_batch16_probe_3090_README_20261102.md)：保持74619858参数独立模型和固定候选初值，16张train图预加载，1次预热+3次计时更新，Adam1e-5/WD5e-5；betas0.9/0.999、eps1e-8为显式工程默认。无epoch、val/test、accuracy、checkpoint或smoke learned-state续训。复用已校验450图缓存中的16图，否则ZIP只恢复16图。它检查目标训练形状的容量/计算，不是单变量性能消融，不能据此比较原smoke分数。

本地隔离环境为精确Torch2.2.2/vision0.17.2/NumPy1.26.4/Pillow10.3.0，Python3.12.13 CPU（目标3.11.8 CUDA另核验）。batch16交付时333项回归与104项身份检查通过；当前正式代码最终386项结果另见下方。

已获得[3090真实batch16运行及status回传](../../results/hgtnet_batch16_probe_3090_20261102.md)：4次更新、五组451参数张量梯度及首步组更新通过，中位0.20135353秒/步、79.4622图/秒计算段。allocated4.40707GiB/reserved4.85742GiB，全probe wall10.059117秒、exit0、config/manifest/候选/UUID一致，新增26项摘要核验通过，无需重跑。

按该三步中位纯更新计算缩放：315次/校准训练轮约63.43秒，394次/refit训练轮约79.33秒；100+100最多70900次更新约3.97小时纯更新计算，不是含IO/增强/验证/保存/启动/test的正式ETA或实际费用，不能将其当作租卡时长上限。正式程序按已观察epoch_wall均值更新当前阶段剩余时间估算。

## 2026-11-03正式实现与交付

[运行包与单行命令](../../output/hgtnet_nwpu_formal_3090_README_20261103.md)、[冻结配方JSON](../../configs/hgtnet_nwpu_formal_v1.json)、[实施计划](../superpowers/plans/2026-11-02-hgtnet-formal-seed42.md)及[完整本地证据](../../results/evidence/hgtnet_formal_prepare_20261102/README.md)已保存。新增独立formal_data/protocol/checkpoints/engine/entry，旧core/config/pretrain/smoke和DF结果保持。

100轮校准、验证正确数严格更大才选最佳，平手保留更早轮；fresh相同初值SHA、pool6300按所选epoch重训，再独立新模型checkpoint数值回读后评估25200test。保存最佳/最终calibration、finalrefit、历史/OA/F1/perclass/CM/逐图预测与身份。best预测packet随epoch快照提交，避免中断后最佳模型与逐图文件错配。

未获原文确认而提前固定：Adam betas(.9,.999)/eps1e-8、constantLR/无scheduler/warmup、无label smoothing/实际clip；训练Resize256bilinear→RandomCrop224→HFlip.5→ImageNetnorm，evalResize256→CenterCrop224→同norm；workers0/drop_lastfalse，FP32/无AMP/TF32。公开工程选择，不以test调这些值，不称作者默认。

两个交替epoch恢复槽和两个best槽保证提交失败时有上一完整状态；每attempt日志/runtime/命令/失败摘要保留。快照核验source/recipe/数据/初值、phase/epoch/history/选轮及恢复后Adam实际policy/moment形状；残缺图片恢复到新attempt目录，receipt原子提交。每epoch42+epoch重启相同RNG/顺序，GPU位级恢复等价未测。可换同软件3090UUID恢复完整epoch；首轮未完成则没有完整epoch快照。

最终386项HGT/DF相关回归通过，0失败/错误/跳过；完整74619858参数模型+实际候选的CPU人工JPEG夹具完成2轮校准→选epoch1→fresh refit1轮→安全保存/独立回读→test2，4次更新、回读误差0，明确cpu_fixture，不是正式NWPU/GPU分数或GPU时间预算。独立审查与实际zipapp禁止ML导入的inspect/status/resume-help通过。

后续用户已回传真实3090正式完成摘要：100轮校准、best95/freshrefit95、25200test，OA94.11904762%/macro-F194.10529173%、exit0。完整身份和边界见[正式结果记录](../../results/hgtnet_nwpu_seed42_20261103.md)，逐图独立复算等待原test文件；不重复训练。本地准备本身未启动GPU/新租卡/外发或Git提交推送。
