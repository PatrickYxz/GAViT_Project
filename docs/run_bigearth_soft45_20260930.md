# BigEarthNet 软分配 seed45：恢复数据与正式实验准备

10-01 资源归档完成：BigEarth soft45总队列14小时44分18.733秒，训练阶段14小时37分58.069秒，测试阶段5分22.637秒；训练CUDA allocated峰值3.659762 GiB、reserved峰值4.029297 GiB。摘要与冻结身份及既有测试指标一致，固定8张val的3份快照归一化/48+32边检查通过。本轮训练、测试和资源/探针归档已收尾，无需补跑或再次回传同批日志。[完整记录](../results/bigearth_soft45_ablation.md)。这仍是单seed混合结果，多seed、新结构公平效率及外部方法比较另列未完项。

10-01 完成回传：BigEarth soft45已完成30轮及119825张test，总/训练/测试退出码均0。mAP70.836919%、macro-F1 64.864823%、micro-F1 76.383354%；较固定45为+0.578683/+0.569346/−0.319103 pp，仍低于Swin。无需重训/续训/解压；[三模型正式结果](../results/bigearth_soft45_ablation.md)已归档，下一步仅读取现有summary补齐总耗时/显存/探针。单seed结果不代表双数据集多seed要求已完成。

10-01 用户报告实例已归还：BigEarth soft45是否已在此前完成待work产物核验。先查总/阶段退出码、日志、已保存轮次和best/last/test产物；不默认从头重训，不先解压。原队列不再视为持续运行，新指标尚未确认。

2026-09-30 启动更新：用户已回传真实CUDA首轮1201/7434（16%）、数据与31,466,766参数符合配置。**当前队列已启动，无需再次执行下方启动命令**；继续看进度，首轮完整验证后回传EPOCH RESOURCE或SOFT EPOCH以更新预算。最终训练/测试和退出验收仍待完成。以下准备记录按时间保留。

2026-09-30 最新状态：用户已回传 `BIGEARTH_SOFT45_PREFLIGHT_PASS`，完整 split/RGB 文件存在性、原对照权重和环境核验通过。当前 RTX2080 Ti 的 UUID 与旧对照不同，型号/驱动/软件/精度相符；该差异保留在记录中。[回传转录](../results/evidence/bigearth_soft45_prepare_20260930/server_preflight_user_return.json)不是下载的原始文件。后台训练→测试命令已准备并做本地验证，**服务器尚未启动训练，也没有新的正式指标**。

## 这次研究什么、为什么只补一个模型

研究问题：在相同 BigEarthNet-19 划分、预训练、优化配方和 seed 下，局部原型软分配相对固定 4×4 分组是否改善主要指标 mAP？同时报告 macro-F1、micro-F1、19 类 AP 与计算开销，保留无提升或退步结果。

主要变量是已发布的分组模块：`attentive_spatial` → `soft_prototype`，包括该模块对应的区域汇总和软归属反馈；这不能进一步拆分为“只改变区域边界”的因果实验。区域数、图构造、GAT、优化和数据协议保持。它补导师要求的 BigEarthNet 三模型消融，不保证 soft 优于固定分组或 Swin，不据结果改温度、挑 seed 或追加训练。

先复用已有 seed45 Swin 与固定 GAViT，只从原 ImageNet 预训练权重训练一个 soft seed45。选择 seed45 是因为双方完整 30/30、权重/测试/退出码/资源记录齐全，不是按测试高低选对照。预检会重新计算服务器 checkpoint 哈希并核对原始记录。旧 checkpoint 无法回答新模块经过训练后的表现；已有双数据集真实 GPU smoke、NWPU 正式实验及可视化已完成，省去重复工程短测。

| 模型 | seed | 正式 test mAP % | macro-F1 % | micro-F1 % | 当前处理 |
|---|---:|---:|---:|---:|---|
| Swin-T | 45 | 71.676930 | 65.084212 | 76.732264 | 复用，服务器预检通过 |
| 固定 4×4 GAViT | 45 | 70.258236 | 64.295477 | 76.702457 | 复用，服务器预检通过 |
| soft_prototype GAViT | 45 | 70.836919 | 64.864823 | 76.383354 | 已完成30轮、完整test及资源/探针归档 |

原数值见[已验收 seed45 结果](../results/bigearth_seed45_comparison.md)。本批完成后也只是一个配对 seed 的消融；不能填成多 seed 稳定性结论，导师完整多 seed 要求仍未完成。

## 冻结身份与资源安排

- soft 代码：`/home/featurize/work/GAViT_SOFT_20260927`，干净 `ed480e60fea03ed786fdba70c144de5e815c34f5`。旧对照为 `deb2b457e86b6fa2feccb5a305e099b6d9ec6dfe`。已核对发布差异保留原数据、优化和固定模型路径；两个版本的 commit 必须如实分别记录，不能写成同一版本。新分组的训练、反馈和探针支持属于本次实现差异。
- 原对照根目录：`/home/featurize/work/bigearth_seed45_20260922_01`；计划新输出：`/home/featurize/work/bigearth_soft45_20260930_01`，目前未创建或启动。
- 固定 split：`/home/featurize/work/GAViT_Project/bigearth_files/splits`；train/val/test = **237871 / 122342 / 119825**。不重新生成划分，不将 val 合入 train。
- 预训练：`/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors`；SHA256 `f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。正式 soft 从它开始，不接着已有固定 GAViT 权重微调。
- seed45，30 轮，batch32，AdamW，lr0.0003，weight_decay0.0001，CosineAnnealingLR/T_max30，BCEWithLogitsLoss，F1阈值0.5，dropout0.1。保留 resize224、水平/垂直翻转和 ImageNet normalization，不套用 AID 的增强。
- 区域16、GAT hidden256/heads4/layers2、sparse_hybrid 48空间+32特征边、token_feedback。soft 使用已发布 `local_prototype_cosine_v1`：dim128、radius1、temperature0.1、iteration1，不另做参数搜索。
- FP32/no AMP，matmul TF32=false、cuDNN TF32=true，8线程；train/val/test workers4，train pin_memory=true、val/test=false。保留与原 BigEarth 对照相同精度，不套用 NWPU 的两项 TF32=false。
- 环境与旧对照核对 GPU 型号、Python/CUDA/cuDNN/包版本和精度；不同实例的 UUID/driver 字符串单独记录，不声称同一物理卡。出现漂移先看具体差异，不自动改环境或重跑旧对照。
- best 只按 val mAP 选取，保存完整30轮曲线、最佳和最终轮、last training state、分配探针及资源记录，随后评估 best 的完整 test，保存精确指标与逐样本预测。正式入口由外部脚本调度，源码和 hash 已打包在下面的单行命令中；不修改服务器训练仓库。

2080 Ti 旧固定 GAViT 实测训练阶段 **14.89 小时**（含验证/权重读写等）；暂为 soft 的训练与测试预留 **16–18 小时**，另加恢复原数据的时间。这是规划估算，未测过完整 soft 吞吐，不能按小样本 smoke 的秒数线性外推。首个完整 epoch 后再用实际耗时修正；费用按用户当前实例实际小时价计算，不能沿用未知或过期租价。

## 现在启动（只复制执行一次）

打开[正式后台启动命令](../results/evidence/bigearth_soft45_prepare_20260930/launch_command.txt)，复制其中完整一行到服务器执行。无需上传文件。它会独占创建 `/home/featurize/work/bigearth_soft45_20260930_01`，先在独立子进程重核预检与本次回传的一致性，再串行执行 soft45 训练30轮和最佳验证权重的119825张 test。不会重跑旧Swin/固定GAViT、smoke或其他seed。

看到 PID 后查看进度：

```bash
tail -f /home/featurize/work/bigearth_soft45_20260930_01/run.log
```

首次等待 `DEVICE`、`Train: 237,871 | Val: 122,342`、`Training GAVIT ... 30 epochs` 和 `Epoch [1/30] Train` 实际前进；用另一个终端检查GPU进程：

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

把设备行和首个实际 batch 的进度回传；第一轮结束后，`SOFT EPOCH`/`EPOCH RESOURCE` 能用于更新时间预算。`Ctrl+C` 只结束当前 `tail`；不要重启/归还实例或重复粘贴启动命令。若失败保留目录和日志，不删除后重跑；脚本不会自动重试或从头续跑付费训练。

结束后一次读取最终摘要：

```bash
cat /home/featurize/work/bigearth_soft45_20260930_01/run.exit /home/featurize/work/bigearth_soft45_20260930_01/summary.json
```

完成须有 `BIGEARTH SOFT45 COMPLETE`、总 `exit_code=0`，预检/训练/测试各阶段 `run.exit=0`、完整30轮与test指标。出现 PID 或进程退出但缺上述材料都不能当作完成。

## 产物与验收范围

- 根目录：完整 `run.log`、`run.pid`、`run.exit`、`summary.json`，独占启动标记、实际及原批准 `preflight.json`、外部脚本。运行中断会保留已产生的文件；SIGKILL/实例关闭无法保证写出退出标记，缺标记不能认定成功。
- `gavit_train/`：`run.log/run.exit/run.json/command.json`，`gavit.pth/gavit.meta.json/gavit.last.train_state.pth`、`history.json` 和 `gavit.assignment_probe/`（完整0–30轮诊断、initial/best/final快照）。原工具与新调度器 hash、正式入口 argv 均记录。
- `gavit_test/`：阶段日志/退出/配置，精确 `metrics.json`（mAP、两项F1、19类AP），`predictions.npz`（119825条原路径、19维标签与概率）。测试只加载 `gavit_train/gavit.pth`，测试前后与训练摘要核对实际权重SHA。
- 吞吐/资源：每轮精确验证分数与loss、检查点间墙钟；入口的 `SOFT EPOCH` 单列排除probe的epoch时间。首轮记录包括初始化，后续检查点间时间包括上轮probe。阶段时间/峰值CUDA显存/主进程RSS及按整个阶段时间计算的有效训练吞吐单独说明口径，不能当纯前向效率；RSS不含所有DataLoader进程。

本地12项针对性检查通过，覆盖错误退出、exit0但记录不全、防覆盖、环境/权重身份、精确历史记录且不消耗训练RNG、真实CPU软分组/图构造张量验收与破坏样例。打包命令的Python/Bash/zsh语法、载荷往返、独立后台实际启动的故障停止与重复运行不改原文件通过；独立审阅无新增问题。[验证摘要](../results/evidence/bigearth_soft45_prepare_20260930/launch_validation.json)、[测试日志](../results/evidence/bigearth_soft45_prepare_20260930/tests.log)。这些是本地调度/CPU证据，不是正式服务器CUDA训练结果。

## 恢复与独立预检（已完成，以下留档）

1. 按之前的 Featurize 数据集恢复方式添加并解压 **BigEarthNet-S2**，等待解压任务完成，目标为 `/home/featurize/data/BigEarthNet-S2`。持久化 `work` 中的代码、划分和权重沿用；160图的 smoke 备份不能替代全量数据。[已有运行手册](featurize_runbook.md)。
2. 恢复完成后，复制[预检命令文件](../results/evidence/bigearth_soft45_prepare_20260930/preflight_command.txt)的**完整一行**到服务器执行。无需上传文件、无需变量、无需嵌套 shell 或 heredoc。
3. 等待最终 `BIGEARTH_SOFT45_PREFLIGHT_PASS`，把最后的 JSON 发回；若报错，发最后的错误。运行中 `RGB CHECK` 是逐 split 文件检查进度，不是训练进度。

该命令只读：检查已有对照、已发布代码、预训练、冻结 CSV/19维二值标签、所有 split 的原 RGB 波段存在性、GPU 空闲与环境，并在指定 work 目录范围内查找既有 formal soft45 产物防止重复付费。会读取权重计算哈希，但不训练、不解压、不写新 checkpoint。RGB 检查不等于逐 TIF 全解码校验，也不是当前服务器 GPU 前向验收。

此前本地验证：Python/Bash/zsh 语法、压缩载荷往返一致；复用的原文件检查用例通过（正确文件可接受，缺波段/错 split 哈希/重复路径/合成图覆盖输入被拒绝）。**服务器现已回传PASS，无需再次手动执行这条独立预检**。证据：[命令验证](../results/evidence/bigearth_soft45_prepare_20260930/command_validation.json)，[可读源码](../results/evidence/bigearth_soft45_prepare_20260930/preflight.py)。

启动成功后仍需设备/GPU占用和首个真实训练 batch 的日志；只出现 PID 不标为训练启动成功。完成以30轮、最佳权重与配置、完整 test 指标/预测、阶段与总退出码均保存为准。
