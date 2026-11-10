# 3080 / 3060 资源短测使用说明

本地准备日期：2026-09-07。对应任务：`docs/thesis_schedule.md` 的 W1-03—W1-06。

**2026-09-08：用户选择在2080 Ti上直接训练，暂停本说明的双卡资源短测流程。本文及旧工具包仅支持3060/3080，不用于2080 Ti启动；当前安排见 `docs/thesis_schedule.md` 最新执行调整。**

## 当前验证边界

- 本地 37 项测试通过（15 项新工具测试、19 项实验身份测试、3 项静态契约测试），覆盖抽样、配置生成、防覆盖、计时、Linux 进程内存解析和失败报告；未在 Mac 安装 PyTorch。这不是 CUDA 运行验证。
- 3080/3060 的真实 CUDA forward/backward、内存峰值、速度及费用均待服务器实测。
- 工具不更改现有模型或训练脚本。服务器训练目录必须是干净的 `bb9f1cff463c996bc5f72695af538898c91490eb`；将本工具放在该目录外。
- 不自动租用实例，不访问已登录浏览器，不自动开始正式训练。每次输出目录必须是新的。

## 1. 上传并解包

将交付的 `gavit-resource-tools-20260907.tar.gz` 上传到服务器 `/home/featurize/work/`。
在服务器终端执行：

```bash
mkdir /home/featurize/work/gavit-resource-tools-20260907 && \
tar -xzf /home/featurize/work/gavit-resource-tools-20260907.tar.gz -C /home/featurize/work/gavit-resource-tools-20260907
```

如果目录已存在，先检查是否已经解包；不要覆盖另一次工具版本。包中只有两个 Python 脚本和本说明，无数据/权重/凭据。

训练目录、原 split、预训练权重和完整解压的数据沿用现有 Featurize runbook。若训练目录有未知改动，先检查，不能通过 reset、清理或自动 stash 绕过检查。

## 2. 先只查看计划

在目标实例上生成计划，路径及 Python 解释器由该实例确定：

```bash
python /home/featurize/work/gavit-resource-tools-20260907/resource_probe.py plan --gpu 3080 --output-dir /home/featurize/work/gavit_resource_runs/rtx3080_smoke_20260907
```

它只输出 JSON，不创建目录、不导入 CUDA、不运行训练。`plan` 中的 smoke 子命令用于说明配置，实际短测应使用下面的 `smoke` 总入口，以完成前置检查。

## 3. 执行 3080 10GB 短测

确认是在 RTX 3080 10GB 实例上且完整数据解压结束后：

```bash
python /home/featurize/work/gavit-resource-tools-20260907/resource_probe.py smoke --gpu 3080 --hourly-rate 0.93 --output-dir /home/featurize/work/gavit_resource_runs/rtx3080_smoke_20260907
```

该命令依次完成：

1. 确认单 GPU 型号/容量、无其他计算进程、干净训练 commit、依赖版本、原 CSV 和预训练权重 SHA256。
2. 检查 train/val/test 行数为 237871/122342/119825，每个 patch 的 B02/B03/B04 均存在，无重复或交叉样本。
3. 运行原训练仓库完整 unittest；任何失败或跳过都不继续。
4. 使用 seed42 确定性抽取 train4096/val1024，独立输出；不复制或评估 test。
5. 依次对 Swin/GAViT 运行 batch32、seed42、一轮 smoke，保存真实训练和资源记录。
6. 检查 best、metadata、last-state 及 epoch/seed/hash，写入 `smoke_report.json` 后退出。

第一次全数据检查可能需要一些时间；在 `data_check.json` 写出以前不会开始训练。短测在前台运行，建议留在 tmux 等持久终端内，防止断连；不要因第一步耗时就重复启动。

## 4. 执行 3060 12GB 短测

在 RTX 3060 12GB 实例上准备好同一份原数据后，使用相同工具和配置：

```bash
python /home/featurize/work/gavit-resource-tools-20260907/resource_probe.py smoke --gpu 3060 --hourly-rate 0.69 --output-dir /home/featurize/work/gavit_resource_runs/rtx3060_smoke_20260907
```

价格是 2026-09-07 查得的参考单价，租用时若不同，传入实际 `--hourly-rate`。一次失败不覆盖旧目录，检查失败原因后使用新的输出名。

## 5. 查看哪些输出

```text
rtx3080_smoke_20260907/
  identity.json              训练 commit、软件/GPU/数据/预训练身份
  data_check.json            三个 split 的完整性检查
  tests.log                  原训练仓库完整测试输出
  splits/manifest.json       确定性抽样及 SHA256
  splits/train.csv
  splits/val.csv
  plan.json                  实际命令
  swin.log / gavit.log       各子进程完整输出
  swin/                     GAViT 对应 gavit/
    training.log             即使单独运行包装器也保存训练日志
    resource_samples.jsonl  每约1秒采样（短暂峰值可能漏采）
    profile.json             资源汇总及执行状态
    swin.pth                 独立 smoke best
    swin.meta.json
    swin.last.train_state.pth
  smoke_report.json          两模型整体通过/失败，不自动启动正式任务
```

优先看两模型 `profile.json` 的：

- `peak_cuda_allocated_gib` / `peak_cuda_reserved_gib`：PyTorch 峰值。
- `sampled_peaks.gpu_process_mib`：nvidia-smi 采样峰值；MiB 与 GiB 不混用。
- `rss_sum_kib`：主进程及 DataLoader 子进程 RSS 之和，可能重复计算共享页；优先同时看 `pss_sum_kib`。取不到的字段为 null，不当作零。
- `minimum_host_available_kib`：主机剩余可用内存的采样最小值。
- `batch_timings`：每轮 train/val 分别剔除前10批后的平均批次耗时，包含数据读取与消费者计算；计时会执行 CUDA 同步。
- `training_wall_seconds`：实际训练入口 wall time；`wall_seconds` 还包含包装器检查及产物验证。
- `estimated_30_epoch_loop_hours` / `estimated_loop_cost`：按正式 train/val 行数外推的循环开销，**不包含**启动、完整验证指标计算、存盘、解压和空闲；不是账单或正式结果。

小 split 的 RAM 不能当作全 split RAM 峰值。正式运行继续记录资源，并检查首轮实际耗时。不要因小样本指标好坏筛选机型或 seed。

## 6. 生成 seed44/45 正式命令（不自动运行）

只有在两模型 smoke 报告通过、成本可接受且确定机型之后，生成相应计划：

```bash
python /home/featurize/work/gavit-resource-tools-20260907/resource_probe.py plan --gpu 3080 --stage formal --seed 44 --smoke-dir /home/featurize/work/gavit_resource_runs/rtx3080_smoke_20260907 --output-dir /home/featurize/work/gavit_resource_runs/seed44_rtx3080_20260907
```

```bash
python /home/featurize/work/gavit-resource-tools-20260907/resource_probe.py plan --gpu 3060 --stage formal --seed 45 --smoke-dir /home/featurize/work/gavit_resource_runs/rtx3060_smoke_20260907 --output-dir /home/featurize/work/gavit_resource_runs/seed45_rtx3060_20260907
```

JSON 中每个模型的 `command` 才是训练命令；默认30轮、batch32、固定原配置。运行时再次检查当前 GPU/软件/数据/预训练身份与通过的 smoke 一致，并重新扫描全数据。只生成 `plan` 不消耗训练资源。没有 `--smoke-dir` 也可查看配置，但执行正式命令会拒绝启动。

逐个运行、检查进程/GPU/首批日志；每个 seed 的两模型在同一硬件配对。训练成功后再执行该模型的 `evaluation_command`，它测试 best checkpoint 并以防覆盖方式写入 `test.log`。评估命令也不会自动执行。

若改为某一种 GPU 承担两个 seed，可给 `--gpu` 指定同一机型，但必须匹配该机型的 smoke 报告。不要在一次正式训练途中换卡。

## 7. 记录与后续

GPU短测通过后，把实际资源与账单记录到研究日记；正式测试完成后再写性能结果表。W1-03 本地工具完成与 W1-04/W1-05 GPU实测通过是不同状态。

seed43 的 epoch21 中断边界仍需独立处理；本工具不自动恢复它。新方法和 learnable soft assignment 的资源需求也不能直接沿用当前模型的短测结论。
