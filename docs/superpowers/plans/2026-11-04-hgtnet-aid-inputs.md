# HGTNet AID Inputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 保存46/47可支持的补录信息，并交付独立AID只读检查包，以继续既有外部比较流程。

**Architecture:** 历史补录只引用现有主指标CSV并保留未知字段为空；AID检查是独立标准库工具，不修改NWPU运行链。两个交付使用独立路径并行，之后统一核对身份和报告。

**Tech Stack:** Python标准库、unittest、zipfile；不安装依赖，不训练。

**Spec:** `docs/external_baselines_20261027/hgtnet_aid_preparation_20261104.md`

## Global Constraints

- 保留现有分数、原CSV及既有验证证据；缺失F1/日志/UUID/checkpoint不补造。
- 固定AID manifest SHA30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22；30类、4000/1000/5000，pool5000。
- 仅检查smoke选定240train/60val字节，不读取test图；不生成新划分。
- 工具只向stdout输出JSON；无训练/安装/写服务器文件；不导入ML。
- 本地新文件独立命名，保留用户dirty工作区，不提交或推送Git。

## Review Focus

- 原manifest缺失/哈希错误必须有明确字段，不能输出资产通过。
- 路径穿越、symlink、重复路径、标签错配、像素组跨split必须拒绝。
- AID每类数量不等，仍按原manifest选择每类8/2，不假设NWPU支持数。
- 本机没有GPU/数据时仍输出有效JSON，不能伪造目标GPU通过。
- 补录只由原主指标支持，四seedF1统计不能引用不同范围F1冒充。

### Task 1: AID只读资产工具

**Files:** Create `tools/inspect_hgtnet_aid_inputs.py`, `tests/test_hgtnet_aid_inputs.py`。

**Interfaces:** CLI接受`inspect --manifest PATH --data-root PATH --weights PATH`，默认采用spec旧服务器路径；stdout为JSON、无文件输出。内部manifest检查允许测试注入expected SHA；CLI固定真实预期SHA，不能随意替换协议。

- [x] 用真实形状的合成manifest夹具验证通过、原件缺失/错SHA、坏路径/标签/像素组、8/2选择和test字节不读取。
- [x] 实现固定taxonomy/协议/counts与流式文件SHA，盘点版本/GPU/磁盘。
- [x] 运行相关unittest和本机无真实资产CLI，确认没有ML导入、文件修改或虚假通过。

### Task 2: 46/47补录

**Files:** Create `results/seed46_seed47_reconstruction_20261104.json/.md`及独立evidence；给旧审计MD加当前状态索引。

**Interfaces:** 读取`results/multiseed_test_results.csv`原有16个主指标；补录JSON标注user_reported与原件lost，secondary metrics null。不得创建run.exit、run.json或仿真逐图预测冒充实测。

- [x] 独立重算八个有结果行的均值、样本SD及对应差值，保留来源打印精度。
- [x] 保存主指标与空字段的补录说明，更新旧审计的当前状态入口，原件和指标不变。
- [x] 验证原CSV SHA保持639a9b45b07625d63c1254a32969637c984a996997248622a647f261d43c0988、所有未知字段为空、没有新增伪造平台工件。

### Task 3: 交付和文档

**Files:** 新`output/hgtnet_aid_inputs_20261104.pyz`及README/evidence；更新research_diary、导师报告当前状态。

- [x] 把仅标准库工具打包，重开检查成员及SHA，运行实际pyz的inspect，确认只读。
- [x] 独立审查工具、补录和当前报告一致性；只修具体问题，不重复已通过工程suite。
- [x] 记录本地验证与尚未进行的目标机检查，提供单行资产检查命令，git diff --check。

执行记录：2026-11-04，本阶段本地交付完成；目标实例资产检查待用户运行只读inspect。源码与包身份、测试及审查见results/evidence/hgtnet_aid_inputs_prepare_20261104/；历史补录和报告证据见results/evidence/seed46_seed47_reconstruction_20261104/。当前不包含AID训练入口或GPU验证。
