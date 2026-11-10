# HGTNet AID Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 交付当前RTX3060可运行的HGTNet×AID独立240/60一轮smoke包。

**Architecture:** 新AID数据与engine身份隔离；复用已核对的30类模型能力和通用诊断。新入口/交付以固定接口对接，独立目录和源码哈希绑定保留旧NWPU产物。

**Tech Stack:** Python标准库、现有Torch2.2.2/vision0.17.2/PIL环境、unittest、zipapp。

**Spec:** `docs/superpowers/specs/2026-11-04-hgtnet-aid-smoke-design.md`

## Global Constraints

- seed42/240train/60val/test0、30类/1epoch/batch2/Adam1e-4/CE、RGB224无增强，FP32/noAMP/双TF32off。
- 固定原manifest30ac5760...aa22和初始候选68c6b09e...1a893；不使用NWPU已训权重、不重新划分。
- 目标实际RTX3060 12288MiB/当前UUID；CUDA总内存允许≤1GiB口径差额，来源/软件/设备不符则停止。
- 复用的旧文件及包不改；独立输出拒绝覆盖；保留失败后的实际更新状态；无Git提交/推送/安装或正式训练。

## Review Focus

- AID每类数量不同与Industrial同像素原组保留，绝不读test图片。
- 真实输入与CPU fixture不可混淆，45类/NWPU身份不能进入新checkpoint。
- CUDA选择先于ML导入，NVML/CUDA容量差异不误报。
- checkpoint键/shape/有限值/身份及独立重载检查不遗漏。
- 真实包成员/hash/源修改/输出占用/锁/中途失败后训练状态与日志一致。

### Task 1: AID数据和单轮engine

**Files:** new `aid_data.py/aid_engine.py`及对应`tests/test_hgtnet_aid_data.py/test_hgtnet_aid_engine.py`。

**Interfaces:** 完整函数/数据字典见spec；与入口按该接口并行实现，均只写各自新文件。

- [x] RED：错SHA/坏标签/路径/split/像素组、test拒绝、30类binding、失败保留实际步数、非有限梯度与checkpoint篡改。
- [x] 实现并GREEN：数据只解码同一份安全已哈希字节；完整1epoch及严格独立重载。

### Task 2: 3060入口和独立交付

**Files:** new `aid_entry.py`、`tools/package_hgtnet_aid_smoke.py`、30类config及对应entry/delivery测试。

- [x] RED→GREEN：stdlib inspect/status、GPU/runtime门禁、固定实际run参数、抢占/覆写保护/失败留痕。
- [x] 独立hash验证安装真实文件，新增AID及复用来源全部绑定，实际pyz inspect/status在禁ML模式通过。

### Task 3: 集成和用户交付

- [x] 归档原始恢复/300图通过输出，更新最新状态而保留旧缺图记录。
- [x] 测试与接口集成，运行必要本地30类真实Tensor夹具并明确范围；保存独立审查及工件保护SHA，不重复无关旧suite。
- [x] 生成新包/README/证据、检查实际包和独立目录拒绝覆盖、更新日记及git diff --check，给用户inspect→实际run的单行命令。

完成记录：2026-11-04本地准备/独立审查完成，P2完成状态问题已修复。最终包a70b6f...3961；目标3060真实smoke待用户执行。证据按data_engine/entry_delivery/integration/review分开保存。
