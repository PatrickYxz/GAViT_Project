# HGTNet AID Formal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 在已通过真实3060 batch16检查后，交付独立AID seed42正式calibration/refit/test运行包。

**Architecture:** 新增AID protocol/data/checkpoint/engine/entry siblings和builder，保留旧源。三组实现有互不重叠文件所有权，通过spec冻结接口连接；根代理负责回传归档、集成和最终交付。

**Tech Stack:** 既有标准库/PyTorch/PIL，服务器Python3.11/Torch2.2.2+cu121，不安装依赖。

**Spec:** docs/superpowers/specs/2026-11-05-hgtnet-aid-formal-design.md

## Global Constraints

- 30类、原4000/1000/5000pool/5000test；manifest/config/初值及recipe冻结，100轮calibration/earliest严格最佳/fresh同初态refit/test一次。
- batch/eval16、Adam1e-5/WD5e-5、FP32、无AMP/TF32/调度；train256→crop224/flip，evalcenter224。
- 新schema/22source/25ZIP成员；旧源/包/成绩表字节保持不变；不Git提交/暂存/推送、租卡、联网下载或远端实际训练。
- 恢复限有效完整epochsnapshot，test事务完整时复用、部分test明确拒绝重复；run/resume独占锁、强status。
- 新包必须复用SHA固定旧batch16包的只读status作为同物理GPU容量证明；pool5000字节、6GiB磁盘与6GiB可用hostRAM门禁。

## Review Focus

- 45类或real_nwpu常量漏改导致错误scope/shape/完成声明。
- test图片在preflight/training或未过reload前被读取，或resume默默再测test。
- 部分epoch已更新后失败被清零，或恢复从头训练而不披露回滚。
- 弱summary/marker完成判定、prediction/order/metrics/checkpoint/初态不一致。
- 新source闭包遗漏、旧runtime/manifest helper引入ML，或并发写/foreign/symlink覆盖。

### Task 1: 原AID全量数据与冻结协议

**Files:** Create aid_formal_protocol.py、aid_formal_data.py（external_baselines/hgtnet/）；configs/hgtnet_aid_formal_v1.json；tests/test_hgtnet_aid_formal_data.py、test_hgtnet_aid_formal_protocol.py。

**Interfaces:** spec里的prepare_inputs/_records/validate_selected/make_dataset及RECIPE/REAL_COUNTS/IDENTITIES/load_recipe/validate_binding；新binding/eval/schema名称必须原样使用。其他两任务依赖这些签名和返回结构，禁止修改旧模块。

- [x] 新行为测试先RED：新模块接口、原全10000元数据/分配/不等类与重复组、默认test拒绝/安全字节及完整配方绑定。
- [x] 最小实现全量选择及train/evaltransform，ML懒导入；新配方与旧NWPU数值同，仅schema/披露范围改AID。
- [x] GREEN并保存范围日志；CPU图像夹具不得冒充原图，root负责相关回归与交付。

### Task 2: 30类正式数值链、恢复与test事务

**Files:** Create aid_formal_checkpoints.py、aid_formal_engine.py；tests/test_hgtnet_aid_formal_checkpoints.py、test_hgtnet_aid_formal_engine.py。

**Interfaces:** run_formal同旧formal_engine签名与主要return键，接Task1协议。newmodel/snapshot/eval/result schema为hgtnet-aid-formal-对应-v1；test_transaction.json字段与spec一致。checkpoint_reload.json、best_val_predictions.json、last_val_predictions.json、history和三主checkpoint保持可交叉验收结构。

- [x] 先RED，覆盖30类/两个phase/earliest tie/fresh相同初态、严格重载，及中断epoch恢复、完整test复用/started无有效packet拒绝。
- [x] 实现新增独立模块，保留原优化/双slot算法并限制test一次；实际更新后失败要回调保存步数，检查五组精确键。
- [x] 小CPU5组模型≤3epochs GREEN，独立验证不间断vsresume、45类拒绝、非有限/篡改checkpoint和错误test packet拒绝；不做完整HGTNet训练。

### Task 3: 3060正式入口、强状态与封装

**Files:** Create aid_formal_entry.py、tools/package_hgtnet_aid_formal.py；tests/test_hgtnet_aid_formal_entry.py、test_hgtnet_aid_formal_delivery.py。

**Interfaces:** parser inspect/run/resume/status、inspect_resources(args)、run_formal_entry(args,resume=False)、status(output)，default output aid_seed42_01；22SOURCE_FILES/新delivery/env见spec。依赖Task1/2接口，GPU ML在run内懒导入。

- [x] 先RED测试stdlib边界/容量子进程证明/全pool只读/foreign与锁；完整artifact fixture与篡改拒绝不能靠skeletal summary。
- [x] 实现自动当前3060绑定/软件与资源门禁、同binding完整epochresume所有权、attempts和两相firststep/实际更新进度；strict completion和test gate，与Task2协议配合。
- [x] 新safe builder四显式动作、22source/25成员；临时bundle/importguard/篡改/额外成员/unsafe path/重复与源码保持测试GREEN。不得先构建最终命名包。

### Task 4: 回传归档、集成审查及用户交付

- [x] 保存本次batch16回传/规范化JSON与算术核对，更新独立工程记录，不写正式成绩表。
- [x] 运行相关新suite及受影响旧回归，独立规格+质量审查，修复本次范围内问题并重跑受影响检查。
- [x] 最终实际命名包inspect/status的stdlib importguard、来源与旧文件保护、Python3.11语法、说明链接、git diff --check；保存SHA/recipe/完整命令/seed/output。
- [x] 中文单行上传与正式run/后续status/resume说明；需要先确认全pool5000/磁盘/容量证明等真实inspect，未回传首批前不声称正式启动。实际正式GPU与test结果仍待用户执行。

本地交付步骤完成。实际服务器预检、首批启动与100轮/refit/test完整结果均待用户执行，不标为已完成实验。
