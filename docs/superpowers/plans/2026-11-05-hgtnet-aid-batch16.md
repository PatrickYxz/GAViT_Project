# HGTNet AID batch16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 给实际3060准备原AID30类的16图四步容量包，作为正式训练前资源门禁。

**Architecture:** 冻结旧文件；新增30类四步engine、复用现有AID只读/设备函数的新入口、独立安全builder。两个实现任务的写入路径互不重叠，入口通过冻结函数签名连接引擎。

**Tech Stack:** Python标准库、现有Torch2.2.2/PIL环境；不安装依赖。

**Spec:** docs/superpowers/specs/2026-11-05-hgtnet-aid-batch16-design.md

## Global Constraints

- 30类/16unique/64processed/4updates/1warmup+3measured，schema hgtnet-aid-batch16-capacity-probe-v1，real_aid/RTX3060。
- 使用原manifest/初值/config SHA与实际GPU身份；无test/val/epoch/checkpoint/accuracy/formal训练。
- 18个源码配置和21个ZIP成员；旧AID/NWPU源及包和所有成绩表保持字节不变。
- 不提交/暂存/推送/租卡/远程训练，不安装依赖。当前用户codex分支与未提交研究产物直接保留；新文件实现隔离，不创建会丢失未提交依赖的新checkout。

## Review Focus

- 45类旧入口或engine混入30类包，必须明确拒绝并测logits维度。
- callback之后诊断失败仍要保留实际更新数，不能回写零。
- 仅summary.completed或伪造exit0不能使status完成，交叉验收完整产物。
- stdlib inspect/status不可触发Torch/CUDA/PIL导入，source闭包不可遗漏依赖或误用旧清单。
- 更换GPU、输入/源码篡改、旧输出/锁/symlink必须阻止运行，不能覆盖历史。

---

### Task 1: 30类数值引擎

**Files:** Create external_baselines/hgtnet/aid_batch_probe_engine.py; tests/test_hgtnet_aid_batch_probe_engine.py。

**Interfaces:** `run_batch_probe(model,images,labels,*,device,batch_size=16,warmup_steps=1,measured_steps=3,on_progress=None)`返回scope明确的dict，沿用旧batch_probe_engine的字段但schema为AID新schema、30类及五组精确检查。

- [x] 写CPU小模型行为测试，先断言新模块可导入并有上述可调用接口，观察RED；再覆盖30类4步/拒45或错误batch/缺组/非有限/更新后失败保留callback。
- [x] 最小实现，复用旧同步计时/四步Adam算法，30类替换与精确五组检查；不改旧引擎。
- [x] 运行新引擎测试保存完整日志，自审说明CPU夹具范围、peak reset边界，不执行完整HGTNet数值训练。

### Task 2: 入口与安全source-only交付

**Files:** Create external_baselines/hgtnet/aid_batch_probe.py; tools/package_hgtnet_aid_batch16_probe.py; tests/test_hgtnet_aid_batch_probe_entry.py; tests/test_hgtnet_aid_batch_probe_delivery.py。

**Interfaces:** `select_probe_records(inputs)`产生label0..15原train16；`inspect_resources(args)`/`run_probe(args)`/`status(output)`，parser inspect/run/status默认路径见spec。入口`SOURCE_FILES`为18源，builder清单相同；runner aid_batch_probe.py，delivery schema hgtnet-aid-batch16-delivery-v1、环境变量HGTNET_AID_BATCH16_DELIVERY_ENTRY。

- [x] 先写stdlib导入、选择、GPU预检/锁和status骨架拒绝测试，看RED；准备完整artifact fixture证明一致产物能完成及逐项篡改拒绝。
- [x] 实现标准库边界、自建来源与完整完成状态，run才调Task1；runtime复用aid_entry，参数/梯度真实模型门禁与18源绑定；留4步durable事件和失败已更新状态。
- [x] builder沿用已验证AID安全封装算法，仅新schema/runner/source清单/entry环境变量；写包安装、篡改、路径冲突、显式动作测试并看RED/GREEN。
- [x] 运行新入口/交付测试和受影响旧AID只读回归，完整日志与边界说明落盘；本地inspect如缺真实资产/GPU应诚实readyfalse。

### Task 3: 集成、独立审查与交付

**Files:** New output/hgtnet_aid_batch16_probe_3060_20261105.pyz及manifest/README；results/evidence/hgtnet_aid_batch16_prepare_20261105/；更新research_diary.md与AID推进文档。

- [x] 归档当前用户status并核对完成字段/输入ID/epoch/步数/UUID，填入真实OA/F1/reload/资源；原summary-pending记录保留快照。
- [x] 执行新相关测试、独立源码审查；修复范围内问题并只重跑受影响检查。
- [x] 构建最终实际包，stdlib importguard执行最终inspect/status、成员/source哈希与旧文件快照核对、git diff --check。
- [x] 用户只上传新小包到work后执行一行run && status；等待实际3060 batch16回传。正式AID trainer与性能尚未实现/执行，不把该阶段称正式完成。

本地步骤已完成，真实RTX3060 batch16四步仍待用户运行回传；正式AID trainer未交付/执行。
