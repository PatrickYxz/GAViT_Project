# AID统一入口：本地工程验证

日期：2026-09-20。状态：**本地代码与CPU合成数据流程通过；真实AID、ImageNet文件、CUDA和双worker服务器短测待验收。** 此报告不提供AID分类性能结论。

## 实现范围

- `run_scene.py`统一prepare/train/evaluate，复用既有Swin backbone与固定4×4 sparse-hybrid GAViT；旧NWPU/BigEarthNet入口未改。
- 按类别及路径排序的清单，外层/内部验证seed独立记录；文件/像素哈希、类名/数量/尺寸/RGB、重复与跨split身份检查；不复制原图。
- 共享增强、AdamW/weight decay、batch、Cosine周期与CE；明确smoke/calibrate/refit来源；不同阶段分别保存checkpoint、metadata、样本清单、日志、退出码和指标。
- 评估检查源运行成功完成、checkpoint/manifest/类别/模型及当前源代码哈希；refit绑定原校准身份，从同一预训练源重新初始化。不同源码不能悄悄重算相同身份的测试结果。

## 已完成的检查

相关回归命令（代码工作树根目录）：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /private/tmp/gavit-aid-qa-20260919/bin/python -m pytest tests/test_scene_data_metrics.py tests/test_scene_runtime.py tests/test_experiment_identity.py tests/test_checkpoint_roundtrip.py tests/test_swin_features.py tests/test_graph_construction.py tests/test_sparse_hybrid_model.py tests/test_sparse_hybrid_static_contract.py tests/test_training_state_helpers.py -q --tb=short
```

最终exit0：**74 tests passed，31 subtests passed**，其中18项为本次新增测试。39条警告来自PyTorch JIT对Python3.12旧AST接口的弃用提示，没有测试失败。[完整测试输出](evidence/aid_entry_local_20260920_cli_cpu/regression_final.log)。

新测试先在缺失实现时失败，之后实现通过。审查补充的源完成状态/身份关联、当前评估代码漂移检查，也观察到修复前失败与修复后通过。用本地生成的权重文件逐张量验证Swin和GAViT加载了完全相同的backbone；该文件不是ImageNet训练权重。

另外，从仓库之外的临时目录，用真实CLI运行30类合成图片（每类10张、全部标`synthetic`），workers=0、CPU、随机初始化、batch2。两个模型各取每类1张train和1张val，完成1轮训练、验证、保存/重建和独立evaluate；prepare、Swin train/evaluate、GAViT train/evaluate共5个命令均exit0。[最终CLI证据](evidence/aid_entry_local_20260920_cli_final/summary.json)。

两模型保存前与重建后的最大logit差均为0；GAViT真实前向核验80条边（48空间+32特征）。从仓库外启动，metadata仍正确记录源代码commit `8f2ed07`、dirty=true和当前文件哈希；dirty意味着本次代码尚未提交，不能拿该短commit单独重建这些改动。参数量只属此次30类实现，不自动填入正式论文效率表。

临时图片和随机checkpoint在QA结束后自动清理；保留命令、日志、配置、哈希、逐轮及验证报告。归档中合成图片产生的低分类分数没有研究意义，不能进入正式comparison表。相关QA辅助脚本为`tmp/aid_cli_qa_20260920.py`，不是服务器依赖。

## 未通过或未执行的部分

- Mac本地workers=2：沙箱内出现`OMP Error #179: Can't open SHM`，训练前退出。[日志](evidence/aid_entry_local_20260920/swin.log)。
- 用户允许在沙箱外重试后，Swin命令180秒超时；该次根因未确定，原QA辅助脚本未保留进程内stdout。[失败记录](evidence/aid_entry_local_20260920_multiprocess/timeout.json)。后续辅助脚本已改为实时写日志，成功的workers=0运行单独存放，没有覆盖失败证据。
- 不能把workers=0的成功当作workers=2成功；Linux/CUDA双worker在服务器短测中必须核验。
- 真实AID尚未准备，正式calibrate30轮/refit没有执行；服务器ImageNet文件、CUDA显存/吞吐与正式训练效果均未验证。
- 后续同步：代码已在独立分支`codex/aid-entry-20260920`提交并推送（`2ba112a`），完整分支89项测试/34个subtest通过，见[发布验证](aid_entry_release_validation.md)。尚未确认服务器部署；用户旧实验结果及未提交改动保留。代码审查识别的4项完成状态/身份/版本问题已修复并回归。

下一步依照[服务器指南](../docs/run_aid.md)准备AID与同步代码，再逐个模型执行smoke；E00整体仍待GPU验收。
