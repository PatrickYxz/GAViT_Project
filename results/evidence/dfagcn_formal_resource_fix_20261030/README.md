# 真实资源门禁修复验收

日期2026-10-30，状态FORMAL_RESOURCE_BLOCKERS_FIXED_OFFLINE_NEW_SERVER_RUN_PENDING。

- [实际用户资源回传](reported_resources.json)：省后缀metadata、新10GiB3080 UUID、work17029922816B、实例本地349GiB，数据/权重/native binary已通过。
- [修复验收](validation.json)、[148项最终测试日志](package_tests.log)。测试本地沿用OMP/MKL线程1和KMP_INIT_AT_FORK=FALSE，不修改正式CUDA配方。
- [独立8项资源TDD/禁止ML导入证据](resource_tests/validation.json)。有效RED重现版本/磁盘/旧UUID误拦；GREEN覆盖分卷、同卷合并、UUID显式选择、真实runtime仍拒错误CUDA等。
- [独立审查](review/independent_review.md)、[机器证据](review/independent_review.json)、[与旧pyz实际源码差异](review/runner_vs_original_pyz.diff)。无待修项，无核心算法/配方改动。
- [最终源码](source_final/)、[21项原文件保护](protected_inputs.json)、[实际本地pyz检查](bundle_local_inspect.json)。完整权重SHA通过，本机缺服务器资源正确拒绝，未创建runs。

修复不把40GiB直接降门槛：cache/images迁至实例scratch，scratch仍40GiB；work独立核算14.374759GiB全SVM上界和1GiB产物余量，同卷预算相加。新output/scratch一律拒绝覆盖。work保留源/配置/history/checkpoint/模型/预测/指标及cache index，TMP中的可再生数组随实例释放丢失。

交付[fix1操作说明](../../../output/dfagcn_nwpu_formal_3080_fix1_README_20261030.md)。未操作远端或启动正式训练；新卡实际CUDA与最终分数待用户执行回传。
