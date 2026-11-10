# DFAGCN独立流程运行说明

2026-10-29：完整独立入口已实现，86项包内测试通过。[最终20图工程检查](../../results/evidence/dfagcn_pipeline_20261029/smoke_run_02/smoke.json)实际运行图片→VGG CPU FC6/卷积→缓存→GCN训练→验证选择→重新拟合→LIBLINEAR→测试。保存重载误差0，总耗时约6.04秒；输入为合成图片，权重随机，不能解释为NWPU表现或耗时。

[默认配方](../../configs/dfagcn_independent_v1.json)和[设计说明](pipeline_design_20261029.md)记录原文依据与独立选择；[NWPU小样本配方](../../configs/dfagcn_nwpu_smoke_v1.json)保留7×7、两层256维GCN、MLP1024→512，限定一轮、batch8、C1。

## 本地可直接复跑

从项目根目录执行，每次换新输出名。环境在临时目录，系统清理后按[依赖版本](../../results/evidence/dfagcn_primitives_20261028/requirements_cpu_test.txt)重建。以下manual输出名尚未使用：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m external_baselines.dfagcn --help
```

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m external_baselines.dfagcn smoke --output /Users/patrick/Projects/GAViT_Project/results/evidence/dfagcn_pipeline_20261029/smoke_manual_01 --train-binary /Users/patrick/Projects/GAViT_Project/results/evidence/dfagcn_pipeline_20261029/liblinear/bin/train --predict-binary /Users/patrick/Projects/GAViT_Project/results/evidence/dfagcn_pipeline_20261029/liblinear/bin/predict
```

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m unittest discover -s tests -p 'test_dfagcn_*.py' -v
```

Mac沙盒中libomp曾偶发共享内存启动错误，日志保留；验收采用上述单线程直接命令。全旧仓库仍缺pytest/timm/torch_geometric，不能称全部回归通过。

## 真实NWPU小样本

需要实际数据根目录、既有冻结manifest、完整SHA校验的VGG权重及目标平台LIBLINEAR。代码不自动下载这些文件。附带二进制仅适用于本地macOS arm64；其他平台用[同一官方v250源码](../../results/evidence/dfagcn_pipeline_20261029/liblinear/provenance.json)重新编译，再传入train/predict路径。

1. `cache`分别提取train、val、pool，统一`--mode engineering --recipe configs/dfagcn_nwpu_smoke_v1.json --train-per-class 8 --val-per-class 2`。pool是这两个子集的并集，最多360/90/450张。传入实际`--manifest`、`--data-root`、`--weights`、`--weights-sha256`、`--weights-kind torchvision_imagenet1k_v1`和独立`--output`。
2. `calibrate`传入同样配方、同样两个限额、train/val缓存及LIBLINEAR路径，只用train拟合、val选epoch和C。
3. `refit`传入pool缓存和calibration/selection.json，重新初始化学习参数。产物标记`train_plus_val_subset`，仅作工程检查。
4. 真实NWPU小样本阶段不运行test；最终测试入口会拒绝subset与calibration产物。

用各子命令的`--help`查看准确参数。本轮没有核验服务器实时数据路径，因此这里不提供虚构路径的远端启动命令，也未启动远端任务。

## 完整实验

`cache(train/val)` → `calibrate` → `cache(pool)` → `refit` → `cache(test)` → `test`。

校准按val CE总和选最早最优epoch；C按val OA选，平局取较小C。refit固定该epoch与C，重置同seed模型并拟合完整train+val。缓存/配方/源码/拟合样本/二进制/checkpoint身份均保存；输出排他创建，失败保留。

正式模式拒绝随机权重、子集或不同manifest，要求既有固定划分与对应torchvision VGG16文件的397923af SHA前缀及完整SHA。这不证明原MatConvNet等价。默认配方面向NWPU；AID需另行预先固定融合与预算，目前无AID结果。

当前test计时只覆盖已池化缓存后的图/融合/线性预测，明确排除VGG。端到端速度、GPU行为及正式成本仍需真实目标设备测量，不能将本轮6秒工程检查当成完整方法成本。
