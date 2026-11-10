# DFAGCN第一批组件验证

日期：2026-10-28。实现身份：`dfagcn_independent_primitives_v0`。

新增53个unittest方法通过，独立代码审查无待修复发现；只验证组件，不代表完整论文方法或准确率复现。来源与未决配置见[原文规格](../external_baselines_20261027/dfagcn/method_spec.json)。

- [汇总及源码/旧文件哈希](validation.json)：19个旧指标文件、12个旧入口源码均保持。
- [最终新增测试日志](primitives_suite.log)：graph26、features13、fusion7、identity6、组合链1；保留原生subTest，不把展开数当方法数。
- [独立审查](review/final_review.md)及[独立标量构图参照](review/numeric_probes.json)。
- [图的RED/GREEN与溢出回归](graph/validation.json)、[VGG的RED/GREEN](features/validation.json)、[融合RED](fusion_red.log)/[最终GREEN](fusion_green.log)、[缓存身份RED](identity_red.log)/[GREEN](identity_green.log)。
- [依赖版本与下载wheel SHA](environment.json)、[可重建环境的依赖版本清单](requirements_cpu_test.txt)。临时环境路径不作为长期运行环境承诺。
- [全仓库unittest日志](repository_unittest.log)及[逐项失败/不可运行明细](repository_unittest_failures.json)：缺pytest/timm/torch_geometric，不能称全仓库通过；unittest也不执行pytest函数式测试。[pytest入口](repository_pytest.log)因未安装框架不能启动。

有效证据包括真实CPU卷积的ReLU前负值、卷积权重保存/重载、GCN手算/梯度、融合顺序和组合链。完整FC6只运行meta形状路径，未验证CPU FC数值或MatConvNet原权重/预处理等价。

没有新增准确率、真实数据smoke、正式训练、GPU任务或完整方法推理成本数据。完整分支层配置/池化、LIBLINEAR、停止/选参、真实张量缓存与训练/refit/test执行门禁仍需后续接入。
