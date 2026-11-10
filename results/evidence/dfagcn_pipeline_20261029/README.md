# DFAGCN完整独立入口验收

2026-10-29。当前状态：完整独立流程已实现，本地合成工程链通过，真实NWPU与预训练表现尚待验证。

- [总验证记录](validation.json)：86项DFAGCN方法通过，独立审查无待修复发现，19个旧指标和12个旧入口源码SHA保持。
- [包内测试](package_tests.log)：原53项组件加config/model11、cache/extraction10、LIBLINEAR7、pipeline4、CLI1，共86。
- [最终20图真实算子smoke](smoke_run_02/smoke.json)：实际随机VGG的CPU FC6/448卷积、缓存、GCN训练、val选epoch/C、fresh refit、LIBLINEAR和合成test；重载误差0。总约6.04秒不代表NWPU成本。
- [独立审查](review/final_review.json)：包含完整49节点45类配置的前向/反向与实际参数更新、C选择、阶段拒绝探针。
- [默认结构参数量](architecture_report.json)：神经部分120436679参数，含实际使用的冻结VGG与图支路/训练头，不含LIBLINEAR，未计入未使用FC7/FC8。这是结构计数，不是显存实测。
- [官方LIBLINEAR来源/编译](liblinear/provenance.json)：v250、固定commit、BSD3、原始源码归档、本地二进制SHA；附带二进制仅macOS arm64。
- RED/GREEN：model、cache、liblinear子目录及pipeline_red.log、pipeline_initial_green.log、subset_red.log、subset_green.log、cli_red.log。原生libomp启动错误与功能RED分开保存。
- [旧仓库逐项失败](repository_failures.json)、[完整日志](repository_tests.log)、[pytest入口](repository_pytest.log)：185项发现检查，9failure/9error，缺pytest/timm/torch_geometric。unittest不执行pytest函数式测试，不能声称全仓库绿色。
- source_final保存当前被审查源码、测试与配方；source_before_subset和smoke_run_01保留早期工程链及对应旧源码，不覆盖历史证据。

仅合成工程训练/评估，无真实NWPU/AID分数、GPU或远端任务。原论文结构/MatConvNet数值等价、目标设备端到端效率、正式训练与多seed均未验证。文件SHA用于本项目实验追踪，不声明对恶意并发写入者的签名认证。

下一步使用[真实NWPU一轮配方](../../../configs/dfagcn_nwpu_smoke_v1.json)接入实际冻结manifest、图片与已核验预训练文件；按[运行说明](../../../docs/external_baselines_20261027/pipeline_usage_20261029.md)先限制每类train8/val2，不读取真实test。
