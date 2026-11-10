# NWPU 冻结权重反馈强度诊断：本地验收

2026-10-03更新：真实服务器诊断已回传并收尾，见[完整结果](nwpu_feedback_sensitivity_seed42.md)。下文保留10-02本地验收时的状态，不再用于表示服务器仍待启动。

2026-10-02。状态：实现、本地检查和单行交付完成；服务器尚未启动，研究结果待测试。

## 问题和固定范围

已有软分配消融在两个数据集呈混合结果，不能直接认定分配器无效或反馈过强。本诊断仅检查：对NWPU seed42两个已训练的校准模型，减弱反馈是否改变全验证集的损益趋势。不是新训练，不将α=0当成独立训练的Swin或无图基线。

固定5040/1260校准阶段best；fixed epoch29、soft epoch28；冻结ed480e6源码与原manifest。只在1260张val上预设α=0/0.25/0.5/0.75/1，不访问test图片或预测，不自动选最优α、不追加训练或seed。先前90图册缓存没有平均token和反馈，不能直接完成这次全验证推断，因此需要各模型一次推理。

正式模型在反馈后立即平均池化。诊断捕获一次原始前向，缓存mean(tokens)、mean(feedback)，仅复算分类头 `head(mean(tokens) + α * mean(feedback))`。α=1必须重建原logits、预测、缓存概率和完整验证OA；不修改正式模型源文件或权重。

## 已运行的本地验证

- 最初7项新增测试在实现前按预期失败；实现后扩展至10项新增检查，与原15项图册检查合计 **25 passed**，exit0，13.86秒。39条警告来自Torch对Python3.12 AST接口的弃用提示。记录见[tests.log](evidence/nwpu_feedback_sensitivity_20261002/tests.log)。
- 新检查覆盖全部val与排除test、每档缓存重放与实际缩放前向的一致性、权重/RNG保持、失败移除hooks、增益/损害计数与指标、错误缓存拒绝、无原图/骨干的保存重载复算、旧输出保护、失败状态与完整流水线产物。完整流水线使用合成1260记录、小型骨干和真实分组/GAT，不代替服务器源身份验证；冻结源码、权重、清单和原图/ZIP拒绝路径由复用工具的原15项测试覆盖。
- 另以真实Swin-T、两种真实GAViT、45类、2×3×224×224合成输入验证五档实际前向及缓存重放。固定/软模型最大logit差分别4.7684e-7/5.9605e-7，保存重载差0，所有argmax一致。随机权重、CPU运行，未下载训练权重，不能解释为准确率实验。见[real_cpu_validation.json](evidence/nwpu_feedback_sensitivity_20261002/real_cpu_validation.json)。
- 单行命令的解码字节与脚本SHA一致、Python编译、bash/zsh语法通过；临时路径下执行完整交付逻辑、以假Popen隔离进程边界，验证写入脚本、只启动一次、重复日志不覆盖、缺checkpoint拒绝。见[delivery_validation.json](evidence/nwpu_feedback_sensitivity_20261002/delivery_validation.json)。没有连接或启动服务器。

本地隔离环境 `/private/tmp/gavit-feedback-qa-20261002`：Python3.12.13、Torch2.2.2、torchvision0.17.2、torch-geometric2.8.0、timm1.0.27、NumPy1.26.4、Pillow10.3.0；CPU/macOS。服务器真实Python3.11/CUDA/原图/训练checkpoint尚待执行检查。本地旧临时环境不完整，另建隔离环境后验证，没有修改服务器或正式模型依赖。

## 交付与判读

[完整执行说明](../docs/run_nwpu_feedback_sensitivity_20261002.md)及[单行启动命令](../docs/launch_nwpu_feedback_20261002.txt)。输出独立目录 `/home/featurize/work/nwpu_feedback42_20261002_01`；原NWPU持久ZIP可直接读取，无需BigEarth数据或解压。

输出保存两模型五档OA/macro-F1/CE、逐类与混淆矩阵、配对纠正/新增错误、预测CSV和可离线复算的PT缓存。先看α=1复现及完整性，再看幅度趋势和类别损益；不能从一个最佳点判根因或按既有test选择α。若无收益，也保留并结束这项假设检查。结果不直接外推至BigEarth，不承诺未来训练能稳定改善。

独立脚本SHA256：`4cd5175261d271638a00e4a4c733f329ecde1c07b646b901c960f0181237eea3`。
复用旧helper SHA256：`35ed63ce1ceda120fe9b5e78b157032c5e89d6b351f0caf67e05eef19d4080ba`。

未Git提交/推送；已有实验表及模型文件的用户修改保持原样。此项本地完成不代表服务器诊断或导师全部实验范围完成。
