# NWPU360/90离线准备证据

2026-10-29，用户明确回复“实例没开启”。因此本轮仅本地准备，未访问Featurize、租用实例或启动真实GPU训练。

- [总验证](validation.json)：105项DFAGCN相关测试通过；[包内日志](package_tests.log)。旧19个指标文件和12个旧入口源码SHA保持。
- [官方VGG来源与完整SHA](weights_provenance.json)、[实际预训练CPU数值核对](pretrained_cpu_validation.json)：553433881字节完整文件，FC6/conv3/conv4/conv5相对标准torchvision实现误差均0，三个卷积分支保留实际负前激活。
- [便携包及操作说明](../../../output/dfagcn_nwpu_smoke_README_20261029.md)：包内源码/配方/完整权重身份/官方LIBLINEAR源码；权重文件单独上传，不混入大包或Mac原生二进制。
- [实际便携包资源检查](bundle_inspect.log)、[包内源码实际编译](bundle_compile.log)：本机仅缺真实服务器manifest和目标GPU，训练未开始、未创建runs目录；当前平台编译与二进制哈希通过，Linux尚待实例检查。
- [GPU上下文拒绝RED](runner_context_red.log)、[配方身份RED](runner_recipe_red.log)、[运行器最终11项GREEN](runner_final.log)。子代理在限额前留下的未完成CUDA上下文检查由主代理复核并修复。
- [真实首batch日志RED](progress_red.log)/[GREEN](progress_green.log)：日志在实际forward/backward/optimizer之后输出，记录设备、样本数和loss，避免用PID或函数调用代替真实batch。
- [便携交付7项最终检查](delivery_final.log)：真实zipapp、复用SHA/已有目录拒绝、路径/重复成员拒绝、官方源码编译、平台/二进制复用校验。原始RED/GREEN在delivery子目录。
- [全旧仓库失败明细](repository_failures.json)：204项发现检查，9failure/9error，仍缺pytest/timm/torch_geometric；unittest不执行pytest函数式测试，未声称全旧仓库通过。
- source_final为最终代码/配方/交付工具快照。GPU UUID、CPU/GPU软件/精度、完整一轮及真实资源必须由实例运行后另存，当前不填造。

主下载域名曾TLS失败，随后使用同属PyTorch的官方下载服务。完整下载与66个Range片段合并的SHA一致，记录在weights_ranges.json；未关闭TLS验证。最终主权重位于Git忽略的checkpoints目录，下载临时片段不是训练成果。
