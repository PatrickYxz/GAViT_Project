# E06 模型推理效率入口：本地验证

日期：2026-09-22。本文件记录启动前本地验证；后续**服务器模型前向测量已回传成功**，见[当前结果](aid_efficiency_summary.md)。本地测试证据与服务器测量证据分别保存。

研究问题：AID 固定split三seed OA平均差+0.0467 pp时，固定区域图实现新增多少延迟、吞吐及显存开销？复用预定第一个seed42的最终refit权重，固定同卡、输入、精度和计时条件，只比较Swin/GAViT完整模型实现。已有权重即可回答，不重新训练，也不根据test挑选模型或seed。

脚本：[benchmark_aid.py](../tools/benchmark_aid.py)。执行：[单行命令与口径](../docs/run_aid_efficiency.md)。命令文本：[paste_aid_efficiency.sh](../docs/paste_aid_efficiency.sh)。脚本在持久化结果目录运行，不放入冻结的deb2b45训练仓库，无须上传文件、Git发布或重新解压原图。

本地证据：

- 先运行新测试，5项因实现缺失失败；实现后5项通过。覆盖异步工作必须同步完成才停表、按总时间/总图片数换算吞吐、非有限/非正计时拒绝、其他GPU进程拒绝、第二阶段失败后停止且保存退出码、已有输出拒绝覆盖，以及配对硬件/阶段完整性检查。见[测试日志](evidence/aid_efficiency_local_20260922/unit-tests.log)。
- CPU真实模型检查使用干净发布仓库deb2b45、**随机权重与合成输入**：参数数目27543960/31375385，输出B×30且有限，保存/重载最大logit差均0；GAViT捕获48条空间边+32条特征边。合成checkpoint身份通过，失败source退出码、损坏权重及错误manifest哈希均被正式校验函数拒绝。见[integration.json](evidence/aid_efficiency_local_20260922/integration.json)。
- 实际生成的一行命令通过bash/zsh语法检查，解码源文件与审阅脚本逐字节相同。在临时目录中以模拟nvidia-smi返回“GPU被占用”，实际后台启动后记录exit1/failure.json，不启动worker、不生成成功摘要；重复粘贴被拒绝且已有文件字节保持不变。见[launcher.json](evidence/aid_efficiency_local_20260922/launcher.json)。
- CPU集成检查与启动验证脚本副本保留在同一证据目录，供复核本地测试边界；它们不是服务器训练或性能结果。发布仓库保持干净，正式模型/训练/评估代码没有修改。

计时范围：固定合成RGB归一化张量预置CUDA；batch1与32分别独立进程、20次预热、5×50次同步主机计时，完整分组/图构建/GAT/反馈均计入。逐次原始值、重复均值/SD、GPU遥测及allocated/reserved峰值保留。输入生成、数据读取/处理/传输、模型加载、图验证hook不计时。显存重置发生在加载/预检缓存清除和真实推理预热之后，包含模型/输入/输出/临时张量及稳定分配缓存。

**上述本地检查当时没有访问服务器、核验正式checkpoint或运行CUDA基准。** 后续用户已回传四阶段exit0与完整成功摘要；CUDA测量数值及核验边界见[独立结果报告](aid_efficiency_summary.md)，不能用本地CPU测试代替该证据。E06的模型前向基准与完整数据处理链区分：后者仍未测量，不把E06全部范围标记完成。
