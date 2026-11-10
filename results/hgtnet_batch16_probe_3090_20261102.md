# HGTNet3090 batch16真实短测回传

状态：**SERVER_REPORTED_BATCH16_PASS_STATUS_IDENTITIES_MATCH**。用户已补回只读status：completed/statecomplete/exit0，4steps与配置、manifest、预训练和UUID均匹配；[26项摘要核验](evidence/hgtnet_batch16_return_20261102/status_validation.json)通过。此前运行命令、CUDA探针、进度和COMPLETE经[29项核验](evidence/hgtnet_batch16_return_20261102/validation.json)及[独立26项核验](evidence/hgtnet_batch16_return_20261102/review/README.md)通过。[原始粘贴](evidence/hgtnet_batch16_return_20261102/reported_output.txt)和[status原值](evidence/hgtnet_batch16_return_20261102/status_reported.json)留存，均为用户回传，非直接远程文件检索。

- 3090，实际输出Torch2.2.2+cu121/CUDA12.1、available=true/device_count1。命令绑定GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1，status UUID已匹配；完整runtime原件未回收。
- batch16，1次预热+3次计时，4steps/64样本呈现。按固定交付路径复用16张真实train图，val/test0，无正式epoch或模型保存。五组451个可训练参数张量梯度均有限、无missing/nonfinite、组norm均正，五组代表参数首步均实际改变。
- 预热1.0480841400003555秒；计时0.1999162600004638、0.20252209699992818、0.20135352999932365秒。中位0.20135352999932365秒，均值0.20126396233323854秒，16/median=79.46222745662192图/秒；总48图/三步总时长=79.49759020200715图/秒，统计口径不同而非矛盾。
- 这些计时仅为已解码/已传GPU固定输入的forward/backward/Adam及有限梯度检查，排除IO、预处理/增强、拷贝、初始化、日志和首步诊断。不能与旧batch2包含读图/首步检查的8.81图/秒相除宣称加速。
- CUDA返回25295323136B=24123.5MiB=23.55810546875GiB，是可见总容量。status补齐allocated峰值4732054016B=4.407068729400635GiB、reserved5215617024B=4.857421875GiB，作用域含单模型/Adam/激活/梯度/诊断，非NVML整卡或进程显存。整个probe wall10.059117196000443秒，含设置/核验/初始化/短测，不作正式epoch或账单时间。此前NVML物理24576MiB为旧清单，不归因两个接口差额。
- loss4.158695→3.343672来自同16图重复更新，不是验证/测试、泛化或方法优劣证据；不进入正式比较表。

按相同三步中位计算时间缩放：calibration5040/16=315次更新，约63.4264秒/轮；refit6300图、不drop_last为394次更新，按每批16等价保守取约79.3333秒/轮。100轮校准+100轮refit最多70900次更新，对应3.96555小时**条件纯更新计算**。最后一批refit实际12图，且不同样本/完整管线/增强和长程行为均未计量，因此这些数不是正式端到端ETA、实际费用或保证上限；实际refit轮数取决于事先固定的校准选轮规则。

本地实际包SHA9563c9b14bb182bb58104480bedee963efda6c44e48cd63fce5f8522a9426799保持，入口与engine逐字节相同。按交付程序正常路径，COMPLETE只在4步/最终参数有限、源码/配置/manifest/权重/所选图片末尾复核、summary写盘、run.exit0和marker之后打印。现status亦明确exit0、configaf6ef410...ac25、manifestce0cfef2...840e、候选68c6b09e...1a893和UUID4f625a02...3cb1一致。接受实际短测及资源摘要，无重复探测理由；仍未直接回收服务器summary/runtime原件。

```sh
/environment/miniconda3/bin/python /home/featurize/work/hgtnet_batch16_probe_3090_20261102.pyz status
```

后续正式seed42协议与独立入口在本地另行实现/验证，完整数据管线每轮耗时由实际正式进度回传；短测本身没有HGTNet benchmark分数。
