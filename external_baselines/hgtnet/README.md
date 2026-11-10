# HGTNet independent v0

已实现完整候选模型，完成真实 CPU 工程验证。当前没有 HGTNet 的数据集准确率、GPU 测量或正式训练结果。

本包是基于论文文字的独立实现。原文未消解的 MR max/average 冲突、第三阶段下采样细节、图分支宽度和融合位置均在[设计说明](../../docs/external_baselines_20261027/hgtnet_implementation_design_20261031.md)中明确。默认 `mr_max`；`relative_mean` 仅为有独立配置身份的算子歧义检查。不能将本版称作作者代码或原文数值复现。

## 结构与接口

- RGB224 → patch4 →128×56×56。
- DA + Swin attention blocks 2/2/18，heads4/8/16；三次下采样后依次256×28×28、512×14×14、512×7×7。
- DW/pointwise transition →256×7×7，6个动态 MR graph/FFN；第二 transition →512×5×5，2个 graph/FFN。两组均 k20、排除自身、每块根据当前投影特征重构图，不裁剪 k。
- 视觉512维和图512维各自 GAP 后相加，再分类。45类主配置；30类配置的构造与头形状已验证，未做AID数据评估。
- 参数总数 **74,619,858**：patch6,528、Transformer60,479,736、DA44,077、graph14,066,432、head23,085。全部可训练。

`model.py` 的 `HGTNet(config)` 返回原始 logits；`forward_with_trace` 额外返回实际形状、实际邻居索引的形状和SHA摘要。摘要描述拓扑，不是边贡献或解释分数。`config.py` 严格验证架构和独立选择；构造函数不下载或隐式加载权重。

## 候选预训练

原文仅说明 partial ImageNet，尚无经过核验的作者权重。本版选择官方 torchvision Swin-B IMAGENET1K_V1 作为候选初始化，复用 stem、前三个 attention stages 和前两个 PatchMerging；DA、第三次下采样、全部 graph 与分类头明确新初始化。

- 来源：[官方候选文件](https://download.pytorch.org/models/swin_b-68c6b09e.pth)。
- 本地文件：`checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth`，351,654,207字节。
- 完整 SHA256：`68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893`。
- 已实际安全读取并验证353个source tensors、全部有限；完整模型实际加载318个兼容 tensors。其59,489,486个state元素占目标79.6670%，含buffer，不能将该数当作参数量。155个target tensors明确初始化，35个source tensors明确跳过，没有缺失或意外键。

`pretrained.py` 区分 schema audit、实际 apply 和有完整文件身份的 load；meta audit不能表示已加载。普通模型和loader无网络副作用。

## 已完成的本地验证

证据入口：[完整本地摘要](../../results/evidence/hgtnet_local_implementation_20261031/full_cpu_01/summary.json)、[相关测试](../../results/evidence/hgtnet_local_implementation_20261031/integration/related_suite.json)、[独立审查](../../results/evidence/hgtnet_local_implementation_20261031/review/independent_review.md)。

- 87项HGTNet测试及148项DFAGCN回归，共235项通过；不是全仓库旧依赖测试通过。
- 全规格 CPU FP32、seed42、batch1/2×224，实际候选初始化后进行一个合成 batch2 的CE反向和Adam更新。451个可训练参数tensor均有有限梯度，五个模块组梯度范数均正，五组代表参数确有变化。
- 实际trace确认graph有49/25节点，每块20邻居，无自身边；不是缩小通道或meta替代。
- `model.pt` 保存后由新进程加载；batch1和batch2最大输出误差均0，容差1e-6。checkpoint绑定完整SHA、配置SHA和六个模型相关本地源码SHA，拒绝覆盖与错误/不完整状态；不承诺optimizer续训。
- Torch2.2.2、torchvision0.17.2、Python3.12.13、macOS CPU、单线程。跨库版本或换CUDA设备后需要重新数值验证。本轮CPU耗时不是3080吞吐、epoch估计或费用。

## 本地复跑

从项目根目录执行。以下是本机现有临时工程环境，不是Featurize命令；新输出目录必须不存在。两行分别启动不同进程，第二行成功才写`completed=true`。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m external_baselines.hgtnet smoke --output /Users/patrick/Projects/GAViT_Project/checkpoints/hgtnet_local_smoke_user_01 --weights /Users/patrick/Projects/GAViT_Project/checkpoints/hgtnet_pretrained_20261031/swin_b-68c6b09e.pth --weights-sha256 68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python -m external_baselines.hgtnet verify-reload --output /Users/patrick/Projects/GAViT_Project/checkpoints/hgtnet_local_smoke_user_01
```

复跑相关测试：

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_INIT_AT_FORK=FALSE /private/tmp/gavit-dfagcn-cpu-20261027/bin/python results/evidence/hgtnet_local_implementation_20261031/integration/run_related_tests.py
```

## 下一阶段

2026-11-02已收到真实3090小样本status：complete/exit0、360/90/test0、180steps、回读误差0，固定身份一致，工程门禁通过。实际wall58.26s、allocator峰值allocated1.43GiB/reserved1.57GiB，smoke val OA36.67%/macro-F129.34%不作正式性能。[结果记录](../../results/hgtnet_nwpu_smoke_3090_20261102.md)与[正式seed42准备](../../docs/external_baselines_20261027/hgtnet_formal_preparation_20261102.md)已保存；下一步分离现有训练耗时、验证论文batch16容量，再制作正式入口。

2026-11-02用户已回传fix1真实工程流程COMPLETE，180steps/360张训练图完成；按交付代码路径接受服务器报告的360/90小样本执行完成，具体指标/显存/耗时/工件身份待现有status短摘要。[完成标记证据](../../results/evidence/hgtnet_3090_smoke_return_20261102/README.md)。无需重跑，此结果仍不是正式准确率比较。

2026-11-02后续3090实机诊断定位到NVML24576MiB与CUDA显示24123MB口径差异，原严格相等门禁误拦截，随后交付[3090 fix1](../../output/hgtnet_nwpu_smoke_3090_fix1_README_20261102.md)。模型/训练设置未改，交付时45项门禁/打包检查通过；服务器最新进展见上条，完整本地数值依赖限制仍保留。

2026-11-02用户改租RTX3090/24GiB时，先交付了[原3090兼容包](../../output/hgtnet_nwpu_smoke_3090_README_20261102.md)，现由上面的fix1取代。本次只改资源/目标传播，模型、数据、引擎与参数配置字节不变；当时39项硬件/打包检查通过。旧临时CPU环境缺Torch/NumPy导致本轮全量数值回归未通过，未将下文历史285项检查当作本轮结果。3090真实CUDA/360+90仍待执行。

此前已交付[独立3080小样本运行包与上传命令](../../output/hgtnet_nwpu_smoke_3080_README_20261031.md)，新增真实图片一轮工程训练/validation/checkpoint入口；工程证据见[2026-10-31记录](../../results/evidence/hgtnet_3080_smoke_prepare_20261031/README.md)。当时相关测试285项通过，完整HGTNet在生成JPEG的CPU夹具上跑通，未开始目标GPU或真实NWPU运行。

下一步依据现有训练history和有界batch16容量检查，冻结正式数据/优化/选轮协议并制作正式入口。目前没有正式训练器或正式准确率；服务器工程结果不写正式准确率比较表。
