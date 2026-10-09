# NWPU 空间先验候选：状态与比较口径

2026-10-08。新候选尚无 GPU 短测、校准或测试结果。

| 模型 | 完整 val 正确数 / 1260 | best val OA % | macro-F1 % | 状态 |
|---|---:|---:|---:|---|
| Swin-T，seed42 | 1187 | 94.206349 | 94.171552 | 已有对照 |
| 固定分组 GAViT，seed42 | 1172 | 93.015873 | 92.976564 | 已有对照 |
| 原 soft GAViT，seed42 | 1181 | 93.730159 | 93.765721 | 已有对照 |
| soft + anchor_gaussian_v1，seed42 | — | — | — | 本地实现；待真实短测与校准 |

已有值来自2026-09-28/29完整校准回传，原 soft 还由2026-10-08 pooling probe 正常模式复现；均不是 test 或多 seed 结果。新候选不写入完成实验均值表。

本次唯一主要变量是 S logits 的固定空间距离项，保留 attention pooling。固定公式、比较与停止规则、待发布服务器短测参数见 [运行设计](../docs/run_nwpu_spatial_prior_20261008.md)。结果将补入本表，工程短测 OA 不进入本表。参数增量为0；服务器吞吐、显存、时间和费用均待测。

本地代码补丁及 SHA 位于 [release_identity.json](evidence/nwpu_spatial_prior_20261008/release_identity.json)。独立审查未发现阻塞问题：温度后加先验、实际不等大分区质心、旧默认/RNG兼容、候选身份与 strict load 均符合设计。失败原因、可用日志与最终测试结果在同目录保存；不将本地 CPU/合成图片测试等同于真实 Swin 预训练与 CUDA 验收。

本地验收：核心64项通过（17.45秒），共享BigEarth入口/身份28项通过（2807.53秒），共92项不同测试通过、退出0。[完整检查状态](evidence/nwpu_spatial_prior_20261008/local_validation.json)。额外耗时是本地离线CPU回归，不能估算服务器短测或训练费用；无需重跑这些已通过检查。
