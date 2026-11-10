# NWPU 空间先验候选：状态与比较口径

2026-10-10 最终收尾：用户已回传身份、三组缓存重算OA/macro-F1、配对和完整30轮压缩曲线；本地摘要算术复核通过。新候选best epoch24，OA92.380952%、macro-F1 92.342474%，均低于旧soft与Swin。按冻结规则关闭候选，无需再回传或训练；没有新增test或稳定性证据。

| 模型 | 完整 val 正确数 / 1260 | best val OA % | macro-F1 % | 状态 |
|---|---:|---:|---:|---|
| Swin-T，seed42 | 1187 | 94.206349 | 94.171552 | 已有对照 |
| 固定分组 GAViT，seed42 | 1172 | 93.015873 | 92.976564 | 已有对照 |
| 原 soft GAViT，seed42 | 1181 | 93.730159 | 93.765721 | 已有对照 |
| soft + anchor_gaussian_v1，seed42 | 1164 | 92.380952 | 92.342474 | 缓存收尾完成；候选关闭 |

前三行来自2026-09-28/29完整校准回传，原soft还由2026-10-08 pooling probe正常模式复现；本次服务器缓存检查重算Swin/soft/spatial三组并返回1164等正确数。本地核对摘要及配对边际，没有下载原始预测或checkpoint。均不是test或多seed结果，不写入多seed均值表。

相对旧soft：OA−1.349206pp、macro-F1−1.423247pp；纠正26张、新增错误43张、双方正确1138、双方错53，净−17。相对Swin：OA−1.825397pp、macro-F1−1.829078pp；纠正21张、新增错误44张、双方正确1143、双方错52，净−23。相对固定分组OA−0.634921pp/少对8张，本次未回传该组配对。

曲线：epoch24为OA和macro-F1最高，loss0.003957；epoch30 OA92.063492%、macro-F1 92.001548%、loss0.003366。最后六轮没有超过最佳值，最终仅较最佳少对4张，没有按本曲线追加轮数的证据。训练loss较低与val平台并存，但不能单凭增强训练loss定位唯一原因，亦不能外推所有空间先验/软分配无效。停止决定不因宏F1或几何指标改变。

本次校准耗时2997.438秒（49.96分钟），峰值allocated3.66257GiB，重载最大logit差9.5367e-7。缓存回传身份为67fe84d干净源码、2080Ti/CUDA12.1/PyTorch2.2.2、8线程、实际图48+32边，manifest/pretrained以及实际best SHA与冻结值一致；相同训练配方检查位于打印输出前。最佳checkpoint SHA为a1c7f3b173e369ec3d40e2122fec91fa9ebc70d39ac6a96f5db90017c76fb579。

从两位小数计时累计train1646.28秒、val221.64秒，训练吞吐按5040×30/总train约91.84图/秒；每轮train32.11–104.00秒、中位48.585秒，val3.52–17.70秒。剩余1129.52秒为训练/验证计时外的各阶段，未分解来源，不能直接归因于网络盘或空间先验开销；不将本实验作为公平效率benchmark。

[初次回传](evidence/nwpu_spatial_calibration_20261010/server_return.json)、[完整缓存回传](evidence/nwpu_spatial_calibration_20261010/closeout_server_return.txt)、[最终算术/曲线与证据边界](evidence/nwpu_spatial_calibration_20261010/closeout_summary.json)、[历史只读命令](../docs/read_nwpu_spatial_closeout_20261010.txt)。所有实质输出已收到，粘贴未含最后PASS标记不再要求重复运行。完整训练S/W几何未回传，也不作为执行停止规则的必需补件。

本次唯一主要变量是 S logits 的固定空间距离项，保留 attention pooling。固定公式、比较与停止规则见 [运行设计](../docs/run_nwpu_spatial_prior_20261008.md)，[历史校准命令](../docs/run_nwpu_spatial_calibrate_20261009.md)已执行，不再重复。参数增量为0；工程短测OA不进入本表。

短测资源：[服务器回传](evidence/nwpu_spatial_smoke_20261009/server_return.json)报告总98.576秒、训练8.261秒、43.578图/秒、验证0.572秒，峰值allocated3.662GiB/reserved4.039GiB，重载最大logit差3.5763e-7。吞吐来自360图首轮，包含启动影响，不代表正式稳态；实际费用未知。本地仅复核摘要算术，没有下载原始权重/快照或再次前向。

本地代码补丁及 SHA 位于 [release_identity.json](evidence/nwpu_spatial_prior_20261008/release_identity.json)。独立审查未发现阻塞问题：温度后加先验、实际不等大分区质心、旧默认/RNG兼容、候选身份与 strict load 均符合设计。失败原因、可用日志与最终测试结果在同目录保存；不将本地 CPU/合成图片测试等同于真实 Swin 预训练与 CUDA 验收。

本地验收：核心64项通过（17.45秒），共享BigEarth入口/身份28项通过（2807.53秒），共92项不同测试通过、退出0。[完整检查状态](evidence/nwpu_spatial_prior_20261008/local_validation.json)。额外耗时是本地离线CPU回归，不能估算服务器短测或训练费用；无需重跑这些已通过检查。
