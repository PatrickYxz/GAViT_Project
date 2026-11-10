# BigEarthNet seed45：Swin / 固定分组 / 软分配正式消融

2026-10-01 验收：用户归还实例后从持久化 work 取回输出；总队列、训练和测试 exit_code 均为0，日志包含完整3745/3745测试、STAGE COMPLETE gavit_test 和 BIGEARTH SOFT45 COMPLETE。训练已到30轮，测试119825张。本次已在归还前完成，无需续训、重新解压或重跑。完整资源/探针摘要现已回传并核验，本轮归档完成。

证据：[原始粘贴字节](evidence/bigearth_soft45_complete_20261001/reported_output.txt)、[结构化核验与逐类差](evidence/bigearth_soft45_complete_20261001/comparison.json)。这是用户终端证据；[完整摘要原始粘贴](evidence/bigearth_soft45_complete_20261001/reported_summary_output.txt)、[结构化摘要](evidence/bigearth_soft45_complete_20261001/summary.json)和[身份/资源核验](evidence/bigearth_soft45_complete_20261001/summary_validation.json)已归档。摘要粘贴缺外层左花括号，原始字节不改，只在解析副本补齐并排除末尾提示符；不是服务器文件的逐字节下载。本地未下载checkpoint、predictions.npz或探针张量。

## 同seed正式测试

| 模型 | Test mAP % | macro-F1 % | micro-F1 % | Best val mAP % | Best / Final epoch |
|---|---:|---:|---:|---:|---:|
| Swin-T | 71.676930 | 65.084212 | 76.732264 | 78.6509 | 6 / 30 |
| 固定4×4 GAViT | 70.258236 | 64.295477 | 76.702457 | 78.2951 | 10 / 30 |
| soft_prototype GAViT | 70.836919 | 64.864823 | 76.383354 | 78.7037 | 11 / 30 |

F1阈值0.5；主要指标为19类macro-mAP。旧对照身份见[seed45原记录](bigearth_seed45_comparison.md)，本次沿用冻结split和预训练，训练轮数、batch32及优化/精度一致；旧对照deb2b45、新模块ed480e6分别记录。同型号RTX2080Ti及相同驱动/包版本，但不同物理GPU UUID。不是完全同一物理环境或多seed稳定性验证。

| 对比 | ΔmAP pp | Δmacro-F1 pp | Δmicro-F1 pp |
|---|---:|---:|---:|
| soft − 固定 | +0.578683 | +0.569346 | -0.319103 |
| soft − Swin | -0.840011 | -0.219389 | -0.348910 |

本seed支持的结论：软分配相对固定分组提高mAP和macro-F1，micro-F1下降；三个test指标仍均低于Swin。不能写成“三指标全面提升”或“稳定优于骨干”。连同NWPU seed42 soft相对fixed的负向结果，目前不支持跨数据集一致优势；NWPU为OA且seed不同，不能将两者数字直接平均。

## 类别变化

相对固定分组，19类AP中12升7降。海岸湿地+15.710764 pp，对macro-mAP贡献+0.826882 pp；其余18类合计贡献−0.248200 pp，形成总净增+0.578683 pp。这里是保留全部19类的算术分解，没有删类后重新报分。它提示主要净增来源集中，不能从AP推断样本标签错误或分组改善的因果。

| 类别 | Swin AP % | 固定 AP % | soft AP % | soft − 固定 pp |
|---|---:|---:|---:|---:|
| Urban fabric | 86.2464 | 86.3446 | 86.4014 | +0.0568 |
| Industrial or commercial units | 49.3535 | 49.4106 | 50.4624 | +1.0518 |
| Arable land | 93.6549 | 93.7962 | 93.8892 | +0.0930 |
| Permanent crops | 63.3516 | 60.8124 | 61.1000 | +0.2876 |
| Pastures | 86.1616 | 86.0899 | 86.1770 | +0.0870 |
| Complex cultivation patterns | 68.8229 | 70.0692 | 70.2285 | +0.1594 |
| Land principally occupied by agriculture, with significant areas of natural vegetation | 72.4131 | 72.8657 | 72.0559 | -0.8098 |
| Agro-forestry areas | 87.2877 | 87.3249 | 87.0858 | -0.2391 |
| Broad-leaved forest | 86.2448 | 85.5848 | 84.8883 | -0.6965 |
| Coniferous forest | 93.6093 | 93.5617 | 93.1714 | -0.3903 |
| Mixed forest | 90.0507 | 89.3596 | 90.2213 | +0.8617 |
| Natural grassland and sparsely vegetated areas | 48.1028 | 47.6140 | 40.1248 | -7.4893 |
| Moors, heathland and sclerophyllous vegetation | 60.4316 | 60.0019 | 60.4438 | +0.4419 |
| Transitional woodland, shrub | 78.9833 | 77.8972 | 78.2832 | +0.3859 |
| Beaches, dunes, sands | 10.3178 | 9.6689 | 12.8162 | +3.1473 |
| Inland wetlands | 62.8743 | 63.2370 | 61.4703 | -1.7667 |
| Coastal wetlands | 33.8542 | 10.6684 | 26.3792 | +15.7108 |
| Inland waters | 90.5851 | 91.1898 | 91.1556 | -0.0343 |
| Marine waters | 99.5160 | 99.4095 | 99.5474 | +0.1379 |

其中草地/稀疏植被AP下降7.489263 pp，沙滩/沙丘/沙地提高3.147280 pp；各类变化均保留。没有逐样本预测配对与类别支持数的独立重算，不将这些差异解释为显著性或结构因果。

## 曲线、产物与资源边界

- best为第11轮val mAP78.7037%；第30轮loss 0.0717331739、val mAP 76.229254%、macro-F1 71.460684%，下一轮LR为0。best到final验证mAP下降 2.474446 pp；测试使用best，不使用final。后期训练loss下降、val低于best的现象保留，不能仅凭片段确定原因。
- history只回传最后3500字符，完整可解析的是epoch17–30共14条，epoch16仅残片；不能宣称本地获得全部30轮曲线或独立重算了best。完整30轮/最佳选取由冻结队列完成检查支持。
- 这14轮检查点间墙钟均值 1744.809秒（约29.08分钟）；不是全程平均，不能乘30冒充实测总时长。测试循环显示05:04，未包含完整加载/保存阶段。本次完整阶段时长见下表；原16–18小时仅为规划，实际费用仍需账单或小时租价。
- metadata和test metrics中的checkpoint SHA一致：`4969f401bf297c3885ee0ac02938ae102413fc2e8c1bc8e0a628e9e918bd35e2`；本地未收到权重字节，不声称独立对权重重新哈希。best.pth实际文件名为gavit.pth。
- 回传文件大小：summary.json 13618字节，gavit.pth 125955898字节，gavit.last.train_state.pth 377875857字节，predictions.npz 9307603字节；仅确认报告的存在/大小，不以大小替代内容完整性核验。
- 摘要与批准预检、冻结完整入口参数、代码/脚本SHA、split及预训练身份一致；训练metadata和测试metrics与上次回传逐字段相同。8张固定val图、初始/best/final共3份快照检查通过；S行归一化最大误差2.384185791e-7、W列2.980232239e-7，实际每图48空间+32特征边。探针覆盖至epoch30，累计probe_seconds=5.398102462。这里只确认工程检查，不证明全数据分组语义、无塌缩或稳定增益。
- 正式训练、测试及本次资源/探针归档均完成，不再要求重复读取同批摘要，也不启动新GPU任务或自动增加seed。导师要求的双数据集多seed和外部方法比较仍未完成。

## 实测资源（2026-10-01补齐）

| 范围 | 墙钟时间 | CUDA allocated峰值 GiB | CUDA reserved峰值 GiB | 主进程RSS峰值 GiB |
|---|---:|---:|---:|---:|
| 训练阶段 | 52678.069086秒 | 3.659762 | 4.029297 | 1.971657 |
| 测试阶段 | 322.636569秒 | 0.595415 | 1.027344 | 1.474846 |

总队列53058.733037秒，即14小时44分18.733秒；训练阶段14小时37分58.069秒，测试阶段5分22.637秒。队列比两阶段计时之和多58.027382秒，属于阶段计时之外的合计开销，未进一步归因。

阶段资源包括checkpoint/探针产物检查；RSS仅主进程，不含DataLoader子进程；allocated与reserved均为运行期峰值，不是模型常驻显存。训练有效吞吐135.466810张/秒的分子为237871×30，分母含验证、初始化、探针、checkpoint IO和产物检查，不能当作纯模型吞吐。两次运行物理GPU UUID不同，不据此宣称soft比旧fixed更快；公平前向效率仍需独立同设备测量。实际费用未知。
