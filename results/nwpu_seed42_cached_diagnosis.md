# NWPU seed42：已有验证预测与训练曲线诊断

**09-29正式refit/test已完成：** 六阶段均exit0，test OA为Swin93.932540%、fixed94.103175%、soft93.869048%。[正式测试结果](nwpu_seed42_test_comparison.md)已归档；下文原启动步骤/建议保留为历史，不重复训练。

2026-09-29，用户返回CACHED_VALIDATION_READ_COMPLETE。原始附件逐字节归档，配对计数、OA换算、每类整数变化和30轮曲线已本地复算。代码ed480e6干净，三模型的training/code/hardware/seed/counts/manifest/pretrained均由服务器命令返回一致；RTX2080Ti、torch2.2.2、CUDA12.1，FP32/无AMP/TF32-off、8CPU线程和既定配方已确认。

| 模型 | 最佳val OA % | 所选checkpoint val macro-F1 % | 最佳轮 | 最终val OA % | 最后5轮平均OA % | 参数量 |
|---|---:|---:|---:|---:|---:|---:|
| Swin-T | 94.2063 | 94.1716 | 29 | 94.2063 | 94.0000 | 27,555,495 |
| 固定4×4 GAViT | 93.0159 | 92.9766 | 29 | 93.0159 | 92.8889 | 31,386,920 |
| 软分配 GAViT | 93.7302 | 93.7657 | 28 | 93.4921 | 93.5873 | 31,486,760 |

## 已能回答的问题

1. **共同身份检查没有发现配置不一致。** OA/macro-F1均是soft高于fixed、低于Swin。不能把所有实现问题都据此排除，但当前没有理由把本批差距归咎于不同划分/预训练/配方/硬件。
2. **收益被新增错误部分抵消。** 对fixed：soft纠正40、破坏31、双方正确1141、双方错误48，净+9；对Swin：纠正30、破坏36、双方正确1151、双方错误43，净−6。这是不同训练模型的预测差异，不是关闭某个模块的因果消融。
3. **不是明显的末期性能崩落。** 三模型训练loss最终约0.002，验证后期接近平台。soft从第28轮峰值到最终少对3张（−0.238095 pp）；选best后仍比Swin少6张，不能说只是选错checkpoint，也没有证据支持盲目加epoch就会改善泛化。低训练loss本身不等于证明过拟合原因。
4. **soft分配确实变化，但区域区分度值得关注。** 固定8val探针从epoch0到28：S变化0.068631；投影空间原型间cosine均值0.33662→0.78264、最大值0.93292→0.99732；分配熵0.67858→1.48263。方向上是原型更相似、分配更分散，尚不能判断这是有害的过度平滑还是任务学习到的相似性。
5. **不能把探针解释为全数据集区域塌缩。** 返回的0/28/30时点低范数原型均0、最小区域质量约1、feedback/token均值1.034→0.515；没有零质量区域证据。源码select_records按路径排序，runner取val_rows[:8]，不是类别分层样本；按NWPU类名顺序预期来自airplane，具体sample_ids此次未返回。其原型指标是128维归一化投影原型，不是768维区域输出特征的相似度。今后论文可视化应补多类别代表样本，不能把这8张图当45类代表，也不为此重训。

## 类别差异：用图片数解释

每个列出的类别只有28个验证样本，一张图相当于3.5714 pp。以下都是逐类召回率差，不是类别precision，且仅为服务器返回的两侧前五名。

| 对照 | 类别 | soft净多/少正确图片 |
|---|---|---:|
| swin | palace | +4 |
| swin | intersection | +3 |
| swin | church | +3 |
| swin | wetland | +2 |
| swin | mobile_home_park | +2 |
| swin | forest | -3 |
| swin | airplane | -2 |
| swin | dense_residential | -2 |
| swin | industrial_area | -2 |
| swin | lake | -2 |
| fixed | church | +4 |
| fixed | wetland | +3 |
| fixed | palace | +3 |
| fixed | railway_station | +2 |
| fixed | railway | +2 |
| fixed | dense_residential | -2 |
| fixed | forest | -2 |
| fixed | mountain | -2 |
| fixed | chaparral | -1 |
| fixed | freeway | -1 |

宫殿、教堂、路口等类别有改善，森林、密集住宅等有退步。当前小样本与局部榜单不支持“图只对人工结构有效”之类机制断言，也不应挑改善类别删去退步类别。

## 资源与下一步决定

30轮纯训练累计：Swin901.44秒、fixed955.07秒、soft1001.80秒；soft仅比fixed多99,840参数（约0.318%），但仍比Swin多3,931,265参数。当前时间是单次同硬件训练观察，不替代独立推理效率测量。

建议保留并冻结本版为首轮机制消融，按既定协议完成seed42的三模型refit/test；价值是检验soft相对fixed的改善能否延伸到未用于选择的test，不是期待test翻盘。由校准预定规则选择Swin/fixed/soft的29/29/28轮，从相同ImageNet预训练重新初始化，在6300张train+val上拟合，Cosine周期仍30，分别用末轮checkpoint评估25200test。禁止沿用校准best续训后声称独立refit。结果无论正负均保留，不以test反馈选温度或seed。

预算推算：将已测训练秒数按6300/5040及29/29/28轮缩放，仅训练约56.87分钟；另需原图核验、存盘、重建、三模型测试，规划约1–1.5小时（同RTX2080Ti、数据已在）。这不是实测保证，也不含实例初始化/解压；费用为实际小时乘租价，租价未知。正式付费批次尚未启动，须在交付执行命令时固定输出目录与来源身份。暂不扩seed43/44或启动耗时更长的BigEarth正式训练。

这完成了本轮已有产物诊断，不继续要求用户重复取同样摘要。E05双数据集多seed和论文多类别分组可视化仍未完成。

训练曲线的90个数据点保存在[cached_curves_rounded.json](evidence/nwpu_calibration42_20260928/cached_curves_rounded.json)，loss保留5位、OA保留4位小数，不能当作原始history.json的完整精度。证据：[原始返回](evidence/nwpu_calibration42_20260928/cached_analysis_server_return.txt)、[精确摘要](evidence/nwpu_calibration42_20260928/cached_analysis_server_summary.json)、[数值核对](evidence/nwpu_calibration42_20260928/cached_analysis_validation.json)。
