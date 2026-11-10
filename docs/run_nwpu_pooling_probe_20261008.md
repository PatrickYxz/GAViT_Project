# NWPU 最终汇总 attention 对照：2026-10-08

**本项已完成，请勿重复启动下方历史命令。** 用户回传1260val全部PASS、完成标记和exit0；仅S汇总OA下降0.555556pp，纠正5/新增错误12。[结果与范围](../results/nwpu_pooling_probe_seed42.md)已归档。保留原attention，关闭此次纯推理替换分支；下文保留原冻结计划及当时的待运行说明。

## 目的与固定范围

90例诊断已完成。soft的S锚点距离低于均匀邻域参照，最终W的距离却更大，同时W的权重更集中；这使“第二次attention重加权的作用”成为更直接的问题。该现象尚未证明有害，空间先验也尚非唯一解法。

本次使用原seed42 soft calibration best与完整1260张val，只比较两种冻结权重的前向：

| 模式 | 最终区域汇总权重 |
|---|---|
| normal | `W = softmax_token(log(S) + attention_scores)` |
| membership_only | `U = S / sum_token(S)` |

保留原型生成时的attention、S、骨干/GAT/分类头参数、16区域、48+32边预算和反馈映射算法。特征top-2边随区域表示变化重新计算，是该干预的下游效应；不宣称边相同。程序使用临时forward hook，正常→干预→恢复均调用真实模型前向，并检查S逐元素相同、真实图边、恢复预测、原验证缓存概率。正式模型源码及旧checkpoint不修改。

该探针对已训练模型进行推理干预，可能改变模型依赖的区域分布。指标升降均不能替代新结构训练后的结论；`U`质心等于归一化`S`质心是数学必然，不能把漂移下降当作研究成功。本次不训练、不追加seed、不扫超参、不评价test，也不自动启动后续实验。

## 运行

打开 [单行启动命令](launch_nwpu_pooling_20261008.txt)，复制唯一一整行到服务器。无需上传文件、git pull或解压数据。默认使用当前实例CUDA做推理，读取持久化原始NWPU ZIP；如果原图目录存在，优先读取并验证原图。所需代码/权重/两个旧helper均在work，缺失或哈希不符则停止保留证据。

输出：`/home/featurize/work/nwpu_pooling42_20261008_01`。启动器独占 `.launch.log`，程序独占输出目录，不自动重跑或覆盖。已存在输出/日志时先检查，不删除它们来重试。

查看进度：

```bash
tail -f /home/featurize/work/nwpu_pooling42_20261008_01.launch.log
```

`PID`不代表完成。需要看到 `SOURCE PASS`、`DEVICE`、首个 `PAIR 32/1260`；随后完整 `PAIR 1260/1260`、`NWPU POOLING PROBE COMPLETE`与`exit_code=0`。完成后Ctrl+C只退出tail。若LOAD暂时没有输出，不重复启动；网络work盘读权重可能占时间。

回传最后几行即可（不会打印混淆矩阵）：

```bash
tail -n 12 /home/featurize/work/nwpu_pooling42_20261008_01.launch.log
```

若异常，回传最后35行日志。完整状态保存在输出目录的 `run.exit` 与 `summary.json`。本次需三次1260张前向、源身份检查与存盘；历史推理时间不能直接当本探针时长，实际总时间和峰值显存将记录。不属于30轮训练，无需BigEarthNet数据。

## 指标、产物与验收

- normal逐样本概率/预测须对齐既有缓存，完整val OA须复现原93.730159%（1181/1260）。否则整项失败，不能解读干预结果。
- 两种模式记录OA、macro-F1、交叉熵、逐类指标和混淆矩阵；屏幕仅打印精简指标及纠正/新增错误/净差。
- `prediction_cache.pt` 保存全部路径/标签、两组logits及分类头输入、分类头参数，可在CPU重放验证；两组实际前向与重放预测一致才完成。
- `selected_cases.pt` 保存原先按路径每类选两张的90例S/W、实际GAT前区域特征、实际边、logits等。保存这些特征是为后续分析复用，避免同一问题反复前向；不是按本次成败重新挑案例。
- `region_diagnostics.json` 记录1260例实际区域特征两两余弦均值、零范数数量，以及32条特征边的共同边数。零范数区域不参与余弦均值，汇总标明有效图像数。此统计是表征描述，不是语义正确性标签。
- `inputs.json` 与 `identity.json` 保存图像/输入张量SHA及冻结来源；`metrics.json`保留完整指标，`summary.json`保留精简结论、产物SHA、资源数据与范围声明。

若去掉最终重加权后预测改善，形成“简化汇总值得独立训练对照”的依据；若退步，说明现有权重依赖它，不证明简化结构重新训练也必然差。若差异很小，不自动追加训练来追求好看分数。读完这一次结果再决定是否值得做一个单变量NWPU候选；空间先验和BigEarthNet扩训均未启动。

## 本地验证

工具：`tools/diagnose_nwpu_pooling.py`。复用冻结的反馈诊断helper SHA `4cd5175261d271638a00e4a4c733f329ecde1c07b646b901c960f0181237eea3` 和原图可视化helper SHA `35ed63ce1ceda120fe9b5e78b157032c5e89d6b351f0caf67e05eef19d4080ba`，均在导入前校验。

验证日志位于 `results/evidence/nwpu_pooling_probe_20261008/tests.log`。本地合成测试使用真实soft grouping和PyG图模块，以小型token骨干替代Swin；它验证干预与直接改pool公式一致、S/RNG/参数保持、异常清理、零attention等价、1260合成样本读写及分类头重放、防覆盖、内嵌脚本一致、Bash/zsh语法和启动拒绝。复用的ZIP验证检查实际JPEG字节/像素身份、无需解压与不掩盖损坏文件。真实预训练Swin、原始1260图与CUDA尚待服务器运行，不能以本地测试冒充实机验收。

最终12项通过，exit0，4.06秒；39条提示是已有Torch/Python3.12 AST弃用提示。独立静态审阅未发现阻断项。工具SHA为 `3a144266ef35b3df3a60cf3e8abd614f5186467510f1bb857c3019eac5b4967a`；完整文件身份与测试范围见[验证记录](../results/evidence/nwpu_pooling_probe_20261008/validation.json)。
