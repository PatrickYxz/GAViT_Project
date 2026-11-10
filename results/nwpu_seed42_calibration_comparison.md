# NWPU seed42 分组消融：校准验证结果

**09-29正式refit/test已完成：** 六阶段均exit0，test OA为Swin93.932540%、fixed94.103175%、soft93.869048%。[正式测试结果](nwpu_seed42_test_comparison.md)已归档；下文原启动步骤/建议保留为历史，不重复训练。

**09-29只读诊断已返回：** 三模型代码/配方/硬件身份一致，macro-F1与OA排序相同，完整[诊断及后续建议](nwpu_seed42_cached_diagnosis.md)已归档。

2026-09-28 用户终端回传。研究问题：在相同NWPU划分、预训练和预定配方下，局部原型软分配是否改善固定4×4 GAViT及Swin的验证表现。只改变模型/分组方案；这一比较不单独分离投影参数、软汇总、软反馈各自的作用。

## 当前结果表

三阶段均process_exit=0，queue.exit为exit_code=0，全部训练30轮。按最高val OA选轮，再报告该checkpoint的OA与macro-F1；后者不是独立最大化macro-F1的选轮结果。当前仅calibrate验证集，尚无最终20%拟合协议的测试成绩；逐类返回只含差值两侧前五名。

| 模型 | 最佳val OA (%) | 所选checkpoint val macro-F1 (%) | 最佳轮 | 正确数/1260 | 对Swin差值(pp) | 全过程分钟 | allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swin-T | 94.206349 | 94.171552 | 29 | 1187 | +0.000000 | 23.24 | 3.5584/3.8301 |
| 固定4×4 GAViT | 93.015873 | 92.976564 | 29 | 1172 | -1.190476 | 24.26 | 3.6559/3.8770 |
| 局部原型软分配 GAViT | 93.730159 | 93.765721 | 28 | 1181 | -0.476190 | 24.73 | 3.6626/4.0391 |

软分配相对固定网格+0.714286 pp，纠正40/破坏31，净多对9张；相对Swin−0.476190 pp，纠正30/破坏36，净少对6张。同路径配对数量已核对。单seed、跨轮次选择的最佳验证值不能证明稳定或显著收益，也不支持据此断言过拟合、区域语义塌缩或图反馈导致差距。

## 协议与证据范围

- 清单：train5040/val1260/test25200；本阶段只训练5040张，未refit到6300张训练池。重复像素组不跨split。新协议不与旧70/15/15表中的约96%合并比较。
- 配方：seed42、30epochs、batch32、workers2、AdamW lr3e-4/weight_decay1e-4、Cosine T_max30、CrossEntropyLoss、RGB224和共同增强、FP32无AMP、双TF32关闭。完整命令已在[启动记录](../docs/run_nwpu_ablation42_20260928.md)启动前固定，发布源码ed480e60fea03ed786fdba70c144de5e815c34f5。09-29返回共同training/code/hardware相等，ed480e6干净，并展开了训练配方及RTX2080Ti、torch2.2.2、CUDA12.1信息。原始完整源码哈希字典未回传，只返回其相等检查。
- 返回architecture确认K16、GAT hidden256/4heads/2layers、sparse_hybrid和token_feedback共同配置；soft为128维共享投影、局部Chebyshev半径1、温度0.1、一次分配。固定/soft真实前向均48空间+32特征边。
- 重载最大logit差：Swin0，fixed/soft均3.5762786865234375e-7，低于入口阈值1e-5。仅确认报告/入口检查，不替代直接下载权重的独立验证。
- 队列4346.86884246秒，即72分26.87秒；包括图像扫描、训练、验证、存盘、重建、soft探针。不能当推理速度，也不能把全部耗时差归因于分组计算。三模型参数量27,555,495/31,386,920/31,486,760，纯训练总秒数901.44/955.07/1001.80，实际费用尚未提供。

| 变体 | 报告的best.pth SHA256 |
|---|---|
| swin | `bbfd4e472739126276e79014f944f911d7b0abbc6f863bb291d799845b9cd0ea` |
| fixed | `1766fe7c330eb37fb350f0348bcd255bafc451b793411a927548ee005a140f67` |
| soft | `674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12` |

共同manifest SHA256：`ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e`。共同pretrained SHA256：`f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36`。

服务器结果根目录：`/home/featurize/work/nwpu_ablation42_20260928_01`；各子目录`{swin,fixed,soft}`中的run.json/history.json/best.pth/validation_*.json或csv是后续依据。用户返回字段完整转录于[结构化证据](evidence/nwpu_calibration42_20260928/server_return.json)，[数值核对](evidence/nwpu_calibration42_20260928/server_return_validation.json)保留未确认事项。没有直接获取服务器完整日志或权重。

## 下一步决定

已有validation_metrics.json、validation_predictions.csv、history.json及soft探针的只读诊断已完成，不重复执行。返还内容及数值复算见[诊断记录](nwpu_seed42_cached_diagnosis.md)。没有用测试集调参。

基于本次相对fixed的收益和三模型消融价值，建议冻结配置，按既定协议做seed42三模型refit29/29/28轮及test，预计同卡1–1.5小时，尚未启动。不改变温度/阈值/划分或根据test挑seed，暂不扩多seed。E05仍是单seed校准与诊断阶段完成，双数据集正式消融和稳定性评估未完成；保留本次负面及正面相对变化。历史comparison_table.csv/ablation_study.csv保持，当前新协议消融表以本页为准。
