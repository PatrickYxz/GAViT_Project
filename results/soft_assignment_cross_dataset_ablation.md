# 可学习软分配的双数据集消融与局限

2026-10-12资源收尾：NWPU43三模型与BigEarth soft43本批训练/test/资源全部归档。BigEarth soft43最佳权重在第8轮，val mAP78.99004691745016%；整批6小时22分21.73秒/约10.64元，两批正式队列合计约12.32元，准备/短测/空闲另计。[完整记录](bigearth_soft43_ablation.md)。无需继续回传同批日志或重跑；最终三seed和公平效率/外部方法缺口保留。

2026-10-12更新：BigEarth soft43正式30轮/test已完成，mAP72.867571%，较旧Swin43约+1.7676pp、fixed43约+1.1676pp。BigEarth共同43/45中soft相对fixed两次mAP/macro-F1升，相对Swin一升一降；[双seed新表](bigearth_three_model_two_seed_summary.md)和[43逐类表](bigearth_seed43_three_model_per_class.csv)已归档。旧43只有打印精度且fixed保存21轮/代码硬件不同；n=2有局限。NWPU42/43 soft均低于Swin，没有跨数据集一致优势。以下10-01/10-11单seed阶段快照保留为历史，当前两数据集三模型均有两seed。

2026-10-11更新：NWPU三模型现有seed42/43两次正式test，[双seed表](nwpu_two_seed_summary.md)显示固定相对Swin平均+0.039683pp、1升1降；soft平均−0.146825pp、两次均降。42为2080Ti、43为4090，描述统计不能把波动仅归因seed或GPU。BigEarth soft仍只有seed45正式结果；43预检PASS、训练/test待完成。[当前机器可读索引](soft_assignment_cross_dataset_update_20261011.json)。下方及旧summary.json为10-01单seed历史快照，n=1文字不代表最新NWPU重复范围；历史数值与分析保留。

2026-10-03补充：[NWPU冻结权重反馈幅度诊断](nwpu_feedback_sensitivity_seed42.md)已完成。全1260val上，固定/软两模型的α=0/0.25/0.5/0.75均比原α=1降低OA/macro-F1并提高CE；关闭反馈各净少对9张。当前结果未支持以简单整体缩小反馈修复差距，但不等于独立训练的图模块有效性证明，也不改变下方正式test表或定位唯一根因。该幅度假设收尾，不自动扩搜索或新增训练。

2026-10-01。当前版本在两个数据集各完成一个seed的三模型正式测试：NWPU为seed42，BigEarthNet为seed45。结论是**分组与预测发生变化，但尚未建立稳定的整体收益**。soft在NWPU未超过固定分组或Swin；在BigEarthNet相对固定分组改善mAP与macro-F1，但micro-F1下降，三个指标仍低于Swin。结果不支持将本版写成“稳定提升”的最终模型，也不能据单seed证明软分配方法整体无效。

本文整合已有产物，不新增训练或测试，不以跨数据集平均数代替重复实验。数值见[机器可读主表](soft_assignment_cross_dataset_summary.json)，来源哈希、配对边际和算术检查见[核验记录](evidence/soft_assignment_synthesis_20261001/validation.json)。历史约96%的另一NWPU协议不进入本表；AID已有固定分组实验，但没有本版soft结果，不作为软分配消融。

## 1. 正式测试表及协议

### NWPU-RESISC45：seed42，主指标OA

| 模型 | 主指标 % | macro-F1 % | micro-F1 % | 参数量 M |
|---|---:|---:|---:|---:|
| Swin-T | 93.932540 | 93.909145 | — | 27.555495 |
| 固定4×4 GAViT | 94.103175 | 94.089933 | — | 31.386920 |
| 局部原型软分配 GAViT | 93.869048 | 93.854987 | — | 31.486760 |

20%训练池、80%测试。先用5040/1260训练/验证选轮，随后从共同ImageNet预训练独立refit全部6300张训练池，分别训练29/29/28轮，评估25200张test。相同选轮规则并不意味着三模型轮数相同；Cosine周期保持30，不使用test选轮。原先校准soft对fixed的val OA差为+0.714286 pp，正式refit/test为−0.234127 pp；这不是同一权重从val转到test，不能仅据排序变化认定过拟合。[完整协议与测试证据](nwpu_seed42_test_comparison.md)。

### BigEarthNet-19：seed45，主指标19类macro-mAP

| 模型 | 主指标 % | macro-F1 % | micro-F1 % | 参数量 M |
|---|---:|---:|---:|---:|
| Swin-T | 71.676930 | 65.084212 | 76.732264 | 27.535501 |
| 固定4×4 GAViT | 70.258236 | 64.295477 | 76.702457 | 31.366926 |
| 局部原型软分配 GAViT | 70.836919 | 64.864823 | 76.383354 | 31.466766 |

沿用237871/122342/119825的train/val/test划分，三个模型各30轮，由验证mAP选择best后完整test；F1阈值0.5。软分配复用已有完整Swin/固定seed45对照，选择依据为完成与可追溯性。旧对照代码deb2b45、新版ed480e6；split、预训练、主要配方一致，RTX2080Ti型号一致但物理GPU UUID不同，不能称完全相同物理环境。[完整对照与资源记录](bigearth_soft45_ablation.md)。

两组均固定ImageNet预训练、RGB224、batch32、AdamW lr3e-4/wd1e-4；NWPU是CE与校准/refit，BigEarthNet是BCE与验证best测试。只有数据集内部作模型差值；OA与mAP不混为同一个指标，seed42与45也不构成同一任务的两次重复。n=1，不填写标准差、置信区间或显著性结论。两套实验均为FP32无AMP；NWPU关闭两项TF32，BigEarthNet保留cudnn_tf32=true、matmul_tf32=false。

| 比较 | NWPU ΔOA pp | NWPU Δmacro-F1 pp | BigEarth ΔmAP pp | BigEarth Δmacro-F1 pp | BigEarth Δmicro-F1 pp |
|---|---:|---:|---:|---:|---:|
| soft − fixed | -0.234127 | -0.234946 | +0.578683 | +0.569346 | -0.319103 |
| soft − Swin | -0.063492 | -0.054158 | -0.840011 | -0.219389 | -0.348910 |

## 2. 已经能解释的现象

**收益与损失抵消。** NWPU正式test上，soft纠正538个fixed错误，同时新增597个错误，净少对59张。工业区、稀疏住宅等有改善，密集住宅、火车站、教堂等有退步；不是所有类别一起退步，也没有“只对人工地物有效”的统一证据。[配对、逐类与混淆分析](nwpu_seed42_closeout.md)。这些是独立训练模型的配对描述，不是同一权重的模块因果干预。

**BigEarth的净增集中。** 相对fixed，19类AP为12升7降。海岸湿地AP提高15.710764 pp，对macro-mAP贡献+0.826882 pp；其余18类合计贡献-0.248200 pp，最后净增+0.578683 pp。草地/稀疏植被下降7.489263 pp，沙滩/沙丘/沙地上升3.147280 pp。全部19类保留；这不是删去某类后重新定义主要指标。AP衡量排序、F1还依赖固定阈值，因此mAP上升不要求F1同步；尚无证据把此次micro-F1下降具体归因于阈值失配、某一类别或标签错误。

**模块确实产生了不同的分配。** 已有真实GPU短测验证分配路径梯度、归一化与重载；正式BigEarth固定8val图的初始/best/final快照检查及每图48+32边通过。NWPU真实90例图册的token主归属变化39.09%，有向特征边共有比例34.44%；说明结构有变化，不等于语义质量更好。图册中的3例纠正、0例新增错误属于校准best的预定验证子集，不能覆盖正式test中的597个新增错误。[跨类别审图记录](nwpu_regions_visual_review_20260930.md)。

**目前没有全局塌缩证据。** NWPU校准8张固定验证探针的投影原型平均余弦相似度0.33662→0.78264、分配熵0.67858→1.48263，提示原型趋同与分配分散；但不是768维区域特征的相似度，样本也不是45类分层抽样。低范数原型为0、区域质量非零；图册里某区域没有argmax赢家也不等于该区域没有软质量。不能写成“已证实过度平滑/区域塌缩”。[探针范围](nwpu_seed42_cached_diagnosis.md)。

## 3. 结构限制与原因假设

以下前两项是代码事实；它们可能限制收益，但尚不是已证实的误差根因。

**内容自适应仍受锚点限制。** 本版输入49个末级token、16个区域；原型从固定4×4区域内注意力汇总生成，最终归属仅允许邻近锚点候选，温度固定0.1、只计算一次。48条空间边仍表示原锚点邻接，区域变化后不随真实质心重建。它不是任意形状分割器，7×7可视化格也不是独立感受野。现有图像尚不足以证明它持续沿地物形成连贯区域。

**反馈后立即全局平均。** 冻结ed480e6源码与本地文件逐字节一致：[分组实现](../models/soft_region_grouping.py)、[反馈与分类](../models/gavit.py)。令token为x_i，区域反馈为F_k，软归属为S_ik，N=49。分类器之前的池化表示满足：

```text
x'_i = x_i + sum_k S_ik F_k
z = (1/N) sum_i x'_i
  = mean_i(x_i) + sum_k (m_k/N) F_k,  m_k = sum_i S_ik
```

所以在给定x和F时，反馈映射传给分类器的是各区域总质量加权的反馈；它后面没有再进行token间交互。这说明“反馈到token”不自动等于新增空间细化过程。本地合成数值核对等式误差小于1e-12，仅为代数检查。**不能推出软分配无作用**：S还改变前面的W、区域特征、离散特征边和F，不能固定这些量后把结论推广到完整模型。

| 待检验解释 | 现有线索 | 尚缺证据 |
|---|---|---|
| 分组自由度不足以改善对象级结构 | 局部候选、7×7表示、固定锚点边；审图未见一致完整区域 | 分组质量的合理评价目标与覆盖样本；分类模型不应被要求等同语义分割 |
| 区域趋同或软汇总稀释了部分判别信息 | NWPU小探针原型相似度/熵上升，部分类别退步 | 真实W/区域特征及与错误的关联；当前不能认定过度平滑 |
| 反馈接口压缩了细粒度分配差异 | 上述平均池化恒等式 | 保持其他因素可比的独立机制检验；现有实验未单独分离此影响 |
| 泛化或训练随机性影响小差距 | 两数据集各只有一个soft seed；BigEarth best→final val mAP下降2.474446 pp | 同协议重复训练或预定机制对照；不能据片段认定错误标签、数据偏移或优化器是根因 |

上述解释保留为讨论，不据此立即改温度、加门控/正则、换层级或试新seed。本次三模型比较改变的是整套软分组机制（原型投影、区域汇总和软反馈），未单独拆分内容依赖、软化和新增参数的作用。

## 4. 复杂度与论文主张

软模块相对fixed增加99,840参数，两个数据集均约0.318%；相对Swin增加约3.93M参数。参数增量小不等于推理代价可忽略。BigEarth soft实际总队列14小时44分19秒，训练allocated峰值3.660 GiB、reserved峰值4.029 GiB；完整阶段包含验证和IO，不能作为纯模型速度或与不同物理GPU的严格效率对照。NWPU refit轮数也不完全相同，不能直接用总训练时长宣称soft更快。新结构同设备推理效率仍待独立验证。

当前可写：已实现可微、内容依赖的局部原型分组，并完成双数据集各一个seed的正式消融与限定案例分析；收益依类别与数据集而异，额外复杂度尚未得到稳定精度收益的支持。当前不可写：稳定超越Swin、三指标全面改善、学到一致语义分割、已定位唯一失败根因，或导师全部实验要求完成。

## 5. 论文正文草稿

**中文。** 我们在NWPU-RESISC45与BigEarthNet-19上比较了Swin-T、固定4×4区域GAViT和局部原型软分配GAViT。NWPU的seed42实验中，软分配相对固定区域的OA下降0.234个百分点；配对统计表明，其纠正538个错误的同时新增597个错误。BigEarthNet的seed45实验中，软分配相对固定区域的mAP与macro-F1分别提高0.579和0.569个百分点，但micro-F1下降0.319个百分点，且三项指标仍低于Swin-T。逐类分解显示，mAP净增主要来自海岸湿地，而其他类别的收益与损失相互抵消。可视化确认软分配改变了区域归属及特征边，却尚未提供一致语义区域改善的证据。因此，当前结果支持该机制具有可学习性和条件性影响，尚不支持稳定的整体性能优势。两数据集各只有一个训练seed，跨seed变异尚未估计。

**English.** We compared Swin-T, fixed-grid GAViT, and GAViT with local-prototype soft grouping on NWPU-RESISC45 and BigEarthNet-19. In the seed-42 NWPU experiment, soft grouping reduced overall accuracy by 0.234 percentage points relative to fixed grouping, correcting 538 errors while introducing 597 new errors. In the seed-45 BigEarthNet experiment, it improved mAP and macro-F1 by 0.579 and 0.569 percentage points, respectively, but reduced micro-F1 by 0.319 percentage points; all three metrics remained below Swin-T. Class-wise decomposition showed that the net mAP gain was primarily attributable to coastal wetlands, with mixed changes across the remaining classes. Qualitative inspection confirmed changes in assignments and feature edges but did not establish consistently more coherent semantic regions. These results show a trainable mechanism with conditional effects, rather than a demonstrated stable overall advantage. Each dataset was evaluated with one training seed for this ablation, so variability across seeds remains unquantified.

表注：数值均为正式test百分比；差值为百分点（pp）。NWPU按验证选轮后独立refit，BigEarthNet使用最佳验证checkpoint。每个数据集n=1，未报告跨seed SD。不同数据集的主指标和训练协议分别说明，未合并计算平均表现。

## 6. 当前收尾与后续顺序

本轮已完成数值整合、原因证据分级和中英文正文；无需再解压数据、读取同批日志或新开GPU。保留本版及所有正负结果，旧结果表不被替换。

下一步优先将这份结果与已有区域/图边图稿纳入实验章节，并把局限写入讨论。外部方法HGTNet/DFAGCN仍按既有请求等待用户提供作者回复/材料；这里只引用09-25邮件已发送的记录，没有重新搜索或宣称当前公开代码一定缺失。拿到材料后先核对模型身份、预训练与协议，再安排低成本接入验证。导师要求的多seed、同协议外部方法对比及新结构效率仍未完成；需要另行固定范围、研究问题和预算，不能将本次单seed归档当作取消这些要求，也不在本轮自动启动额外训练。
