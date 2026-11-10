# 论文结构与图表布局只读审查 — 2026-11-08

审查范围：当前 `thesis/ntu_gavit/chapters/03-method.tex`、`04-experimental-setup.tex`、`05-external-comparisons.tex`，已有内部实验/有界候选说明和BEN四例图。未重新计算另两agent负责的主表数值；未改论文、代码、旧表、raw或图像，没有网络/SSH/GPU/前向。页数为排版预算，需最终编译验证。

## 推荐结构与总页数

围绕三个读者问题组织：**相对共享Swin是否有稳定收益 → 哪些结构比较/诊断支持或不支持收益解释 → 代价与结论**。把现有8页外部补充章并入结果章，避免外部方法先于内部主结果占据中心。

| 部分 | 目标页 | 内容与布局 |
|---|---:|---|
| 前置页/目录 | 4 | 先沿用现有预算；正式前置页如增加，使用末尾预留 |
| 1 Introduction | 4 | 研究问题、受支持贡献、论文路线 |
| 2 Literature Review | 11 | 本轮可保持，无需靠新引用/删字号挤页 |
| 3 Method | 6 | 当前固定/soft、稀疏图、反馈定义保留 |
| 4 Experimental Setup | 5 | 从7页压缩，见下节 |
| 5 Results and Efficiency | 7 | 内部主结果约3页；三项外部单seed约2页；真实AID配对/成本约2页；不重复内部表 |
| 6 Architecture Comparisons and Diagnostics | 7 | fixed/soft及单seed拓扑/K比较约2页；冻结反馈/FFN/空间先验约2页；BEN四例完整主图2页；范围与解释约1页 |
| 7 Discussion | 3 | 条件性收益、变化与语义/因果区分、复杂度和局限；不重抄数字 |
| 8 Conclusion | 1 | 回答研究问题、已做/未做、有限后续方向 |
| References | 7 | 保持现有 |
| Appendix: complete AID diagnostics | 2 | 两幅已验收完整图各一页；必须计入总页数 |
| **合计** | **57** | 预留3页容纳正式前置页/目录增长、浮动体和结论溢出；不能把附录移出统计来“达标” |

若两幅AID图留在结果正文，则结果章变9页、附录为0页，合计仍57页。全文不能通过更小字体、页边距、行距达标。新增每章带来的目录增长也占3页预留，不应把7页结果写成7页纯文字后再追加浮动图。

## Ch4：7页压缩至4–5页的具体位置

当前约8个section、多个短subsection和4组指标/统计公式，内容正确但边界反复出现。建议合成4节：

1. **Questions, datasets and partitions（约1.3页）**：研究问题缩成一段；保留当前partition表与三数据集来源。精简逐数据集的重复文字。必须保留固定partition seed与training seed不同、NWPU/AID重复组在同split、BEN来自已有metadata分区/可用RGB而非新70/15/15，以及RGB三波段/非全多光谱范围。
2. **Training and model selection（约1.2页）**：一张紧凑表列输入/增强、优化器、损失、预算、选轮、refit/不refit。共享ImageNet归一化只写一次。单标签fresh refit/Tmax=30与BEN最佳验证checkpoint/阈值0.5是正文重点；AdamW默认betas/epsilon等可放表注或已有复现记录，不应重复占段。
3. **Metrics and repetitions（约1.1页）**：保留OA、macro-F1、mAP、micro-F1和样本SD定义；可把OA写为行内定义，F1与mAP公式组合呈现。保留absent/zero denominator约定、百分比与pp区别、实际seed组由结果表声明。不要在每条公式后再重复一次任务适用范围。
4. **Efficiency, visualisation and evidence scope（约0.9页）**：同卡/驻留合成输入/20 warm-up和5×50计时、allocated/reserved范围写成一个紧凑段；归档/重现层级用一段统一声明。视觉边界（membership、pooling、构造边、GAT attention不等同）由Ch3定义后只交叉引用。

可压缩的重复：Ch4开头和末尾反复说明外部单seed非统一配方；repetition末3段与reproducibility末段反复说明不能等效/显著/seed纯方差；qualitative段与Ch3/未来图注重复同一注意力解释。**不能删掉**校准validation与fresh-refit test不是同一权重，因此排名反转不等于同checkpoint过拟合；历史条件不同/中断不能冒充严格重复；source选项不等于正式实验已完成。

## Ch5外部8页：合并后约3页文字/表 + 2页完整图

来源：`thesis/ntu_gavit/chapters/05-external-comparisons.tex`。

- 第18–26行split计数/选轮/refit与Ch4重复，换成一次表号引用。第6–16、35–38、42–46、79–96、131–145、187–189、289–299行多次重申“单seed、异配方、非官方、不能因果”，集中在开头一段+配方表caption+Discussion一段。不是删范围，而是让同一限制只在读者需要处出现。
- 现配方表保留真正影响比较的backbone/预处理/optimizer/budget/选轮差异；细节改为紧凑表注。DFAGCN staged/frozen VGG与HGTNet partial Swin-B都必须显式保留，不能只标方法名。
- NWPU/AID单seed表合为一个双panel float；内部Swin/GAViT seed42若已在主表出现，可引用主表，仍在外部panel保留读者能直接看出差距的参照。原始证据不够的NWPU“净多对4张”仅aggregate；不可生成配对纠错数。三项真实测量齐备，不扩DFAGCN×AID。
- **真实AID 75 corrected / 76 introduced / 78 shared errors必须保留**；与Swin的108/79可用两行小配对表，保留候选/参照方向。净-1不能替代151个正确性不一致样本。现第181–207行的长解释可缩成表+4句。
- 第231–241行恢复细节不需要半页工程叙事：正文一句说明fresh96 refit与恢复不构成新seed；55,048接受步、重载是在服务器而非本地执行和原始收据可留来源注。attempt2-only墙钟/峰值与全实验未知在效率/限制段出现一次即可。
- 两张完整AID图使用已验收`images/external_aid_20261108/`版本，各占一页；保留100cal+96refit、best96/final100与完整30类CM。不能为省页缩回3–4pt。若放附录，正文只保留选择数和主要混淆的一段，明确图交叉引用。图放附录不会节省总页数。

## 已完成的结构比较与诊断：如何命名而不混淆

| 项目 | 适合论文中的位置/名称 | 必须带的限制与来源 |
|---|---|---|
| Swin / fixed / local-soft完整test | 主结果、architecture comparison | soft同时改变投影、聚合、反馈，不能单独归因“softness”；AID没有soft结果。主表seed资格由数值审查确定 |
| BEN corrected-kNN vs sparse-hybrid seed42 | 单seed topology comparison | `results/bigearth_comparison.csv`；已正式test，但历史代码/设备与打印精度限制必须注，不能把早期方向bug当当前方法消融 |
| BEN K9/K16 seed42 | 历史跨批次region-count comparison + 四例诊断 | `docs/ben_grid3_grid4_seed42_20261020.md`与`docs/ben_gridpair42_visual_analysis_20261020.md`；完整test与真实捕获完成，但非多seed严格同环境K因果实验 |
| NWPU反馈幅度α五档 | Frozen-checkpoint validation sensitivity | `.worktrees/sparse-hybrid-4n-top2/results/nwpu_feedback_sensitivity_seed42.md`；α=0仍保留联合训练的backbone/head，绝不是独立训练no-graph baseline；非test |
| 池化前FFN（Swin/soft四格） | Bounded calibration candidate, negative result | 同工作树`results/nwpu_ffn_calibration42_comparison.md`；两新配置30轮已完成，预设继续条件失败；没有refit/test，不写进正式test主表 |
| soft+anchor Gaussian spatial prior | Bounded calibration candidate, negative result | 同工作树`results/nwpu_spatial_prior_trial.md`；30轮校准/收尾完成，候选关闭；非test/非稳定性证据 |

后3项可用**一张分块表**明确Evaluation split=validation，并保留原比较参照与方向，约1页；不为每个候选安排完整独立曲线页。主表不要使用根目录`results/comparison_table.csv`的早期≈96%或`results/ablation_study.csv`空白计划行。

尚缺或不能宣称已完成：当前主协议下完整baseline→region-only→graph独立训练链；K4/K9/K16系统受控多seed；空间/特征边分别移除与GAT层数/头数系统消融；soft投影/软化/反馈分别隔离；soft公平推理benchmark；外部多seed/统一配方。同一训练网络关闭反馈的诊断、旧日记小节或源码支持不能填这些格子。March历史链条不应与现在20%pool/refit体系拼成完整ablation。

## BEN四例可读插入：全保留，不能直接缩旧页

当前图源：`output/pdf/ben_gridpair42_20261020/graphs_predictions_300dpi.png`与`pooling_300dpi.png`；真实捕获：`results/evidence/ben_gridpair42_visual_return_20261020/server_return/captures.json`及`inputs/row_{97389,14680,28450,4439}.png`。本次已目视检查两张图。

旧导出版面是180mm×约293mm；文字默认6–6.8pt、节点文字还受panel缩放影响。若整张缩到145mm，文字约4.8–5.5pt、图高约236mm，再加caption将超过正文高。因此不能直接`width=\textwidth`插入整张旧页。

建议正文两页，每页2例、每例RGB / K9 / K16三列，使用实际capture重排并保留原图像像素与边，不重新模型前向。最终宽145mm，所有行标题/预测文字8pt，节点字号至少7pt。概率结果放每例下方独立短表，避免将细小文字烙进图内。两页顺序固定：97389+14680；28450+4439。四例包括：

- 97389：pasture分数增加仍未过0.5，没有任何阈值翻转；必须保留这个未解决错误。
- 14680：grassland纠正，同时pasture误报仍在。
- 28450：shrub纠正、marine water新增漏判同时展示，不能只展示有利标签。
- 4439：urban fabric纠正，sand接近但仍低于阈值；不能列为已纠正。

正文小表列全部4个阈值翻转（3纠正+1新增漏判），加上述未解决项；完整76判断保留在已归档机器可读工件，不必把19类×4全部概率占正文。图注明确为四张**预定历史validation**输入、同seed的不同受训模型、箭头neighbor→query；仅画真实特征边（K9 18、K16 32），空间边原记录保留，颜色/节点不是语义标签或GAT attention。4例不估计总体获益率。

pooling页不必再花两页重复相同RGB；正文用4行浓度计数表或1段描述即可，必须排除K16单token区域，不能据“更集中”判断语义质量。若要完整pooling图进论文，需再预留2页或替换别的图，仍计入57–60页。另有历史seed44反馈on/off四例（`.worktrees/sparse-hybrid-4n-top2/output/pdf/seed44_feedback_figure/README.md`），它是另一实验，不可与seed42 K9/K16拼成同一组；本轮页预算建议仅交叉引用，不重复插图。

## Discussion与结论应回答的问题

Discussion用3页连接既有事实：跨seed增减与历史证据层级；不同grouping确实改变分配/边但未建立语义质量或稳定收益；受限候选的负面结果收窄了哪些具体解释；AID同卡实测的额外代价与外部参数更少是不同参照。保留AID负面seed44、BEN44/45和soft不利seed，不能用四seed汇报子集覆盖完整历史。把额外复杂度是否值得作为最终判断，不把“比HGTNet参数少”当相对Swin有价值的证明。

Ch3已有反馈后平均池化的恒等式，可在Discussion引用，避免重新推一整页；它不等于soft无效。小探针原型相似/熵变化、固定边界更细、构造边变化均不能写成已确定塌缩、过平滑、语义连贯或唯一失败原因。结论控制1页，回答条件性结果和未完成外部统一/multiseed/机制隔离，而非宣布导师所有实验要求已完成。
