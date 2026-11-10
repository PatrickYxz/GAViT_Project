# 相关工作：视觉表征、图推理与区域分组

2026-10-06中文初稿。按导师要求覆盖遥感视觉Transformer、图神经网络用于图像理解、区域分组三条主线，并将文献机制与本项目实际实现分开。文中R1–R12对应末尾参考文献和[可导入的BibTeX](thesis_related_work_20261006.bib)。本稿讨论相关性与差异，不把原论文成绩当作本项目统一协议下的复现实测。

## 1. 从视觉Transformer到遥感场景表示

视觉Transformer将图像转化为token序列，通过注意力学习token之间的关系。Dosovitskiy等人的ViT表明，将图像patch嵌入后输入Transformer，可以在预训练与迁移框架下进行图像分类。其基本单元来自规则切分，而不是先验给定的地物区域；学习到的表征与输入切分单位应加以区分。[R1：ViT](https://arxiv.org/abs/2010.11929)

Swin Transformer采用层次化表示，并结合窗口内注意力与移位窗口实现跨窗口连接。这为不同尺度的信息交互提供了结构基础，同时约束单层注意力的计算范围。[R2：Swin](https://arxiv.org/abs/2103.14030) 因此，在Swin之上增加图模块时，不能将研究动机写成“骨干无法建模区域间关系”。合理的问题是：在已有上下文表征上，显式汇总区域并施加另一种关系结构，是否能够补充有用的归纳偏置，以及这种补充是否值得额外复杂度。

在遥感场景分类中，Bazi等人采用不重叠patch的嵌入序列、Transformer encoder与分类头，使用多头注意力整合图像信息。该工作说明了Transformer用于遥感场景表示的可行路线，但它不等同于显式学习地物边界或构建区域图。[R3：遥感ViT](https://www.mdpi.com/2072-4292/13/3/516) 本项目进一步关注表征的组织方式：保留预训练Swin-T作为骨干，在其末级49个token上构建16个区域节点，再进行关系推理。这样能够与同骨干基线比较，但增加模块本身并不构成性能收益的证据。

HGTNet是更直接相关的图增强Transformer方法。Li等人将三阶段分层Transformer、通道与空间双重注意力以及ViG图特征提取结合，并通过跳跃连接融合视觉与结构信息。[R4：HGTNet](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification) 它与GAViT共享“视觉表征与显式图关系结合”的研究目标，但模块位置、分层组织及节点表示不同。GAViT是在末级Swin token上增加区域汇总、稀疏区域图和反馈；HGTNet不能简化为“Swin后加一个GAT”。两者的比较属于完整方法比较，不能仅把分数差归因于某一个图模块。

HGTNet原文III-C与IV-B分别出现max-relative graph convolution和average aggregation表述，具体复现算子仍需材料核对。相关工作在此采用“ViG图特征提取”的已确认描述，不擅自消除原文歧义；文中的细粒度或pixel-level表述也不自动意味着原始输入逐像素构图。

## 2. 图神经网络在图像理解中的作用

### 2.1 节点、邻接与聚合是三个不同的设计选择

将图像表示为图至少需要回答三个问题：节点表示什么，哪些节点相连，以及邻居信息如何汇总。GAT通过邻域内的注意力计算，为不同邻居分配不同贡献权重，提供了可学习的关系聚合机制。[R5：GAT](https://arxiv.org/abs/1710.10903) 这种权重学习与边集合的构造并不是同一件事。即使聚合权重可学习，预先指定的邻接仍可能限制信息来源；反之，内容依赖的近邻选择也不等同于学习了正确的语义关系。

GAViT当前构图包含4×4锚点的48条有向空间邻接边，以及每个查询节点从非自身、非空间邻居中选出的两条特征边，共32条。该80条指构图阶段的显式非自环边；空间边保留锚点先验，特征边依赖区域特征。余弦相似度仅用于选择拓扑，代码返回的相似度权重并未传入GAT参与边权调制。因此，图册中画出的边既不是相似度加权消息的直接可视化，也不是GAT注意力强度。[本项目构图](../models/graph_construction.py)、[图推理](../models/graph_reasoning.py)

### 2.2 视觉图骨干与区域图补充分支

Han等人的Vision GNN（ViG）将patch作为节点，基于近邻建立图，用Grapher模块进行信息聚合，再用FFN变换节点特征，形成用于视觉任务的网络骨干。其FFN与特征变换用于缓解作者所研究的特征同质化问题。[R6：ViG](https://arxiv.org/html/2206.00272v3) GAViT采用已有Swin表征上的低节点数区域图，属于不同的结构位置与计算分配，不能视为ViG的同配置复现。

这一联系也有助于限定本项目FFN实验的解释：本项目尝试的是反馈后、平均池化前的轻量逐token残差FFN，隐藏维度192、末层零初始化，并设置相同Swin+FFN对照。它与ViG骨干中的Grapher/FFN堆叠、归一化和宽度不同。本项目一次校准未改善，并不能推翻ViG的原结论；反过来，ViG中的积极结果也不保证同名FFN在本任务中有效。[本项目FFN结果](../results/nwpu_ffn_calibration42_comparison.md)

### 2.3 遥感图特征聚合与完整方法比较

Xu等人的DFAGCN使用预训练VGG16提取多层卷积特征与全连接特征，通过GCN刻画patch之间的关系，再将图聚合特征与其他特征加权拼接，最终用LIBLINEAR分类。其流程涉及不同输入尺度和分阶段特征处理，不能概括为普通端到端CNN–GCN分类器。[R7：DFAGCN](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)

DFAGCN与GAViT都试图通过关系聚合补充局部表示，但二者的骨干、节点来源、融合位置和训练链不同。GAViT由Swin末级token汇总区域，在区域图上推理后将结果送回token；DFAGCN则围绕多层CNN特征聚合与最终特征融合组织分类。这样的差异具有比较价值，也意味着外部方法的训练、预训练和分类器必须完整保留并明确记录。只复用一个GCN层再接本项目分类头，不足以称为DFAGCN复现。

HGTNet与DFAGCN都需要作为相关完整方法进行实验比较。共享名义训练比例不等于共享划分、预训练或模型选择过程，原论文结果应与本项目重跑值分开。截至本稿，这两项统一协议下的实测比较尚未完成；文献分析不能代替实验结果。[材料状态与接入条件](external_methods_readiness_20260924.md)

## 3. 区域分组：固定划分、聚类与可学习归属

### 3.1 固定网格和区域内注意力

固定网格以空间位置决定成员关系，结构确定、计算简单，也为区域编号及邻接提供直接含义。其限制在于区域边界不随图像内容变化。GAViT的固定版本在每个既定区域内学习token的注意力权重，因此区域表示可学习，但成员集合仍固定。将它称作“注意力分组”时，需要明确学习的是区域内的汇总权重，不能据此宣称边界已经自适应。[固定分组代码](../models/region_grouping.py)

固定划分是否足够取决于任务，不应仅凭形状规则就认定它较差。对于已有上下文的末级token，规则显示格也不是严格独立的感受野。本项目因此保留固定4×4版本作为对照，以实际分类结果检验增加分组自由度的价值，而不是预先认定软分配更适合。

### 3.2 k-means式分组与局部超像素

k-means式方法交替进行成员分配和中心更新，以特征相似性组织区域。本项目历史`KMeansGrouping`使用余弦相似性、硬argmax和均值更新；分配过程被显式detach，区域均值汇总仍能把梯度传回已选token。这个实现与通常的欧氏距离k-means并不完全相同，也不能把“硬分配不反传”扩大为整个模型都不可训练。[历史聚类实现](../models/region_grouping.py)

SLIC是图像分组中的一个相关例子：通过颜色和空间信息上的局部聚类形成超像素，使区域具有局部性，并在边界贴合与紧凑性之间权衡。[R8：SLIC](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/) 它说明空间先验与内容相似性可以结合，但低层像素超像素与Swin末级token区域具有不同含义。超像素边界表现不能直接转换为场景分类收益，更不等同于对象或语义类别的分割。

### 3.3 可学习软分配的不同路线

Jampani等人的Superpixel Sampling Networks（SSN）将可微聚类与特征学习结合，让超像素形成过程能够服务于下游目标。它在局部候选内形成软关联，并通过关联更新区域表示。[R9：SSN](https://varunjampani.github.io/ssn/) 这一思路为“内容改变归属且保留空间限制”提供了直接参照。不过，本项目不运行像素级SSN，而是在已有Swin特征上用投影后的当前图像区域原型计算一次局部分配；距离形式、汇总方式、分辨率与后续任务均不同。

Slot Attention通过竞争式注意力，把输入特征分配给一组slots并迭代更新slots，以形成可交换的对象表示。[R10：Slot Attention](https://arxiv.org/abs/2006.15055) 其slot维度归一化与随后按输入汇总的计算，提示需要区分“每个输入属于哪个组”与“一个组如何汇总输入”。GAViT同样区分归属矩阵S与汇总矩阵W，但不使用完整Slot Attention迭代或其slot更新过程；本项目区域还绑定空间锚点，而不是可任意交换的无位置编号slots。

GroupViT使用group tokens和分层分组，并借助图像—文本监督形成语义分组，为从视觉token学习分组提供另一种路线。[R11：GroupViT](https://arxiv.org/abs/2202.11094) 其官方实现包含Gumbel和直通硬分配机制，不能把“可训练的分配”一概写成“前向始终保留全软归属”。本项目使用图像分类标签下的软归属，不使用文本监督或该分层合并结构；因此不会把GroupViT的语义分割能力当作本项目区域语义质量的证据。[固定版本源码核对](soft_assignment_github_review_20260926.md)

STViT与本项目的数据流更相近：从规则网格初始化super tokens，在附近候选间计算关联，汇总后进行super-token注意力，再将信息映射回原token。[R12：STViT](https://arxiv.org/abs/2211.11167) 这一先汇总、再交互、再回映的组织方式已有先例，不能作为GAViT未经限定的首创主张。二者的关键区别在于，本项目只处理末级49→16的局部原型汇总，区域交互使用固定预算的稀疏GAT，而不是STViT的super-token自注意力。此前锁定版本的源码核对还显示，STViT关联计算位于no-grad路径；GAViT的S计算保留梯度。这是具体实现的优化路径差异，并不证明哪种设计效果更好。[STViT固定提交及代码范围](soft_assignment_github_review_20260926.md)

### 3.4 本项目软分配的实际范围

当前`soft_prototype`从固定区域内的注意力权重形成每张图像的初始原型，用共享LayerNorm和128维线性投影后的归一化特征计算余弦相似性，再在邻近锚点候选上按温度0.1得到S。原型随输入变化，它们不是直接学习的一组独立持久slot向量。S沿区域维度归一化，控制token的软归属和反馈；W结合token注意力与log S沿token维度归一化，用于区域表示汇总。[软分配代码](../models/soft_region_grouping.py)

分配允许一个token贡献给多个邻近区域，但没有取消位置约束。48条空间边仍表示原4×4锚点的邻接，不按软区域真实质心或接壤关系更新。分配路径可学习也不代表所有离散操作均可微：区域特征top-2选边仍是离散选择。因此，本稿采用“局部原型软分组与稀疏区域图反馈”的具体表述，不称其为完全自由的区域分割或全可微图拓扑学习。

## 4. 相关设计与GAViT的定位

| 方法/路线 | 表示与关系组织 | 与GAViT相关的启示 | 不能直接等同的部分 |
|---|---|---|---|
| ViT / Swin（R1–R2） | patch/token注意力；Swin提供层次与跨窗口连接 | 骨干已具上下文交互能力 | 新增区域图收益须由对照验证 |
| HGTNet（R4） | 分层Transformer、双重注意力与ViG结构融合 | 遥感图增强Transformer的直接相关方法 | 完整多模块结构不等于末级Swin+GAT |
| GAT / ViG（R5–R6） | 邻域注意力聚合；图像patch图骨干 | 分清边选择、消息权重和节点内变换 | ViG的FFN结果不保证本项目FFN有效 |
| DFAGCN（R7） | 多层CNN特征、图聚合、加权融合和线性分类 | 图关系可以补充多层视觉特征 | 分阶段完整流程不能替换成一个GCN层 |
| SLIC / SSN（R8–R9） | 空间局部聚类；SSN引入可学习特征与可微分组 | 内容适应与位置约束可以共存 | 像素超像素质量不等于末级token分类收益 |
| Slot Attention（R10） | 竞争归属与迭代slot更新 | 归属与区域汇总的归一化需区分 | 无固定空间身份的slots不能直接沿用4×4邻接 |
| GroupViT（R11） | 文本监督下的分层视觉分组 | 分组可作为表征学习的一部分 | 监督来源、直通硬分配与层次不同 |
| STViT（R12） | 局部关联、super-token注意力及回映 | 汇总—交互—回映具有明确先例 | 区域交互算子、所在层级及关联梯度路径不同 |
| 本项目固定/软GAViT | Swin末级区域汇总、48+32构造边、GAT与token反馈 | 通过共享骨干与协议检验具体分组机制 | 当前结果不能建立一般性稳定优势或语义分割结论 |

GAViT的研究定位应由可核对的设计和证据界定。首先，固定与软分组共享骨干和区域图接口，能够检验更灵活的归属机制在这一具体实现中是否有益；但软版本同时改变投影、区域汇总和反馈映射，尚未独立分离每一项作用。其次，稀疏区域图将空间先验和特征近邻结合，提供明确的构造结构；这种可显示结构不等于已证实语义合理或因果可解释。最后，多数据集对照、效率及负面诊断约束了可以提出的性能主张。

据已归档实验，当前版本尚未建立稳定超越Swin的收益，因此本章不以“已有方法不能处理某问题，而GAViT已经解决”作为论证。更准确的表述是：已有研究提供了注意力、显式图关系和可学习分组等不同组织视觉信息的方式；本项目在遥感分类中实现并检验它们的一种受约束组合，报告其收益条件、计算代价和未得到支持的假设。[实验与讨论](thesis_experiments_discussion_20261006.md)

## 参考文献

以下按正文首次使用编号；期刊卷期年与预印本年不混用。尚未核实的字段未补猜。来源核对范围、代码定位与引用键见[来源索引](../results/evidence/related_work_20261006/sources.json)。

- R1. Dosovitskiy, A., et al. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.** ICLR, 2021. [作者论文](https://arxiv.org/abs/2010.11929)；BibTeX键：`dosovitskiy2021vit`。
- R2. Liu, Z., et al. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.** ICCV, 2021. [作者论文](https://arxiv.org/abs/2103.14030)；键：`liu2021swin`。
- R3. Bazi, Y., Bashmal, L., Al Rahhal, M. M., Al Dayil, R., and Al Ajlan, N. **Vision Transformers for Remote Sensing Image Classification.** Remote Sensing, 13(3):516, 2021. [出版社原文](https://www.mdpi.com/2072-4292/13/3/516)，DOI：10.3390/rs13030516；键：`bazi2021remotevit`。
- R4. Li, Z., et al. **A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification.** IEEE JSTARS, 17:20315–20330, 2024. [DOI](https://doi.org/10.1109/JSTARS.2024.3491335)；键：`li2024hgtnet`。
- R5. Veličković, P., et al. **Graph Attention Networks.** ICLR, 2018. [作者论文](https://arxiv.org/abs/1710.10903)；键：`velickovic2018gat`。
- R6. Han, K., Wang, Y., Guo, J., Tang, Y., and Wu, E. **Vision GNN: An Image is Worth Graph of Nodes.** NeurIPS, 2022. [作者论文](https://arxiv.org/abs/2206.00272)；键：`han2022vig`。
- R7. Xu, K., Huang, H., Deng, P., and Li, Y. **Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing.** IEEE TNNLS, 33(10):5751–5765, 2022（2021在线发表）. [DOI](https://doi.org/10.1109/TNNLS.2021.3071369)；键：`xu2022dfagcn`。
- R8. Achanta, R., et al. **SLIC Superpixels Compared to State-of-the-Art Superpixel Methods.** IEEE TPAMI, 34(11):2274–2282, 2012. [作者项目页](https://www.epfl.ch/labs/ivrl/research/slic-superpixels/)；键：`achanta2012slic`。
- R9. Jampani, V., Sun, D., Liu, M.-Y., Yang, M.-H., and Kautz, J. **Superpixel Sampling Networks.** ECCV, 2018. [作者项目页](https://varunjampani.github.io/ssn/)；键：`jampani2018ssn`。
- R10. Locatello, F., et al. **Object-Centric Learning with Slot Attention.** NeurIPS, 2020. [作者论文](https://arxiv.org/abs/2006.15055)；键：`locatello2020slot`。
- R11. Xu, J., et al. **GroupViT: Semantic Segmentation Emerges from Text Supervision.** CVPR, 2022. [作者论文](https://arxiv.org/abs/2202.11094)；键：`xu2022groupvit`。
- R12. Huang, H., Zhou, X., Cao, J., He, R., and Tan, T. **Vision Transformer with Super Token Sampling.** CVPR, 2023. [作者论文](https://arxiv.org/abs/2211.11167)；键：`huang2023stvit`。

## 编辑与验证说明（定稿时移至工作记录）

本稿完成三条主线的初稿覆盖与具体方法差异论证，不声称是穷尽所有最新工作的系统综述或独创性检索。HGTNet/DFAGCN实测、soft多seed、新结构公平效率和3×3/4×4受控比较仍按原任务保留。作者未回复的状态不改变两篇论文已经发表的事实。

本轮没有运行文献代码或新增实验。GAViT机制依据当前本地源码；STViT/GroupViT的特定代码行为依据锁定commit的静态核对，不能外推到所有版本。引用、BibTeX键、链接及当前源码哈希的本地核验见[验证记录](../results/evidence/related_work_20261006/validation.json)。正式论文仍需按学校格式统一参考文献、章节编号，并结合导师反馈修订。
