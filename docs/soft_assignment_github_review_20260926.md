# 软分配GitHub实现核对与GAViT设计建议

检索开始于2026-09-25，整理完成于09-26。响应用户要求，核对5个仓库的11个文件；固定commit及文件路径见[来源索引](../results/evidence/soft_assignment_github_20260925/source_index.json)。这是定向静态阅读，不是完整文献综述或运行验证；未安装外部环境、执行外部代码、下载模型权重或启动训练。

## 1. 先看哪些实现

| 实现及身份 | 核心做法 | 对本项目的用途 |
|---|---|---|
| [STViT，论文作者公开代码](https://github.com/hhb072/STViT) | 局部token与附近super token计算关联，汇总→super token attention→映射回token | **最接近现有数据流**，但关联计算有stop-gradient，不能直接称为可微分配头 |
| [SSN，作者官方代码](https://github.com/NVlabs/ssn_superpixels) | 学习特征，在局部候选内按距离软聚类，更新中心并重建特征 | **分配机制的重要依据**；官方为Caffe/自定义CUDA，实现思想比整库移植更合适 |
| [Slot Attention，Google参考实现](https://github.com/google-research/google-research/tree/master/slot_attention) | slot竞争归一化、加权汇总、GRU迭代更新 | 核对归一化和梯度路径；全局slots没有现成4×4区域身份 |
| [GroupViT，作者官方PyTorch代码](https://github.com/NVlabs/GroupViT) | 学习group tokens，用分组attention分层合并token，支持Gumbel/直通硬分配 | 学习分组的直接参考；不能把其默认硬分配称为普通全软前向 |
| [DiffPool，PyG库实现](https://github.com/pyg-team/pytorch_geometric/blob/79d33965a40b7fa83616a9f598a0f8619f25d939/torch_geometric/nn/dense/diff_pool.py) | 学习归属S，同时计算节点汇总与粗化邻接`SᵀAS` | 回答自由分组后怎样定义图；直接采用会改变当前48+32边协议 |

STViT的论文摘要直接链接该仓库；SSN作者项目页链接NVlabs仓库；Slot Attention README自述reference implementation；GroupViT README自述official implementation。DiffPool此处明确是PyG维护的库实现，不冒充原论文作者仓库。[STViT论文](https://arxiv.org/abs/2211.11167)、[SSN作者项目](https://varunjampani.github.io/ssn/)。

## 2. STViT：最相近，但要看清梯度

固定commit `3a499f0a633ad41ede49f58e98c473476018c9ba`。

`StokenAttention.stoken_forward`先平均汇总初始super tokens，计算token与附近3×3 super tokens的点积关联，在9个候选上softmax；按区域接收的总关联归一化后更新区域特征。区域交互使用attention，再用关联矩阵映射回原token。[源码199–249行](https://github.com/hhb072/STViT/blob/3a499f0a633ad41ede49f58e98c473476018c9ba/models/stvit.py#L199-L249)。

**关键区别：**206行开始的`torch.no_grad()`覆盖关联计算与迭代；227行之后区域汇总仍在梯度路径中。即关联会随输入特征变化，但当前前向不通过关联系数计算路径反传；不能说整个模块不学习，也不能说分配头端到端可微。直接去掉no_grad将改变优化方式和资源开销，须单独验证，不能当作无影响的移植。

另一个适配点：它按规则小块reshape并在不整除时补零；我们是7×7 token对应4×4不等大区域。若直接补到8×8按2×2分块，零token和边界候选会参与计算，需有效性mask；优先保留本项目显式49×16 membership表达。原实现未用我们草案的0.9/0.1先验。

判断：借鉴“局部关联→归一化汇总→区域交互→软映射回去”的组织方式；不直接复制关联梯度策略、padding或完整骨干。

## 3. SSN：按内容相似性软聚类

固定commit `082918950d768f29466148666b7f434c78663dc6`。

`create_net.py`通过特征网络变换含位置/颜色的信息，初始化区域中心；随后重复“计算关联→更新区域特征”。`Passoc`得到负平方距离，再softmax形成局部软关联。区域中心按关联权重平均，重建阶段再将区域特征按关联映射回像素。最终展示的硬标签另走ArgMax。[网络组合与关联](https://github.com/NVlabs/ssn_superpixels/blob/082918950d768f29466148666b7f434c78663dc6/create_net.py#L84-L177)、[特征与初始中心](https://github.com/NVlabs/ssn_superpixels/blob/082918950d768f29466148666b7f434c78663dc6/create_net.py#L203-L230)。

进一步核对了自定义CUDA：距离层处理局部9候选及边界，并实现对像素/区域特征的backward；区域汇总层实现对特征和关联的backward。小质量区域有其专门数值处理，不能未经验证照搬阈值。[距离层](https://github.com/NVlabs/ssn_superpixels/blob/082918950d768f29466148666b7f434c78663dc6/lib/src/caffe/layers/passoc_layer.cu#L17-L211)、[汇总层](https://github.com/NVlabs/ssn_superpixels/blob/082918950d768f29466148666b7f434c78663dc6/lib/src/caffe/layers/spixel_feature2_layer.cu#L81-L241)。

判断：SSN支持“保留位置约束，同时允许内容改变区域归属”的方向；不支持把0.9先验、单次768→128→16 MLP说成已发表验证的配置。可借鉴其特征与区域中心距离的打分方式，使用现有Swin特征做本项目适配；不需要另装Caffe。像素级超像素质量不能直接迁移为49-token场景分类结论。

## 4. Slot Attention与GroupViT：归一化相近，分配方式不同

Slot Attention固定commit `d36068b845da4c2b24927fee2cea1e6ef98dadda`。`model.py`先在slot维度softmax，让每个输入在slots之间竞争，再沿输入维度归一化以求加权均值，之后GRU和MLP更新slots；初始化含高斯随机采样。[源码71–103行](https://github.com/google-research/google-research/blob/d36068b845da4c2b24927fee2cea1e6ef98dadda/slot_attention/model.py#L71-L103)。

这支持我们区分“归属S”与“汇总W”，但源码会修改同一个attention张量，若还需反馈，应另保留归属矩阵。搬入完整Slot Attention会增加迭代、随机初始化与更新机制，且slots无固定空间编号；不是替换一个softmax即可完成的接入。

GroupViT固定commit `13b786155a1dfffe4703f40d028c92be58e1178d`。`AssignAttention`在group维度归一化；随后按分到该组的权重总和及`assign_eps`缩放汇总。`GroupingBlock`默认hard/gumbel均为true，训练使用Gumbel分配，hard模式前向one-hot、反向用软梯度估计；eval仍可硬分配。[AssignAttention与直通估计](https://github.com/NVlabs/GroupViT/blob/13b786155a1dfffe4703f40d028c92be58e1178d/models/group_vit.py#L50-L170)、[GroupingBlock默认值](https://github.com/NVlabs/GroupViT/blob/13b786155a1dfffe4703f40d028c92be58e1178d/models/group_vit.py#L181-L236)。

GroupViT本身以文本监督学习分层区域；不直接把其损失/预训练迁入当前CE/BCE消融。它说明“可训练分配”和“前向保持软分配”是两件事；本项目若选择全软前向，需要明确自己的配置及区别。

## 5. DiffPool：自由分组后图不必再绑定编号

PyG固定commit `79d33965a40b7fa83616a9f598a0f8619f25d939`。核心为`S=softmax(logits)`、`X'=SᵀX`、`A'=SᵀAS`，另返回link prediction和entropy损失；`dense_diff_pool`接收外部计算的分配logits，本函数本身不定义分配网络。[源码68–84行](https://github.com/pyg-team/pytorch_geometric/blob/79d33965a40b7fa83616a9f598a0f8619f25d939/torch_geometric/nn/dense/diff_pool.py#L68-L84)。

判断：这是无需人为给新区域编号赋空间含义的一条有出处的路线。但是它先要求输入图A，所得粗图通常为加权稠密图；汇总也不是我们的注意力加权均值。套用到当前GAViT会同时改分组、图定义及可能的损失，不能算“只替换分组”的首轮实验。它作为后续自由分组方向记录，不扩充首轮训练矩阵。

## 6. 对09-25草案的修正建议

**继续保留：**局部分配候选；S/W分别归一化；软汇总与软反馈一致；GAT及图预算先固定；梯度、占用和资源诊断。这些有相近实现可参考，但整体组合仍是本项目适配。

**不应提前冻结：**0.9/0.1先验、输出16个固定编号分数的MLP，以及相应102032新增参数。它们是初稿候选，不是上述代码共同采用或已验证更优的做法。

建议在纸面设计中增加“区域原型相似度”候选：

1. 用原固定分区汇总出当前图片的16个初始区域原型。
2. 比较每个token与附近原型的特征相似性/距离，形成候选区域分数；学习投影是否共享、是否迭代必须明确后再实现。
3. 保持分配路径可微，得到S；保留原注意力汇总W，接原GAT，并用S送回反馈。

与初稿的区别是：分配依据不仅是“这个token对区域编号k的分数”，还直接取决于“这张图片第k个区域里是什么内容”。这是设计推断与推荐理由，不是已证实的收益。建议优先细化这条路线，再决定是否保留初稿MLP为备选，不同时发起两套完整训练。

**对照也须随之调整：**若分配由图像特征与原型产生，即使冻结投影头，S仍会因骨干更新而改变；不能再把该版本称为“固定分配不学习”。原草案M2=`S=P0`仍可作为纯空间平滑对照，但不能单独隔离原型方案全部学习作用。正式设计须说明比较到底隔离什么，再冻结实验矩阵。

现有4×4空间边可以继续作为局部锚点先验；以上参考均不能替我们证明这些边就是学习区域真实接壤。论文也不能将“汇总→交互→还原”本身当作全新结构，须引用相近方法并说明本项目的稀疏区域图、反馈方式和遥感任务证据。

## 7. 检索边界

- 额外发现`perrying/ssn-pytorch`，其README明确区分原版与PyTorch移植；本轮未把该移植当作官方源码行为证据，也未验证其依赖。
- SPFormer存在3D超点、超光谱Self-Pooling等同名结果。针对Mei等人的《Enhancing Vision Transformer with Superpixel Representation》，本轮定向查询未确认作者源码；论文存在不等于已核验可用实现，不据其他同名仓库下结论。
- 本次链接锁定commit；不以star数、README效果图或外部ImageNet成绩替代本项目验证。源码静态核对已经回答可借鉴哪些计算步骤，运行可用性、GAViT开销及收益仍未测量。
