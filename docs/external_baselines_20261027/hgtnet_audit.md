# HGTNet 原文核对与独立重实现规格草案

核对日期：2026-10-27。目标论文为 Ziwei Li 等，*A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification*，IEEE JSTARS，17:20315-20330，2024，DOI [10.1109/JSTARS.2024.3491335](https://doi.org/10.1109/JSTARS.2024.3491335)。[合作者主页的 2024 年论文条目](https://ovoyu.github.io/)与论文标题、作者和 DOI 一致。

本文件只准备原文证据、模块规格和最低工程验证。未取得经过身份核验的 HGTNet 作者实现或预训练文件，未运行模型、安装依赖、启动 GPU、训练、测试或修改历史结果。用户此次同意准备材料和规格，不构成付费训练授权。

建议将后续实现标为 **HGTNet-independent-reimplementation-v0（依据论文独立重实现）**，命令中的模型名可用 `hgtnet_independent_v0`。它不是 `official`。若采取下文尚未证实的架构选择，应在配置和结果标题中同时列出；不能把独立实现结果直接标成作者原版结果。

## 1. 本次能核对到哪一层

技术信息直接依据[原论文公开全文文字层](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification)，不是其他论文的转引表格或搜索摘要。已打开并读取 III-A/B/C、IV-B/C、V-C/D 对应原文。

本次网页工具能读到正文和表题，但表 II、III 的单元格未展开。两个公开 PDF 链接读取失败，IEEE 页面/PDF 直接读取不可用，本地公开 PDF 下载因网络解析失败未取得文件。浏览器访问原页显示 DataDome 访问受限，随后停止该路径，未操作验证码。因此：

- 没有保存 PDF，`pdf_sha256=null`；不把网页文字层的 SHA 冒充 PDF SHA。
- 没有直读原图 3、4 的像素，也没有原表 II、III 的截图。
- 页码依据公开全文中的原论文页眉/分页标记定位，尚未与 PDF 文件独立交叉核验。
- 保存了直接打开原文得到的短摘录、行号、表号、页码及来源 URL，见[原文证据](../../results/evidence/external_baselines_20261027/hgtnet/source_excerpts.json)；保存了访问边界，见[来源清单](../../results/evidence/external_baselines_20261027/hgtnet/source_provenance.json)。摘录是核对时手工转录的选段，不是全文字节副本。

旧审计用于确定接续范围，本次结论不依赖旧计划已经完成：已读当前 `research_diary.md` 最新条目、主结果/消融 CSV，以及研究工作树的 `external_methods_audit_20260919.md`、`external_methods_readiness_20260924.md`、`external_methods_refresh_20261006.md`。旧审计中需等待作者的表述不再作为独立工程验证的前置条件；无法确认的配置必须在独立实现中公开登记。

## 2. 原作者报告值：均值已确认，标准差缺失

以下全部是 `reported`，不是本项目重新训练或重新测试。原论文 IV-C 说明每项实验重复五次；可读正文未说明这五次是否每次重新划分、固定划分仅换训练随机性、使用哪些随机种子，或标准差按总体还是样本公式计算。这些不能自行补齐。

- **AID，20% 训练 / 80% 测试**：OA 均值原始字符串为 `96.91%`；均值小数精度为两位。标准差 `null`，完整 `mean±SD` 字符串 `null`。均值证据为 IV-C.1，印刷页 20323；对应原表 II 也在印刷页 20323，但表格单元格本次不可直读。
- **AID，50% 训练 / 50% 测试**：OA 均值原始字符串为 `98.47%`；标准差 `null`，完整 `mean±SD` 字符串 `null`。均值证据为 IV-C.1，印刷页 20323；对应原表 II，印刷页 20323。
- **NWPU-RESISC45，10% 训练 / 90% 测试**：OA 均值原始字符串为 `94.52%`；标准差 `null`，完整 `mean±SD` 字符串 `null`。均值证据为 IV-C.2，印刷页 20323；对应原表 III 位于印刷页 20324。
- **NWPU-RESISC45，20% 训练 / 80% 测试**：OA 均值原始字符串为 `95.75%`；标准差 `null`，完整 `mean±SD` 字符串 `null`。均值证据为 IV-C.2，印刷页 20323；对应原表 III，印刷页 20324。

机器可读记录在 [reported_results.json](../../results/evidence/external_baselines_20261027/hgtnet/reported_results.json)。缺少标准差表示本次无法直读，不表示原表没有报告标准差。没有从其他论文补数、从图估数、反算标准差或把本项目其他协议的结果填入该文件。

## 3. 模块到张量的对应关系

以下统一使用 PyTorch 的 `B×C×H×W`；论文描述常用 `H×W×C`。标为“已确认”的内容来自当前可读原文；标为“工程选择/假设”的内容是可执行规格，不冒充作者配置。详细字段另存 [module_spec.json](../../results/evidence/external_baselines_20261027/hgtnet/module_spec.json)。

### 3.1 输入与 patch embedding

已确认：IV-B，印刷页 20322，输入 RGB，增强后统一为 `224×224`，patch 为 `4×4`，embedding 通道为 128。

可执行接口：输入 `x: B×3×224×224`。非重叠 patch 投影产生 `B×128×56×56`，或等价 `B×3136×128`。用 `Conv2d(3,128,kernel_size=4,stride=4)` 可表达这一映射；bias、归一化层与初始化方式仍须在配置中公开选定，不能仅凭输入形状称为与作者数值一致。

### 3.2 DA：并行的通道与空间注意力

已确认：III-B.1，公式 (1)-(7)，印刷页 20319-20320；三阶段 DA 输入通道 `(128,256,512)`，各自保持输入通道和空间大小。两条分支并行使用同一输入，不是先通道再空间的串联 CBAM。

输入 `F: B×C×H×W`：

- 通道分支：沿 `H,W` 分别取平均和最大，得到两个 `B×C×1×1` 张量；共享 MLP 后融合，形成 `Mc: B×C×1×1`；广播乘回 `F`。
- 空间分支：沿 `C` 分别取平均和最大，拼成 `B×2×H×W`；连续三个 `3×3` 空洞卷积，dilation 为 `1,2,3`；输出 `Ms: B×1×H×W`，广播乘回 `F`。
- 合并：`F_out=F*Mc+F*Ms`，形状仍为 `B×C×H×W`，对应 (1)-(3)。不能额外除以 2，或自动再加一次 `F` 而不登记差异。

工程选择草案：共享通道 MLP 采用 `C→max(1,C//16)→C`、中间 ReLU；空间卷积采用 `2→1→1→1`，padding 分别为 `1,2,3` 以保持大小，卷积间不加归一化/激活，仅末端 sigmoid。这些通道宽度、reduction、bias 和中间激活均未由本次可读原文确认，应全部写入配置。

额外文本歧义：正文明确说通道/空间权重经过 sigmoid，但紧邻 (7) 的符号说明把 `σ` 解释为 GELU；(4)/(5) 在网页文字层还有括号/MLP 位置错乱。独立版本可优先遵循正文的 sigmoid 门控及“共享 MLP 后相加”，并记录 `da_gate=sigmoid`。这只能消除实现歧义，不能证明作者源码就是该式。原图 4 或 PDF 公式仍有进一步核对价值。

### 3.3 STB 与三阶段 Transformer

已确认：III-B.2，公式 (8)-(11)，印刷页 20320；一个 STB 在本文公式中包含一套 W-MSA+MLP 和一套 SW-MSA+MLP，各自有 LN 和残差。因此 `(1,1,9)` 个 STB 对应候选 `(2,2,18)` 个单注意力 Swin block，不能直接在只把一次 attention 算一个 block 的库里填 `(1,1,9)`。

已确认的通道与深度配置：DA 输入 `(128,256,512)`；STB 数 `(1,1,9)`；对应 attention head 数 `(4,8,16)`；阶段输出通道 `(256,512,512)`。按输入通道计算的 head dimension 均为 32，这是通道/头数的算术结果，不是另一个被原文直接写出的参数。

单个 stage 内的形状约束：注意力和 MLP 保持 `B×H_i×W_i×C_i`；`Q,K,V` 每头宽度为 `C_i/h_i`。若窗口宽为 `M`，window attention 的分数张量为 `B*n_windows×h_i×M²×M²`。窗口宽、移位量、attention mask、relative-position bias、qkv bias、MLP expansion、dropout、stochastic depth、LN epsilon 与具体 downsample 运算没有在本次已读文字中固定。

工程选择草案：窗口 `M=7`、shift `3`、MLP ratio `4`、pre-LN、GELU，attention/MLP dropout 先取 `0`，drop-path 先取 `0`，便于模块等价和保存重载验证；这些是独立实现默认值。后续如果采用公开 Swin 模块，应把“一个论文 STB=两个库 block”的映射单独测试，不自动继承整套 Swin-T/Swin-B 拓扑。

### 3.4 空间分辨率不能机械照搬四阶段 Swin

III-A 说明三阶段包含下采样，IV-B 给出通道，却没有在可读文字层给出每一处 stride/padding。原图 3 本次未能直读，因此不能把下述空间路径写成已确认：

`56→28→14→7`，随后图分支再 `7→4`。

原因是最后 `4×4` 只有 16 个节点，不能直接为每个节点取 `k=20` 个不同候选；排除自身时还需至少 21 个节点。实现前必须以 `N>=k`（含自身）或 `N>=k+1`（不含自身）为门禁，禁止用 `min(k,N)` 静默改小邻居数。

可行的独立工程候选是让第三阶段仅保持 `14×14`、通道 512，图分支先 `14×14` 再下采样到 `7×7`；此时节点数 `196→49` 可容纳 `k=20`。但“第三阶段不再降低空间大小”是尚未证实的核心假设，配置必须标成 `shape_policy=independent_14_to_7`。另一条候选是修改图分支下采样位置；也必须另立身份，不能为了使代码运行就宣称原图已得到还原。

本轮不冻结上述候选为作者结构。可先实现配置化模块和约束检查；若后续选择 HGTNet 开始独立工程验证，则在首个可执行配置中明确选用一个路径，并保存逐层 shape trace。

### 3.5 ViG / Grapher / FFN

已确认：III-A/C，图 3(c)、公式 (12)-(13)，印刷页 20319、20321；Transformer 输出进入 GNN extractor，第一处 DWConv 减少通道，后续 DWConv 减少空间大小；两个 ViG 组的堆叠次数为 `(6,2)`，邻居数 `k=20`，GELU。图上的节点是特征图的规则非重叠块，不是本项目 K16 区域节点，也不是原 RGB 像素原样送进 GAT。

可执行接口：图特征 `X: B×Cg×Hg×Wg`，展平 `B×N×Cg`，其中 `N=Hg*Wg`。邻居索引可保存为 `B×N×20`；显式边方向为 `neighbor→query`。Grapher 前后线性映射和 FFN 都应回到 `Cg`，使残差加法有定义。公式约束为：

`Xg = FC_out(GELU(GraphConv(FC_in(X)))) + X`

`Xout = FC2(GELU(FC1(Xg))) + Xg`

第一/第二图组的具体 `Cg`、DWConv kernel/stride/padding、是否带 pointwise projection、BN/LN、Grapher 隐藏宽度、FFN expansion、图重建频率、距离度量、L2 normalize、relative position、dilation、自环和并列距离的处理，都未由本次可读 HGTNet 原文确认。

最低实现可采用配置化通道 `graph_channels=(256,512)`、depthwise `3×3`+pointwise `1×1`、第二处 stride 2/padding 1，并将所有选择标为工程假设；这与前述 `14→7` 候选配套，不能与 `7→4` 路径混用。必须显式检查组内残差与图组间通道投影，而非通过不透明 reshape 消除错误。

### 3.6 融合与分类头

已确认：III-A 和图 3(a) 描述跳连融合 Transformer 视觉特征与图结构特征后分类。当前文字层不足以唯一确认跳连来自哪一处、先池化还是先融合、加法还是拼接、pooling/FC 宽度及分类头 dropout。

可执行独立候选：分别对 `Fvisual: B×512×H×W` 与 `Fgraph: B×512×H'×W'` 做全局平均池化，得到两个 `B×512`，相加后线性映射到 `B×30`（AID）或 `B×45`（NWPU）。标记 `fusion=gap_sum`。这只是使不同空间大小的分支能稳定连接的工程候选；若采用 concat 或多尺度跳连，必须记录为不同实现假设，不能把它们混进一个“忠实复现”标签。

## 4. 图聚合冲突如何处理

原论文 III-C 明确写基于邻居差分的最大相对图卷积；IV-B 又写平均聚合。两个位置的短原文均保存在 `source_excerpts.json`。仅凭文字无法确定它们是笔误、不同模块的两个操作，还是源码另有实现。

查读 [ViG 作者仓库的 MRConv2d](https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/gcn_lib/torch_vertex.py)可确认参考算子的数学含义：对每个 query 逐通道计算 `max_j(x_j-x_i)`，与 `x_i` 拼接，再投影。它不是邻居特征均值。参考仓库也有图候选降采样用的平均池化；这能说明“模型出现 average 并不一定代表 MRConv 本身求均值”，但不能据此断言 HGTNet 的 IV-B 指该池化。该 ViG 仓库不是 HGTNet 官方代码。

独立实现可明确选择两个配置：

- 主规格候选 `graph_operator=mr_max`：优先按 III-C 的命名及差分描述实现最大相对聚合。
- 歧义敏感性候选 `graph_operator=relative_mean`：只把相同邻居差分的 reduction 从 max 改为 mean，其余形状、投影、边、初始化完全相同。这是为定位歧义定义的本项目候选，不是声称已经发现作者的平均版本。

先用一个手算小图验证：query 为 `0`，两个邻居为 `1,3`，差分 max 为 `3`、mean 为 `2`；在无偏置、固定投影下二者应按预期不同，交换邻居顺序应不改变结果。该验证不需数据集或 GPU，不应为解决术语冲突直接跑两个完整训练。

## 5. 预训练与原论文训练评估协议

原论文 IV-B/C，印刷页 20322-20323，已确认设置如下：

- 环境：Python 3.8、PyTorch 1.11.0、CUDA 11.3、Ubuntu 20.04；RTX 3090 24 GB。它们是作者报告环境，当前本项目未复测该速度或内存。
- 训练：100 epochs、batch 16、Adam、learning rate `1e-5`、weight decay `5e-5`、交叉熵。
- 输入：RGB 224，增强提到 crop/flip；精确 crop 分布、概率、插值、归一化均值/标准差、train/eval resize 次序未确认。
- 预训练：仅说明部分 ImageNet 预训练，没有可核验的文件名、训练集版本、下载 URL、SHA 或逐层加载映射。
- 协议：AID 20/80 与 50/50；NWPU 10/90 与 20/80；重复五次；指标 OA 和混淆矩阵。
- 学习率 scheduler/warmup、Adam betas/epsilon、梯度裁剪、label smoothing、混合精度、验证集大小、best/final 选择规则、随机种子及重复划分清单均未确认。

预训练工程规则：先输出模型所有参数名/shape；对候选权重逐 tensor 建立 `loaded / missing / unexpected / shape_mismatch / intentionally_skipped` 清单。128/256/512 和 `(2,2,18)` 单注意力 block 与某些 Swin-Base 阶段维度相容，只能作为候选来源线索，不能证明作者用的是 Swin-Base，亦不能证明当前本项目的 Swin-T 权重可用。DA、图分支、改动的下采样与分类头需明确初始化，禁止 `strict=False` 后只打印“预训练成功”。

在尚无权重材料时，可先随机初始化完成单次合成前向/反向与保存重载；该阶段只证明程序能运行，不证明预训练方案或文献 OA 可复现。预训练映射是正式比较前必须冻结的核心假设，不必等作者回信才可开始工程检查。

## 6. 独立实现中可自行选定与必须披露的事项

可以由项目自行决定且通常不改变研究问题的事项：文件/类名、张量布局转换、配置和 metadata schema、日志格式、异常提示、路径、SHA 记录、纯确定性的边索引 tie-break、批量图的存储方式、测试夹具及 checkpoint 保存格式。仍需在实现记录中写清，以便复核。

会影响模型身份或公平性的事项必须显式冻结，不能当作普通清理：最后 Transformer 阶段的空间下采样、两处 DWConv 的通道/stride、图算子 max/mean、kNN 距离/归一化/自环/relative-position/dilation、DA 的 sigmoid 和 MLP 细节、Swin 窗口与 block 计数、融合和分类头、具体预训练映射、增强/归一化、选模规则和数据划分重复机制。

当前可继续做独立模块工程；不能声称“原论文所有核心细节已确认”。若采用本文件的 `independent_14_to_7 + mr_max + gap_sum` 路径，就在模型标题中注明这些假设。若之后取得原图或源码，再作对照并另立版本，不覆盖已产生的失败或负面证据。

## 7. 最低工程验证顺序与停止条件

以下是后续实施清单，本轮尚未执行。无需付费 GPU即可先完成前五项。

1. 建立独立目录/配置和来源清单；实现 patch、DA、STB pair、构图、MR/mean、Grapher/FFN、融合头，保留所有未决项为显式配置，不从 GAViT 改名拼装。
2. 用 `B=1,2` 的固定合成 RGB 输入保存每层 shape；断言 DA 不改形状、论文 STB 数能映射成两倍 attention block、残差两端 shape 相同、输出为 `B×30/45`。若出现末级 16 节点配 `k=20`，在构图前立即失败，不能截断 k。
3. DA 检查广播、通道池化维度和并行分支；人为固定 gate 时核对加法而非串联乘法。对 `56/28/14` 和非方形夹具均保持大小，避免只因 7 能整除而掩盖布局错误。
4. 用手算小图核对相对差分方向、max/mean、邻居置换不变性、batch 隔离、索引范围、自环政策与 k 数；同输入重复构图确定，变化输入时能观察到动态图是否按配置更新。
5. 固定随机种子，检查全模型前向有限、交叉熵反向梯度有限且 backbone/DA/图/头都收到梯度；保存后在新进程重载，eval logits 在预先约定容差内一致。记录参数量而不预填作者参数量。
6. 冻结预训练候选文件 SHA 和逐层加载清单；核对实际加载覆盖率及关键几层权重哈希。若无法确定映射，仍可保留随机初始化工程验证结果，正式文献数值复现状态继续标未知。
7. 在用户选定该方法并批准实际执行范围后，复用已核验清单做极少样本 forward/backward/validation/checkpoint smoke，记录设备、首批、峰值内存与耗时。仅在首轮资源证据存在后估算正式成本；本文件不提供现在就会启动付费任务的命令。
8. 正式实验另行冻结完整命令、source snapshot/SHA、预训练/数据 SHA、seed、输出名、预算和选模规则。先单 seed 回答可行性；是否多 seed 按项目规则和结果价值另定。

成功层级依次是“材料审计完成”“独立规格冻结”“本地工程门禁通过”“真实 smoke 通过”“正式运行完成”。它们不能互相替代。正式指标需要日志、配置、best/final checkpoint 身份和逐样本评估证据；本轮没有任何 `rerun` 分数。

## 8. 与本项目比较时如何使用

文献报告值独立呈现，当前四项仅有直接核对的均值，SD 待获得原表。统一协议重跑须另立结果身份，不覆盖 `reported_results.json`。

旧研究工作树计划的 AID 固定训练池/内部验证/固定 test 和三训练 seed，与原论文“五次实验、训练/测试比例”并不自动相同。即使比例相同，仍需披露 validation 占比、split、重复机制、预训练及预算差异。HGTNet 与 GAViT 的骨干和流程不同，完整方法分数差不能全部归因于图模块。BigEarthNet 多标签不在本次 HGTNet 单标签规格内。

当前最有价值的下一步是选定是否接受这些明确的独立实现假设，并完成最低工程门禁；不需要以作者是否回信作为工程开始条件，也不应以补齐标准差为理由推迟可并行的形状和算子检查。

## 9. 本轮交付与证据入口

- [原作者报告值 JSON](../../results/evidence/external_baselines_20261027/hgtnet/reported_results.json)：四项均值字符串、SD=null、五次重复、来源和证据状态。
- [原文短摘录](../../results/evidence/external_baselines_20261027/hgtnet/source_excerpts.json)：直接打开论文正文后的摘录与行号，不是搜索 snippet。
- [来源/访问边界](../../results/evidence/external_baselines_20261027/hgtnet/source_provenance.json)：公开全文、作者主页、PDF/图像读取失败与无 PDF hash 的边界。
- [模块规格 JSON](../../results/evidence/external_baselines_20261027/hgtnet/module_spec.json)：已确认字段、工程候选和核心未决项。
- [本轮静态核对](../../results/evidence/external_baselines_20261027/hgtnet/validation.json)：只核对文档/JSON/路径和空值，不代表模型或训练验证。

原论文许可页面标注 CC BY-NC-ND 4.0；这只能说明论文内容许可，不能据此推定 HGTNet 代码或预训练权重许可。本轮没有复制、导入或执行候选 HGTNet/ViG 源码。
