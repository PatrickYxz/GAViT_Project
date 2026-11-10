# GAViT项目进度汇报

更新日期：2026-11-08。本次加入HGTNet在AID上的独立实现seed42结果，以及Swin／固定GAViT／HGTNet三模型原始预测的本地重算和逐图配对。原定最低矩阵HGTNet×NWPU/AID、DFAGCN×NWPU的三项单seed实测均已齐备；统一训练/评估协议与多seed外部比较仍未完成。内部多seed统计保持原10-20归档及11-04补录范围，主表仍采用用户指定的seed42/43/46/47，全量历史结果另列。46/47缺失F1及原始记录继续保留空项；外部单seed不并入原四seed均值和SD。

展示统一四舍五入至两位小数，均值、样本SD和配对差先按原始可得精度计算，再舍入显示；不先舍入各seed再做统计。NWPU seed46/47的六个OA已由用户确认为两位小数的四舍五入回传值，未取得更多小数。BEN部分来源只有一位小数，补尾零仅用于统一格式，不增加测量精度。显示值直接相减可能与单独舍入的差值相差0.01pp。

目前已实现固定区域图GAViT和局部原型软分配，完成AID、NWPU-RESISC45、BigEarthNet-19的内部模型比较。新增RTX4090上seed46/47的回传主指标中，固定版在三个数据集均高于Swin；软分配在NWPU低于两项对照，在BigEarthNet表现混合。结合此前正负seed，当前能够报告条件性收益、方差和开销，尚不能主张三个数据集上稳定优于Swin。BEN受训3×3与4×4的同图分析及图稿现已完成。DFAGCN与HGTNet独立实现的NWPU seed42正式实验和25200条预测离线复核均完成，OA分别为86.53%与94.12%；HGTNet×AID seed42也已完成，OA96.92%、macro-F1 96.77%。AID两项核心模型共10份原始评估文件已回收，与HGTNet的5000条预测逐图核对同路径/真值后完成配对，无新增模型评估。主要后续工作是外部统一协议与多seed要求、完整内部结果/消融/讨论的论文整合，以及Swin／固定／soft的同图对照和具体图边贡献的独立验证。

## 1. 与导师要求的对应进度

以下按[已归档导师邮件](../.worktrees/sparse-hybrid-4n-top2/docs/supervisor_latest_email_received_20260924.md)核对。

增加AID第三基准

- 已完成内容：固定划分上Swin/固定版42/43/44三seed已归档；新增46/47 OA回传并更正；本次回收seed42两模型10份原始评估记录、重算全部预测并与HGTNet完成配对
- 当前范围：46/47保留已确认OA，macro-F1及资源未提供；已有42/43/44完整辅助指标按其范围单列。AID未运行soft，邮件最低soft要求针对NWPU/BEN

BEN增加至少两个seed，全部seed报告mAP、macro-F1、micro-F1均值和SD

- 已完成内容：Swin/固定版42/43/44/45三项统计已归档；新增46/47 mAP
- 当前范围：46/47两项F1缺失，历史可用F1按实际seed范围单列；历史部分轮数、硬件与退出状态不同，不能当作完全一致的训练重复

NWPU与BEN比较Swin、固定4×4、可学习soft

- 已完成内容：NWPU42/43三模型；BEN42/43/45三模型均有归档，soft42已完成full30及完整test；新增46/47双数据集三模型主指标
- 当前缺口：同设备效率和可用工件分析；多seed已有主指标，46/47按历史分数补录，不再以原件回收为推进条件

HGTNet与DFAGCN实测

- 已完成内容：HGTNet×NWPU/AID及DFAGCN×NWPU三项独立实现seed42均完成验证选择、fresh refit与完整test，并取回预测独立复算。NWPU HGTNet OA94.12%、macro-F1 94.11%，选定95轮；DFAGCN OA86.53%、macro-F1 86.46%，选定51轮、C10。AID HGTNet OA96.92%、macro-F1 96.77%，完整100轮校准、选定96轮并fresh refit96轮；Swin／固定GAViT／HGTNet真实逐图配对已完成
- 当前缺口：统一训练/评估协议及正式多seed外部比较；模型二进制未全部本地回收。三项是披露配方下的单seed补充比较，不能称官方复现、原论文数值复现或导师全部外部要求已完成

区域覆盖图与实际图边，讨论3×3/4×4及语义关系

- 已完成内容：已补BEN受训K9/42，与K16/42在同四张原图上捕获真实边、汇总权重和全部19类预测；两页PDF/PNG及分析已完成，3次标签纠正、1次新增漏判
- 当前缺口：四例为历史诊断验证子集；Swin／固定／soft同图及图边独立因果贡献、语义区域质量量化尚未完成

三条相关工作与本方法差异

- 已完成内容：遥感Transformer、图网络、区域分组的中文初稿与BibTeX已整理；实验讨论初稿已有
- 当前缺口：更新新结果、导师反馈、引用覆盖与学校格式

## 2. 本次汇报主表：seed42、43、46、47

本次主表按用户指定采用42、43、46、47四个seed，报告均值±样本SD（n−1）。这是在已有结果上确定的汇报子集；全部历史seed及其正负结果保留在第3节，不能将本表称为完整预设seed的稳定性证据。历史硬件/代码/训练轮数差异仍适用；46/47来源为用户确认的历史主指标，见[补录及统计范围](../results/seed46_seed47_reconstruction_20261104.md)。

AID

- 指标：OA
- n：4
- Swin均值±SD %：96.37±0.48
- 固定GAViT均值±SD %：96.83±0.51
- 固定−Swin均值 pp：+0.46

NWPU-RESISC45

- 指标：OA
- n：4
- Swin均值±SD %：93.81±0.11
- 固定GAViT均值±SD %：94.00±0.19
- 固定−Swin均值 pp：+0.20

BigEarthNet-19

- 指标：mAP
- n：4
- Swin均值±SD %：70.90±0.22
- 固定GAViT均值±SD %：71.65±0.39
- 固定−Swin均值 pp：+0.75

在这个四seed子集中，固定版AID与BEN的四次配对主指标均为正；NWPU为三次正、seed43负。不得据此改写为三个数据集每次均提升，或忽略历史44/45来证明总体稳定。

### 软分配：只在有三模型结果的相同seed上比较

NWPU与BEN在指定42/43/46/47四seed上均已有三模型主指标。BEN soft42正式30轮与完整119825张测试已核对，test mAP70.17%（统计使用JSON原始精度）；seed45继续保留在历史补充，不替换指定四seed。AID未运行soft。46/47主指标为用户确认的历史回传值。

NWPU-RESISC45

- 三模型共同seeds：42/43/46/47
- n：4
- Swin均值±SD %：93.81±0.11
- 固定均值±SD %：94.00±0.19
- soft均值±SD %：93.65±0.15
- soft−Swin均值 pp：-0.16
- soft−固定均值 pp：-0.36

BigEarthNet-19

- 三模型共同seeds：42/43/46/47
- n：4
- Swin均值±SD %：70.90±0.22
- 固定均值±SD %：71.65±0.39
- soft均值±SD %：71.19±1.25
- soft−Swin均值 pp：+0.29
- soft−固定均值 pp：-0.46

在指定BEN四seed中，soft对Swin为2升2降（43/46正、42/47负）；对固定版仅43为正。soft样本SD1.25pp高于固定版0.39pp，不据均值+0.29pp宣称稳定提升。

46/47的F1缺失，指定四seed不计算完整F1均值和SD。现有历史F1分数及其统计范围见第4节，不能直接作为本次四seed汇总。

### 新增RTX4090批次：seed46/47（补充）

用户确认配置和测试数量与此前一致，物理GPU UUID不同。以下为已确认历史主指标的算术汇总，来源级别为用户回传。AID两项记录已按用户确认更正：固定46 **97.13→97.12%**；Swin47 **95.73→95.72%**。原表和更正前记录保留。

均值±样本SD（n=2）；AID/NWPU为test OA，BEN为test mAP，差值为百分点（pp）。

AID

- Swin %：96.14±0.59
- 固定GAViT %：96.60±0.74
- soft %：未运行
- 固定−Swin pp：+0.46
- soft−Swin pp：—

NWPU-RESISC45

- Swin %：93.72±0.06
- 固定GAViT %：94.08±0.16
- soft %：93.55±0.01
- 固定−Swin pp：+0.36
- soft−Swin pp：-0.18

BigEarthNet-19

- Swin %：70.80±0.28
- 固定GAViT %：71.45±0.49
- soft %：70.85±0.78
- 固定−Swin pp：+0.65
- soft−Swin pp：+0.05

固定版AID两次增益为+0.56/+0.36pp，NWPU为+0.20/+0.51pp，BEN为+0.80/+0.50pp。soft相对固定版在NWPU均值−0.53pp、BEN−0.60pp。BEN只回传一位小数，soft−Swin均值+0.05pp的保守舍入区间[−0.05,+0.15]pp含零，不据此写成提升或统计等效。三数据集的配对组数不作为同一任务的独立重复。

## 3. 完整历史记录（本次四seed主表之外的补充）

仅纳入当前选定固定4×4稀疏混合图版本，不混入历史未修正/纯kNN版本、smoke或val指标。“已归档”指现有回传证据已保存和核对，不表示本地下载了全部checkpoint及预测数组。新46/47仅有用户回传分数。单标签soft缺项不补猜值。

### AID：test OA（%），test=5000

seed 42

- Swin：96.34
- 固定GAViT：96.94
- soft：—
- 固定−Swin pp：+0.60
- soft−Swin pp：—
- 证据状态：已归档

seed 43

- Swin：96.86
- 固定GAViT：97.16
- soft：—
- 固定−Swin pp：+0.30
- soft−Swin pp：—
- 证据状态：已归档

seed 44

- Swin：96.84
- 固定GAViT：96.08
- soft：—
- 固定−Swin pp：-0.76
- soft−Swin pp：—
- 证据状态：已归档

seed 46

- Swin：96.56
- 固定GAViT：97.12
- soft：—
- 固定−Swin pp：+0.56
- soft−Swin pp：—
- 证据来源：用户确认的历史主指标

seed 47

- Swin：95.72
- 固定GAViT：96.08
- soft：—
- 固定−Swin pp：+0.36
- soft−Swin pp：—
- 证据来源：用户确认的历史主指标

AID使用固定数据清单：4000/1000校准，以各模型最佳验证轮数在5000训练池refit后测试5000图。42/43/44为同协议历史三seed；训练seed的重复不是三个独立数据划分。

### NWPU-RESISC45：test OA（%），test=25200

seed 42

- Swin：93.93
- 固定GAViT：94.10
- soft：93.87
- 固定−Swin pp：+0.17
- soft−Swin pp：-0.06
- 证据状态：已归档

seed 43

- Swin：93.86
- 固定GAViT：93.77
- soft：93.63
- 固定−Swin pp：-0.09
- soft−Swin pp：-0.23
- 证据状态：已归档

seed 46

- Swin：93.76
- 固定GAViT：93.96
- soft：93.54
- 固定−Swin pp：+0.20
- soft−Swin pp：-0.22
- 证据来源：用户确认的历史主指标

seed 47

- Swin：93.68
- 固定GAViT：94.19
- soft：93.55
- 固定−Swin pp：+0.51
- soft−Swin pp：-0.13
- 证据来源：用户确认的历史主指标

NWPU现行清单5040/1260校准，以最佳验证轮数在6300训练池refit后测试25200图。42用RTX2080Ti，43用RTX4090；46/47也由用户确认为RTX4090，具体UUID与物理卡配对未独立核验。当前四个有soft结果的seed，soft主指标均低于Swin和固定版。

### BigEarthNet-19：test macro-mAP（%），test=119825

seed 42

- Swin：70.90
- 固定GAViT：72.00
- soft：70.17
- 固定−Swin pp：+1.10
- soft−Swin pp：-0.73
- 证据状态：历史对照*；soft正式full30/test已归档

seed 43

- Swin：71.10
- 固定GAViT：71.70
- soft：72.87
- 固定−Swin pp：+0.60
- soft−Swin pp：+1.77
- 证据状态：历史归档*

seed 44

- Swin：71.40
- 固定GAViT：70.20
- soft：—
- 固定−Swin pp：-1.20
- soft−Swin pp：—
- 证据状态：历史归档*

seed 45

- Swin：71.68
- 固定GAViT：70.26
- soft：70.84
- 固定−Swin pp：-1.42
- soft−Swin pp：-0.84
- 证据状态：历史归档*

seed 46

- Swin：71.00
- 固定GAViT：71.80
- soft：71.40
- 固定−Swin pp：+0.80
- soft−Swin pp：+0.40
- 证据来源：用户确认的历史主指标

seed 47

- Swin：70.60
- 固定GAViT：71.10
- soft：70.30
- 固定−Swin pp：+0.50
- soft−Swin pp：-0.30
- 证据来源：用户确认的历史主指标

BEN使用RGB输入与19标签，train/val/test=237871/122342/119825，测试各模型最佳验证checkpoint；F1阈值0.5。*历史42–44的Swin/固定对照主指标原始回传保留一位小数，本表补尾零显示两位；soft42/43使用正式JSON完整精度统计再显示；fixed43/44保存到21/26轮，Swin44最终退出未确认；45三模型完整30轮。42/43三模型存在代码/历史硬件差异；soft42/43分别在记录UUID的RTX4090上full30。soft42 best10/val78.75%、final30/val75.71%，测试最佳验证checkpoint。这些限制不被相同seed消除。

### 全记录描述性统计

下表为全量历史统计，范围不同于第2节指定的四seed主表，包含用户回传的新46/47与全部既有正负seed。**硬件、部分代码/训练终止条件和证据完整度不同；该混合汇总不能作为最终统一条件的核心性能表，也不是纯seed方差或显著性证据。** 统计按原始可得打印精度复算，表格统一显示两位小数。

AID

- seeds：42/43/44/46/47
- n：5
- Swin均值±SD %：96.46±0.47
- 固定均值±SD %：96.68±0.55
- 配对固定−Swin均值±SD pp：+0.21±0.56
- 固定正向次数：4/5

NWPU

- seeds：42/43/46/47
- n：4
- Swin均值±SD %：93.81±0.11
- 固定均值±SD %：94.00±0.19
- 配对固定−Swin均值±SD pp：+0.20±0.25
- 固定正向次数：3/4

BEN

- seeds：42/43/44/45/46/47
- n：6
- Swin均值±SD %：71.11±0.38
- 固定均值±SD %：71.18±0.79
- 配对固定−Swin均值±SD pp：+0.06±1.09
- 固定正向次数：4/6

软分配仅在同一组具备三模型结果的seed内统计，避免将BEN soft的五seed均值与Swin的六seed均值直接相减。

NWPU

- 三模型共同seeds：42/43/46/47
- Swin均值±SD %：93.81±0.11
- 固定均值±SD %：94.00±0.19
- soft均值±SD %：93.65±0.15
- soft−Swin均值 pp：-0.16
- soft−固定均值 pp：-0.36

BEN

- 三模型共同seeds：42/43/45/46/47
- Swin均值±SD %：71.06±0.39
- 固定均值±SD %：71.37±0.71
- soft均值±SD %：71.12±1.09
- soft−Swin均值 pp：+0.06
- soft−固定均值 pp：-0.26

以上仍为混合条件描述性汇总。BEN soft43/45相对固定的mAP提高，而42/46/47下降；无法将局部原型soft写成一致优于固定分组。NWPU中该具体soft实现持续为负，作为消融结果保留，不外推为所有可学习分组方法无效。

## 4. 辅助指标与完整归档

以下均来自已归档历史结果；46/47的macro-F1、BEN micro-F1缺失，不用主指标推算或沿用其他seed。各行统计范围列出，不能直接替代“所有seed三项指标”的最终表。

AID

- 模型：Swin
- 已有F1的seeds：42/43/44
- macro-F1均值±SD %：96.52±0.26
- micro-F1均值±SD %：—

AID

- 模型：固定GAViT
- 已有F1的seeds：42/43/44
- macro-F1均值±SD %：96.54±0.58
- micro-F1均值±SD %：—

NWPU

- 模型：Swin
- 已有F1的seeds：42/43
- macro-F1均值±SD %：93.87±0.05
- micro-F1均值±SD %：—

NWPU

- 模型：固定GAViT
- 已有F1的seeds：42/43
- macro-F1均值±SD %：93.92±0.24
- micro-F1均值±SD %：—

NWPU

- 模型：soft
- 已有F1的seeds：42/43
- macro-F1均值±SD %：93.72±0.19
- micro-F1均值±SD %：—

BEN

- 模型：Swin
- 已有F1的seeds：42/43/44/45
- macro-F1均值±SD %：64.77±0.76
- micro-F1均值±SD %：76.51±0.34

BEN

- 模型：固定GAViT
- 已有F1的seeds：42/43/44/45
- macro-F1均值±SD %：65.30±0.75
- micro-F1均值±SD %：76.83±0.19

BEN

- 模型：soft
- 已有F1的seeds：42/43/45
- macro-F1均值±SD %：65.20±1.34
- micro-F1均值±SD %：76.70±0.28

46/47按已确认主指标补录，缺失的F1、逐图预测和运行字段保留空项，不再以回收原件作为下一阶段前提。AID的46/47更正值已明确；这两seed的正确数仍仅为OA派生检查，不冒充独立预测证据。与此分开，AID seed42的Swin和固定GAViT原始预测本次已实际回收并独立复算，完成与HGTNet的真实逐图配对。新回收不填补46/47的证据空项。

## 5. 方法、可视化与效率

当前可视化优先项已完成到受训3×3／4×4阶段（2026-10-20）：BEN固定K9/42已补训，原119825张test mAP71.08%，历史K16/42为72.00%，约高0.92pp。两套权重在同四张归档原始RGB验证输入上完成CPU真实捕获，全部8组观察/重建检查误差0，37个返回文件SHA和全部76项判断已核对。[正式图稿及逐图分析](ben_gridpair42_visual_analysis_20261020.md)同时展示原图、实际特征边和区域内汇总权重；4×4纠正3个标签判断，新增1个漏判。草地42.85%→51.59%、过渡林地44.74%→91.82%、建成区41.38%→76.57%为纠正；另一图的海洋水域64.96%→5.84%为新增漏判。保留未越阈值的牧场和沙地漏判，不能把3/1解释为整图成败或总体获益率。旧[几何覆盖](ben_grid3_grid4_analysis_20261017.md)作为前一阶段材料保留；Swin／固定／soft同图对照仍待继续，NWPU原90例fixed/soft图册保持原范围。

当前固定版在Swin末级49个token上汇总16个区域，构建48条空间4邻接和32条特征top-2有向边，经两层四头GAT后反馈到token。80条是显式构造的非自环边；图册展示的是构造边，不是GAT注意力或因果归因。soft用共享归一化/128维投影、图像内局部原型与温度0.1的余弦软归属替代硬成员集合，增加99,840参数；空间锚点邻接仍保留。

NWPU预定验证90例、180张fixed/soft SVG已制作、核对和审阅。比较对象为固定GAViT与软分配GAViT：82张双方正确，3张固定错而soft对，0张固定对而soft错，5张双方都错；“5例共同错误”不表示5例新增退化。39.09%的token主归属改变。这些是seed42校准best权重下的验证案例观察，不是refit测试总体收益，也没有直接比较Swin。正式seed42的25200张test中，固定GAViT相对Swin纠正586张、同时新增543张错误，净多对43张，OA约93.93%→94.10%；该总体结果与90例fixed/soft图册分别报告。图册用于分析区域归属、特征边变化和分类正反例；目前没有语义区域标注下的量化质量评估。人工审阅中也未观察到跨案例一致的地物区域完整性改善，因此不能由分组改变或个别预测纠正直接推出普遍语义改善。

这里的“语义一致性”指汇总区域是否更连贯地对应图中相关地物，例如建筑、林地或河道；颜色本身只是区域ID，连线只是捕获的构造边。教堂纠正样本中的建筑仍跨多个软区域，河道共同错误样本的分组/图边变化后仍未正确分类。这些观察说明可视化的解释范围，不把一个地物跨多个区域视为分类模型必然失败，也不把分类纠正当作语义分割质量证据。论文应同时展示纠正、共同错误及完整测试中的损害统计。

BigEarthNet固定模型的图反馈诊断另有四例论文图稿，覆盖预定24例中的全部阈值翻转：2例纠正误报、2例引入漏报，已导出PDF和300dpi PNG。该图使用历史seed44权重进行反馈开/关诊断，与本次42/43/46/47性能主表、NWPU固定/软分配图册以及新增K9/K16 seed42对照分别标注。3×3/4×4的受训模型捕获和图稿现已补齐，图边单独的因果贡献仍未隔离；两次训练批次的限制保留。已有诊断不重复运行。

### 逐seed类别对比：BEN seed42与seed43，两种模型分别对Swin

按用户要求，每次固定一个数据集和seed，并统一以Swin-T为参照，分别展示42和43，不合并两seed类别均值，详见[逐seed独立对Swin的类别表](ben_seed42_seed43_vs_swin_20261013.md)。这里“明显”指数值幅度，没有进行类别级统计显著性检验；全部可得19类均保留。

固定GAViT对Swin（BEN42）：荒原/石南灌丛/硬叶植被+5.00pp、海岸湿地+4.40pp、沙滩/沙丘/沙地+3.50pp、多年生作物+2.40pp、复杂耕作+2.10pp、内陆湿地+1.40pp；工业或商业区−1.30pp。总体mAP约+1.10pp。

软分配GAViT对Swin（BEN42）：天然草地/稀疏植被+6.24pp、多年生作物+2.76pp、荒原/石南灌丛/硬叶植被+2.64pp、阔叶林+1.04pp；海岸湿地−19.64pp、沙滩/沙丘/沙地−4.43pp、工业或商业区−2.32pp。总体mAP−0.73pp。该次正式test macro-F1为64.05%、micro-F1为76.80%。海岸湿地、沙滩和草地的变化方向与soft43相反，不能写成跨seed一致的类别改善。

固定GAViT对Swin（BEN43）：荒原/石南灌丛/硬叶植被+4.20pp、多年生作物+3.30pp、内陆湿地+2.90pp、海岸湿地+1.60pp、天然草地/稀疏植被+1.30pp；沙滩/沙丘/沙地−2.70pp。总体mAP约+0.60pp。

软分配GAViT对Swin（同一BEN43）：沙滩/沙丘/沙地+12.47pp、海岸湿地+9.81pp、多年生作物+4.70pp、荒原/石南灌丛/硬叶植被+4.59pp、内陆湿地+2.64pp、城市建成区+1.00pp；天然草地/稀疏植被−1.32pp、复杂耕作−1.02pp。总体mAP约+1.77pp。该段全部差值均对Swin，避免将soft低于fixed误写为soft低于Swin。

旧的soft−fixed和其他数据集/seed资料移作[完整附录](per_class_findings_20261013.md)。单seed打印精度、历史训练轮数/代码/设备差异仍保留；46/47逐类缺失，不据此推断四seed共同类别趋势。

现有AID同卡RTX2080Ti、FP32、驻留GPU合成输入的模型前向测量：参数量27.544M→31.375M（+13.91%），batch1平均延迟14.891→20.016ms（+34.41%），batch32吞吐564.892→478.433图/秒（−15.31%），batch32峰值allocated显存594.482→609.353MiB（+2.50%）。这份效率记录可进入论文；它不是完整数据读取流程，也不代表新4090三模型的效率对照。

反馈幅度、空间先验、轻量FFN的有界候选检查及负面结果已有记录。它们不作为新46/47结果的一部分，也不据此追加结构搜索。当前论文更合适的研究问题是：显式区域关系在何种条件下有收益、对随机性和分组方式是否敏感、收益是否值得额外复杂度。

## 6. 外部方法与后续工作

当前原定最低矩阵HGTNet×NWPU/AID、DFAGCN×NWPU已完成三项独立实现seed42正式实测。NWPU两项均完成6300训练池fresh refit和25200测试；AID HGTNet完成5000训练池fresh refit和5000测试。外部原始预测均已取回独立复算；AID的Swin与固定GAViT两模型10份原始评估文件也已回收，包含各自5000条预测、全部逐类指标、metrics、run与退出记录。没有为本次分析重新训练或重做模型评估。原论文报告值仍单列作文献参考。

同划分NWPU seed42完整方法对照（OA / macro-F1）：

- Swin-T：93.9325% / 93.9091%；23671/25200正确。
- 固定4×4 GAViT：94.1032% / 94.0899%；23714/25200正确。
- 局部软分配GAViT：93.8690% / 93.8550%；23655/25200正确。
- DFAGCN独立实现：86.5317% / 86.4589%；21806/25200正确。
- HGTNet独立实现：94.1190% / 94.1053%；23718/25200正确。

HGTNet的25200条预测、45类指标与全部混淆单元重算通过。完整100轮校准和95轮fresh refit历史已核对；验证集唯一最佳为95轮，末轮100仍保留在曲线中，见[NWPU离线分析](../results/hgtnet_nwpu_seed42_analysis_20261104.md)。HGTNet比固定GAViT高0.01587302pp OA，净正确数差4张，属于单seed点估计接近。固定版31,386,920参数，HGTNet74,619,858，前者少57.94%；两者NWPU逐图配对原件仍未齐，不能从净4张推断纠正/新增错误数。

HGTNet与DFAGCN的NWPU路径/类别逐项一致：双方都正确21303张、HGTNet纠正2415张、HGTNet新增错误503张、双方都错误979张，净多对1912张。DFAGCN召回率最低的palace为58.04%、church65.36%、tennis_court66.96%；最弱五类共910次错误，占26.81%。最大的双向混淆包括palace/church142张（84/58）、railway/railway_station85张（65/20）、basketball_court/tennis_court79张（39/40）。完整方向均保留；这些是错误分布描述，不能据此识别图传播等机制的因果作用。

同划分AID seed42完整方法对照（OA / macro-F1）：

- Swin-T：96.34% / 96.22088514124282%；4817/5000正确；校准30轮、选定并fresh refit21轮。
- 固定4×4 GAViT：96.94% / 96.7811271561119%；4847/5000正确；校准30轮、选定并fresh refit20轮。
- HGTNet独立实现：96.92% / 96.77286814931982%；4846/5000正确；校准100轮、选定并fresh refit96轮。

AID三模型已按原manifest、全部5000条路径和真值验证一致后真实配对，以HGTNet为候选模型：相对固定GAViT，双方都正确4771张、HGTNet纠正75张、新增76张错误、双方都错误78张，净少对1张；相对Swin，双方都正确4738张、HGTNet纠正108张、新增79张错误、双方都错误75张，净多对29张。固定GAViT比HGTNet高0.02pp OA、0.0082590068pp macro-F1。75/76表明净差1张背后存在两个方向的预测变化，不把它写成仅1张预测不同；也不据单seed配对描述宣称统计显著或稳定优势。见[原预测配对核对](../results/aid_seed42_three_model_paired_analysis_20261106.json)。

HGTNet AID验证第96轮唯一最高，972/1000正确、OA97.20%、macro-F1 97.074557%；末轮100为971/1000、OA97.10%、macro-F1 96.990538%。完整100+96轮历史、30类指标、混淆矩阵和错误明细均保留，最终test不是refit训练100%准确率。AID最低召回率为Square86.67%、Center86.92%、Resort88.97%、School90.00%；Center/Square与Park/Resort双向混淆各12张。离线分析只描述错误结构，不把它解释成已证实的图模块原因。

固定GAViT的AID参数31,375,385，HGTNet74,612,163，前者少57.95%。相对同一AID内部Swin-T的27,543,960参数，固定版增加13.91%；额外复杂度的价值仍需结合核心多seed结果和同条件效率判断。这些参数差可报告，但速度和总费用不能用异设备记录直接比较。AID HGTNet恢复后的attempt2墙钟4986.588285893秒、峰值allocated约4.406GiB/reserved约4.820GiB仅覆盖该调用；全实验总墙钟、全程峰值和费用未获得。服务器完成检查与checkpoint重载证据已核对，未声称本地加载了模型二进制。

三项均为披露不同recipe的完整方法比较。内部Swin/GAViT使用Swin-T、224直接resize、AdamW和30轮校准；HGTNet使用部分Swin-B初始化、256resize后224crop、Adam和100轮校准；DFAGCN使用冻结VGG16、224/448多尺度输入、三GCN及LIBLINEAR。GPU分别涉及历史2080Ti、DFAGCN3080、HGTNet NWPU3090与AID3060。相同split与信息隔离不等于全部训练条件一致，不能把差值归因单个图模块，不能称官方复现或同设备效率优势。

后续落实外部主比较的统一训练/评估协议、多seed均值与样本SD，并继续整合完整内部结果/消融/讨论。三项单seed齐备不代表导师的全部外部要求已完成。AID单seed结果不并入内部42/43/46/47四seed统计；其他历史正负结果、46/47缺失F1和原始记录继续保留。当前最低矩阵不包含DFAGCN×AID，不为本次分析新增该训练项。

## 7. 可直接复用的归档材料

- [11月8日外部比较可复制更新](supervisor_external_results_update_20261108.md)、[三项完成比较与边界](external_baselines_20261027/completed_comparison_20261106.md)、[统一机器可读比较记录](../results/external_methods_comparison_20261106.json)。
- [HGTNet AID完整离线分析](../results/hgtnet_aid_seed42_analysis_20261106.md)、[AID三模型原预测配对](../results/aid_seed42_three_model_paired_analysis_20261106.json)、[核心10份原始记录](../results/evidence/external_results_integration_20261106/aid_core_raw/)。

- [HGTNet完整离线分析与图表](../results/hgtnet_nwpu_seed42_analysis_20261104.md)、[可复制结果](../output/hgtnet_seed42_verified_summary_20261104.txt)、[同划分五模型CSV](../results/external_baselines_nwpu_seed42_20261030.csv)。
- [46/47历史主指标补录](../results/seed46_seed47_reconstruction_20261104.md)、[AID阶段记录](external_baselines_20261027/hgtnet_aid_preparation_20261104.md)。

- [DFAGCN逐类误差分析与图稿](../results/dfagcn_nwpu_seed42_error_analysis_20261031.md)、[45类明细CSV](../results/dfagcn_nwpu_seed42_per_class_20261031.csv)。

- [DFAGCN NWPU seed42正式结果与可复制说明](../results/dfagcn_nwpu_seed42_20261030.md)、[摘要与身份核验](../results/evidence/dfagcn_formal_3080_return_20261030/validation.json)。

- [BEN soft42完成及指定四seed三模型汇总](ben_four_seed_three_model_20261017.md)、[四seed CSV](../results/ben_four_seed_three_model_20261017.csv)、[soft42正式回传核对](../results/evidence/bigearth_soft42_complete_20261017/validation.json)。

- [新增46/47当前数值表](../results/multiseed_test_results.csv)、[更正/算术核对](../results/seed46_seed47_audit_20261012.md)、[原内部结果完整精度与来源SHA](../results/supervisor_progress_data_20261012.json)。
- [AID三seed](../.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.md)、[NWPU42/43](../.worktrees/sparse-hybrid-4n-top2/results/nwpu_two_seed_summary.md)、[BEN四seed](../.worktrees/sparse-hybrid-4n-top2/results/bigearth_four_seed_summary.md)、[BEN43/45三模型](../.worktrees/sparse-hybrid-4n-top2/results/bigearth_three_model_two_seed_summary.md)。
- [NWPU图册分析](../.worktrees/sparse-hybrid-4n-top2/results/nwpu_regions_visual_review_20260930.md)、[NWPU纠正与共同错误对照](../.worktrees/sparse-hybrid-4n-top2/deliverables/nwpu_regions_review_20260930/correction_and_failure_example.png)、[BEN图反馈四例图稿](../.worktrees/sparse-hybrid-4n-top2/output/pdf/seed44_feedback_figure/README.md)、[AID效率](../.worktrees/sparse-hybrid-4n-top2/results/aid_efficiency_summary.md)、[外部材料状态](../.worktrees/sparse-hybrid-4n-top2/docs/external_methods_refresh_20261006.md)。
- [相关工作初稿](../.worktrees/sparse-hybrid-4n-top2/docs/thesis_related_work_20261006.md)、[实验与讨论初稿](../.worktrees/sparse-hybrid-4n-top2/docs/thesis_experiments_discussion_20261006.md)、[BEN42/43分别对Swin的类别表](ben_seed42_seed43_vs_swin_20261013.md)、[完整逐类资料附录](per_class_findings_20261013.md)、[配套英文邮件草稿](supervisor_email_draft_20261012.md)。
