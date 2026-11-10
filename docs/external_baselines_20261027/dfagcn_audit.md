# DFAGCN 原文证据与独立实现准备

核对日期：2026-10-27。范围是原文转录、模块规格和有限工程准备；本轮没有实现模型、安装依赖、训练、测试、发信或改动历史结果。

## 1. 本轮直接查到什么

论文为 Xu、Huang、Deng、Li 的 *Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing*，DOI [10.1109/TNNLS.2021.3071369](https://doi.org/10.1109/TNNLS.2021.3071369)，2021年在线发表、2022年卷33期10印刷页5751-5765。本轮重新读取[作者 Hong Huang 上传的原文页面](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)，页面显示2022-11-15上传；旧项目审计仅用于定位，没有作为本轮数值来源。

网页全文正文可读，但表格单元格没有进入文本抽取。原PDF下载链接报错；标准库直接访问页面先遇到沙箱DNS限制，获只读网络执行后返回HTTP403；正常Chrome打开该原网址停在设备检查，未处理挑战或绕过限制。因此没有取得原PDF字节、表格截图或本地PDF渲染。可复查的本轮原文短摘录、印刷页和网页行定位保存于[primary_web_evidence.json](../../results/evidence/external_baselines_20261027/dfagcn/primary_web_evidence.json)。

## 2. 作者报告值：保留已知数，不猜标准差

AID，50%训练、50%测试，OA作者报告点值为94.88%；相关比较表为Table V，印刷页5762。正文IV-F直接写明该OA，原文IV-C规定十次随机训练/测试划分。标准差未从原表核对，记为null。本轮未直接看到表内均值符号，不能将正文点值升级为已逐格核对的“均值±SD”。

NWPU-RESISC45，20%训练、80%测试，OA作者报告点值为89.29%；相关比较表为Table VII，印刷页5762。正文IV-F直接写明该OA，重复方式同为十次随机划分；标准差同样记为null，均值/SD单元格仍未视觉核对。

[reported_results.json](../../results/evidence/external_baselines_20261027/dfagcn/reported_results.json)以字符串保留两位小数、比例、十次重复、表号、印刷页、来源URL和短摘录；不以其他论文引用值补SD。这是文献报告记录，不是本项目重跑。两项缺失SD不阻止先做已明确模块的工程准备；补数时只补原表直接证据。

## 3. 完整方法链及论文对应位置

以下来自原文III-A至III-D、IV-B至IV-D；细项和空缺可机读于[method_spec.json](../../results/evidence/external_baselines_20261027/dfagcn/method_spec.json)。

第一步是ImageNet VGGNet-16特征提取链。原文给出MatConvNet权重来源；224×224输入取第一个FC层ReLU后的4096维向量。另一输入尺寸448×448，在Conv3-3、Conv4-3、Conv5-3取ReLU之前的激活，形状分别为112×112×256、56×56×512、28×28×512。不能将卷积特征默认改成ReLU之后，也不能把全部输入统一到224后仍称原方法。依据为III-A印刷页5755、IV-B印刷页5757。

第二步是每张图、每个卷积层单独构图。III-B说明先平均池化到适当大小，再把每个位置的通道向量作为节点。式(1)规定k=10，只要任一方向属于k近邻便置边，得到二值、无向的kNN并集图。本轮正文没有明确邻居距离度量、坐标是否参与、池化核/步长/最终节点数；这些不能用本项目48空间边+32特征边规则替换。图6和Table I的图像细节仍需直接读取，不能把未读到说成论文没有给。尤其是“先池化到7×7、49个节点”本轮没有原文依据，pooling目标和节点数明确为null；不能继承GAViT末级7×7设置。依据为印刷页5755。

第三步是分别训练三支图级特征处理器。式(2)-(4)为A加单位矩阵、计算度矩阵、以D的负二分之一次幂作两侧归一化，再传播节点特征并施加激活。式(5)对节点做全局平均；后续多层FC、ReLU、dropout和softmax用于图级分类。IV-B明确最终送线性分类器的是Table I标记的最后ReLU输出，每支512维。完整GCN/FC层数、隐藏维度和dropout率来自Table I，本轮未读到单元格，分别记为null；唯一核实的输出是各支最后ReLU的512维，不能凭现有GAViT的两层四头GAT填入。依据为印刷页5755-5758。

第四步是加权拼接。式(6)的顺序为Fconv5、alpha×Fconv4、beta×Fconv3；式(7)再拼gamma×Ffc。它不是三支相加。按三支各512和FC4096推导，总长度为5632；5632是本轮维度推导，不冒充原文打印数。是否分支归一化、是否拼接后再归一化，当前正文不足以确定。依据为印刷页5756。

第五步是LIBLINEAR监督分类，不能删除该阶段改接当前端到端交叉熵分类头后沿用原方法名。C网格为10的-4至4次幂；具体solver、bias、多分类策略、特征缩放和调参数据规则未在已读段落明确。依据为IV-C印刷页5757。

## 4. 原文已给出的范围与最终系数

GCN训练使用Adam默认参数、batch100、每50轮学习率乘0.1、训练到收敛、不使用数据增强。原文段落没有给出能直接执行的收敛判据和历史依赖版本；不能把今天框架的默认值默认为作者当年的准确配置。初始化学习率可在独立实现中显式选定并披露，不能写成原文已核实数。

alpha与beta的公开网格为0、0.1、0.2、…、0.9、1、1.2、1.4、1.8、2；本轮依据可读原文列表转录，没有擅自补1.6。gamma网格为0至1.4、步长0.1。原文后续AID采用alpha=0.4、beta=0.3、gamma=1.1；NWPU采用alpha=0.7、beta=0.3、gamma=0.4。C网格、融合网格以及这些最终系数均已有公开文字，不能再笼统记作“作者未公开融合配置”。依据为IV-C印刷页5757、IV-D印刷页5758-5759。

原文对“根据什么内部验证集选择C、融合系数和收敛点”没有在已读段落说明清楚。仅凭这个披露缺口不能断言作者使用测试集调参。本项目则明确禁止用test选择C、alpha、beta、gamma、停止轮次或checkpoint。可以预先固定上述论文系数作为一个注册配置，再只用train/val选择C和停止规则；若另调融合系数，也只用train/val，并保留调参预算。

## 5. 可以自行决定的实现假设与核心未决项

无需等作者回信，可以先使用独立实现身份，完成确定模块；不称official、不称已复现论文分数。现在能够不猜核心结构就开始的是特征hook、数学算子、融合和接口；不能先任意填GCN层宽/7×7池化再宣称完成完整DFAGCN。

可合理决定并预先记录的工程约定包括：特征缓存键包含原图SHA、split身份、权重SHA、预处理和层位置；图节点按行优先展开；kNN并集边排序去重；自环只在GCN归一化阶段添加一次；邻居同分按节点索引稳定选择；浮点精度、误差阈值和保存重载规范显式固定。这些是可复查的独立实现约定，不声称作者实现相同。

可先写成参数化而不选正式默认的部分包括：池化目标大小、邻居度量、GCN层宽与dropout、特征归一化、LIBLINEAR solver和收敛上限。合成fixture可自行选小尺寸，只验证算子，不能把fixture参数带入正式复现身份。

决定完整模型身份的核心未决项是Table I完整结构、池化目标、精确MatConvNet checkpoint及输入处理；此外要冻结LIBLINEAR配置和可执行的train/val停止/选参规则。先取得可直接阅读的Table I或原PDF能解除一部分缺口，并不以作者回信作为唯一途径。若仍不得不选择未明确的细节，逐项披露，最终名称保留“依据论文独立实现”，不把临时近似结构冒充原版。

## 6. 有限的下一步与完成标准

1. 完成VGG层位置/激活时机规格和权重转换映射审查。最小fixture要求224分支4096维、448三层形状正确；权重不可得时只能标结构fixture，不能声称数值等价。
2. 完成k=10对称并集图、GCN归一化、节点均值及融合的纯算子检查。至少覆盖边对称/无重复、单次自环、手算小图传播、样本间无边、三支512加FC4096的5632维和固定拼接顺序。无需完整数据或GPU训练。
3. 直接补读Table I，列出全部层和池化配置；确认MatConvNet来源文件、归一化和转换差异。检查材料本身，不为弥补材料缺口启动训练。
4. 冻结独立实现配置及验证规则，接入现有manifest和预测/混淆矩阵/OA/macro-F1接口。当前NWPU是已生成的5040/1260校准、6300训练池重新拟合、25200测试；AID是4000/1000校准、5000拟合、5000测试。split42/val4242固定；这些固定split训练seed重复与作者十次随机划分不同。依据见[当前NWPU正式结果](../../.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed43_test_comparison.md)和[AID完成结果](../../.worktrees/sparse-hybrid-4n-top2/results/aid_multiseed_summary.md)，不沿用旧审计“NWPU尚待生成”状态。
5. 完整配置与材料门禁通过后，下一阶段才是小样本前向、反向、验证与存盘重载；它只验工程链，不进正式性能表。记录首轮时间/内存后再提出单seed成本，正式多seed另按既定论文问题安排。本轮未运行该阶段。

当前准备工作的完成条件是：本轮原文数值和缺失项可追溯、完整链未被简化、实现假设与核心缺口分开、有明确的无训练模块任务。现有交付已达到这个准备条件；模型代码和正式外部方法比较尚未完成。

## 7. 来源与可机读记录

- [作者上传的原文全文](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)：本轮数值、公式和公开参数的唯一外部技术来源。
- [直接原文证据与访问范围](../../results/evidence/external_baselines_20261027/dfagcn/primary_web_evidence.json)。包含原文摘录、网页行定位、印刷页及获取限制。
- [作者报告值JSON](../../results/evidence/external_baselines_20261027/dfagcn/reported_results.json)。两项OA点值已核，SD均为null。
- [完整链与超参数JSON](../../results/evidence/external_baselines_20261027/dfagcn/method_spec.json)。表格未读字段明确未决。

本轮未读到原表SD，不将任意二手引文填成完整均值±SD；也不将文献值当作同一固定split下的实测比较。
