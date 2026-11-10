# HGTNet / DFAGCN：2026-10-06公开材料复查

用户最新反馈仍未收到作者回复。该状态来自用户陈述，本次未读取邮箱或发送消息。E02外部方法实测比较仍未完成。

本次进行6条网页检索，并打开HGTNet合作者主页、两篇论文全文页面，定向核查代码链接。**在这一有界范围内，未取得新可确认作者归属的完整实现、预训练权重或代码许可证。** 这不证明实现不存在，也不是全GitHub扫描；2026-09-24的直接GitHub仓库检索结论仍作为历史记录保留。

## 核对结果

| 入口 | 本次可确认信息 | 处理 |
|---|---|---|
| [Shiyu Yang主页](https://ovoyu.github.io/) | 合作者论文列表仍列出目标HGTNet、JSTARS 2024与对应DOI；该条目未见代码链接 | 可核对论文身份，不能视为代码材料 |
| [HGTNet论文全文页面](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification) | 搜索命中的ECA-MSDWNet代码链接出现在另一篇推荐论文摘要中 | 不是HGTNet仓库，不接入 |
| [DFAGCN论文全文页面](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing) | GitHub脚注指向PyTorch Geometric依赖；另列MatConvNet与LIBLINEAR | 依赖库不能替代完整DFAGCN实现 |
| [另一项同名DFAGCN](https://aaronlee2019.github.io/publication/DFAGCN) | 交通预测论文，非Xu等人的遥感分类论文 | 排除同名误匹配 |

两篇目标论文均已有发表身份，当前限制是可运行完整方法的材料和复现细节。原论文分数与本项目不同划分/训练条件的分数必须分开呈现，不能用文献最高值填补统一协议重跑栏。

## 一个需要澄清的技术缺口

DFAGCN并非完全没有公开超参数范围：论文IV-C/D给出LIBLINEAR的C网格以及部分融合系数范围，也说明GCN训练至收敛。仍待核对的是准确权重/预处理、完整融合与求解器配置、可执行停止规则，以及超参数选择使用什么数据。不能把这些细节自动换成当前Swin端到端交叉熵入口。HGTNet的图聚合表述与预训练映射缺口沿用[已有审计](external_methods_readiness_20260924.md)，本次没有新材料解决它们。

## 下一步交付边界

已有内部实验可继续形成[实验与讨论初稿](thesis_experiments_discussion_20261006.md)。收到作者材料后，先记录来源、版本、权重和许可，逐模块核对论文，完成小样本前向/训练/保存重载，再落实统一协议与多seed预算。若仍无材料，独立逐式重实现需单独明确假设与验证范围；不能悄悄简化后沿用原论文方法名。

本轮不发送催信、不克隆或安装候选仓库、不启动GPU实验、不移除导师的比较要求。本文是材料状态更新，不代表外部比较已完成。

检索词（6条）：`"HGTNet" "Hierarchical Graph-Enhanced" github`、`"DFAGCN" github Xu Huang`、`site:ovoyu.github.io HGTNet`、`"Deep Feature Aggregation Framework" code github`、`"HGTNet" "3491335" code`、`"DFAGCN" "3071369" code`。前4条存在无关结果，后2条返回空；空结果仅描述本次检索。
