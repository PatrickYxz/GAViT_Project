# AID / HGTNet / DFAGCN 复现可行性核对

09-24后续：[最新接入核对](external_methods_readiness_20260924.md)与[作者材料请求草稿](external_methods_requests_20260924.md)已完成，GitHub直接检索排除同名无关仓库，仍未核实作者实现。AID两模型三seed及效率测量已完成；下文09-19/09-20的AID待执行状态为历史。统一协议要求以[最新导师邮件](supervisor_latest_email_received_20260924.md)为准，E02尚未启动训练。

核对日期：2026-09-19。W1-09 的文献与现有代码审计已完成；AID 接入、外部方法复现和 GPU smoke 尚未执行。下一步见[实验矩阵](experiment_matrix_20260919.md)。

09-20进展：新增统一单标签入口和本地合成数据测试，见[运行指南](run_aid.md)；以下旧脚本审计保留，新AID实验使用新增入口。真实数据与CUDA验证待执行。

本地依据为 `sparse-hybrid-4n-top2` 工作区当前文件，HEAD `8f2ed07671ed50387998a46a2d2167a7a4dfc2ed`，工作区有未提交修改。网页结论以本次检查为限；“未找到已核实的官方仓库”不表示代码一定不存在。

## 1. 论文身份与原始协议

| 项目 | 已核实身份 | 与本项目有关的原始协议 |
|---|---|---|
| AID | Aerial Image Dataset，30 类、10,000 张 RGB、600×600，单标签；不是 Million-AID 或多标签数据集 | 官方基准采用 20% 或 50% 训练，其余测试；随机划分重复 10 次，报告 OA 均值和标准差 [S1] |
| HGTNet | Ziwei Li 等，*A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification*，IEEE JSTARS 17:20315–20330，2024；DOI 10.1109/JSTARS.2024.3491335 | AID 20/80、50/50；NWPU 10/90、20/80；重复 5 次，报告 OA 和混淆矩阵 [S2–S3] |
| DFAGCN | Kejie Xu、Hong Huang、Peifang Deng、Yuan Li，*Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing*，IEEE TNNLS 33(10):5751–5765，2022；2021 年在线发表；DOI 10.1109/TNNLS.2021.3071369 | AID 50/50、NWPU 20/80；随机划分重复 10 次，报告 OA、AA、Kappa 和混淆矩阵 [S4] |

共同的最小比较范围是 **AID 50% 训练 / 50% 测试**；如扩展到 NWPU，共同设置是 **20% / 80%**。这是本次归纳，不代表已生成数据划分。两篇方法论文的相关实验是单标签任务；本项目尚未决定把它们迁移到 BigEarthNet 多标签任务。

## 2. 实现差异及缺口

### HGTNet

原文结合分层 Transformer、双重注意力和 ViG。实验配置为 RGB224、部分 ImageNet 预训练、Adam、lr=1e-5、weight decay=5e-5、batch16、100 epochs、交叉熵，并使用裁剪/翻转增强。[S3，III、IV-B]

目前未核实官方实现、预训练文件及其加载映射、数据划分清单、最佳权重选择规则。原文 III 节写 max-relative graph convolution，IV-B 又提 average aggregation；复现前需核对具体模块，不能凭名称拼装一个近似网络。

工程判断：它不能当作现有 Swin-T 后面新增一个 GAT。若最终自行实现，结果应标为“依据论文重新实现”，列出未确定细节和实现选择；取得作者代码并验证之前，不写成“官方复现”。

### DFAGCN

原文使用 MatConvNet 的 ImageNet VGG16 权重：224 输入提取 FC 特征，448 输入提取多层卷积特征，PyTorch/PyG GCN 聚合后加权拼接，最终通过 LIBLINEAR 分类。GCN 使用 Adam、batch100、每 50 epochs 学习率乘 0.1、训练至收敛，不使用数据增强；线性分类器的 C 通过网格搜索确定。[S4，IV-B/C]

目前未核实官方代码、依赖锁定、预训练数值对齐、收敛判据、调参所用验证划分。论文给出的 PyG、MatConvNet、LIBLINEAR 链接是依赖，不是 DFAGCN 源码。

工程判断：这是分阶段的特征聚合方法。若换成 torchvision 权重、统一224输入、去掉线性分类器或改为端到端训练，均需明确标成改编版本，不能沿用原方法复现标签。效率测量必须覆盖实际特征提取、GCN 和分类器，不能只计缓存特征之后的分类时间。

## 3. 代码与许可证检查记录

| 对象 | 本次检查范围 | 核对结果 |
|---|---|---|
| AID | 官方项目页的数据与基准代码下载链接、说明、页尾 | 下载入口可见；本次未下载数据/代码包，页面未提供可据以确认包内代码许可证的文本。取得数据时保存来源和随包说明，不把下载入口当成明确的代码许可证 |
| HGTNet | 原文、作者 Shiyu Yang 的论文列表、精确标题/缩写与 GitHub 定向检索 | 已确认论文归属，未找到经过归属核验的官方代码仓库，代码许可证和权重许可均待核实。GitHub 仓库搜索页面一次读取失败，不以此证明没有代码 |
| DFAGCN | 原文、精确标题/缩写与作者组合检索、GitHub 仓库搜索 | 当前缩写仓库搜索页返回0项；结合其他检索仍未核实官方仓库和代码许可证。该结果不能排除其他名称仓库或非公开代码 |

HGTNet 公开论文页面标注的 CC BY-NC-ND 4.0 是论文内容许可，不作为代码许可证。DFAGCN 的公开全文由作者 Hong Huang 上传，但也不能由此推定代码许可。[S3–S4]

可继续完成数据协议、我们模型的接入和 smoke。外部代码取得后再记录仓库 URL、commit、LICENSE 和权重来源，核对 forward、训练与测试入口。需要联系作者时先准备具体缺失项；本次没有发送邮件或联系任何人，也没有导入第三方源码。

## 4. 当前仓库不能直接承担新比较的原因

| 当前文件 | 观察到的行为 | 新实验需要补齐的内容 |
|---|---|---|
| `baselines/swin_baseline/split_nwpu.py:12–26` | 70/15/15；`os.listdir` 未排序后 shuffle；向已有目录复制 | 新建显式、分层、可校验的样本清单；保存排序规则、类别映射、split seed 和哈希；旧划分保留，不靠同一个 seed 重新生成来替代旧证据 |
| `train_gavit.py:75,170`；`baselines/swin_baseline/train_swin_baseline.py:38,60` | 类别数45与 NWPU metadata 写死 | AID 使用30类及正确 dataset 身份；不能只改 `DATA_ROOT` |
| 两个训练文件的 train transform | GAViT 有水平/垂直翻转和 ColorJitter；Swin 只有水平翻转 | 新的 Swin/GAViT 结构对比共享增强配置与实现 |
| `train_gavit.py:159–162`；Swin训练文件`:131` | GAViT 显式 wd=1e-4；Swin AdamW 未显式传 weight_decay | 两边明确记录相同的 lr、wd、调度和轮数，不依赖默认值 |
| `train_gavit.py:46` | 默认 edge_type 为 knn | 固定模型显式指定 K16、attentive_spatial、sparse_hybrid、token_feedback，并验证48+32条边 |
| `test_gavit.py`；`baselines/swin_baseline/test_swin.py` | 类别身份固定为 NWPU；当前评估主体只输出 Accuracy | 输出 OA、macro-F1、逐类正确率/支持数、混淆矩阵与逐样本预测，保存完整精度及 exit 状态 |

这些是当前代码观察，不足以反推每次历史服务器运行都使用相同文件。`results/comparison_table.csv` 的部分旧数字为近似值、部分 test 为空；它们不能升级成新协议下的正式比较结果。

## 5. 论文呈现与结论边界

实验表区分三种来源：原论文报告值、可追溯实现的本地重跑、我们自行改编的模型。原论文不同训练比例、重复次数与预训练条件须显式列出；即使比例相同，也不能称为同一 split 的配对结果。

我们自己的 Swin / 固定分组 / 可学习分组用于结构消融：共享 backbone、split 和训练流程，主要变量是分组或图模块。HGTNet 与 DFAGCN 的 backbone 和流程不同，用于完整方法比较，不能把性能差异全部归因于图推理。

本次不抄录外部 OA 数字进入正式结果表；完成协议与实现核验后，另以原文表格逐项核对数值及标准差。当前 BigEarthNet 的正式测试记录、seed44诊断和图稿均不变。

## 来源

- [S1：AID 官方项目页](https://captain-whu.github.io/AID/)，数据、Evaluation protocols、Reference。
- [S2：HGTNet 作者论文列表](https://ovoyu.github.io/)，2024 年条目；[DOI](https://doi.org/10.1109/JSTARS.2024.3491335)。
- [S3：HGTNet 公开论文全文](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification)，III、IV-B/C。IEEE页面本次直接读取失败，实验设置根据公开全文核对。
- [S4：DFAGCN 作者上传论文全文](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)，标题页、IV-B/C；[DOI](https://doi.org/10.1109/TNNLS.2021.3071369)。
- [S5：DFAGCN GitHub 仓库名称检索](https://github.com/search?q=DFAGCN&type=repositories)，仅作本次检索记录，不作为论文技术结论来源。
