# HGTNet / DFAGCN 作者材料请求草稿

**2026-09-25状态更新：** 用户确认已发送作者请求邮件，现等待回复。发送状态依据用户回报；实际收件人、正文、投递回执及回复未在本地核验。以下保留09-24准备的草稿，不作为已发送邮件的逐字副本。

2026-09-24草稿记录：当时尚未发送。姓名及邮箱取自两篇论文首页的公开作者联系信息；未通过实际投递验证地址当前是否有效。正文不包含本项目数据、权重或尚未公开的实验数值。

## 1. HGTNet

To: xwming2@126.com（通讯作者 Weiming Xu）

可选抄送：225520010@fzu.edu.cn（第一作者 Ziwei Li）。默认只联系通讯作者，不自动抄送其他作者。

Subject: Request for HGTNet implementation and pretrained weights for thesis comparison

Dear Dr. Xu,

My name is Xingze Yang. I am working on a thesis on graph-enhanced remote sensing scene classification and would like to include HGTNet in a controlled comparison on AID and NWPU-RESISC45.

I have read your paper, “A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification” (DOI: 10.1109/JSTARS.2024.3491335), but have not located a verified author implementation. Could you please share the code or its repository link, pretrained-weight source and loading instructions, and training/evaluation configuration? Please also let me know any conditions for research use and whether modifications may be shared.

Two details would particularly help ensure a faithful implementation:

1. Which ImageNet model supplies the partial pretrained weights, and which layers are loaded or reinitialized?
2. Section III describes max-relative graph convolution, while the experimental settings mention average aggregation. Which operator and configuration were used in the reported experiments?

Any split-generation scripts, checkpoint-selection rules, or environment requirements would also be very helpful. I will clearly distinguish our controlled-protocol results from the original paper's results and cite your work appropriately.

Thank you for your time and assistance.

Best regards,
Xingze Yang

来源：[HGTNet作者论文全文](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification)，首页作者联系信息、III-C与IV-B。作者身份另与[合作者主页](https://ovoyu.github.io/)核对。

## 2. DFAGCN

To: hhuang@cqu.edu.cn（通讯作者 Hong Huang）

可选抄送：xukejie@cqu.edu.cn（第一作者 Kejie Xu）。默认只联系通讯作者。

Subject: Request for DFAGCN implementation and feature-extraction pipeline

Dear Professor Huang,

My name is Xingze Yang. I am working on a thesis on graph-enhanced remote sensing scene classification and would like to include DFAGCN in a controlled comparison on AID and NWPU-RESISC45.

I have read “Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing” (DOI: 10.1109/TNNLS.2021.3071369). Could you please share the author implementation or repository, along with any research-use and redistribution conditions?

To reproduce the complete method faithfully, I would particularly appreciate:

- The exact MatConvNet VGG16 checkpoint and preprocessing, feature-layer, and normalization settings.
- The GCN architecture, graph construction, training configuration, and stopping criterion.
- The weighted feature-fusion settings and LIBLINEAR solver configuration, including how the fusion weights and C are selected without using test labels.
- Any split-generation scripts, pretrained or trained weights, and environment requirements.

Our comparison will use a shared data split and a clearly documented evaluation protocol. We will preserve and disclose method-specific stages and distinguish controlled adaptations from the original reported results.

Thank you for your time and assistance.

Best regards,
Xingze Yang

来源：[DFAGCN作者论文全文](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)，首页作者联系信息与IV-C/D。

## 3. 对方回复后怎样使用

先保存回复与材料来源，再核对代码版本、完整前向/分类流程、预训练映射和使用条件。对不能共享的材料仅记录必要身份，不擅自公开。先做本地工程检查和服务器小样本验证，再冻结正式比较方案；不直接启动全量训练。

09-24准备草稿时，助手没有发送邮件、创建GitHub issue或联系任何作者。09-25用户自行发送的状态见顶部记录。
