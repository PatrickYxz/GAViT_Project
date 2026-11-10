# HGTNet独立本地实现设计

用户已批准上一轮提出的HGTNet实现和本地工程验证。沿用独立实现路线，先完成可核验模块、完整模型、候选预训练映射及保存重载，不启动Featurize或正式训练。原模型/DFAGCN产物及用户未提交内容保持；新代码只落external_baselines/hgtnet、对应新config/tests和文档，当前checkout原位新增，不提交推送。

## 来源与实现选择

原文事实见[hgtnet原文审计](hgtnet_audit.md)及[本轮主来源复核](../../results/evidence/hgtnet_local_implementation_20261031/source_review/primary_review.json)。DA并行相加、通道128/256/512、STB1/1/9（每STB两attention）、heads4/8/16、stage输出256/512/512、ViG组深度6/2、实验k20、skip融合有文字证据。MRmax与average冲突仍未消解；首版明确mr_max，relative_mean只用于单算子歧义检查，不追加完整训练。

原图像素/完整作者代码与权重仍未核验，不能把工程选择称为作者结构。此前14→7候选会跳过第三stage下采样。首个可执行版本采用更贴合“每stage下采样”文字的候选：

- RGB224 → Conv4/stride4+LN →128×56×56。
- stage0：DA128→两个Swin attention block（shift0/3）→torchvision PatchMerging→256×28×28。
- stage1：DA256→两个block→PatchMerging→512×14×14。
- stage2：DA512→十八个block→Conv2/stride2，同512通道，后LN→512×7×7。
- 第一graph transition：DWConv3/stride1/pad1+pointwise512→256+节点LN，仍7×7。6个Grapher/FFN，k20。
- 第二transition：DWConv3/stride1/pad0+pointwise256→512+节点LN，7→5。2个Grapher/FFN，k20。
- visual512×7×7和graph512×5×5各GAP，相加后Linear512→45（NWPU）或30（AID），输出原始logits。

shape_policy=independent_text_7_to_5。第三stage Conv2、第二DWConv无padding、graph宽256/512、GAP相加均是本项目明确选择，非原图确认值。另一个14→7候选不在此版默认实现中；不按将来test分数切换。

DA channel共享MLP reduction16/ReLU，avg/max池化后共享两次并相加再sigmoid。空间分支2→1→1→1三个3×3卷积，dilation/padding1/2/3，无中间激活，最终sigmoid。Fout=F*Mc+F*Ms；不平均，不额外加F。

Swin使用已安装torchvision0.17.2的SwinTransformerBlock、PatchMerging，窗口7/shift3、MLP ratio4、dropout/attention_dropout/drop_path0、LN eps1e-5，均显式记录。源库复用，不复制大段实现；不假装是完整Swin-T或Swin-B拓扑。

图以B×N×C节点表示。kNN基于当前Grapher映射后的特征，欧氏距离、排除自身、stable argsort按原索引破同距，no_grad构图，每个block重新构图。必须N>=k+1，禁止静默裁k。MR消息max_j(x_j-x_i)，与中心拼接再投影。Grapher为C→C/LN→MRConv(C→2C)→GELU→Linear2C→C+原输入；FFN为LN→LinearC→4C→GELU→Linear4C→C+残差。归一化/宽度为独立选择，无BN或随机边丢弃。

## 独立模块接口

attention.py：

- PatchEmbedding(in_channels=3,embed_dim=128,patch_size=4,norm_eps=1e-5)，属性projection/norm，BCHW→BCHW。
- DualAttention(channels,reduction=16)，属性channel_mlp/spatial，BCHW→同形。
- TransformerStage(channels,stbs,heads,*,output_channels,downsample_kind,window_size=7,mlp_ratio=4.0,dropout=0.0,attention_dropout=0.0,stochastic_depth_prob=0.0,norm_eps=1e-5)。属性da/blocks/downsample。downsample_kind为patch_merge或conv2_same_channels，forward BCHW→BCHW。blocks为Sequential单attention模块，交替shift0/3。

graph.py：

- knn_indices(nodes,k,exclude_self=True)→LongTensor B×N×k。
- relative_aggregate(nodes,neighbours,reduction='max'|'mean')→B×N×C；显式消息方向neighbor−query，batch隔离。
- MRConv(in_channels,out_channels,reduction='max')，属性projection=Linear(2Cin,Cout)，forward(nodes,neighbours)。
- GraphBlock(channels,k,operator='mr_max',ffn_ratio=4,norm_eps=1e-5,exclude_self=True)，forward(nodes,*,return_neighbors=False)默认Tensor，True返回(output,actual_indices)。
- DepthwiseProjection(in_channels,out_channels,*,stride=1,padding=1,norm_eps=1e-5)，DW3+pointwise1+节点LN，BCHW→BCHW。

model.py：HGTNet(config)，属性patch_embed、stages(ModuleList)、graph_transition1、graph_blocks1、graph_transition2、graph_blocks2、head；forward(x)→logits，forward_with_trace(x)→(logits,JSON-safe shape/actual-neighbour-shape trace)。所有模块全新建，配置/类数30或45明确；输入dtype/device/shape/finite严格校验。

## 候选预训练

原文只写partial ImageNet，没有已核验的作者来源。torchvision Swin-B IMAGENET1K_V1仅作候选，因为前三阶段维度/深度匹配；不能套本项目Swin-T。当前安装源枚举URL为https://download.pytorch.org/models/swin_b-68c6b09e.pth。本轮后续实际取得351654207字节文件，完整SHA为68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893；该值只表示本轮候选文件身份，不证明是作者使用的权重。

映射命名固定：

- patch_embed.projection.* ←features.0.0.*；patch_embed.norm.* ←features.0.2.*。
- stages.i.blocks.j.* ←features.(1+2i).j.*，i=0,1,2。
- stages.i.downsample.* ←features.(2+2i).*，仅i=0,1。
- stage2新downsample、全部DA/graph/fusion head明确初始化；source features6/7、norm/head为明确不使用项。

pretrained.py实现audit_pretrained_state(model,state,*,validate_values=True)、apply_pretrained_state(model,state)、load_pretrained_file(model,path,expected_sha256)。audit输出compatible、missing、unexpected、shape_mismatch、dtype_mismatch、nonfinite、intentionally_initialized、source_intentionally_skipped，数值/布局不完整时不能apply；schema-only/meta检查不得标加载成功。validate全部在模型变更之前；只加载确定映射，并核验实际missing恰好为新模块，unexpected为空。实际文件loader须完整SHA且匹配68c6b09e前缀、weights_only安全加载，记录候选来源而非HGTNet官方预训练。

若能在有界时间取得官方候选文件，做实际CPU加载和映射一致性检查并冻结完整SHA；否则明确交付“映射代码与schema已验证、真实文件尚待获取”，不能把meta或随机source当成ImageNet已加载。没有权重材料也先完成真实完整随机模型的数值前后向/重载工程检查；正式GPU训练必须等待实际预训练身份核验。

## 完成本轮的证据

TDD先RED再实现：DA固定门控/非方形，STB数和shift/形状，手算MR max3 vsmean2、排列/批次/自环/同距/小N拒绝、真实梯度、完整B1/B2×224输出、45/30头、主要模块梯度与一步参数更新、实际checkpoint新进程重载。保存配置/逐层trace/参数量/软件/源码SHA/本地时间及测试日志。单元测试与合成local smoke不产生遥感准确率或GPU资源宣称。

当前scope不接正式train/val/test CLI、不下载数据/租卡、不改旧入口、不发送邮件、不更改历史结果表。完成后更新research_diary与本包README，提供可复制本地工程命令及下一阶段预训练/目标GPU仍待事项。

本轮实现与本地工程检查已完成，实际运行见[本地证据](../../results/evidence/hgtnet_local_implementation_20261031/README.md)。候选实际CPU加载已通过，目标GPU与真实数据门禁仍待下一阶段。
