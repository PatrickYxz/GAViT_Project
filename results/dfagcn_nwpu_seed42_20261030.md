# DFAGCN独立实现：NWPU seed42正式结果

实验批次：2026-10-30。来源：用户回传服务器run.exit及summary，exit_code=0、completed=true。正式流程验收状态为SERVER_REPORTED_FORMAL_COMPLETE_SUMMARY_IDENTITIES_MATCH；不是原作者官方实现或多seed结果。

## 本次结果

- Test OA：86.53174603174604%，显示为86.53%。
- Test macro-F1：86.45890874660422%，显示为86.46%。
- 测试样本：25200；2026-10-31已取回完整test.json逐图复算，确认21806张正确、3394张错误，45类各560张。
- 验证集选定训练轮数：51；C：10。校准配方上限100轮；按已选51轮从同seed新模型在6300池上重新拟合。
- 划分：5040 train、1260 val，fresh6300 train+val refit、25200 test。45类；固定manifest SHA与既有NWPU三模型一致。
- 重载特征最大误差：0.0。

## 同划分、同seed42的完整方法比较

以下是单seed实际结果，不合并进原四seed均值/SD，也不使用旧约96%的不同协议占位数。

- Swin-T：OA93.932540%，macro-F1 93.909145%。
- 固定4×4 GAViT：OA94.103175%，macro-F1 94.089933%。
- 局部软分配GAViT：OA93.869048%，macro-F1 93.854987%。
- DFAGCN独立实现：OA86.531746%，macro-F1 86.458909%。

固定GAViT比本次DFAGCN独立实现高7.571429个百分点OA、7.631025个百分点macro-F1，对应净多1908张正确。Swin比DFAGCN高7.400794个百分点OA，软分配版高7.337302个百分点。该净正确数差不等于逐图配对的“纠正/新增错误”数，配对统计尚未计算。

三项旧对照来源：[已归档seed42正式比较](../.worktrees/sparse-hybrid-4n-top2/results/nwpu_seed42_test_comparison.md)。精确机器表：[四模型结果CSV](external_baselines_nwpu_seed42_20261030.csv)、[本次完整结果记录](dfagcn_nwpu_seed42_20261030.json)。

## 比较能说明什么

共同之处是固定数据清单、标签映射、5040/1260内部选择、fresh6300 refit及25200最终test的信息隔离原则。本次可报告披露配置下完整方法的单seed表现差异。

DFAGCN使用冻结torchvision VGG16、224/448输入、三独立GCN、固定加权融合及LIBLINEAR；池化、层宽、预处理和部分优化选项是事先披露的独立选择。Swin系列使用不同骨干/预训练、增强及端到端优化。DFAGCN按验证CE选epoch、验证OA选C，旧三模型按验证OA选epoch。三项旧seed42对照使用RTX2080Ti，本次DFAGCN使用RTX3080。因此，不能把7.57pp全部归因于图模块，也不能据此比较同设备效率或建立跨seed稳定优势。

已归档原论文报告点值89.29%仅作文献参考；本次低约2.758254pp。原文十次随机划分、MatConvNet及未完全核定的层配置与当前独立配方不同，不能声称已复现原文数值，也不据差距推断某个原因已经成立。文献范围见[原文数值说明](../docs/external_baselines_20261027/literature_comparison.md)。

## 资源与耗时

- 实际GPU：RTX3080 10GiB，UUID GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed，驱动610.57.04。
- Python3.11.8，Torch2.2.2+cu121，torchvision0.17.2+cu121，CUDA12.1/cuDNN8902，CPU8线程。seed42、FP32，AMP/双TF32关闭，cudnn deterministic=true/benchmark=false。
- run记录wall：2035.63461228秒，即33分55.63秒；起点在预检之后，不包括上传、预检和租卡空闲，不等于计费总时长。
- pool提取阶段168.95秒；train/val切片13.60秒；校准790.69秒；refit255.99秒；test缓存阶段695.65秒；最终test阶段100.97秒。阶段合计2025.86秒，其他run内开销约9.78秒。
- PyTorch各阶段allocated最大值616995328字节=588.41MiB；reserved最大值698351616字节=666MiB。两个最大值来自不同阶段，不是nvidia-smi整进程显存。
- 缓存后的测试子段66.66秒，排除VGG和部分前置加载，包含图/融合、SVM文本及预测；不是纯神经前向延迟。test缓存生成加完整test阶段796.62秒，含恢复/加载/缓存读写，不直接与旧GPU模型推理速度比较。
- 冻结VGG提取器117479232参数；三图分支含CE训练头2957447参数；合计120436679神经参数，约120.44M，另有LIBLINEAR。不能只把2.96M写作完整模型大小。
- 实际租赁费率未提供，不估算费用。

## 验收与尚未回收到本地的内容

独立29/29项核验通过，主代理另复算runtime/source身份SHA和OA整数计数；六阶段身份、配方、源码及权重报告与交付包一致。原始回传、规范化JSON、完整检查见[证据索引](evidence/dfagcn_formal_3080_return_20261030/README.md)。没有重新训练或重新运行test。

2026-10-31已下载完整test.json，在本地从路径类别与预测重建全部混淆矩阵和指标，独立32项核验通过。复算macro-F1 86.45890874660425%与服务器报告仅差2.84×10^-14。逐图/逐类/混淆矩阵已归档；原manifest/图像、selection/refit历史和checkpoint原件仍未回本地，不声称对这些外部工件完成字节审计。[误差分析与全部明细](dfagcn_nwpu_seed42_error_analysis_20261031.md)。

本次逐类复核与误差分析已完成。旧三模型的完整预测未在本地，尚不能做与DFAGCN的逐图纠错/新增错误配对；现有净差仍只作总体计数差。HGTNet实测和外部比较的重复实验范围仍是后续独立工作，不因本次分数临时修改配置，也不为本次离线复核追加训练。

## 可直接复制到导师汇报

已完成DFAGCN独立实现的NWPU seed42正式实验。在既有固定划分下，模型经5040/1260训练验证选择后，在6300张训练池上重新拟合，并完成25200张测试；测试OA为86.53%，macro-F1为86.46%。同一划分上，固定4×4 GAViT的OA为94.10%，高7.57个百分点。本次结果体现已披露配置下完整方法的单seed差异，DFAGCN与Swin系列的骨干、预训练、优化和硬件不同，尚不能视为原论文数值复现或GAViT稳定优势的证明。逐类预测、混淆矩阵和OA/macro-F1现已完成本地复算；后续继续其余外部方法比较。
