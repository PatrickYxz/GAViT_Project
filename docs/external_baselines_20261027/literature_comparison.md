# HGTNet与DFAGCN：原论文报告值及可复制说明

整理日期：2026-10-27。以下均为原论文报告值，来源类型为`paper_reported`，不是本项目重新训练或重新测试的结果。

## 1. HGTNet

论文：Li等，A Hierarchical Graph-Enhanced Transformer Network for Remote Sensing Scene Classification，IEEE JSTARS，2024，DOI：10.1109/JSTARS.2024.3491335。

AID，20%训练、80%测试：原文报告OA均值96.91%。

AID，50%训练、50%测试：原文报告OA均值98.47%。

NWPU-RESISC45，10%训练、90%测试：原文报告OA均值94.52%。

NWPU-RESISC45，20%训练、80%测试：原文报告OA均值95.75%。

论文采用5次重复。具体重复方式、模块和协议核对见[HGTNet审计](hgtnet_audit.md)；[原论文公开全文](https://www.researchgate.net/publication/385542196_A_Hierarchical_Graph-Enhanced_Transformer_Network_for_Remote_Sensing_Scene_Classification)。

本轮从可读取的原文正文核对上述OA均值；相关表格图片未能直接读取，标准差暂未转录。这不是声称论文未报告标准差；标准差不得填0，也不从其他论文的对比表补入。

## 2. DFAGCN

论文：Xu等，Deep Feature Aggregation Framework Driven by Graph Convolutional Network for Scene Classification in Remote Sensing，IEEE TNNLS，2022卷期、2021在线发表，DOI：10.1109/TNNLS.2021.3071369。

AID，50%训练、50%测试：作者报告OA点值94.88%。

NWPU-RESISC45，20%训练、80%测试：作者报告OA点值89.29%。

论文采用10次随机划分重复。模块、优化和完整分类链见[DFAGCN审计](dfagcn_audit.md)；[作者上传原论文](https://www.researchgate.net/publication/350910107_Deep_Feature_Aggregation_Framework_Driven_by_Graph_Convolutional_Network_for_Scene_Classification_in_Remote_Sensing)。

本轮从原文IV-F的结果讨论核对上述点值，对应Table V和Table VII，印刷页5762；均值和标准差所在表格单元格尚未直接读取，暂不声称完成均值±SD逐格核对。保留论文原有打印精度，不补造更高精度、标准差、AA或Kappa。

## 3. 与本项目的关系

两篇方法与本项目共有的名义设置是AID 50%训练，以及NWPU 20%训练。本项目对应的真实固定清单已经生成，不能再按旧材料说NWPU清单仍待创建。

本项目AID在4000/1000内部校准后，用5000张训练池重新拟合并测试5000张；NWPU在5040/1260内部校准后，用6300张训练池重新拟合并测试25200张。详见[协议与阶段验收](protocol_and_acceptance.md)。

DFAGCN十次随机划分，与本项目固定划分上的多个训练seed包含不同随机因素；HGTNet五次重复是否每次重划分，本轮尚未确认。比例相同也不等于共享同一测试图片清单。文献报告值可以作为背景比较，暂不用于逐样本配对差、显著性判断或“统一协议重跑已完成”的声明。

HGTNet、DFAGCN与GAViT的骨干和完整流程也不同；外部方法对比评估完整方法表现，不把差值全部归因于某种图模块。以后独立复现结果与这里的文献值分别标注来源和协议，不混填已有本项目性能表。

## 4. 可直接复制进论文的说明

作为相关方法的文献参考，HGTNet在AID 50%训练和NWPU-RESISC45 20%训练设置下，分别报告98.47%和95.75%的OA；DFAGCN在相应设置下分别报告94.88%和89.29%。这些数值取自原论文，是各自实验协议下的报告值。本研究采用固定划分、内部验证选择和最终训练池重新拟合流程；DFAGCN采用十次随机划分，HGTNet的五次重复机制仍待进一步核对。因此，文献结果用于描述已有方法的表现背景，统一数据与评估流程下的直接比较将单独报告本文独立复现的结果。

正式论文中在HGTNet和DFAGCN首次出现处分别引用两篇原论文；本段没有声称GAViT已经优于其中任何方法。

## 5. 机器可读记录

- [HGTNet逐项来源与打印值](../../results/evidence/external_baselines_20261027/hgtnet/reported_results.json)。
- [DFAGCN逐项来源与打印值](../../results/evidence/external_baselines_20261027/dfagcn/reported_results.json)。

缺失标准差用JSON的`null`表示。HGTNet正文OA均值已核对；DFAGCN正文OA点值已核对，均值/SD单元格尚未视觉确认。二者都不显示成已完整转录的均值±标准差。
