# 3×3与4×4性能结果可比性审查

2026-10-17。任务仅为检查已有结果并配对，不启动训练。

## 结论

本地归档中尚未找到可与当前固定4×4 GAViT配对的BEN固定3×3正式性能结果。服务器持久化目录尚未直接检查，因此不能断言服务器没有此实验。当前3×3的BEN test mAP及其相对4×4的差值保持空缺。

| 数据集及记录 | 3×3 | 4×4 | 是否能回答区域数的影响 |
|---|---:|---:|---|
| 早期NWPU历史日记，test accuracy | 95.9%（原记录一位小数） | 95.8%（原记录一位小数） | 否：同时更换池化和反馈结构，且不是当前测试协议 |
| BEN当前固定稀疏混合图，seed42，test mAP | 待确认 | 72.00% | 尚无3×3配对，不能计算差值 |

旧NWPU数值只是历史线索，未重新测试、未由本地权重或原始测试输出复核。不能因显示差0.1个百分点就判断3×3更好，也不能据此证明4×4更好。

## 本地查到的材料

1. [早期比较表](../results/comparison_table.csv)包含K9 spatial+GAT验证96.5%、K9 kmeans+GAT验证96.0%，对应test栏为空；region-only验证96.2%也不是完整GAViT测试结果。[消融表](../results/ablation_study.csv)中K4/K9/K16行没有性能数据，是计划表。
2. [研究工作树日记](../.worktrees/sparse-hybrid-4n-top2/research_diary.md)的2026-03-22及04-04条目记录K9 SpatialGrouping/均值池化及K16 AttentiveSpatialGrouping/token_feedback。旧K16 test为4526/4725，不能混入当前25200张NWPU test汇总。日记中的推测性原因不作为此次实验结论。
3. 旧K9权重名线索为best_gavit_K9_spatial.pth、best_gavit_K9_spatial_knn_fusion.pth；本地包括研究工作树未找到模型权重文件。未读取或加载服务器checkpoint。
4. [研究工作树BEN表](../.worktrees/sparse-hybrid-4n-top2/results/bigearth_comparison.csv)的13条记录没有K9条目。70.4%标为历史v2未修正kNN；70.6%是修正kNN K16；72.0%是稀疏混合图K16。不能把70.4%或70.6%改标为3×3并与72.0%计算“区域数提升”。70.4%独立metadata尚未取得。
5. Git首个BEN提交0b1e02c（2026-04-04）的train_bigearth.py默认num_regions=16；jobs/run_bigearth_train.sh明确指定K16、attentive_spatial、token_feedback及kNN。此为历史代码配方，不能证明从未有人用覆盖参数运行K9。针对BigEarth相关文件的K9/num_regions=9历史差异搜索未找到匹配提交。
6. 已回传的固定BEN42 metadata为K16/attentive_spatial/sparse_hybrid/token_feedback，cd39be9、seed42、best13/final30；[原回传](../results/evidence/bigearth_soft42_preflight_20261016/server_return.json)保存了checkpoint及测试日志SHA。test mAP72.0%、macro-F1 66.0%、micro-F1 76.7%。历史训练4090、测试3060，来源限制保留。
7. 当前研究工作树的GAViT及sparse_hybrid构图显式要求K16。将既有4×4 checkpoint强行换成9区域，或只更改分区覆盖图，都不产生受训3×3性能结果。

## 什么样的3×3结果可以进入对照

必须核对BEN-19数据版本、19类映射、RGB预处理及同一train/val/test划分（237871/122342/119825及split SHA），再核对预训练来源、seed、区域池化、GAT、反馈方式、构图规则及消息方向、优化配方、训练预算和验证选best规则。K变化自然引起节点/边数变化应如实报告；不能同时把旧kNN换成新稀疏混合图，再把收益归给K。

- 若发现配置可配对的K9测试结果：核验退出状态、测试数量、checkpoint与日志绑定后对照。
- 若发现匹配K9权重但没有正式测试：先核验身份，再考虑只补测试。
- 若只有旧NWPU K9，或BEN K9同时改变多个关键变量：列为历史材料，不纳入纯区域数消融。
- 若服务器也没有匹配材料：将缺项如实记录；新增训练是后续独立决定，本次不启动。

## 服务器库存补查

已准备[两条单行命令](../results/evidence/ben_k9_comparability_20261017/server_inventory_commands.txt)。第一条列出权重路径及带K9/3x3的文件名，包含普通命名best.pth；第二条在JSON/log/out/err中找确切K9/num_regions=9/3×3线索。均跳过缓存、依赖、Git和数据目录，不导入torch、不加载权重、不读原图，不需要GPU或重新解压数据。

命中只表示候选，可能是计划或其他数据集，需要继续核对内容。文本检索仅覆盖小于20 MiB的上述文件；更大日志、未包含配置的普通命名权重、其他服务器目录和原NTU服务器库存仍有盲区。grep无命中退出1表示未匹配，不代表训练失败。命令不写服务器文件；请保留错误输出，权限或路径问题不能视为“没有结果”。

[结构化审查与来源SHA](../results/ben_k9_k16_comparability_audit_20261017.json)。现有几何图仍仅用于解释分区；本审查没有修改性能CSV，也没有建立“4×4优于3×3”的结论。
