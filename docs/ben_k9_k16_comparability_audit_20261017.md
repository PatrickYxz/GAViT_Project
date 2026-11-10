# 3×3与4×4性能结果可比性审查

2026-10-17。任务仅为检查已有结果并配对，不启动训练。

2026-10-20补充状态：后续授权的K9 seed42已回传队列exit0与完整119825张测试指标，mAP71.08%，历史K16为72.00%，约+0.92pp支持4×4；单seed跨批次及旧打印精度限制保留，summary待核。见[新性能对照](ben_grid3_grid4_seed42_20261020.md)。下文“未找到”是10月17日存量审查的历史结论，不代表当前仍没有K9分数。

## 结论

已核对本地归档及用户回传的Featurize库存，未识别出可与当前固定4×4 GAViT配对的BEN固定3×3正式性能结果。当前没有可复用的3×3对照候选，3×3的BEN test mAP及其相对4×4的差值保持空缺。这是本次扫描范围内的结论，不等于证明所有历史服务器从未运行过K9。

| 数据集及记录 | 3×3 | 4×4 | 是否能回答区域数的影响 |
|---|---:|---:|---|
| 早期NWPU历史日记，test accuracy | 95.9%（原记录一位小数） | 95.8%（原记录一位小数） | 否：同时更换池化和反馈结构，且不是当前测试协议 |
| BEN当前固定稀疏混合图，seed42，test mAP | 本次库存未找到 | 72.00% | 尚无3×3配对，不能计算差值 |

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

## 服务器库存回传已核对

用户回传了[两条命令的完整粘贴文本](../results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/server_return.txt)，39390字节，SHA-256为2d032da46b31c8da1e3aa1e8c7e3e0f238530c88b8b2a5227e08042042853246。两条命令与准备版本一致，命令后均回到shell提示符，粘贴中未见错误；没有单独回传进程退出码。本轮只读取这份回传，没有直接访问服务器文件系统。

- 第一条列出510个不同文件路径：136个.pth、373个.pt、1个.safetensors；没有文件名含K9或3x3的候选。路径数量不是模型/实验数量，其中包括训练状态、assignment_probe及可视化缓存。
- 第二条配置内容检索没有输出匹配行，直接回到提示符；在其覆盖的JSON/log/out/err中没有找到K9、num_regions=9或3×3线索。
- 主项目checkpoints里能认出K16 kNN、sparse_hybrid正式/工程权重及Swin等已有路径，未出现旧best_gavit_K9_spatial.pth或best_gavit_K9_spatial_knn_fusion.pth。普通命名gavit.pth、best.pth和diagnostic权重不能仅凭文件名完成架构鉴定；本轮没有加载它们。

[逐路径结构化库存及核对边界](../results/evidence/ben_k9_comparability_20261017/server_inventory_return_20261017/parsed_inventory.json)已归档。本轮已有结果审查的结论为：没有识别出可直接进行BEN K9/K16性能对照的材料，不能填入3×3分数或声称4×4胜出。

## 库存命令与范围

[两条单行命令](../results/evidence/ben_k9_comparability_20261017/server_inventory_commands.txt)现已收到回传。第一条列出权重路径及带K9/3x3的文件名，包含普通命名best.pth；第二条在JSON/log/out/err中找确切K9/num_regions=9/3×3线索。均跳过缓存、依赖、Git和数据目录，不导入torch、不加载权重、不读原图，不需要GPU或重新解压数据。

命中只表示候选，可能是计划或其他数据集，需要继续核对内容。文本检索仅覆盖小于20 MiB的上述文件；更大日志、未包含配置的普通命名权重、其他服务器目录和原NTU服务器库存仍有盲区。grep无命中时可能返回1，不能解释成训练失败。命令不写服务器文件，权限或路径问题也不能视为“没有结果”。

## 后续性能实验的最小问题

若继续量化“4×4是否比3×3好”，需要单独的K9/K16受控消融。先固定seed42及BEN划分、注意力池化、token_feedback、GAT、构图规则与消息方向、训练配方和验证选best规则，仅改变区域数及其自然引起的节点/边数。已有4×4能否直接复用还需核对配方身份。当前sparse_hybrid实现拒绝K9，需要先做实现和小规模验证，不能仅把正式命令的16改成9直接训练。本次没有修改模型、准备正式启动命令或启动付费训练；单seed结果只能作为初步消融证据。

2026-10-18更新：用户已授权补跑，现已准备[独立K9 seed42快照与运行队列](run_ben_grid3_seed42_20261018.md)。只新增K9，历史K16为明确保留跨批次限制的参考；服务器门禁和正式训练仍待执行。上述2026-10-17审查结论及空缺指标保持，没有用新准备的代码冒充已有性能。

[结构化审查与来源SHA](../results/ben_k9_k16_comparability_audit_20261017.json)。现有几何图仍仅用于解释分区；本审查没有修改性能CSV，也没有建立“4×4优于3×3”的结论。
