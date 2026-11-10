# DFAGCN 返回测试记录独立离线复核

- 日期：2026-10-31；复核实验：2026-10-30；状态：PASS，32/32 项检查通过。
- 方法：仅用 Python 标准库，未导入项目实现，未重训或运行模型。
- 测试记录原始字节 SHA256：`771dee2dcd1e676e4884c36654cc8fbcc4c067811362dbf9f3e997c903ccd4e9`。
- 去掉 record_sha256、sort_keys=True、separators=(",",":")、allow_nan=False 的 canonical JSON SHA256：`cf17f54c41421b61b2e3f816e0cad39ecc4ce0b2a6cf7aabaf7052b515fe700f`，与记录 seal 一致。
- schema / formal / test / completed、manifest 引用、test cache content SHA 引用与 cached time 均与归档摘要一致。
- 25200 条路径唯一、已排序且为规范化相对路径；45 类索引为 0–44，每类路径重建真值 support=560；所有预测为 0–44 整数。
- 重建混淆矩阵全部 2025 个单元、45 类 support/accuracy/F1 与报告一致。trace=21806，错误=3394；OA=86.53174603174604%，macro-F1=86.45890874660425%，与 test.json 和归档 summary 一致（绝对容差 1e-10）。

## 最弱 recall 的 5 类

| 类别 | 正确/560 | 错误 | Recall % | F1 % |
|---|---:|---:|---:|---:|
| palace | 325 | 235 | 58.035714 | 62.984496 |
| church | 366 | 194 | 65.357143 | 67.094409 |
| tennis_court | 375 | 185 | 66.964286 | 71.428571 |
| basketball_court | 408 | 152 | 72.857143 | 74.588665 |
| railway | 416 | 144 | 74.285714 | 78.787879 |

## 有向混淆前 10（真值 → 预测）

| 真值 | 预测 | 张数 | 占全部错误 % | 占该真值类 % |
|---|---|---:|---:|---:|
| palace | church | 84 | 2.474956 | 15.000000 |
| railway | railway_station | 65 | 1.915144 | 11.607143 |
| church | palace | 58 | 1.708898 | 10.357143 |
| rectangular_farmland | terrace | 53 | 1.561579 | 9.464286 |
| tennis_court | medium_residential | 43 | 1.266942 | 7.678571 |
| church | commercial_area | 42 | 1.237478 | 7.500000 |
| tennis_court | basketball_court | 40 | 1.178550 | 7.142857 |
| wetland | lake | 40 | 1.178550 | 7.142857 |
| basketball_court | tennis_court | 39 | 1.149087 | 6.964286 |
| industrial_area | railway_station | 34 | 1.001768 | 6.071429 |

## 双向混淆 pair 前 5（按两方向错误之和排序）

| A | B | A→B | B→A | 合计 | 占全部错误 % |
|---|---|---:|---:|---:|---:|
| church | palace | 58 | 84 | 142 | 4.183854 |
| railway | railway_station | 65 | 20 | 85 | 2.504420 |
| basketball_court | tennis_court | 39 | 40 | 79 | 2.327637 |
| church | commercial_area | 42 | 26 | 68 | 2.003536 |
| lake | wetland | 26 | 40 | 66 | 1.944608 |

## 错误集中度

- 最弱 5 类：910/3394，占 26.812021%。
- 前 10 有向单元：498/3394，占 14.672952%。
- 前 5 无序类别对：440/3394，占 12.964054%。
- 三种统计相互重叠，不能相加。

## 证据边界

- 真值来自返回路径首目录，索引来自返回记录 per_class。原始 manifest 完整字节及原图未返回，本次只验证引用哈希一致，不能声称独立验证划分成员、原图字节或标签。
- refit checkpoint / refit 记录未返回，refit_record_sha256 仅为未独立核验的引用；未验证模型字节或复现推理。
- cache 字节未返回，仅核对 cache content SHA 与归档摘要一致。record seal 证明记录自洽，不证明独立来源真实性。
- 误差结构只作本次独立实现、单 seed 的描述，不作机制因果、跨 seed 稳定性或原作者性能结论。

完整检查项和所有逐类重建指标保存在同目录 audit.json。
