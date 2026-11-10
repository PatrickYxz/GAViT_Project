# NWPU seed43 三模型正式测试（2026-10-11）

用户回传总exit0、complete、六训练阶段与三模型25200张test的紧凑摘要。固定版与soft版均低于Swin；保留全部负面结果，不重试选分。

| 模型 | test OA % | macro-F1 % | ΔOA vs Swin pp | 正确数/25200 | refit轮数 |
|---|---:|---:|---:|---:|---:|
| Swin-T | 93.857143 | 93.834530 | +0.000000 | 23652 | 23 |
| 固定4×4 GAViT | 93.765873 | 93.743513 | -0.091270 | 23629 | 26 |
| soft_prototype GAViT | 93.626984 | 93.587821 | -0.230159 | 23594 | 29 |

固定相对Swin净少对23张，soft净少对58张；soft相对fixed净少对35张。这是正确数净差，尚未读取43的逐样本预测，不能拆成纠正/新增错误或指定受益类别。

协议：ed480e60fea03ed786fdba70c144de5e815c34f5；冻结manifest SHA ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e；ImageNet SHA f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36。5040/1260各校准30轮，以最高val OA首个同分选23/26/29；重新初始化6300池各自refit；各自refit末轮测试。batch32/workers2，CE/AdamW3e-4/wd1e-4/cosine周期30，FP32/no AMP/双TF32-off。完整预先命令见[执行说明](../docs/run_seed43_three_models_20261011.md)。紧凑摘要确认选轮身份；固定/soft的校准OA与校准macro-F1未回传，不填写。

GPU：GPU-cede40c0-f93d-7c8e-1006-cf6183e53592, NVIDIA GeForce RTX 4090, 610.57.04, 24564 MiB。输出`/home/featurize/work/nwpu_three43_20261011_01`；各模型`*_refit/last.pth`与`*_test`报告的SHA对应，完整SHA见[JSON](nwpu_seed43_test_summary.json)。

| 模型 | 参数量 | calibrate秒 | refit秒 | refit allocated / reserved GiB |
|---|---:|---:|---:|---:|
| Swin-T | 27555495 | 501.472 | 421.103 | 3.5587 / 3.8887 |
| 固定4×4 GAViT | 31386920 | 638.539 | 586.543 | 3.6539 / 3.9355 |
| soft_prototype GAViT | 31486760 | 638.652 | 617.456 | 3.6646 / 3.9531 |

总队列3617.327秒=1小时0分17.33秒，按1.67元/h折算1.6780元（约1.68元）；这是本次运行时间折算，不是包含恢复/短测/空闲的账单。训练阶段资源包括检查点重建等；不同refit轮数与IO耗时不用于推导公平推理加速比。

证据：[原文](evidence/seed43_nwpu_final_20261011/server_return.txt)、[解析](evidence/seed43_nwpu_final_20261011/reported_summary.json)、[核验](evidence/seed43_nwpu_final_20261011/validation.json)。调度器之前已检查原报告/权重/完整history/混淆和预测文件；本轮本地仅审计回传字段，没有下载原权重、完整配置、逐类准确率或混淆矩阵。

与seed42合并见[双seed比较](nwpu_two_seed_summary.md)。seed42为2080Ti、43为4090，变化不能仅归因随机性或GPU；两seed仍不满足最终三seed要求。BigEarth soft43已通过预检，尚未训练。
