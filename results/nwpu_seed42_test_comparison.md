# NWPU-RESISC45 seed42：三模型正式测试结果

2026-10-01后续更新：本批配对、90图册审阅及BigEarth soft45正式运行均已完成，双数据集各单seed结果见[整合稿](soft_assignment_cross_dataset_ablation.md)。下文09-29的“待完成”清单保留当时范围，以整合稿和最新排期为准；多seed、外部方法与新结构公平效率仍未完成。

2026-09-29，用户回传queue.exit=0、六阶段process_exit=0/checks_passed=true及六份run.exit=0。三模型正式拟合与25200张test已完成，原始附件逐字节保存，轮数/架构/权重来源与OA整数计数在本地复算通过。仅单seed，不能声称稳定或显著差异。

## 正式测试表

| 模型 | Test OA (%) | Test macro-F1 (%) | 正确/25200 | 对Swin OA差(pp) | refit轮数 | refit分钟 | allocated/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Swin-T | 93.932540 | 93.909145 | 23671 | +0.000000 | 29 | 18.48 | 3.5584/3.8301 |
| 固定4×4 GAViT | 94.103175 | 94.089933 | 23714 | +0.170635 | 29 | 20.95 | 3.6550/3.8770 |
| 局部原型软分配 GAViT | 93.869048 | 93.854987 | 23655 | -0.063492 | 28 | 20.67 | 3.6616/4.0391 |

- fixed−Swin：OA +0.170635 pp，macro-F1 +0.180788 pp，净多对43张。
- soft−Swin：OA −0.063492 pp，macro-F1 −0.054158 pp，净少对16张。
- soft−fixed：OA −0.234127 pp，macro-F1 −0.234946 pp，净少对59张。

配对统计已完成：fixed对Swin纠正586/新增543；soft对Swin纠正570/新增586；soft对fixed纠正538/新增597，与净差完全一致。[配对表及收尾结论](nwpu_seed42_closeout.md)已归档。不同训练模型之间的纠正/破坏描述不是单模块因果干预，也不能据单seed宣称稳定性。

## 与校准结果的关系

校准val OA为Swin94.206349%、fixed93.015873%、soft93.730159%，soft−fixed +0.714286 pp。正式test则fixed最高，soft的验证优势没有延续。本批没有证据支持“soft比固定网格更好”的结论，固定网格也只有这一个seed的小幅优势。

不能简单称作同一权重在test上掉点：校准仅用5040张训练并在1260张val选轮；refit按已定预算从原预训练重新初始化，在6300张train+val上拟合，测试25200张不同样本。训练池、学习轨迹与评价样本都发生了预定变化，排序变化本身既不证明数据错误，也不证明过拟合或区域塌缩。之前8张探针的同质化迹象仍只是假设线索。

## 身份、产物与成本

- 原协议scene-pool-group-v2；seed42，三模型训练池6300、test25200，固定manifest SHA ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e，重复像素组不跨split。
- 发布ed480e60fea03ed786fdba70c144de5e815c34f5、同ImageNet预训练SHA f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36；RGB224与共同增强、CE/AdamW lr3e-4/wd1e-4、batch32/workers2、FP32无AMP/双TF32-off、Cosine T_max30。模型结构及参数量沿用校准记录，三模型分别27,555,495/31,386,920/31,486,760参数。
- 按校准val OA选择29/29/28轮；最终均评估last.pth，training_phase=refit、fit_scope=train_plus_val。命令与冻结依据见[执行记录](../docs/run_nwpu_refit_test42_20260929.md)。没有用test选轮。
- 当前回传子集未展开training/code/hardware字典；六个checks_passed=true对应已交付队列的共同身份、权重实际SHA与来源哈希等检查，含与校准RTX2080Ti硬件一致的断言。架构、轮数、校准best权重报告SHA和refit/test权重SHA关系已另外本地核对；未把返回子集冒充直接获取服务器全部权重或原始日志。
- 固定/soft真实前向均48空间+32特征边；最大重载差Swin0、两个GAViT均9.5367431640625e-7，低于入口阈值1e-5。
- 本次六阶段3965.277424567秒，即66分05.28秒，处于此前1–1.5小时估计范围。三项refit分别1108.58/1257.25/1240.20秒；剩余时间包括测试及队列开销，不能当纯推理耗时。连同此前三模型校准共约138.54分钟（2小时18分32秒），不含恢复/空闲。租价未知，不推算实际费用。

| 变体 | 最终last.pth SHA256 | 校准run.json SHA256 |
|---|---|---|
| swin | `0c89fd0d5c374b85cdba68893d2fe29c1a9c803ba582b2deb8975015f5da1c4d` | `e2f0b8d5154cd345906442e35c35bb336d9f8373af736abff39078be5a3b7119` |
| fixed | `38aa3d94c16a4d22710174ddc6cb6519cb5fdd272c4215ca7e15c8902a8f544e` | `4d10169e3618252f079bce85094750eb0477e6caa8ad9a37f13394eecfbb81d8` |
| soft | `2d25957190f3988136a721ae3c266b5785e99c0f9564ece611303a2186813b69` | `9c1fa474d3f2e79203a1c245acfd15692a9a6e766acb71ef2566baff3d1bfd33` |

结果根目录`/home/featurize/work/nwpu_refit_test42_20260929_01`。模型与来源在`{swin,fixed,soft}_refit`；metrics.json、predictions.csv、per_class.csv、含混淆矩阵的metrics及退出记录在各`*_test`。队列检查确认CSV产物存在；随后服务器从CSV复算的OA/macro-F1、配对数、部分逐类及前五混淆摘要已回传。完整CSV及完整混淆矩阵未下载到本地。

## 下一步与论文边界

本seed训练、测试和缓存配对收尾均完成，[最终分析](nwpu_seed42_closeout.md)已归档。不重复执行[历史读取命令](../docs/analyze_nwpu_test42_20260929.md)。固定本次模型与全部分数，不根据测试结果调参或重跑seed42；本次统计只用于描述误差结构。

暂不自动追加多seed或BigEarth完整训练，也不因seed42结果选择性取消最终消融要求。E05的NWPU单seed三模型正式链路已完成；双数据集、多seed、独立效率和跨类别区域/图边可视化仍为未完成项。旧NWPU约96%的不同协议成绩保留，不混入本表。

证据：[原始返回](evidence/nwpu_refit_test42_20260929/server_return.txt)、[解析摘要](evidence/nwpu_refit_test42_20260929/server_return.json)、[本地核对](evidence/nwpu_refit_test42_20260929/server_return_validation.json)。
