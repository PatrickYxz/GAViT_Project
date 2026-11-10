# BigEarthNet seed43 三模型正式对比（2026-10-12）

soft43已完成30轮与119825张test，总/训练/test退出码均0，两个STAGE COMPLETE及总完成标记返回。支持完整训练和评估已结束，不需重跑。

| 模型 | test mAP % | macro-F1 % | micro-F1 % | 保存轮数 |
|---|---:|---:|---:|---:|
| Swin-T | 71.1 | 64.1 | 76.0 | 30 |
| 固定4×4 GAViT | 71.7 | 65.7 | 76.8 | 21 |
| soft_prototype GAViT | 72.867571 | 66.673892 | 76.905971 | 30 |

| soft比较 | ΔmAP约 pp | Δmacro-F1约 pp | Δmicro-F1约 pp |
|---|---:|---:|---:|
| soft − Swin-T | +1.7676 | +2.5739 | +0.9060 |
| soft − 固定4×4 GAViT | +1.1676 | +0.9739 | +0.1060 |

旧Swin/fixed43只返回一位小数，因此差值为近似；对应参考舍入误差约±0.05pp。此seed中soft三项指标均高于两个旧对照，最小micro-F1提升也大于舍入误差。旧fixed43仅保存21轮/best8，Swin43完整30/best6；旧bb9f1cf、历史4090训练/3060固定test，新soft ed480e6/current4090，不能宣称三模型完全同条件或仅分组改动的因果提升。完整30轮的seed45另保留，不重训旧43。

| 类别 | 固定43打印AP % | soft43 AP % | 差约 pp |
|---|---:|---:|---:|
| Beaches, dunes, sands | 8.6 | 23.7739 | +15.1739 |
| Coastal wetlands | 34.4 | 42.6134 | +8.2134 |
| Permanent crops | 63.3 | 64.6977 | +1.3977 |
| Natural grassland and sparsely vegetated areas | 48.5 | 45.8823 | -2.6177 |
| Industrial or commercial units | 50.8 | 49.8134 | -0.9866 |
| Complex cultivation patterns | 69.0 | 68.3804 | -0.6196 |

相对fixed43，舍入范围可确定8类升、8类降，另外3类方向不确定（农业与自然植被、针叶林、海洋水域）。soft沙滩/沙丘/沙地与海岸湿地改善较大，草地/稀疏植被、工业商业区和复杂耕作退步；不是所有场景同时提升。soft精确AP与旧打印AP作差时按±0.05pp范围判断，不能把接近0的差写成确定涨跌；这些是预测结果描述，不是机制/显著性证明。[全部19类CSV](bigearth_seed43_three_model_per_class.csv)。

最佳权重为第8轮，val mAP78.99004691745016%、macro-F1 71.95576173181863%；metadata按四位小数保存78.9900%，原训练尾日志的79.0%为打印舍入。最终第30轮loss0.0600898171、val mAP75.69968198%、macro-F1 71.32167879%、nextLR0，val mAP相对best下降3.290365pp。测试加载gavit_train/gavit.pth（按val mAP选best），checkpoint SHA c01403d67df77ec03f173877ca3466e1211d06015cd83dc89672629595e387a7，与训练、测试和之前精确metrics一致。

训练与测试设备/软件/精度均与批准4090预检相同：GPU-cede40c0-f93d-7c8e-1006-cf6183e53592，FP32/noAMP/matmul TF32false/cuDNN TF32true。完整阶段资源已回传并核验：

| 范围 | 墙钟秒 | CUDA allocated峰值 GiB | CUDA reserved峰值 GiB | 主进程RSS峰值 GiB |
|---|---:|---:|---:|---:|
| 30轮训练阶段 | 22679.135110 | 3.662695 | 3.943359 | 2.047615 |
| 最佳权重完整test阶段 | 197.353206 | 0.595415 | 1.027344 | 1.475349 |

整批22941.731309秒，即6小时22分21.73秒；训练阶段6小时17分59.14秒，test阶段3分17.35秒。整批比两阶段合计多65.242993秒，未进一步分解。按用户单价1.67元/h折算约10.64元；加上NWPU43三模型的3617.327163秒/约1.68元，两批正式队列合计7小时22分39.06秒、约12.32元。这是正式运行时间折算，解压、预检、短测和租赁空闲另计，不能代替实际账单。[费用口径与精确值](evidence/bigearth_soft43_complete_20261012/seed43_formal_cost.json)。

有效训练吞吐为237871×30/22679.135110=314.656179图/秒，分母含验证、IO、探针和工件检查，不是纯训练或公平模型速度；RSS不含所有loader worker。8张固定val的初始/best/final共3份探针已检查，S行/W列归一化误差最大2.98e-7，实际48空间+32特征=80边；这支持工程路径正常，不证明语义分组更优。本地回传包含1–30轮编号及首轮/best/末轮三行；best来自服务器完整history的只读命令断言，本地没有独立复算全部曲线，也未下载原权重、逐样本预测或探针张量。

证据：[原指标字节](evidence/bigearth_soft43_complete_20261012/reported_output.txt)、[指标核验](evidence/bigearth_soft43_complete_20261012/validation.json)、[资源原始回传](evidence/bigearth_soft43_complete_20261012/reported_summary_output.txt)、[资源与身份核验](evidence/bigearth_soft43_complete_20261012/resource_validation.json)、[完整JSON](bigearth_soft43_test_summary.json)。本次训练/test/资源归档全部收尾，不需再索取同批日志或重跑。[43/45双seed三模型描述表](bigearth_three_model_two_seed_summary.md)：soft对fixed两次mAP/macro-F1提升，对Swin一升一降；NWPU42/43的soft仍均低于Swin，不宣称稳定跨数据集收益。最终三seed与公平效率/外部方法实测仍有缺口。
