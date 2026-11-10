# BigEarthNet seed45：GAViT尾日志回传，配对摘要待验收

**2026-09-24已由完整摘要补齐并验收：见[最终配对结果](bigearth_seed45_comparison.md)。以下保留首次部分回传时的证据边界，不作为当前待办。**

收到用户终端尾日志，来源目录`/home/featurize/work/bigearth_seed45_20260922_01`。原消息尾部重复，按同一次运行记录；[归档摘录](evidence/bigearth_seed45_20260922_01/reported_excerpt.txt)规范化下划线并去重，不声称是服务器原始日志的逐字节副本。

| 项目 | 当前回传 |
|---|---|
| 实际测试设备 | RTX 2080 Ti；CUDA12.1、cuDNN8902、8线程 |
| 精度 | FP32，无AMP；matmul TF32=False、cuDNN TF32=True |
| GAViT训练进度 | 第30/30轮，7434/7434训练批；STAGE COMPLETE gavit_train |
| 最佳验证mAP | 78.3%（打印舍入值）；最佳epoch、精确值待摘要 |
| 最后一轮验证 | mAP 76.61513448708149%；macro-F1 71.71295447541755% |
| 最后一轮loss | 0.08172658044316794；next_epoch_lr=0 |
| 最后一轮耗时 | 训练24:21；含验证/保存的epoch_wall_seconds=1824.7612908370065（约30.41分钟） |
| GAViT测试 | 119825样本、3745/3745批；STAGE COMPLETE gavit_test |
| GAViT Test mAP / macro-F1 / micro-F1 | **70.3% / 64.3% / 76.7%**（均打印舍入值） |
| 测试循环耗时 | 05:11；12.00批/秒；不是完整测试命令墙钟 |
| 队列标志 | BIGEARTH SEED45 COMPLETE |
| 仍缺的直接回传 | 总及六阶段run.exit、Swin45结果、精确metrics/best/身份/资源摘要 |

完成标志与本地冻结队列的控制流一致：正常打印它前须六阶段进程退出0、状态complete及共同硬件/精度一致。当前只是返回日志的间接证据；服务器脚本SHA与退出文件尚未回传，不替用户伪造exit_code=0或Swin分数。也不把单轮耗时乘30写成实际总耗时或账单。

GAViT45的70.3%与44的70.2%接近。沙滩9.7%、海岸湿地10.7%仍低，自然草地47.6%；不能仅凭AP低判断类别支持不足或图模块因果。缺少本次Swin45，不能判断seed45的配对差值，暂不改四seed均值，也不新增训练。先读取work里的现成JSON和退出记录完成本轮收尾；42/43/44及AID数值保留。

19类AP见归档摘录及`reported_metrics.json`。数值只保留日志提供的精度；精确metrics返回后另存来源，不覆盖原摘录。

本地检查19个类别名称/顺序与代码一致，打印AP平均70.26315789473684%，与汇总70.3%的舍入相容。只读回传命令见[单行命令](../docs/read_bigearth_seed45_summary.sh)；它读取summary与退出文件，输出各阶段实测时长之和（不含预检/导入/间隔/解压/闲置），不将该数直接称作实际费用。
