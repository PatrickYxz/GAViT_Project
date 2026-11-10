# AID seed44曲线与三seed重点类别混淆诊断

2026-09-22。106个逐轮四元组与12条真实类别混淆行核验通过。没有发现缺轮、错误选轮或学习率周期变化；当前证据支持保留seed44负面结果。此轮只解释既有实验，不改测试结果、不追加训练、不按test重新选epoch。

来源：[粘贴原文](evidence/aid_seed44_curve_confusion_20260922/reported_output.txt)、[提取JSON](evidence/aid_seed44_curve_confusion_20260922/summary.json)、[派生分析](evidence/aid_seed44_curve_confusion_20260922/analysis.json)、[来源范围](evidence/aid_seed44_curve_confusion_20260922/provenance.json)、[核验](evidence/aid_seed44_curve_confusion_20260922/validation.json)。只回传四列曲线和指定两类行，未获得完整history、全部混淆矩阵或逐图片预测，不冒充全量原始文件。

## 训练过程与选轮

| 阶段 | 完成轮数 | 训练loss首轮→末轮 | 末轮使用的LR | 选中epoch | 并列最高val轮 | 末轮val OA% |
| --- | ---: | --- | ---: | ---: | --- | --- |
| swin calibrate | 30 | 0.817972 → 0.000822 | 0.000000822 | 25 | [25, 26] | 96.6 |
| gavit calibrate | 30 | 0.957828 → 0.000875 | 0.000000822 | 21 | [21, 22, 26] | 96.7 |
| swin refit | 25 | 0.762180 → 0.001007 | 0.000028647 | — | — | 无独立val |
| gavit refit | 21 | 0.811568 → 0.002999 | 0.000075000 | — | — | 无独立val |

Swin校准第25/26轮最高97.0%，按同分最早选25；GAViT第21/22/26轮最高97.0%，选21。各自refit完整25/21轮，与已记录规则一致，21轮结束不是提前中断。refit从相同ImageNet来源重新初始化、合并5000训练池，不能把它拼接在校准第30轮后，也不能把calibrate与refit loss视作一条连续曲线。

全部106条LR符合`0.0003 * (1 + cos(pi * (epoch - 1) / 30)) / 2`，最大绝对误差约8.13e-20；refit未将T_max悄悄改为25或21。GAViT第21轮LR为0.000075，Swin第25轮为0.000028647，但这是预先规定的选轮与周期策略，不因测试下降临时延长。

两模型calibrate末轮loss约0.000822/0.000875；val末轮96.6%/96.7%，比各自最佳97.0%下降0.4/0.3 pp，变化没有单独显示GAViT明显崩溃。refit末轮loss为Swin0.001007、GAViT0.002999；GAViT第9→10轮0.116852→0.197366、第13→14轮0.028954→0.064490有反弹，此后仍下降。不能仅靠epoch平均loss把这些波动定性为错误、证明欠拟合/过拟合，或推断图反馈导致退步；尚无逐batch和梯度信息。

## 具体误分方向

每行是真实类别，字典键是预测类别；表中只列错误去向。两类支持数分别175和130，每一行总和与既有逐类指标匹配。

| seed | 模型 | 真实类别 | 正确/支持数 | 错误预测类别及数量 |
| --- | --- | --- | ---: | --- |
| 42 | swin | Commercial | 145/175 | Industrial: 4, RailwayStation: 1, School: 22, Square: 3 |
| 42 | swin | RailwayStation | 126/130 | Industrial: 3, School: 1 |
| 42 | gavit | Commercial | 168/175 | Industrial: 1, School: 4, Square: 2 |
| 42 | gavit | RailwayStation | 128/130 | Center: 1, Square: 1 |
| 43 | swin | Commercial | 166/175 | Church: 1, DenseResidential: 1, Industrial: 3, School: 2, Square: 2 |
| 43 | swin | RailwayStation | 125/130 | Center: 2, Industrial: 1, School: 1, Viaduct: 1 |
| 43 | gavit | Commercial | 169/175 | Center: 1, Industrial: 1, RailwayStation: 1, School: 2, Square: 1 |
| 43 | gavit | RailwayStation | 128/130 | Center: 1, Commercial: 1 |
| 44 | swin | Commercial | 171/175 | Center: 1, Industrial: 1, School: 1, Viaduct: 1 |
| 44 | swin | RailwayStation | 126/130 | Center: 2, Industrial: 1, School: 1 |
| 44 | gavit | Commercial | 150/175 | Center: 1, Church: 3, DenseResidential: 1, Industrial: 1, Resort: 3, School: 8, Square: 2, StorageTanks: 6 |
| 44 | gavit | RailwayStation | 116/130 | Center: 1, Church: 1, Industrial: 10, Resort: 1, Viaduct: 1 |

seed44的错误计数变化如下，正数表示GAViT该误分类计数更多。它们是两个模型的行边际计数之差，不能据此断言恰好对应同一批由正确变错误的图片。

| 真实类别 | 预测类别 | Swin错误数 | GAViT错误数 | 差 |
| --- | --- | ---: | ---: | ---: |
| Commercial | School | 1 | 8 | +7 |
| Commercial | StorageTanks | 0 | 6 | +6 |
| Commercial | Church | 0 | 3 | +3 |
| Commercial | Resort | 0 | 3 | +3 |
| Commercial | Square | 0 | 2 | +2 |
| Commercial | DenseResidential | 0 | 1 | +1 |
| Commercial | Center | 1 | 1 | +0 |
| Commercial | Industrial | 1 | 1 | +0 |
| Commercial | Viaduct | 1 | 0 | -1 |
| RailwayStation | Industrial | 1 | 10 | +9 |
| RailwayStation | Church | 0 | 1 | +1 |
| RailwayStation | Resort | 0 | 1 | +1 |
| RailwayStation | Viaduct | 0 | 1 | +1 |
| RailwayStation | Center | 2 | 1 | -1 |
| RailwayStation | School | 1 | 0 | -1 |

Commercial正确171→150，错误4→25（净+21）；主要增加为School+7、StorageTanks+6、Church/Resort各+3。RailwayStation正确126→116，错误4→14（净+10）；主要是误分Industrial由1→10（+9），其余方向有增有减。两类合计多31次错误，其余28类合计净多7次，重现seed44全局净少正确38。

跨seed看，Commercial→School在seed42为Swin22/GAViT4，seed43为2/2，seed44为1/8；相同模型结构并未保持一致的纠错方向。RailwayStation→Industrial分别3/0、1/0、1/10。描述上支持类别区分随训练seed变化，不构成图消息、特定边或背景依赖的因果证据。

## 本轮收尾与下一步

预定三seed结论保持：OA平均差+0.0467 pp、macro-F1平均差+0.0151 pp，当前固定区域图版本没有显示稳定的整体收益。此次已排查的轮次、学习率和来源边界未发现协议错误；不声称排除了全部软件或数据问题。保留负面结果与错误分析作为论文讨论材料，不按test改预算或挑seed。

E03指标与本次限定范围的错误分析收尾。下一步按排期进入E06，用已有匹配checkpoint测同GPU/输入/batch/精度下的推理延迟、吞吐与峰值显存，明确是否含数据读取/传输，并保留预热、同步和多次测量。已有参数开销+13.91%，实际推理代价仍待测；本轮未编写或启动GPU计时任务。外部方法E02和分组候选E05保持独立待办，不直接根据这两类test错误设计修补。
