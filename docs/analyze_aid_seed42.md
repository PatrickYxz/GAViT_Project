# AID seed42 测试后的离线分析

2026-09-21完成更新：用户已回传配对计数和30类表，全部指标核对通过。GAViT纠正93/新增错误63、净多正确30；类内准确率17类提高/7类下降/6类持平，Commercial和Park分别净增23/7张。见[分析报告](../results/aid_seed42_per_class_analysis.md)。下方为已执行的只读命令，不需重复运行；下一步[seed43重复实验](run_aid_seed43.md)。

本步只读取持久化work中的predictions.csv和per_class.csv，不加载模型、原图或GPU，不改变任何训练结果。目的：查看净增益如何组成、哪些类别改善或退步；结果用于论文逐类表和失败讨论，不能用于修改已冻结模型后重复选择同一测试集结果。单seed和同一固定划分的案例分析不证明跨seed稳定性。

## 1. 按图片路径对齐预测

整行执行。先检查两个文件均5000条且路径唯一、路径集合一致、逐样本真值一致，再核对正确总数与已有OA一致。即使两文件排序不同也能配对，不按行号硬拼接。

```bash
python -c 'import csv,json,sys; from collections import Counter; A=list(csv.DictReader(open(sys.argv[1]))); B=list(csv.DictReader(open(sys.argv[2]))); a={r["path"]:r for r in A}; b={r["path"]:r for r in B}; assert len(A)==len(a)==len(B)==len(b)==5000,"Wrong count or duplicate paths"; assert a.keys()==b.keys(),"Different image paths"; assert all(a[p]["label"]==b[p]["label"] for p in a),"Different labels"; c=Counter((a[p]["prediction"]==a[p]["label"],b[p]["prediction"]==b[p]["label"]) for p in a); assert c[True,True]+c[True,False]==4817 and c[True,True]+c[False,True]==4847,"Predictions disagree with reported OA"; print(json.dumps({"samples":len(a),"both_correct":c[True,True],"gavit_corrects":c[False,True],"gavit_breaks":c[True,False],"both_wrong":c[False,False],"net_correct":c[False,True]-c[True,False]},indent=2))' /home/featurize/work/aid_swin_test42_20260921_01/predictions.csv /home/featurize/work/aid_gavit_test42_20260921_01/predictions.csv
```

字段含义：gavit_corrects为Swin错/GAViT对，gavit_breaks为Swin对/GAViT错；这两个独立训练模型之间的差异不等同于图反馈分支的因果贡献。both_wrong包含两者都错但预测类别可能不同的情况。任何断言失败先检查文件身份，不删行或强行调整数据让统计通过。

## 2. 读取30类指标

整行执行，输出先Swin的表，再GAViT的表，各有表头和30行。字段class_index/class_name/support用于对齐；accuracy_percent是类内正确率，不能当AP。F1同样是百分数。

```bash
cat /home/featurize/work/aid_swin_test42_20260921_01/per_class.csv /home/featurize/work/aid_gavit_test42_20260921_01/per_class.csv
```

回传以上两段输出后，核对30类类名和支持数相同、支持数和5000、按支持数加权正确率重现各自OA、类F1平均重现macro-F1，再计算逐类差值。完整metrics.json内混淆矩阵及predictions.csv继续保留，尚未下载到本地；需要具体类别混淆时再读已有矩阵，不重新推理。

本地先用独立合成CSV核查统计命令的四种计数、乱序配对及缺行/重复路径/标签不一致拒绝；随后用户真实回传已到并与OA、macro-F1核对一致。原始逐样本预测和完整混淆矩阵仍在服务器。当前诊断收尾，不新增测试集调参。
