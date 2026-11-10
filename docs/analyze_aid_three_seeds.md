# AID三seed结果后的只读检查

09-22完成更新：用户已回传下方命令结果，四条曲线/106个epoch四元组和12条混淆行核验通过；选轮及LR计划一致，具体误分类已记录。见[诊断报告](../results/aid_seed44_curve_confusion_analysis.md)。本页命令作为已完成检查的复现记录，无需再次回传同份数据；下一步E06测推理效率。

2026-09-22。三seed总体指标已经完成；本检查用于事后解释，不重新选择模型或训练预算。seed44负向结果保留。

当前已有摘要显示协议、权重来源、重载和总指标一致。下一步读取seed44四个训练阶段的history，以及三seed各模型test混淆矩阵中Commercial/RailwayStation两行，核对校准最早最佳轮、观察loss/lr/val曲线、确定实际误分类方向。曲线正常不能排除所有故障，类别关联也不能证明图结构因果。

只读取work下的JSON和退出码，不使用原图、torch、CUDA或网络；不修改服务器文件，也不需要重新解压。每个run.exit必须为0，history长度/轮次与run.json一致，校准best符合最高val OA且同分最早，混淆矩阵维度/非负计数/样本总数/对角线OA通过后才输出。refit的val为null符合其训练范围。

复制以下一整行并回传输出：

```bash
python -c 'import json,pathlib,sys; b=pathlib.Path(sys.argv[1]); read=lambda p:json.loads(p.read_text()); hp=[b/("aid_"+m+"_"+phase+"44_20260921_01") for phase in ("calibrate","refit") for m in ("swin","gavit")]; tp=[b/("aid_"+m+"_test"+str(s)+"_20260921_01") for s in (42,43,44) for m in ("swin","gavit")]; assert all((p/"run.exit").read_text().strip()=="exit_code=0" for p in hp+tp); rr=[read(p/"run.json") for p in hp]; hh=[read(p/"history.json") for p in hp]; tt=[read(p/"metrics.json") for p in tp]; assert all(r["status"]=="complete" and len(h)==r["final_epoch"] and [x["epoch"] for x in h]==list(range(1,len(h)+1)) for r,h in zip(rr,hh)); assert all(r["phase"]!="calibrate" or max(h,key=lambda x:x["validation"]["oa_percent"])["epoch"]==r["best_epoch"] for r,h in zip(rr,hh)); assert all(t["samples"]==5000 and t["split"]=="test" and len(t["confusion_matrix"])==30 and all(len(row)==30 and all(isinstance(v,int) and v>=0 for v in row) for row in t["confusion_matrix"]) for t in tt); assert all(sum(map(sum,t["confusion_matrix"]))==5000 and abs(sum(t["confusion_matrix"][i][i] for i in range(30))/50-t["oa_percent"])<1e-8 for t in tt); assert all(t["class_names"][7]=="Commercial" and t["class_names"][21]=="RailwayStation" for t in tt); print(json.dumps({"curve_columns":["epoch","train_loss","lr","val_OA_percent"],"seed44_curves":[{"directory":p.name,"best_epoch":r["best_epoch"],"final_epoch":r["final_epoch"],"curve":[[x["epoch"],x["train_loss"],x["lr"],x["validation"]["oa_percent"] if x["validation"] is not None else None] for x in h]} for p,r,h in zip(hp,rr,hh)],"confusion_orientation":"rows=true, columns=predicted","focus_test_rows":[{"directory":p.name,"rows":[{"true_class":t["class_names"][i],"predicted_counts":{t["class_names"][j]:n for j,n in enumerate(t["confusion_matrix"][i]) if n}} for i in (7,21)]} for p,t in zip(tp,tt)]},indent=2))' /home/featurize/work
```

本地使用合成history/混淆矩阵检查了shell/Python语法、四曲线/六模型测试行、真实标签行与预测列方向、同分最早epoch规则和错误选轮拒绝。真实服务器曲线/混淆行仍待回传；不把合成检查计入实验结果。
