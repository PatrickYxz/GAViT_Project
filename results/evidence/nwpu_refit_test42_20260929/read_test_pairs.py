import csv,json,sys
from pathlib import Path
from collections import Counter
p=Path(sys.argv[1])
names=("swin","fixed","soft")
q=json.loads((p/"queue_summary.json").read_text())
assert q["status"]=="complete" and (p/"queue.exit").read_text().strip()=="exit_code=0"
m={n:json.loads((p/(n+"_test")/"metrics.json").read_text()) for n in names}
a={n:list(csv.DictReader((p/(n+"_test")/"predictions.csv").open())) for n in names}
d={n:{v["path"]:v for v in a[n]} for n in names}
assert all(len(a[n])==len(d[n])==25200 and d[n].keys()==d["swin"].keys() for n in names),"Count or path mismatch"
assert all(d[n][s]["label"]==d["swin"][s]["label"] for n in names for s in d[n]),"Labels differ"
classes=m["swin"]["class_names"]
assert len(classes)==45 and all(m[n]["class_names"]==classes and m[n]["samples"]==25200 and m[n]["training_phase"]=="refit" and m[n]["fit_scope"]=="train_plus_val" and m[n]["split"]=="test" and m[n]["manifest_sha256"]==q["tests"][n]["manifest_sha256"] and m[n]["checkpoint_sha256"]==q["tests"][n]["checkpoint_sha256"] for n in names),"Test identity mismatch"
assert all(0<=int(v[k])<45 for n in names for v in a[n] for k in ("label","prediction")),"Invalid class index"
cm={n:Counter((int(v["label"]),int(v["prediction"])) for v in a[n]) for n in names}
support={i:sum(cm["swin"][i,j] for j in range(45)) for i in range(45)}
assert all(support[i]>0 for i in support)
correct={n:sum(cm[n][i,i] for i in range(45)) for n in names}
mf={n:sum(200*cm[n][i,i]/(support[i]+sum(cm[n][j,i] for j in range(45))) for i in range(45))/45 for n in names}
assert [correct[n] for n in names]==[23671,23714,23655] and all(abs(correct[n]*100/25200-m[n]["oa_percent"])<1e-8 and abs(mf[n]-m[n]["macro_f1_percent"])<1e-8 for n in names),"Metrics mismatch"
pairs=(("swin","fixed"),("swin","soft"),("fixed","soft"))
paired={b+"_vs_"+a:Counter((d[a][s]["prediction"]==d[a][s]["label"],d[b][s]["prediction"]==d[b][s]["label"]) for s in d[a]) for a,b in pairs}
changes={b+"_vs_"+a:sorted([{"class":classes[i],"support":support[i],"base_correct":cm[a][i,i],"candidate_correct":cm[b][i,i],"net_correct":cm[b][i,i]-cm[a][i,i]} for i in range(45)],key=lambda v:v["net_correct"]) for a,b in pairs}
out={"samples":25200,"metrics_recomputed":{n:{"correct":correct[n],"OA":correct[n]*100/25200,"macro_F1":mf[n]} for n in names},"pairs":{k:{"both_correct":v[True,True],"candidate_corrects":v[False,True],"candidate_breaks":v[True,False],"both_wrong":v[False,False],"net_correct":v[False,True]-v[True,False]} for k,v in paired.items()},"class_extremes":{k:{"losses":[v for v in vals if v["net_correct"]<0][:4],"gains":[v for v in reversed(vals) if v["net_correct"]>0][:4]} for k,vals in changes.items()},"top_confusions":{n:[{"true":classes[i],"predicted":classes[j],"count":v} for (i,j),v in cm[n].most_common() if i!=j][:5] for n in names}}
print(json.dumps(out,indent=2))
print("NWPU_TEST_PAIRS_COMPLETE")
