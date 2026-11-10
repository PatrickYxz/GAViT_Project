import csv,json,hashlib
from pathlib import Path
from collections import Counter
n=("old_swin","old_soft","swin_ffn","soft_ffn")
root=Path("/home/featurize/work")
p=dict(zip(n,[root/"nwpu_ablation42_20260928_01"/x for x in ("swin","soft")]+[root/"nwpu_ffn42_20261003_01"/x for x in n[2:]]))
read=lambda k,f:json.loads((p[k]/f).read_text())
r={k:read(k,"run.json") for k in n}
m={k:read(k,"validation_metrics.json") for k in n}
h={k:read(k,"history.json") for k in n}
assert all((p[k]/"run.exit").read_text().strip()=="exit_code=0" and r[k]["status"]=="complete" and r[k]["phase"]=="calibrate" and r[k]["seed"]==42 and r[k]["counts"]=={"train":5040,"val":1260} and r[k]["final_epoch"]==30 for k in n),"run identity/exit"
assert all(r[k][f]==r[n[0]][f] for k in n for f in ("manifest_sha256","pretrained_sha256","training")),"shared inputs/training"
assert all([v["epoch"] for v in h[k]]==list(range(1,31)) and max(h[k],key=lambda v:v["validation"]["oa_percent"])["epoch"]==r[k]["best_epoch"] for k in n),"history/first maximum"
a={k:list(csv.DictReader((p[k]/"validation_predictions.csv").read_text().splitlines())) for k in n}
d={k:{v["path"]:(int(v["label"]),int(v["prediction"])) for v in a[k]} for k in n}
assert all(len(a[k])==len(d[k])==1260 and d[k].keys()==d[n[0]].keys() for k in n),"paths/duplicates"
assert all(d[k][s][0]==d[n[0]][s][0] and all(0<=v<45 for v in d[k][s]) for k in n for s in d[k]),"labels/predictions"
c={k:Counter(d[k].values()) for k in n}
t={k:Counter(y for y,z in d[k].values()) for k in n}
u={k:Counter(z for y,z in d[k].values()) for k in n}
correct={k:sum(c[k][i,i] for i in range(45)) for k in n}
f1={k:sum(200*c[k][i,i]/(t[k][i]+u[k][i]) if t[k][i]+u[k][i] else 0 for i in range(45))/45 for k in n}
classes={k:{v["class_index"]:v["class_name"] for v in m[k]["per_class"]} for k in n}
assert all(len(m[k]["per_class"])==45 and set(classes[k])==set(range(45)) and classes[k]==classes[n[0]] and all(t[k][i]==28 for i in range(45)) for k in n),"classes/support"
assert all(m[k]["samples"]==1260 and abs(correct[k]*100/1260-m[k]["oa_percent"])<1e-8 and abs(f1[k]-m[k]["macro_f1_percent"])<1e-8 and abs(m[k]["oa_percent"]-r[k]["best_val_oa_percent"])<1e-8 and abs(m[k]["oa_percent"]-h[k][r[k]["best_epoch"]-1]["validation"]["oa_percent"])<1e-8 and abs(f1[k]-h[k][r[k]["best_epoch"]-1]["validation"]["macro_f1_percent"])<1e-8 for k in n),"cached/selected metrics"
actual={k:hashlib.sha256((p[k]/"best.pth").read_bytes()).hexdigest() for k in n[2:]}
assert all(actual[k]==r[k]["checkpoint_sha256"] for k in n[2:]),"best SHA"
pairs=((n[1],n[3]),(n[2],n[3]),(n[0],n[3]),(n[0],n[2]))
paired={(b,q):Counter((d[b][s][0]==d[b][s][1],d[q][s][0]==d[q][s][1]) for s in d[b]) for b,q in pairs}
[print(k,json.dumps(dict(OA=m[k]["oa_percent"],macroF1=f1[k],correct=correct[k],best=r[k]["best_epoch"],finalOA=h[k][-1]["validation"]["oa_percent"],last5mean=sum(v["validation"]["oa_percent"] for v in h[k][-5:])/5,finalLoss=h[k][-1]["train_loss"],parameters=r[k]["total_parameters"],trainSeconds=sum(v["train_seconds"] for v in h[k]),valSeconds=sum(v["validation_seconds"] for v in h[k]),**({"code_git":r[k].get("code",{}).get("git"),"hardware":r[k].get("hardware"),"actualBestSHA":actual[k]} if k in n[2:] else {})),separators=(",",":"))) for k in n]
[print(q+" vs "+b,json.dumps(dict(both_correct=paired[b,q][True,True],corrects=paired[b,q][False,True],new_errors=paired[b,q][True,False],both_wrong=paired[b,q][False,False],net=correct[q]-correct[b],class_net={classes[b][i]:c[q][i,i]-c[b][i,i] for i in range(45) if c[q][i,i]!=c[b][i,i]}),separators=(",",":"))) for b,q in pairs]
print("COMPACT_CLOSEOUT_PASS; class_net omitted classes=0; scope=cached validation and two actual best hashes only; no tensor probe, last/state byte audit, local curve mirror, new forward, training or test")
