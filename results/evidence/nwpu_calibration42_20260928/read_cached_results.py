import csv, json, sys
from pathlib import Path
from collections import Counter
p = Path(sys.argv[1])
names = ("swin", "fixed", "soft")
read = lambda n, f: json.loads((p/n/f).read_text())
r = {n: read(n, "run.json") for n in names}
m = {n: read(n, "validation_metrics.json") for n in names}
h = {n: read(n, "history.json") for n in names}
shared = {k: all(r[n][k] == r["swin"][k] for n in names) for k in ("seed", "counts", "manifest_sha256", "pretrained_sha256", "training", "code", "hardware")}
assert all(shared.values()), shared
assert all(r[n]["status"] == "complete" and r[n]["phase"] == "calibrate" and r[n]["seed"] == 42 and r[n]["counts"] == {"train":5040,"val":1260} and r[n]["final_epoch"] == 30 for n in names), "Run identity mismatch"
a = {n: list(csv.DictReader((p/n/"validation_predictions.csv").open())) for n in names}
d = {n: {v["path"]:v for v in a[n]} for n in names}
assert all(len(a[n]) == len(d[n]) == 1260 and d[n].keys() == d["swin"].keys() for n in names), "Path mismatch or duplicates"
assert all(d[n][s]["label"] == d["swin"][s]["label"] for n in names for s in d[n]), "Labels differ"
correct = {n: sum(v["label"] == v["prediction"] for v in a[n]) for n in names}
assert all(m[n]["samples"] == 1260 and abs(correct[n]*100/1260 - m[n]["oa_percent"]) < 1e-8 and abs(m[n]["oa_percent"] - r[n]["best_val_oa_percent"]) < 1e-8 for n in names), "OA mismatch"
assert all([v["epoch"] for v in h[n]] == list(range(1,31)) for n in names), "Incomplete history"
assert all(r[n]["best_epoch"] == max(h[n], key=lambda v:v["validation"]["oa_percent"])["epoch"] for n in names), "Best epoch mismatch"
c = {n: Counter((d[n][s]["prediction"] == d[n][s]["label"], d["soft"][s]["prediction"] == d["soft"][s]["label"]) for s in d[n]) for n in ("swin", "fixed")}
paired = {n:{"both_correct":v[True,True],"soft_corrects":v[False,True],"soft_breaks":v[True,False],"both_wrong":v[False,False],"net_correct":v[False,True]-v[True,False]} for n,v in c.items()}
pc = {n:{v["class_index"]:v for v in m[n]["per_class"]} for n in names}
assert all(set(pc[n]) == set(range(45)) and sum(v["support"] for v in pc[n].values()) == 1260 for n in names), "Class count mismatch"
assert all((pc[n][i]["class_name"],pc[n][i]["support"]) == (pc["swin"][i]["class_name"],pc["swin"][i]["support"]) for n in names for i in range(45)), "Class identity mismatch"
changes = {n:sorted([{"class":pc[n][i]["class_name"],"support":pc[n][i]["support"],"base_OA":round(pc[n][i]["accuracy_percent"],4),"soft_OA":round(pc["soft"][i]["accuracy_percent"],4),"delta_pp":round(pc["soft"][i]["accuracy_percent"]-pc[n][i]["accuracy_percent"],4)} for i in range(45)],key=lambda v:v["delta_pp"]) for n in ("swin","fixed")}
probe = [json.loads(s) for s in (p/"soft"/"assignment_probe"/"assignment_diagnostics.jsonl").read_text().splitlines() if s.strip()]
assert [v["epoch"] for v in probe] == list(range(31)), "Incomplete probe"
selected = [probe[i] for i in (0,r["soft"]["best_epoch"],30)]
probe_summary = [{**{k:v[k] for k in ("epoch","entropy_mean","mean_abs_delta_S","low_norm_prototypes","prototype_norm_min","prototype_cosine_off_diagonal_mean","prototype_cosine_off_diagonal_max")},"min_region_mass":min(x for row in v["region_mass"] for x in row),"max_region_mass":max(x for row in v["region_mass"] for x in row),"feedback_ratio_mean":sum(v["feedback_to_token_ratio"])/len(v["feedback_to_token_ratio"]) if all(x is not None for x in v["feedback_to_token_ratio"]) else None} for v in selected]
summary = {"shared_identity":shared,"code_git":r["swin"]["code"]["git"],"hardware":r["swin"]["hardware"],"training":r["swin"]["training"],"validation":{n:{"OA":m[n]["oa_percent"],"macro_F1":m[n]["macro_f1_percent"],"correct":correct[n],"best_epoch":r[n]["best_epoch"],"final_OA":h[n][-1]["validation"]["oa_percent"],"last5_OA_mean":sum(v["validation"]["oa_percent"] for v in h[n][-5:])/5,"parameters":r[n]["total_parameters"],"train_seconds_sum":sum(v["train_seconds"] for v in h[n])} for n in names},"soft_vs":paired,"class_extremes":{n:{"losses":[v for v in changes[n] if v["delta_pp"]<0][:5],"gains":[v for v in reversed(changes[n]) if v["delta_pp"]>0][:5]} for n in changes},"soft_probe":probe_summary}
print(json.dumps(summary,indent=2))
print("CURVES epoch:train_loss:val_OA")
print("\n".join(n+" "+" | ".join(str(v["epoch"])+":"+format(v["train_loss"],".5f")+":"+format(v["validation"]["oa_percent"],".4f") for v in h[n]) for n in names))
print("CACHED_VALIDATION_READ_COMPLETE")
