import json,math
from pathlib import Path
p=Path("/home/featurize/work/bigearth_soft43_20261011_01")
s=json.loads((p/"summary.json").read_text())
assert (p/"run.exit").read_text().strip()=="exit_code=0" and s["status"]=="complete" and s["seed"]==43
train=s["stages"]["gavit_train"]
test=s["stages"]["gavit_test"]
assert train["status"]==test["status"]=="complete" and train["final_epoch"]==30
assert train["checkpoint_sha256"]==test["checkpoint_sha256"]==test["metrics"]["checkpoint_sha256"]
assert test["metrics"]["samples"]==119825
h=json.loads((p/"gavit_train/history.json").read_text())
assert [row["epoch"] for row in h]==list(range(1,31))
best=max(h,key=lambda row:row["val_mAP_percent"])
assert best["epoch"]==train["best"]["epoch"] and abs(best["val_mAP_percent"]-train["best"]["value"])<=.000051
print(json.dumps({"status":s["status"],"seed":s["seed"],"queue_elapsed_seconds":s["elapsed_seconds"],"hourly_yuan":1.67,"queue_time_estimated_yuan":s["elapsed_seconds"]*1.67/3600,"train":{k:train.get(k) for k in ("final_epoch","best","parameters","counts","elapsed_seconds","peak_cuda_allocated_bytes","peak_cuda_reserved_bytes","main_process_peak_rss_kib","effective_train_images_per_second","checkpoint_sha256","graph_capture","probe_summary","commit","pretrained_sha256","splits","hardware","precision")},"test":{k:test.get(k) for k in ("elapsed_seconds","peak_cuda_allocated_bytes","peak_cuda_reserved_bytes","main_process_peak_rss_kib","checkpoint_sha256","hardware","precision")},"history":{"epochs":[row["epoch"] for row in h],"first":h[0],"best":best,"last":h[-1]},"stage_exits":{stage:(p/stage/"run.exit").read_text().strip() for stage in ("gavit_train","gavit_test")}},indent=2,allow_nan=False))
