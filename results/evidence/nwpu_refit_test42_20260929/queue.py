import json,hashlib,subprocess
from pathlib import Path
repo=Path("/home/featurize/work/GAViT_SOFT_20260927")
out=Path("/home/featurize/work/nwpu_refit_test42_20260929_01")
source=Path("/home/featurize/work/nwpu_ablation42_20260928_01")
manifest=Path("/home/featurize/work/nwpu_protocol_20260927_02.json")
pre=Path("/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors")
data=Path("/home/featurize/data/NWPU-RESISC45")
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit="ed480e60fea03ed786fdba70c144de5e815c34f5"
mh="ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e"
ph="f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36"
names=("swin","fixed","soft")
import os,sys,time,atexit
info=json.loads((out/"queue_preflight.json").read_text())
assert info["status"]=="ready" and info["commit"]==commit
assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==commit and not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip()
assert sha(manifest)==mh and sha(pre)==ph and all(sha(source/n/"run.json")==info["calibration_run_sha256"][n] for n in names),"Frozen inputs changed"
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy"
(out/"queue.started").open("x").close()
status={"exit_code":1}
atexit.register(lambda:(out/"queue.exit").write_text("exit_code="+str(status["exit_code"])+"\n"))
os.chdir(repo)
start=time.monotonic()
cal={n:json.loads((source/n/"run.json").read_text()) for n in names}
common=["--manifest",str(manifest),"--data-root",str(data),"--device","cuda","--batch-size","32","--workers","2"]
entry=[sys.executable,"-u",str(repo/"run_scene.py")]
stages=[(n,"refit",entry+["train","--phase","refit","--seed","42","--model","swin" if n=="swin" else "gavit","--calibration",str(source/n),"--pretrained-path",str(pre),"--output",str(out/(n+"_refit"))]+common+([] if n=="swin" else ["--grouping","attentive_spatial" if n=="fixed" else "soft_prototype"])) for n in names]+[(n,"test",entry+["evaluate","--checkpoint",str(out/(n+"_refit")/"last.pth"),"--split","test","--output",str(out/(n+"_test"))]+common) for n in names]
(out/"commands.json").write_text(json.dumps(stages,indent=2)+"\n")
read=lambda n,k,f:json.loads((out/(n+"_"+k)/f).read_text())
check_refit=lambda n,r:r["phase"]=="refit" and r["counts"]=={"train":6300,"val":0} and r["final_epoch"]==info["selected_epochs"][n] and r["best_epoch"] is None and r["best_val_oa_percent"] is None and r["reload_max_logit_diff"]<=1e-5 and r["refit_source"]=={"run_json_sha256":info["calibration_run_sha256"][n],"best_checkpoint_sha256":cal[n]["checkpoint_sha256"]} and all(r[k]==cal[n][k] for k in ("model","seed","training","architecture","code","hardware","manifest_sha256","pretrained_sha256","graph_capture")) and sha(out/(n+"_refit")/"last.pth")==r["checkpoint_sha256"]
check_test=lambda n,r:(m:=read(n,"test","metrics.json"))["samples"]==r["samples"]==25200 and m["split"]==r["split"]=="test" and m["training_phase"]=="refit" and m["fit_scope"]=="train_plus_val" and m["manifest_sha256"]==mh and m["checkpoint_sha256"]==r["checkpoint_sha256"]==read(n,"refit","run.json")["checkpoint_sha256"] and r["evaluation"]["code"]==cal[n]["code"] and r["evaluation"]["hardware"]==cal[n]["hardware"] and all((out/(n+"_test")/f).is_file() for f in ("predictions.csv","per_class.csv"))
check=lambda n,k:(out/(n+"_"+k)/"run.exit").read_text().strip()=="exit_code=0" and (r:=read(n,k,"run.json"))["status"]=="complete" and (check_refit(n,r) if k=="refit" else check_test(n,r))
results=[]
stage=lambda s:(print("START",s[0],s[1],flush=True),rc:=subprocess.run(s[2]).returncode,passed:=rc==0 and check(s[0],s[1]),results.append({"model":s[0],"stage":s[1],"process_exit":rc,"checks_passed":passed}),print("STAGE_COMPLETE" if passed else "STAGE_FAILED",s[0],s[1],flush=True),passed)[-1]
ok=all(stage(s) for s in stages)
summary={"status":"complete" if ok else "failed","elapsed_seconds":time.monotonic()-start,"stages":results,"tests":{n:{k:read(n,"test","metrics.json")[k] for k in ("samples","oa_percent","macro_f1_percent","training_phase","fit_scope","manifest_sha256","checkpoint_sha256")} for n in names} if ok else {}}
(out/"queue_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
status["exit_code"]=0 if ok else 1
print("NWPU_REFIT_TEST_COMPLETE" if ok else "NWPU_REFIT_TEST_FAILED",out,flush=True)
sys.exit(status["exit_code"])
