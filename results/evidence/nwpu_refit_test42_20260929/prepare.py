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
assert not out.exists(),"Output exists; inspect it before another launch"
assert subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()==commit,"Wrong source commit"
assert not subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True).strip(),"Dirty source tree"
assert sha(manifest)==mh and sha(pre)==ph,"Input hash changed"
m=json.loads(manifest.read_text())
assert m["dataset"]=="NWPU-RESISC45" and m["counts"]=={"train":5040,"val":1260,"test":25200} and len(m["records"])==31500
missing=[v["path"] for v in m["records"] if not (data/v["path"]).is_file()]
assert not missing,("DATA_MISSING",missing[:5])
assert (source/"queue.exit").read_text().strip()=="exit_code=0"
r={n:json.loads((source/n/"run.json").read_text()) for n in names}
assert [r[n]["best_epoch"] for n in names]==[29,29,28]
expected=["bbfd4e472739126276e79014f944f911d7b0abbc6f863bb291d799845b9cd0ea","1766fe7c330eb37fb350f0348bcd255bafc451b793411a927548ee005a140f67","674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12"]
assert all(r[n]["status"]=="complete" and r[n]["phase"]=="calibrate" and r[n]["seed"]==42 and r[n]["final_epoch"]==30 and r[n]["manifest_sha256"]==mh and r[n]["pretrained_sha256"]==ph and (source/n/"run.exit").read_text().strip()=="exit_code=0" and sha(source/n/"best.pth")==r[n]["checkpoint_sha256"]==expected[i] for i,n in enumerate(names)),"Calibration changed"
assert all(r[n][k]==r["swin"][k] for n in names for k in ("training","code","hardware")),"Calibration identities differ"
prior=[str(f.parent) for f in set(out.parent.glob("nwpu*/run.json"))|set(out.parent.glob("nwpu*/*/run.json")) if (v:=json.loads(f.read_text())).get("status")=="complete" and v.get("phase")=="refit" and v.get("seed")==42 and v.get("manifest_sha256")==mh]
assert not prior,("Existing refits: inspect for reuse",prior)
gpu=subprocess.check_output(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],text=True).strip()
assert gpu==r["swin"]["hardware"]["device_name"],("GPU differs from calibration",gpu)
assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU busy"
out.mkdir()
info={"status":"ready","commit":commit,"manifest_sha256":mh,"pretrained_sha256":ph,"calibration_run_sha256":{n:sha(source/n/"run.json") for n in names},"selected_epochs":{n:r[n]["best_epoch"] for n in names},"gpu":gpu,"scope":"seed42 refit 6300 images; test 25200 images; no calibration rerun"}
(out/"queue_preflight.json").write_text(json.dumps(info,indent=2)+"\n")
print("NWPU_REFIT_TEST_READY",out,info["selected_epochs"],flush=True)
