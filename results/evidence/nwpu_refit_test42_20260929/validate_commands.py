"""Validate pasted command syntax and orchestration with explicit fake child outputs."""
import ast, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

e=Path(__file__).resolve().parent
c=json.loads((e/'commands.json').read_text())
for key in ('prepare_code','queue_code','result_code'):
    ast.parse(c[key])
for key in ('prepare','queue','result'):
    for shell in ('bash','zsh'):
        subprocess.run([shell,'-n','-c',c[key]],check=True)
base=Path(tempfile.mkdtemp(prefix='nwpu-refit-queue-test-',dir='/private/tmp'))
repo=Path('/private/tmp/gavit-soft-release-git-20260927')
source=base/'calibration'; source.mkdir()
(source/'queue.exit').write_text('exit_code=0\n')
data=base/'data'; data.mkdir(); (data/'image.jpg').write_bytes(b'synthetic path-presence placeholder')
manifest=base/'manifest.json'
# Duplicated placeholder paths are intentional: only queue control is tested.
manifest.write_text(json.dumps({'dataset':'NWPU-RESISC45','counts':{'train':5040,'val':1260,'test':25200},'records':[{'path':'image.jpg'}]*31500}))
pre=base/'pretrained'; pre.write_bytes(b'pretrained identity fixture')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((e.parent/'nwpu_calibration42_20260928/server_return.json').read_text())['runs_returned_subset']
shared=json.loads((e.parent/'nwpu_calibration42_20260928/cached_analysis_server_summary.json').read_text())
replacements={
    '/home/featurize/work/GAViT_SOFT_20260927':str(repo),
    '/home/featurize/work/nwpu_ablation42_20260928_01':str(source),
    '/home/featurize/work/nwpu_protocol_20260927_02.json':str(manifest),
    '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors':str(pre),
    '/home/featurize/data/NWPU-RESISC45':str(data),
    old['swin']['manifest_sha256']:sha(manifest),
    old['swin']['pretrained_sha256']:sha(pre)}
for n,r in old.items():
    d=source/n; d.mkdir(); (d/'best.pth').write_bytes(('fixture '+n).encode())
    replacements[r['checkpoint_sha256']]=sha(d/'best.pth')
    r=dict(r,phase='calibrate',checkpoint_sha256=sha(d/'best.pth'),manifest_sha256=sha(manifest),pretrained_sha256=sha(pre),training=shared['training'],hardware=shared['hardware'],code={'git':shared['code_git'],'files_sha256':{'fixture':'not real model code'}})
    (d/'run.json').write_text(json.dumps(r)); (d/'run.exit').write_text('exit_code=0\n')
patch='''import subprocess,json,os,hashlib
from pathlib import Path
from types import SimpleNamespace
actual_run=subprocess.run
actual_check_output=subprocess.check_output
def output(argv,**kwargs):
    if argv[0]=="nvidia-smi":
        return "" if "--query-compute-apps=pid" in argv else "NVIDIA GeForce RTX 2080 Ti\\n"
    return actual_check_output(argv,**kwargs)
subprocess.check_output=output
def fake_run(argv,**kwargs):
    if argv[0]=="git":
        return actual_run(argv,**kwargs)
    assert argv[2].endswith("run_scene.py")
    get=lambda k:argv[argv.index(k)+1]
    out=Path(get("--output")); n,kind=out.name.split("_")
    assert get("--device")=="cuda" and get("--batch-size")=="32" and get("--workers")=="2"
    source=Path(os.environ["FIXTURE_SOURCE"])
    cal=json.loads((source/n/"run.json").read_text())
    with (out.parent/"calls.jsonl").open("a") as f:
        f.write(json.dumps({"stage":out.name,"argv":argv})+"\\n")
    if os.environ.get("FAIL_STAGE")==out.name:
        return SimpleNamespace(returncode=9)
    out.mkdir()
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if kind=="refit":
        assert argv[3]=="train" and get("--phase")=="refit" and get("--seed")=="42"
        assert get("--calibration")==str(source/n)
        assert get("--model")==cal["model"]
        assert (("--grouping" not in argv) if n=="swin" else get("--grouping")==cal["architecture"]["grouping"])
        (out/"last.pth").write_bytes(out.name.encode())
        r=dict(cal,status="complete",phase="refit",counts={"train":6300,"val":0},final_epoch=cal["best_epoch"],best_epoch=None,best_val_oa_percent=None,checkpoint_sha256=h(out/"last.pth"),refit_source={"run_json_sha256":h(source/n/"run.json"),"best_checkpoint_sha256":cal["checkpoint_sha256"]})
        if os.environ.get("BAD_IDENTITY")==out.name:
            r["counts"]={"train":5040,"val":1260}
    else:
        assert argv[3]=="evaluate" and get("--split")=="test"
        assert get("--checkpoint")==str(out.parent/(n+"_refit")/"last.pth")
        training=json.loads((out.parent/(n+"_refit")/"run.json").read_text())
        r={"status":"complete","samples":25200,"split":"test","checkpoint_sha256":training["checkpoint_sha256"],"evaluation":{"code":cal["code"],"hardware":cal["hardware"]}}
        m=dict(r,training_phase="refit",fit_scope="train_plus_val",manifest_sha256=cal["manifest_sha256"],oa_percent=1.0,macro_f1_percent=1.0)
        (out/"metrics.json").write_text(json.dumps(m))
        (out/"predictions.csv").write_text("fixture only"); (out/"per_class.csv").write_text("fixture only")
    (out/"run.json").write_text(json.dumps(r)); (out/"run.exit").write_text("exit_code=0\\n")
    return SimpleNamespace(returncode=0)
subprocess.run=fake_run
'''
all_stages=['swin_refit','fixed_refit','soft_refit','swin_test','fixed_test','soft_test']
def adapted(key,out):
    text=c[key].replace('/home/featurize/work/nwpu_refit_test42_20260929_01',str(out))
    for original,target in replacements.items():
        text=text.replace(original,target)
    return text
results=[]
for label,fail,bad,count in [('success','','',6),('first_failure','swin_refit','',1),('middle_failure','fixed_refit','',2),('late_failure','fixed_test','',5),('bad_identity','','fixed_refit',2)]:
    out=base/label
    env={**os.environ,'FIXTURE_SOURCE':str(source),'FAIL_STAGE':fail,'BAD_IDENTITY':bad}
    prepared=subprocess.run([sys.executable,'-c',patch+'\n'+adapted('prepare_code',out)],env=env,text=True,capture_output=True)
    assert prepared.returncode==0,prepared.stderr
    result=subprocess.run([sys.executable,'-c',patch+'\n'+adapted('queue_code',out)],env=env,text=True,capture_output=True)
    calls=[json.loads(s)['stage'] for s in (out/'calls.jsonl').read_text().splitlines()]
    assert calls==all_stages[:count],(label,calls,result.stderr)
    expected=0 if label=='success' else 1
    assert result.returncode==expected,result.stderr
    assert (out/'queue.exit').read_text()==f'exit_code={expected}\n'
    summary=json.loads((out/'queue_summary.json').read_text())
    assert summary['status']==('complete' if expected==0 else 'failed')
    if label=='success':
        report=subprocess.run([sys.executable,'-c',adapted('result_code',out)],capture_output=True,text=True)
        assert report.returncode==0,report.stderr
        before=(out/'queue_summary.json').read_bytes()
        twice=subprocess.run([sys.executable,'-c',patch+'\n'+adapted('queue_code',out)],env=env,capture_output=True,text=True)
        assert twice.returncode!=0 and 'FileExistsError' in twice.stderr
        assert (out/'queue.exit').read_text()=='exit_code=0\n' and (out/'queue_summary.json').read_bytes()==before
    (e/(label+'.log')).write_text(prepared.stdout+prepared.stderr+result.stdout+result.stderr)
    results.append(dict(case=label,stages=calls,exit=result.returncode))
# A missing original file must fail before output creation or paid child execution.
(data/'image.jpg').unlink()
missing=base/'missing_data'
failed=subprocess.run([sys.executable,'-c',patch+'\n'+adapted('prepare_code',missing)],capture_output=True,text=True)
assert failed.returncode!=0 and 'DATA_MISSING' in failed.stderr and not missing.exists()
validation=dict(scope='Local syntax and fake-child orchestration only; no actual refit/test inference',syntax='Python, Bash, zsh PASS',cases=results,duplicate_launch_preserves_completed_result=True,missing_data_rejected_before_creation=True,fixture=str(base))
(e/'local_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
