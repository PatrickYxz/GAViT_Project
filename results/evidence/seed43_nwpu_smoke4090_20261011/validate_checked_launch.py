from pathlib import Path
import base64,copy,hashlib,importlib.util,json,subprocess,sys,tempfile,time,zlib
from unittest.mock import patch
root=Path.cwd(); e=root/'results/evidence/seed43_nwpu_smoke4090_20261011'
spec=importlib.util.spec_from_file_location('nwpu43',root/'tools/run_nwpu_seed43.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
gate={}; exec(compile((e/'check_smoke.py').read_text(),'<gate>','exec'),gate)

def fixture(work):
    repo=work/'source'; repo.mkdir(); (repo/'run_scene.py').write_text('fixture source')
    smoke=work/'smoke'; smoke.mkdir()
    for name in ('best.pth','last.pth','last.train_state.pth','assignment_probe/epoch_000.pt','assignment_probe/epoch_001.pt','assignment_probe/best.pt','assignment_probe/final.pt'):
        p=smoke/name; p.parent.mkdir(exist_ok=True); p.write_bytes(b'fixture checkpoint')
    code=dict(git={'commit':m.COMMIT[:7],'dirty':False},files_sha256={'run_scene.py':m.digest(repo/'run_scene.py')})
    training=dict(seed=43,batch_size=32,workers=2,optimizer='AdamW',lr=.0003,weight_decay=.0001,scheduler_t_max=30,loss='CrossEntropyLoss',precision='float32_no_amp',tf32=False)
    r=dict(status='complete',phase='smoke',model='gavit',dataset='NWPU-RESISC45',seed=43,counts={'train':360,'val':90},final_epoch=1,best_epoch=1,manifest_sha256=m.MANIFEST_SHA,pretrained_sha256=m.PRE_SHA,
        architecture=dict(grouping='soft_prototype',num_classes=45,edge_type='sparse_hybrid',integration='token_feedback'),code=code,training=training,
        hardware=dict(device='cuda',device_name=m.GPU_NAME,torch_num_threads=8,python=sys.version,packages={}),
        graph_capture=dict(edges_per_image=80,spatial_edges=48,feature_edges=32,actual_forward_checked=True),reload_max_logit_diff=1e-6,checkpoint_sha256=m.digest(smoke/'best.pth'))
    meta=dict(checkpoint_sha256=r['checkpoint_sha256'],execution={'phase':'smoke'},data={'counts':r['counts']},training=training,code=code,architecture=r['architecture'])
    for name,obj in (('run.json',r),('best.meta.json',meta),('history.json',[dict(epoch=1,train_loss=3.334170)])):
        (smoke/name).write_text(json.dumps(obj))
    (smoke/'run.exit').write_text('exit_code=0\n')
    return smoke,{**m.__dict__,'REPO':repo},r

results=[]
for failure in ('none','nonzero','seed','recipe','graph','reload_nan','weight','source','metadata','probe','loss'):
    with tempfile.TemporaryDirectory() as td:
        work=Path(td); smoke,ns,r=fixture(work)
        if failure=='nonzero': (smoke/'run.exit').write_text('exit_code=1\n')
        if failure=='seed': r['seed']=42
        if failure=='recipe': r['training']['lr']=.03
        if failure=='graph': r['graph_capture']['actual_forward_checked']=False
        if failure=='reload_nan': r['reload_max_logit_diff']=float('nan')
        if failure=='weight': (smoke/'best.pth').write_bytes(b'changed')
        if failure=='source': (ns['REPO']/'run_scene.py').write_text('changed')
        if failure=='metadata':
            meta=json.loads((smoke/'best.meta.json').read_text());meta['execution']['phase']='calibrate';(smoke/'best.meta.json').write_text(json.dumps(meta))
        if failure=='probe': (smoke/'assignment_probe/final.pt').unlink()
        if failure=='loss': (smoke/'history.json').write_text(json.dumps([dict(epoch=1,train_loss=float('nan'))]))
        (smoke/'run.json').write_text(json.dumps(r))
        try: receipt=gate['check_smoke'](ns,smoke)
        except (RuntimeError,FileNotFoundError): assert failure!='none',failure
        else: assert failure=='none' and receipt['status']=='NWPU43_SMOKE_ARTIFACTS_PASS',failure
        results.append(dict(case=failure,passed=True))

# Exercise the real detached bootstrap with an inert worker and local artifacts.
original=(root/'tools/run_nwpu_seed43.py').read_bytes(); blob=base64.b64encode(zlib.compress(original,9)).decode(); sha=hashlib.sha256(original).hexdigest()
raw=(e/'launch_nwpu_checked.py').read_text(); assert raw.count(blob)==raw.count(sha)==1
for reject in (False,True):
    with tempfile.TemporaryDirectory() as td:
        work=Path(td); smoke,ns,r=fixture(work); out=work/'formal'
        if reject: (smoke/'run.exit').write_text('exit_code=1\n')
        fake='''import hashlib,json,sys\nfrom pathlib import Path\nREPO=Path(REPO_PATH)\nCOMMIT=COMMIT_VALUE\nMANIFEST_SHA=MANIFEST_VALUE\nPRE_SHA=PRE_VALUE\nGPU_NAME="NVIDIA GeForce RTX 4090"\ndef read(p): return json.loads(Path(p).read_text())\ndef digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()\ndef require(ok,message):\n if not ok: raise RuntimeError(message)\ndef preflight(): return {"status":"FIXTURE","gpu_profile":{"uuid":"GPU-cede40c0-f93d-7c8e-1006-cf6183e53592"}}\nif __name__=="__main__":\n p=Path(sys.argv[1]); (p/"run.exit").write_text("exit_code=0\\n"); print("FIXTURE NO GPU",flush=True)\n'''
        for key,val in dict(REPO_PATH=str(ns['REPO']),COMMIT_VALUE=m.COMMIT,MANIFEST_VALUE=m.MANIFEST_SHA,PRE_VALUE=m.PRE_SHA).items():fake=fake.replace(key,repr(val))
        fb=base64.b64encode(zlib.compress(fake.encode(),9)).decode(); fh=hashlib.sha256(fake.encode()).hexdigest()
        src=raw.replace(blob,fb).replace(sha,fh).replace('/home/featurize/work/nwpu_three43_20261011_01',str(out)).replace('/home/featurize/work/gpucheck_nwpu_soft43_4090_20261011_01',str(smoke))
        p=work/'launch.py'; p.write_text(src)
        q=subprocess.run([sys.executable,str(p)],capture_output=True,text=True)
        if reject: assert q.returncode!=0 and not out.exists(),(q.stdout,q.stderr)
        else:
            assert q.returncode==0,(q.stdout,q.stderr)
            end=time.monotonic()+5
            while time.monotonic()<end and (not (out/'run.exit').exists() or 'FIXTURE NO GPU' not in (out/'run.log').read_text()):time.sleep(.02)
            assert (out/'run.exit').read_text()=='exit_code=0\n'
            assert json.loads((out/'smoke_validation.json').read_text())['status']=='NWPU43_SMOKE_ARTIFACTS_PASS'
            before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
            repeat=subprocess.run([sys.executable,str(p)],capture_output=True,text=True)
            assert repeat.returncode!=0 and before=={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
        results.append(dict(case='bootstrap_reject' if reject else 'bootstrap_accept_duplicate_preserved',passed=True))
(e/'local_validation.json').write_text(json.dumps(dict(status='CHECKED_LAUNCH_FIXTURES_PASS',cases=results,gpu_calls=0,formal_helper_sha256=sha,helper_unchanged=True),indent=2)+'\n')
print(json.dumps(dict(cases_passed=len(results),gpu_calls=0,formal_helper_unchanged=True)))
