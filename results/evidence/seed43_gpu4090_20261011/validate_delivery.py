from pathlib import Path
import ast,base64,hashlib,json,shlex,subprocess,sys,tempfile,time,zlib
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
e=root/'results/evidence/seed43_gpu4090_20261011'
fixtures=[]
fake='''import sys\nfrom pathlib import Path\ndef preflight(): return {"status":"FIXTURE_PREFLIGHT_PASS"}\nif __name__=="__main__":\n p=Path(sys.argv[1]); (p/"run.exit").write_text("exit_code=0\\n"); print("FIXTURE ONLY, NO GPU",flush=True)\n'''
fake_blob=base64.b64encode(zlib.compress(fake.encode(),9)).decode(); fake_sha=hashlib.sha256(fake.encode()).hexdigest()
for label,tool,target in [('nwpu','run_nwpu_seed43.py','nwpu_three43_20261011_01'),('bigearth','run_bigearth_soft43.py','bigearth_soft43_20261011_01')]:
    raw=(e/('launch_'+label+'.py')).read_text()
    original=(root/'tools'/tool).read_bytes()
    blob=base64.b64encode(zlib.compress(original,9)).decode(); sha=hashlib.sha256(original).hexdigest()
    assert raw.count(blob)==1 and raw.count(sha)==1
    for failure in (False,True):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/target
            s=raw.replace('/home/featurize/work/'+target,str(out)).replace(blob,fake_blob).replace(sha,fake_sha)
            if label=='nwpu' and failure:
                bad=fake.replace('return {"status":"FIXTURE_PREFLIGHT_PASS"}','raise RuntimeError("fixture preflight rejection")')
                s=s.replace(fake_blob,base64.b64encode(zlib.compress(bad.encode(),9)).decode()).replace(fake_sha,hashlib.sha256(bad.encode()).hexdigest())
            if label=='bigearth':
                tree=ast.parse(s)
                arg=next(n.value.args[0].elts[4].value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='process' for t in n.targets))
                replacement='print("RGB CHECK fixture"); print(\'READYJSON {"status":"BIGEARTH_SOFT43_PREFLIGHT_PASS","seed":43}\'); raise SystemExit('+('7' if failure else '0')+')'
                assert repr(arg) in s
                s=s.replace(repr(arg),repr(replacement))
            p=Path(td)/'launcher.py';p.write_text(s)
            q=subprocess.run([sys.executable,str(p)],capture_output=True,text=True)
            if failure:
                assert q.returncode!=0 and not out.exists(),(label,q.stdout,q.stderr)
            else:
                assert q.returncode==0,(label,q.stdout,q.stderr)
                end=time.monotonic()+10
                while not (out/'run.exit').exists() and time.monotonic()<end: time.sleep(.02)
                assert (out/'run.exit').read_text()=='exit_code=0\n'
                # Wait for fake process to close its log before measuring preservation.
                end=time.monotonic()+10
                while 'FIXTURE ONLY' not in (out/'run.log').read_text() and time.monotonic()<end:time.sleep(.02)
                before={str(f.relative_to(out)):f.read_bytes() for f in out.rglob('*') if f.is_file()}
                repeat=subprocess.run([sys.executable,str(p)],capture_output=True,text=True)
                after={str(f.relative_to(out)):f.read_bytes() for f in out.rglob('*') if f.is_file()}
                assert repeat.returncode!=0 and before==after,'Duplicate changed artifacts'
            fixtures.append(dict(queue=label,preflight_rejected=failure,passed=True,gpu_calls=0,real_detached_child=not failure))
# Shell-delivery decoded bytes exactly match the current readable sources.
for name,expected in [('nwpu_preflight_command.txt',root/'tools/run_nwpu_seed43.py'),('bigearth_preflight_command.txt',e/'preflight_bigearth43.py'),('nwpu_launch_command.txt',e/'launch_nwpu.py'),('bigearth_launch_command.txt',e/'launch_bigearth.py')]:
    code=shlex.split((e/name).read_text())[-1]
    constants=[n.value for n in ast.walk(ast.parse(code)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>256]
    assert len(constants)==1
    data=zlib.decompress(base64.b64decode(constants[0]))
    assert data==expected.read_bytes(),name
    ast.parse(data)
(e/'delivery_validation.json').write_text(json.dumps(dict(fixtures=fixtures,payload_roundtrip=True,gpu_work_performed=False),indent=2)+'\n')
print(json.dumps(fixtures,indent=2))
