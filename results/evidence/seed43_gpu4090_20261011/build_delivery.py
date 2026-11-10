from pathlib import Path
import base64, hashlib, json, zlib, ast, subprocess
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
e=root/'results/evidence/seed43_gpu4090_20261011'
def packed(p):
    data=p.read_bytes()
    return base64.b64encode(zlib.compress(data,9)).decode(),hashlib.sha256(data).hexdigest()
def wrapper(p,body):
    b,h=packed(p)
    return 'import base64,zlib,hashlib,json; s=zlib.decompress(base64.b64decode('+repr(b)+')); assert hashlib.sha256(s).hexdigest()=='+repr(h)+'; ns={"__name__":"frozen_handoff"}; exec(compile(s,"<frozen_handoff>","exec"),ns); '+body
# Use only one shell quoting layer, with double quotes inside Python.
def command(src):
    ast.parse(src)
    return "python -B -u -c '"+src.replace("'",'"')+"'\n"
nw=root/'tools/run_nwpu_seed43.py'
be=root/'tools/run_bigearth_soft43.py'
pf=e/'preflight_bigearth43.py'
(e/'nwpu_preflight_command.txt').write_text(command(wrapper(nw,'print(json.dumps(ns["preflight"](),indent=2))')))
blob,sha=packed(nw)
script=f'''import base64,zlib,hashlib,json,subprocess,sys,os
from pathlib import Path
p=Path("/home/featurize/work/nwpu_three43_20261011_01")
assert not p.exists(), "Output exists: preserve and inspect"
s=zlib.decompress(base64.b64decode("{blob}"))
assert hashlib.sha256(s).hexdigest()=="{sha}"
ns={{"__name__":"frozen_handoff"}}
exec(compile(s,"<nwpu43>","exec"),ns)
print(json.dumps(ns["preflight"](),indent=2),flush=True)
p.mkdir()
f=p/"run_nwpu_seed43.py"
f.write_bytes(s)
with (p/"run.log").open("xb") as log:
    q=subprocess.Popen([sys.executable,"-B","-u",str(f),str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","PYTHONDONTWRITEBYTECODE":"1"}})
(p/"run.pid").write_text(str(q.pid)+"\\n")
print("PID",q.pid,"OUTPUT",p,flush=True)
'''
(e/'launch_nwpu.py').write_text(script)
# Compress the multiline bootstrap, keeping the command shell-simple.
def bootstrap(p):
    blob,sha=packed(p)
    return command('import base64,zlib,hashlib; s=zlib.decompress(base64.b64decode("'+blob+'")); assert hashlib.sha256(s).hexdigest()=="'+sha+'"; exec(compile(s,"<launcher>","exec"),{"__name__":"__main__"})')
(e/'nwpu_launch_command.txt').write_text(bootstrap(e/'launch_nwpu.py'))
(e/'bigearth_preflight_command.txt').write_text(command(wrapper(pf,'pass')))
blob,sha=packed(be)
pfblob,pfsha=packed(pf)
pfcode=wrapper(pf,'print("READYJSON "+json.dumps(ns["report"]),flush=True)')
# repr() is Python data, never interpolated directly into a shell command.
script=f'''import base64,zlib,hashlib,json,subprocess,sys,os
from pathlib import Path
p=Path("/home/featurize/work/bigearth_soft43_20261011_01")
assert not p.exists(), "Output exists: preserve and inspect"
process=subprocess.Popen([sys.executable,"-B","-u","-c",{pfcode!r}],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
ready=None
for line in process.stdout:
    if line.startswith("READYJSON "):
        ready=json.loads(line[len("READYJSON "):])
    else:
        print(line,end="",flush=True)
assert process.wait()==0 and ready is not None, "Preflight failed: no training started"
assert ready["status"]=="BIGEARTH_SOFT43_PREFLIGHT_PASS"
s=zlib.decompress(base64.b64decode("{blob}"))
check=zlib.decompress(base64.b64decode("{pfblob}"))
assert hashlib.sha256(s).hexdigest()=="{sha}" and hashlib.sha256(check).hexdigest()=="{pfsha}"
p.mkdir()
(p/"preflight.py").write_bytes(check)
(p/"approved_preflight.json").write_text(json.dumps(ready,indent=2)+"\\n")
f=p/"run_bigearth_soft43.py"
f.write_bytes(s)
with (p/"run.log").open("xb") as log:
    q=subprocess.Popen([sys.executable,"-B","-u",str(f),str(p)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={{**os.environ,"OMP_NUM_THREADS":"8","MKL_NUM_THREADS":"8","PYTHONDONTWRITEBYTECODE":"1"}})
(p/"run.pid").write_text(str(q.pid)+"\\n")
print("PID",q.pid,"OUTPUT",p,flush=True)
'''
(e/'launch_bigearth.py').write_text(script)
(e/'bigearth_launch_command.txt').write_text(bootstrap(e/'launch_bigearth.py'))
validation={}
for name in ('nwpu_preflight_command.txt','nwpu_launch_command.txt','bigearth_preflight_command.txt','bigearth_launch_command.txt'):
    p=e/name
    import shlex
    argv=shlex.split(p.read_text())
    assert argv[:4]==['python','-B','-u','-c'] and len(argv)==5
    ast.parse(argv[4])
    for shell in ('bash','zsh'):
        result=subprocess.run([shell,'-n',str(p)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    validation[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),python_ast=True,bash_syntax=True,zsh_syntax=True,shell_layers=1)
(e/'command_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({k:{'bytes':(e/k).stat().st_size,**v} for k,v in validation.items()},indent=2))
