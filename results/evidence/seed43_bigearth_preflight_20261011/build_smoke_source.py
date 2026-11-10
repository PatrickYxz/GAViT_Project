import ast
import base64
import hashlib
import json
import shlex
import subprocess
import zlib
from pathlib import Path

R = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
E = R / 'results/evidence/seed43_bigearth_preflight_20261011'
ready = json.loads((E/'user_return.json').read_text())
prior = json.loads((R/'results/evidence/bigearth_soft_preflight_20260928/launch_commands.json').read_text())
old = '/home/featurize/work/bigearth_soft_smoke42_20260928_01'
out = '/home/featurize/work/gpucheck_bigearth_soft43_4090_20261011_01'
prepare = prior['prepare_code'].replace(old,out)
# Engineering subset uses the previously validated Random(42); only training seed becomes43.
train = prior['run_code'].replace(old,out).replace('"--seed","42"','"--seed","43"').replace('soft_prototype_seed42_20260928_01','soft_prototype_seed43_4090_20261011_01')
tail = 'status["exit_code"]=0; print("BIGEARTH_SOFT_TRAIN_RELOAD_PASS",json.dumps(summary),flush=True)'
assert train.count(tail)==1
train = train.replace(tail,'')
helper = (R/'tools/run_bigearth_soft43.py').read_bytes()
sha = hashlib.sha256(helper).hexdigest()
assert sha=='7346c6aa25426998f44eeb55aa9aa33c1f43357d4cbabb3951413e5aa8d1b832'
extra = '''
import math
assert meta['architecture']==helper_ns['ARCH']
expected_training={**helper_ns['TRAINING'],'epochs':1}
assert meta['training']==expected_training and meta['execution']['run_stage']=='smoke'
assert meta['git']=={'commit':'ed480e6','dirty':False}
assert ns['total_params']==31466766
assert all(math.isfinite(ns[k]) for k in ('avg_loss','val_mAP','val_f1'))
state=torch.load(out/'gavit.last.train_state.pth',map_location='cpu',weights_only=True)
assert state['epoch']==1
del state
graph_probe=helper_ns['check_probe'](out/'gavit.assignment_probe',torch,
    ns['model'].region_grouping.candidate_mask.cpu(),ns['val_patches'][:8],1,final_epoch=1)
hardware_after,precision_after=helper_ns['environment'](torch,b)
assert hardware_after==ready['hardware'] and precision_after==ready['precision']
summary.update(seed=43,commit=helper_ns['COMMIT'],hardware=hardware_after,precision=precision_after,
    metadata=meta,parameters=ns['total_params'],train_loss=ns['avg_loss'],val_mAP_percent=ns['val_mAP'],
    val_macro_f1_percent=ns['val_f1'],graph_probe=graph_probe,
    source_preflight='approved_preflight.json',resource_scope='whole smoke including reload and artifact checks',
    status='complete')
torch.cuda.synchronize()
summary['total_seconds_including_reload']=time.perf_counter()-start
summary['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
summary['cuda_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\\n')
status['exit_code']=0
print('BIGEARTH_SOFT43_SMOKE_COMPLETE',json.dumps(summary),flush=True)
'''
train += '\n' + extra
script = f'''import os,sys,hashlib,json,base64,zlib
from pathlib import Path
sys.dont_write_bytecode=True
assert not sys.flags.optimize, 'Disable PYTHONOPTIMIZE'
os.environ['OMP_NUM_THREADS']='8'
os.environ['MKL_NUM_THREADS']='8'
out=Path({out!r})
assert not out.exists(), 'Smoke output exists: preserve and inspect it; do not relaunch'
ready=json.loads({json.dumps(ready)!r})
helper=zlib.decompress(base64.b64decode({base64.b64encode(zlib.compress(helper)).decode()!r}))
assert hashlib.sha256(helper).hexdigest()=={sha!r}
helper_ns={{'__name__':'frozen_bigearth43_smoke_helper'}}
exec(compile(helper,'<frozen_bigearth43_helper>','exec'),helper_ns)
b=helper_ns['load_base']()
b.identity()
b.idle_gpu()
import torch
hardware,precision=helper_ns['environment'](torch,b)
assert hardware==ready['hardware'] and precision==ready['precision'], 'GPU/software/precision changed since passed preflight'
prepare_code={prepare!r}
train_code={train!r}
ns={{'__name__':'__main__','helper_ns':helper_ns,'b':b,'ready':ready}}
exec(compile(prepare_code,'<prepare_bigearth43_smoke>','exec'),ns)
(out/'approved_preflight.json').write_text(json.dumps(ready,indent=2)+'\\n')
(out/'helper_source.py').write_bytes(helper)
(out/'prepare_source.py').write_text(prepare_code)
(out/'train_reload_source.py').write_text(train_code)
exec(compile(train_code,'<train_bigearth43_smoke>','exec'),ns)
'''
ast.parse(script)
ast.parse(prepare)
ast.parse(train)
# Validate actual frozen CLI without executing imports/models.
source=subprocess.check_output(['git','-C',str(R),'show','ed480e6:train_bigearth.py'],text=True)
tree=ast.parse(source)
nodes=[]
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='parser' for t in n.targets):
        nodes.append(n)
    elif isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and isinstance(n.value.func.value,ast.Name) and n.value.func.value.id=='parser':
        nodes.append(n)
parser_ns={}
import argparse
parser_ns['argparse']=argparse
import os
parser_ns['os']=os
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'<frozen_cli>','exec'),parser_ns)
tn=ast.parse(train)
argv_node=next(n for n in tn.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Attribute) and ast.unparse(n.targets[0])=='sys.argv')
argv=eval(compile(ast.Expression(argv_node.value),'<smoke_argv>','eval'),{'str':str,'repo':Path('/home/featurize/work/GAViT_SOFT_20260927')})
args=parser_ns['parser'].parse_args(argv[1:])
assert args.seed==43 and args.epochs==1 and args.batch_size==32 and args.run_stage=='smoke'
assert args.grouping=='soft_prototype' and args.edge_type=='sparse_hybrid' and args.integration=='token_feedback'
assert args.checkpoint_path==out+'/gavit.pth' and args.data_dir==out+'/splits'
assert 'Random(42)' in prepare and out in prepare and old not in prepare+train
assert train.index("helper_ns['check_probe']") < train.index("status['exit_code']=0")
(E/'prepare_smoke_source.py').write_text(prepare)
(E/'train_reload_smoke_source.py').write_text(train)
(E/'smoke_handoff.py').write_text(script)
encoded=base64.b64encode(zlib.compress(script.encode())).decode()
script_sha=hashlib.sha256(script.encode()).hexdigest()
payload='import base64,zlib,hashlib; s=zlib.decompress(base64.b64decode("'+encoded+'")); assert hashlib.sha256(s).hexdigest()=="'+script_sha+'"; exec(compile(s,"<bigearth43_smoke>","exec"))'
cmd='python -B -u -c '+shlex.quote(payload)+'\n'
(E/'bigearth_smoke_command.txt').write_text(cmd)
for shell in ('bash','zsh'):
    subprocess.run([shell,'-n',str(E/'bigearth_smoke_command.txt')],check=True)
assert zlib.decompress(base64.b64decode(encoded)).decode()==script
(E/'smoke_command_validation.json').write_text(json.dumps(dict(status='delivery_static_validation_pass',
    checks=['Python AST for combined/prepare/train scripts','actual ed480e6 CLI parse','Bash/zsh syntax',
        'compressed payload byte equality/SHA','GPU/software/precision equality gate before creating smoke output',
        'independent output; smoke stage/96+64 subset/seed43/epoch1','probe/finite loss/state/metadata/reload checks before exit0'],
    script_sha256=script_sha,helper_sha256=sha,argv=argv,
    limits=['No local CUDA/model run this turn','Original96/64 wrapper was CPU and server2080Ti validated previously; current4090 GPU result pending',
        'No formal training launched','Sampling remains Random42 engineering subset; formal data unchanged']),indent=2)+'\n')
print(json.dumps(dict(command=str(E/'bigearth_smoke_command.txt'),command_characters=len(cmd),argv=argv,script_sha256=script_sha),indent=2))
