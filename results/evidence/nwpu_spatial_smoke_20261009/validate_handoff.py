import copy, hashlib, json, os, shlex, subprocess, sys, tempfile
from pathlib import Path
from unittest.mock import patch
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
release=Path('/Users/patrick/.codex/worktrees/token-ffn-nwpu-release/GAViT_Project')
e=root/'results/evidence/nwpu_spatial_smoke_20261009'
source=(e/'server_launch.py').read_text()
command=(root/'docs/launch_nwpu_spatial_calibrate_20261009.txt').read_text().strip()
assert shlex.split(command)[:2]==['python','-c']
assert compile(shlex.split(command)[2],'<delivered>','exec')
# Comparing AST ignores newline-vs-semicolon formatting only.
import ast
assert ast.dump(ast.parse(shlex.split(command)[2]))==ast.dump(ast.parse(source))
for shell in ('/bin/bash','/bin/zsh'):
    subprocess.run([shell,'-n','-c',command],check=True)
sys.dont_write_bytecode=True
sys.path.insert(0,str(release))
import torch
from PIL import Image
from grouping_identity import soft_assignment_identity
from models.soft_region_grouping import SoftPrototypeGrouping
from scene_classification.runner import SceneDataset
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
results=[]
with tempfile.TemporaryDirectory(prefix='spatial-handoff-',dir='/private/tmp') as tmp:
    base=Path(tmp); smoke=base/'smoke'; data=base/'data'; q=smoke/'assignment_probe'
    q.mkdir(parents=True); data.mkdir(); out=base/'calibrate'; log=Path(str(out)+'.launch.log')
    manifest=base/'manifest.json'; pretrained=base/'weights'; pretrained.write_bytes(b'fixture identity only')
    ids=[f'image_{i}.png' for i in range(8)]
    for i,name in enumerate(ids):Image.new('RGB',(12,13),(20*i,9*i,7*i)).save(data/name)
    records=[{'path':name,'label':i} for i,name in enumerate(ids)]
    manifest.write_text(json.dumps({'records':records}))
    ds=SceneDataset(data,records,False); x=torch.stack([ds[i][0] for i in range(8)])
    ident={'split':'val','sample_ids':ids,'soft_assignment':soft_assignment_identity('anchor_gaussian_v1'),
           'input_shape':list(x.shape),'input_dtype':str(x.dtype),'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()}
    (q/'probe_manifest.json').write_text(json.dumps(ident))
    mask=SoftPrototypeGrouping(soft_assignment=ident['soft_assignment']).candidate_mask
    s=mask.float()/mask.sum(-1,keepdim=True); w=mask.float()/mask.sum(0,keepdim=True)
    tensors={'S':s.expand(8,-1,-1).clone(),'W':w.expand(8,-1,-1).clone()}
    def snapshots():
        for epoch in (0,1):torch.save({'epoch':epoch,'identity':ident,'tensors':tensors},q/f'epoch_{epoch:03d}.pt')
        for name in ('best.pt','final.pt'):(q/name).write_bytes((q/'epoch_001.pt').read_bytes())
    snapshots()
    (q/'assignment_diagnostics.jsonl').write_text('\n'.join(json.dumps({'epoch':i,'split':'val','sample_ids':ids,'mean_abs_delta_S':0.0}) for i in (0,1)))
    (smoke/'selected_samples.json').write_text(json.dumps({'val':ids}))
    (smoke/'run.exit').write_text('exit_code=0\n')
    runtime=['run_scene.py','utils.py','experiment_identity.py','grouping_identity.py','soft_assignment_diagnostics.py']+[str(p.relative_to(release)) for directory in ('models','scene_classification') for p in sorted((release/directory).glob('*.py'))]
    r={'status':'complete','phase':'smoke','seed':42,'counts':{'train':360,'val':90},'final_epoch':1,
       'code':{'files_sha256':{n:sha(release/n) for n in runtime}},'manifest_sha256':sha(manifest),'pretrained_sha256':sha(pretrained),
       'architecture':{'soft_assignment':ident['soft_assignment']}}
    (smoke/'run.json').write_text(json.dumps(r))
    replacements={
    '/home/featurize/work/GAViT_SPATIAL_20261008':str(release),
    '/home/featurize/work/nwpu_spatial42_smoke_20261008_01':str(smoke),
    '/home/featurize/work/nwpu_spatial42_calibrate_20261009_01':str(out),
    '/home/featurize/data/NWPU-RESISC45':str(data),
    '/home/featurize/work/nwpu_protocol_20260927_02.json':str(manifest),
    '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors':str(pretrained),
    'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e':sha(manifest),
    'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36':sha(pretrained)}
    test_source=source
    for before,after in replacements.items():test_source=test_source.replace(before,after)
    def run_case(name,modify=None,error=None,dirty=False,busy=False):
        calls=[]
        if log.exists():log.unlink()
        undo=modify() if modify else None
        def fake_check(cmd,**kw):
            if cmd[0]=='git':
                if cmd[3]=='rev-parse':return '67fe84d431685cf2c7788edfae21fa4e9eb040a4\n'
                if cmd[3]=='status':return ' M run_scene.py\n' if dirty else ''
            if cmd[0]=='nvidia-smi':
                if cmd[1].startswith('--query-gpu'):return 'NVIDIA GeForce RTX 2080 Ti, 20000, 0\n'
                return '123\n' if busy else ''
            raise AssertionError(cmd)
        def fake_launch(cmd,**kw):
            assert cmd[cmd.index('--phase')+1]=='calibrate'
            assert cmd[cmd.index('--spatial-prior')+1]=='anchor_gaussian_v1'
            assert '--calibration' not in cmd and '--token-refinement' not in cmd
            assert kw['start_new_session'] is True and kw['env']['OMP_NUM_THREADS']=='8'
            assert kw['env']['MKL_NUM_THREADS']=='8' and kw['env']['PYTHONDONTWRITEBYTECODE']=='1'
            calls.append(cmd)
            return type('FakeChild',(),{'pid':12345})()
        try:
            with patch('subprocess.check_output',fake_check),patch('subprocess.Popen',fake_launch),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                try:
                    exec(compile(test_source,'<fixture>','exec'),{})
                except (AssertionError,FileNotFoundError) as exc:
                    if not error or error not in str(exc):raise
                    assert not calls and not log.exists(), (name,calls)
                else:
                    assert error is None and len(calls)==1
                    assert 'SMOKE_CACHE_PASS' in log.read_text()
            results.append({'case':name,'passed':True})
        finally:
            if undo:undo()
    run_case('valid real-schema CPU snapshots; launches one calibration')
    run_case('dirty source refuses before launch',error='Wrong or modified source',dirty=True)
    run_case('GPU occupied refuses before launch',error='GPU busy or changed',busy=True)
    def corrupt_input():
        path=data/ids[0]; b=path.read_bytes(); Image.new('RGB',(12,13),(255,0,0)).save(path)
        return lambda:path.write_bytes(b)
    run_case('eight-image byte identity rejects changed pixels',corrupt_input,error='Probe pixels mismatch')
    def corrupt_norm():
        path=q/'epoch_000.pt'; b=path.read_bytes(); v=torch.load(path,weights_only=True);v['tensors']['S']*=.5;torch.save(v,path)
        return lambda:path.write_bytes(b)
    run_case('normalization rejects invalid tensor',corrupt_norm,error='Snapshot normalization')
    def corrupt_alias():
        path=q/'final.pt'; b=path.read_bytes(); v=torch.load(path,weights_only=True);v['epoch']=2;torch.save(v,path)
        return lambda:path.write_bytes(b)
    run_case('alias epoch rejects mismatch',corrupt_alias,error='Snapshot identity mismatch')
    def existing_output():out.mkdir();return lambda:out.rmdir()
    run_case('existing output refused',existing_output,error='Output/log exists')
    def missing_image():
        path=data/ids[0]; b=path.read_bytes();path.unlink();return lambda:path.write_bytes(b)
    run_case('missing data refuses before launch',missing_image,error='NWPU images missing')
validation={'date':'2026-10-09','scope':'Local synthetic cached tensors/images with published CPU classes; subprocess GPU/Git/launch mocked, no model forward/training/server access',
            'syntax':['Python compile and AST equality','bash -n','zsh -n'], 'cases':results,
            'command_sha256':sha(root/'docs/launch_nwpu_spatial_calibrate_20261009.txt'),'actual_server_calibration_started':False}
(e/'handoff_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
