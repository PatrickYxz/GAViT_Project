import ast, contextlib, hashlib, io, json, os
from pathlib import Path
import subprocess, tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
E=ROOT/'results/evidence/nwpu_ffn_release_20261003'
launch=(E/'launch_payload.py').read_text()
readback=(E/'readback_payload.py').read_text()
checked=[]
with tempfile.TemporaryDirectory() as tmp:
    d=Path(tmp); repo=d/'repo'; repo.mkdir(); data=d/'data'; data.mkdir(); weights=d/'pre.pth'; weights.write_bytes(b'fixture_weights')
    records=[]
    for split, n in [('train',5040),('val',1260),('test',25200)]:
        records.extend({'path':f'{split}/{i}.jpg','label':i%45,'split':split} for i in range(n))
    manifest=d/'manifest.json'; manifest.write_text(json.dumps({'dataset':'NWPU-RESISC45','class_names':list(map(str,range(45))),'records':records,'counts':{'train':5040,'val':1260,'test':25200}}))
    swaps={
      '/home/featurize/work/GAViT_FFN_20261003':str(repo),
      '/home/featurize/work/nwpu_protocol_20260927_02.json':str(manifest),
      '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors':str(weights),
      '/home/featurize/data/NWPU-RESISC45':str(data),
      'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e':hashlib.sha256(manifest.read_bytes()).hexdigest(),
      'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36':hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    def compiled(text,out):
        mapped={**swaps,'/home/featurize/work/nwpu_ffn42_20261003_01':str(out)}
        class Substitute(ast.NodeTransformer):
            def visit_Constant(self,node):
                if isinstance(node.value,str) and node.value in mapped:
                    return ast.copy_location(ast.Constant(mapped[node.value]),node)
                return node
        return compile(ast.fix_missing_locations(Substitute().visit(ast.parse(text))),'delivery_fixture','exec')
    actual_is_file=Path.is_file
    def run(name, *, missing=False,busy=False,dirty=False):
        out=d/name; captures=[]
        def output(args,**kwargs):
            if args[0]=='git': return 'M changed.py\n' if args[3]=='status' and dirty else '' if args[3]=='status' else '058f1d8b79d0be538ec358eb742d2cd3992921d4\n'
            if '--query-compute-apps=pid,process_name,used_memory' in args: return '123,python,3000\n' if busy else ''
            return '0, NVIDIA GeForce RTX 2080 Ti, GPU-fixture, 610.57.04, 22528, 519, 0\n'
        def spawn(args,**kwargs): captures.append((args,kwargs));return SimpleNamespace(pid=12345)
        def is_file(p): return not missing if p.is_relative_to(data) else actual_is_file(p)
        with patch('subprocess.check_output',output),patch('subprocess.Popen',spawn),patch.object(Path,'is_file',is_file),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),contextlib.redirect_stdout(io.StringIO()):
            exec(compiled(launch,out),{})
        return out,captures
    out,caps=run('success')
    assert len(caps)==1 and len(json.loads((out/'launch.json').read_text())['commands'])==2
    command,options=caps[0]
    assert command[:2]==['bash','-c'] and ' && ' in command[2]
    assert options['start_new_session'] and options['stdin']==subprocess.DEVNULL
    assert options['env']['OMP_NUM_THREADS']==options['env']['MKL_NUM_THREADS']=='8'
    for argv in json.loads((out/'launch.json').read_text())['commands']:
        assert argv[argv.index('--phase')+1]=='smoke' and argv[argv.index('--token-refinement')+1]=='residual_ffn_v1'
    checked.append('sequential_smoke_arguments_and_exclusive_claim_with_519MiB_idle_gpu')
    before=(out/'launch.json').read_bytes()
    try:run('success')
    except AssertionError:pass
    else:raise AssertionError('duplicate launch accepted')
    assert (out/'launch.json').read_bytes()==before
    checked.append('duplicate_preserves_previous_claim')
    for name,kwargs in [('missing',{'missing':True}),('busy',{'busy':True}),('dirty',{'dirty':True})]:
        try:run(name,**kwargs)
        except AssertionError:pass
        else:raise AssertionError(name+' accepted')
        assert not (d/name).exists()
        checked.append(name+'_rejected_before_output_or_spawn')
    for rc in (0,7):
        marker=d/f'second_{rc}'
        import shlex,sys
        first=shlex.join([sys.executable,'-c',f'raise SystemExit({rc})'])
        second=shlex.join([sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).write_text("second")'])
        got=subprocess.run(['bash','-c',first+' && '+second],check=False)
        assert got.returncode==rc and marker.exists()==(rc==0)
    checked.append('real_bash_stops_second_stage_when_first_fails')
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        try:exec(compiled(readback,d/'unfinished'),{})
        except SystemExit as exc:assert exc.code==2
        else:raise AssertionError('unfinished accepted')
    assert 'NWPU_FFN_SMOKE_CHECKS_PASS' not in captured.getvalue()
    checked.append('readback_does_not_claim_completion_for_missing_outputs')
print(json.dumps({'status':'pass','checks':checked,'scope':'Local command/control-flow fixtures; no server, CUDA, or real training'},indent=2))
