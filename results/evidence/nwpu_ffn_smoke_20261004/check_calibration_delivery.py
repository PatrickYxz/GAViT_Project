"""Offline delivery checks: fake GPU/Git/launch; no training or server access."""
import ast, contextlib, hashlib, io, json, os, shlex, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

E = Path(__file__).resolve().parent
source = (E / 'calibration_launch_payload.py').read_text()
reported = json.loads((E / 'server_return.json').read_text())['runs']
release = '058f1d8b79d0be538ec358eb742d2cd3992921d4'
checks = []
with tempfile.TemporaryDirectory() as temp:
    base = Path(temp)
    def attempt(mode):
        d = base / mode; d.mkdir(); out = d / 'out'; out.mkdir(); repo = d / 'repo'; repo.mkdir()
        data = d / 'data'; data.mkdir(); (data / 'one.jpg').write_bytes(b'fixture')
        manifest = d / 'manifest.json'; manifest.write_text(json.dumps({'records':[{'path':'one.jpg'}]}))
        weights = d / 'pretrained'; weights.write_bytes(b'fixture_weights')
        for n,r in reported.items():
            q = out/n; q.mkdir(); (q/'best.pth').write_bytes(n.encode()); (q/'run.exit').write_text(r['exit'])
            (q/'run.json').write_text(json.dumps({**r, 'code':{'git':r['git']}, 'checkpoint_sha256':hashlib.sha256(n.encode()).hexdigest()}))
        if mode == 'missing_image': (data/'one.jpg').unlink()
        if mode == 'occupied': (out/'soft_ffn').mkdir()
        if mode == 'existing_claim': (out/'calibration_launch.json').write_text('preserve')
        if mode == 'bad_checkpoint': (out/'soft_ffn_smoke'/'best.pth').write_bytes(b'changed')
        if mode == 'failed_smoke': (out/'soft_ffn_smoke'/'run.exit').write_text('exit_code=1')
        swaps = {'/home/featurize/work/nwpu_ffn42_20261003_01':str(out), '/home/featurize/work/GAViT_FFN_20261003':str(repo), '/home/featurize/data/NWPU-RESISC45':str(data), '/home/featurize/work/nwpu_protocol_20260927_02.json':str(manifest), '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors':str(weights), 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e':hashlib.sha256(manifest.read_bytes()).hexdigest(), 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36':hashlib.sha256(weights.read_bytes()).hexdigest()}
        class Remap(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.copy_location(ast.Constant(swaps[node.value]),node) if isinstance(node.value,str) and node.value in swaps else node
        code = compile(ast.fix_missing_locations(Remap().visit(ast.parse(source))), '<fixture>', 'exec')
        def output(args, **kwargs):
            if args[0]=='git':
                if 'rev-parse' in args: return 'wrong' if mode=='bad_release' else release
                return 'M changed.py' if mode=='dirty' else ''
            if '--query-compute-apps=pid' in args: return '999' if mode=='busy' else ''
            return '0, NVIDIA GeForce RTX 2080 Ti, GPU-fixture, 22000, 0'
        spawned=[]
        def spawn(args,**kwargs): spawned.append((args,kwargs)); return SimpleNamespace(pid=98765)
        before={str(p.relative_to(out)):p.read_bytes() for p in out.rglob('*') if p.is_file()}
        with patch('subprocess.check_output',output),patch('subprocess.Popen',spawn),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}),contextlib.redirect_stdout(io.StringIO()):
            try: exec(code,{})
            except AssertionError:
                assert mode!='success'
            else: assert mode=='success'
        if mode=='success':
            assert len(spawned)==1
            meta=json.loads((out/'calibration_launch.json').read_text()); cmds=meta['commands']
            assert meta['pid']==98765 and len(cmds)==2
            assert [c[c.index('--model')+1] for c in cmds]==['swin','gavit']
            assert [c[c.index('--output')+1] for c in cmds]==[str(out/'swin_ffn'),str(out/'soft_ffn')]
            assert cmds[1][cmds[1].index('--grouping')+1]=='soft_prototype'
            for c in cmds:
                assert c[c.index('--phase')+1]=='calibrate' and c[c.index('--seed')+1]=='42'
                assert '--epochs' not in c and '--random-init' not in c and '--calibration' not in c
            args,kw=spawned[0]; assert args[:2]==['bash','-c'] and ' && ' in args[2]
            assert kw['start_new_session'] and kw['env']['OMP_NUM_THREADS']==kw['env']['MKL_NUM_THREADS']=='8'
            assert kw['env']['CUDA_VISIBLE_DEVICES']=='0' and kw['cwd']==repo
        else:
            assert not spawned and not (out/'calibration.log').exists()
        assert all((out/name).read_bytes()==b for name,b in before.items())
        checks.append(mode)
    for case in ('success','missing_image','occupied','existing_claim','bad_checkpoint','failed_smoke','bad_release','dirty','busy'): attempt(case)
    for rc in (0,7):
        marker=base/f'after_{rc}'
        a=shlex.join([sys.executable,'-c',f'raise SystemExit({rc})'])
        b=shlex.join([sys.executable,'-c',f'from pathlib import Path; Path({str(marker)!r}).write_text("ok")'])
        result=subprocess.run(['bash','-c',a+' && '+b])
        assert result.returncode==rc and marker.exists()==(rc==0)
    checks.append('real_shell_first_failure_stops_second_stage')
command="python -c '"+'; '.join(source.splitlines())+"'"
assert shlex.split(command)[2]=='; '.join(source.splitlines())
compile(shlex.split(command)[2],'<single-line>','exec')
for shell in ('bash','zsh'): subprocess.run([shell,'-n','-c',command],check=True)
checks.append('bash_zsh_python_single_line_syntax')
print(json.dumps({'status':'pass','checks':checks,'scope':'Offline fixtures only; no server/GPU/training; each existing smoke file preserved'},indent=2))
