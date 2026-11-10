import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

evidence = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2/results/evidence/nwpu_calibration42_20260928')
c = json.loads((evidence / 'commands.json').read_text())
for k in ('prepare_code', 'queue_code', 'result_code'):
    ast.parse(c[k])
for k in ('restore', 'prepare', 'queue', 'result'):
    for shell in ('bash', 'zsh'):
        r = subprocess.run([shell, '-n'], input=c[k] + '\n', text=True, capture_output=True)
        assert r.returncode == 0, (k, shell, r.stderr)
base = Path(tempfile.mkdtemp(prefix='nwpu-queue-check-', dir='/private/tmp'))
repo = Path('/private/tmp/gavit-soft-release-git-20260927')
manifest = base / 'manifest.json'
pre = base / 'pretrained-fixture'
manifest.write_text('{}')
pre.write_bytes(b'fixture only')
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
info = {'status': 'ready', 'commit': 'ed480e60fea03ed786fdba70c144de5e815c34f5', 'manifest_sha256': h(manifest), 'pretrained_sha256': h(pre)}
patch = '''import subprocess,json,os
from pathlib import Path
from types import SimpleNamespace
real_run=subprocess.run
def fake_run(argv, **kwargs):
    if argv[0]=='git':
        return real_run(argv, **kwargs)
    assert argv[2].endswith('run_scene.py') and argv[3]=='train'
    get=lambda key:argv[argv.index(key)+1]
    assert get('--phase')=='calibrate' and get('--seed')=='42'
    assert get('--device')=='cuda' and get('--batch-size')=='32' and get('--workers')=='2'
    output=Path(get('--output'))
    name=output.name
    assert get('--model')==('swin' if name=='swin' else 'gavit')
    assert (('--grouping' not in argv) if name=='swin' else get('--grouping')==('attentive_spatial' if name=='fixed' else 'soft_prototype'))
    with (output.parent/'calls.jsonl').open('a') as f:
        f.write(json.dumps({'stage':name,'argv':argv})+'\\n')
    if os.environ.get('FAIL_STAGE')==name:
        return SimpleNamespace(returncode=9)
    output.mkdir()
    (output/'run.exit').write_text('exit_code=0\\n')
    info=json.loads((output.parent/'queue_preflight.json').read_text())
    (output/'run.json').write_text(json.dumps({'status':'complete','phase':'calibrate','seed':42,'final_epoch':30,'counts':{'train':5040,'val':1260},'manifest_sha256':info['manifest_sha256'],'pretrained_sha256':info['pretrained_sha256']}))
    return SimpleNamespace(returncode=0)
subprocess.run=fake_run
'''
results=[]
for fail, expected in [('', ['swin','fixed','soft']), ('fixed',['swin','fixed']), ('swin',['swin'])]:
    out = base / ('success' if not fail else 'fail_' + fail)
    out.mkdir()
    (out/'queue_preflight.json').write_text(json.dumps(info))
    code=c['queue_code'].replace('/home/featurize/work/GAViT_SOFT_20260927',str(repo)).replace('/home/featurize/work/nwpu_ablation42_20260928_01',str(out)).replace('/home/featurize/work/nwpu_protocol_20260927_02.json',str(manifest)).replace('/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors',str(pre))
    r=subprocess.run([sys.executable,'-c',patch+'\n'+code],env={**os.environ,'FAIL_STAGE':fail},text=True,capture_output=True)
    assert (out/'calls.jsonl').is_file(),r.stderr
    calls=[json.loads(x)['stage'] for x in (out/'calls.jsonl').read_text().splitlines()]
    assert calls==expected,(calls,expected,r.stderr)
    assert r.returncode==(1 if fail else 0),r.stderr
    assert (out/'queue.exit').read_text()==f'exit_code={r.returncode}\n'
    report=json.loads((out/'queue_summary.json').read_text())
    assert report['status']==('failed' if fail else 'complete')
    results.append({'failure_stage':fail or None,'stages_called':calls,'exit':r.returncode})
    (evidence/('local_'+('success' if not fail else 'fail_'+fail)+'.log')).write_text(r.stdout+r.stderr)
result={'scope':'Local syntax and queue control-flow only, subprocess training replaced by explicit fake outputs; no GPU/30-epoch training executed','fixture':str(base),'results':results}
(evidence/'local_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('QUEUE_CONTROL_FLOW_PASS: success order, first-stage failure, middle-stage failure, exits; no GPU training')
