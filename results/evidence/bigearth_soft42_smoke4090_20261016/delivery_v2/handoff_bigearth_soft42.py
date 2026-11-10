"""Three explicit steps for the authorized soft42 supplement; frozen source only."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

FORMAL = Path('/home/featurize/work/bigearth_soft42_20261013_01')
SMOKE = Path('/home/featurize/work/gpucheck_bigearth_soft42_4090_20261013_01')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(root, name):
    spec = importlib.util.spec_from_file_location(name, root / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def package_identity(root):
    manifest = read(root / 'manifest.json')
    require(manifest['seed'] == 42 and manifest['status'] == 'prepared_local_only', 'Wrong handoff identity')
    for name, sha in manifest['files'].items():
        require(digest(root/name) == sha, 'Handoff file changed: ' + name)


def check_smoke(root, ready, helper):
    require((root/'run.exit').read_text().strip() == 'exit_code=0', 'Smoke did not exit0')
    r = read(root/'summary.json')
    require(r['status'] == 'complete' and r['seed'] == 42 and r['commit'] == helper.COMMIT, 'Wrong smoke identity')
    require(r['counts'] == dict(train=96,val=64) and r['epochs'] == 1 and r['parameters'] == 31466766, 'Wrong smoke scope')
    require(r['hardware'] == ready['hardware'] and r['precision'] == ready['precision'], 'Smoke GPU/software/precision differs from preflight')
    require(math.isfinite(r['reload_max_logit_diff']) and 0 <= r['reload_max_logit_diff'] <= 1e-5, 'Smoke reload failed')
    require(all(math.isfinite(r[k]) for k in ('train_loss','val_mAP_percent','val_macro_f1_percent')), 'Nonfinite smoke metrics')
    require(r['graph'] == dict(edges_per_image=80,spatial_edges=48,feature_edges=32,actual_forward_checked=True), 'Wrong actual forward graph')
    require(r['graph_probe']['actual_forward_checked'] and
            max(r['graph_probe']['normalization_errors'].values()) < 1e-5, 'Assignment probe failed')
    ckpt = root/'gavit.pth'
    meta = read(ckpt.with_suffix('.meta.json'))
    require(meta == r['metadata'] and digest(ckpt) == meta['checkpoint_sha256'] == r['checkpoint_sha256'], 'Smoke checkpoint bytes/metadata/summary mismatch')
    require(meta['architecture'] == helper.ARCH and meta['training'] == {**helper.TRAINING,'epochs':1}, 'Wrong smoke architecture/recipe')
    require(meta['execution']['run_stage'] == 'smoke' and meta['git'] == dict(commit='ed480e6',dirty=False), 'Wrong smoke code/stage')
    return r


def check_preflight_continuity(ready, fresh):
    require(ready['status'] == fresh['status'] == 'BIGEARTH_SOFT42_PREFLIGHT_PASS' and ready['seed'] == fresh['seed'] == 42,
            'Preflight did not pass for soft42')
    # A successful engineering run legitimately adds artifacts to the inventory.
    old_identity = {k:v for k,v in ready.items() if k != 'inventory'}
    new_identity = {k:v for k,v in fresh.items() if k != 'inventory'}
    require(old_identity == new_identity, 'Data/control/GPU/software changed since smoke: preserve artifacts and inspect')


def preflight(root, destination):
    require(not destination.exists(), 'Preflight snapshot already exists; inspect it before repeating')
    gate = load(root, 'preflight_bigearth_soft42')
    r = gate.preflight()
    print(json.dumps(r, indent=2, allow_nan=False), flush=True)
    if r['status'] != 'BIGEARTH_SOFT42_PREFLIGHT_PASS':
        return 2
    with destination.open('x') as f:
        f.write(json.dumps(r, indent=2, allow_nan=False) + '\n')
    return 0


def smoke(root):
    require(not SMOKE.exists(), 'Smoke output exists: preserve/inspect it, do not overwrite or start a retry')
    ready = read(root/'ready.json')
    require(ready['status'] == 'BIGEARTH_SOFT42_PREFLIGHT_PASS', 'Initial preflight not passed')
    helper = load(root, 'run_bigearth_soft42')
    b = helper.load_base()
    b.identity()
    b.idle_gpu()
    import torch
    hardware, precision = helper.environment(torch, b)
    require(hardware == ready['hardware'] and precision == ready['precision'], 'GPU/software/precision changed since preflight')
    for c in ready['controls'].values():
        require(digest(c['checkpoint']) == c['checkpoint_sha256'] and digest(c['test_log']) == c['test_log_sha256'], 'Existing42 control changed')
    ns = dict(__name__='__main__',helper_ns=vars(helper),b=b,ready=ready)
    prepare = (root/'prepare_smoke_source.py').read_text()
    train = (root/'train_reload_smoke_source.py').read_text()
    exec(compile(prepare,str(root/'prepare_smoke_source.py'),'exec'), ns)
    (SMOKE/'approved_preflight.json').write_text(json.dumps(ready,indent=2)+'\n')
    (SMOKE/'helper_source.py').write_bytes((root/'run_bigearth_soft42.py').read_bytes())
    (SMOKE/'prepare_source.py').write_text(prepare)
    (SMOKE/'train_reload_source.py').write_text(train)
    exec(compile(train,str(root/'train_reload_smoke_source.py'),'exec'), ns)
    # The frozen worker writes run.exit in atexit. Return so it can finish;
    # launch() checks that marker and the saved artifacts in a new process.


def launch(root):
    # This parent remains stdlib-only. CUDA is initialized only by a finished child.
    require(not FORMAL.exists(), 'Formal output exists: inspect/reuse; do not relaunch')
    ready = read(root/'ready.json')
    helper = load(root,'run_bigearth_soft42')
    check_smoke(SMOKE,ready,helper)
    current = root/'preflight_pre_launch.json'
    subprocess.run([sys.executable,'-B','-u',str(Path(__file__).resolve()),'gate_launch'],check=True)
    fresh = read(current)
    check_preflight_continuity(ready, fresh)
    with (root/'launch.claim').open('x') as claim:
        claim.write(str(os.getpid())+'\n')
    FORMAL.mkdir(exist_ok=False)
    (FORMAL/'preflight.py').write_bytes((root/'preflight_bigearth_soft42.py').read_bytes())
    (FORMAL/'approved_preflight.json').write_text(json.dumps(fresh,indent=2)+'\n')
    script = FORMAL/'run_bigearth_soft42.py'
    script.write_bytes((root/'run_bigearth_soft42.py').read_bytes())
    (FORMAL/'handoff_manifest.json').write_bytes((root/'manifest.json').read_bytes())
    with (FORMAL/'run.log').open('xb') as log:
        p = subprocess.Popen([sys.executable,'-B','-u',str(script),str(FORMAL)],stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    (FORMAL/'run.pid').write_text(str(p.pid)+'\n')
    print(json.dumps(dict(status='BIGEARTH_SOFT42_LAUNCH_DISPATCHED',pid=p.pid,output=str(FORMAL),
          training_start_verified=False,next_action='Check GPU and gavit_train.process.log for first training batch'),indent=2),flush=True)


def status():
    result = dict(output=str(FORMAL),queue_exit=(FORMAL/'run.exit').read_text().strip() if (FORMAL/'run.exit').exists() else 'pending')
    for stage in ('gavit_train','gavit_test'):
        p = FORMAL/(stage+'.process.exit')
        result[stage+'_exit'] = p.read_text().strip() if p.exists() else 'pending'
    if (FORMAL/'summary.json').exists():
        s = read(FORMAL/'summary.json')
        result.update(status=s['status'],seed=s['seed'],queue_seconds=s['elapsed_seconds'])
        for stage, r in s['stages'].items():
            result[stage] = {k:r[k] for k in ('status','seed','commit','hardware','precision','checkpoint_sha256',
                'elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes')}
            if stage == 'gavit_train':
                result[stage].update(best=r['best'],final_epoch=r['final_epoch'],counts=r['counts'],parameters=r['parameters'])
            else:
                result[stage]['metrics'] = r['metrics']
    else:
        log = FORMAL/'run.log'
        result['log_tail'] = log.read_text().splitlines()[-20:] if log.exists() else []
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    require(not sys.flags.optimize, 'Disable PYTHONOPTIMIZE')
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    root = Path(__file__).resolve().parent
    package_identity(root)
    require(len(sys.argv) == 2, 'Use preflight, smoke, launch or status')
    command = sys.argv[1]
    if command in ('preflight','gate_launch'):
        sys.exit(preflight(root,root/('ready.json' if command == 'preflight' else 'preflight_pre_launch.json')))
    elif command == 'smoke':
        smoke(root)
    elif command == 'launch':
        launch(root)
    elif command == 'status':
        status()
    else:
        raise RuntimeError('Unknown handoff step')
