"""External NWPU seed43 replay of the frozen seed42 three-model protocol."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path('/home/featurize/work/GAViT_SOFT_20260927')
COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
MANIFEST = Path('/home/featurize/work/nwpu_protocol_20260927_02.json')
MANIFEST_SHA = 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
PRE = Path('/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors')
PRE_SHA = 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
DATA = Path('/home/featurize/data/NWPU-RESISC45')
NAMES = ('swin', 'fixed', 'soft')
GPU_NAME = 'NVIDIA GeForce RTX 4090'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def output(argv):
    return subprocess.check_output(argv, text=True).strip()


def identity():
    require(output(['git', '-C', str(REPO), 'rev-parse', 'HEAD']) == COMMIT, 'Wrong frozen commit')
    require(not output(['git', '-C', str(REPO), 'status', '--porcelain', '--untracked-files=all']), 'Dirty source; preserve it and stop')
    require(digest(MANIFEST) == MANIFEST_SHA and digest(PRE) == PRE_SHA, 'Input hash changed')


def idle_gpu():
    require(not output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits']), 'GPU busy')


def existing_runs(work):
    prior = []
    for p in sorted(set(work.glob('nwpu*/run.json')) | set(work.glob('nwpu*/*/run.json'))):
        r = read(p)
        if r.get('seed') == 43 and r.get('manifest_sha256') == MANIFEST_SHA:
            prior.append({'path': str(p), 'phase': r.get('phase'), 'status': r.get('status')})
    for p in sorted(set(work.glob('nwpu*/*.meta.json')) | set(work.glob('nwpu*/*/*.meta.json'))):
        r = read(p)
        if (r.get('training', {}).get('seed') == 43 and r.get('data', {}).get('manifest_sha256') == MANIFEST_SHA
                and r.get('execution', {}).get('phase') in ('calibrate', 'refit')):
            prior.append({'path': str(p), 'phase': r['execution']['phase'], 'status': 'checkpoint_present'})
    return prior


def preflight():
    identity()
    m = read(MANIFEST)
    require(m['dataset'] == 'NWPU-RESISC45' and m['counts'] == {'train': 5040, 'val': 1260, 'test': 25200}
            and len(m['records']) == 31500, 'Wrong manifest counts')
    missing = [r['path'] for r in m['records'] if not (DATA / r['path']).is_file()]
    require(not missing, 'NWPU DATA_MISSING: ' + str(missing[:5]))
    prior = existing_runs(Path('/home/featurize/work'))
    require(not prior, 'Existing seed43 artifacts: inspect for reuse before launch: ' + json.dumps(prior))
    idle_gpu()
    gpu = output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader'])
    rows = list(csv.reader(gpu.splitlines(), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == 4, 'One RTX 4090 GPU identity required: ' + gpu)
    uuid, name, driver, memory = [field.strip() for field in rows[0]]
    require(uuid.startswith('GPU-') and name == GPU_NAME and bool(driver), 'Expected RTX 4090 GPU: ' + gpu)
    memory_parts = memory.split()
    require(len(memory_parts) == 2 and memory_parts[0].isdigit() and memory_parts[1] == 'MiB'
            and int(memory_parts[0]) > 0, 'Invalid GPU memory identity: ' + gpu)
    return dict(status='NWPU43_PREFLIGHT_PASS', commit=COMMIT, manifest_sha256=MANIFEST_SHA,
                pretrained_sha256=PRE_SHA, seed=43, counts=m['counts'], gpu=gpu,
                gpu_profile=dict(uuid=uuid, name=name, driver=driver, memory_mib=int(memory_parts[0])),
                stages=[s for s, _ in jobs(Path('/planned_output'))], training_started=False)


def jobs(root):
    common = ['--manifest', str(MANIFEST), '--data-root', str(DATA), '--device', 'cuda', '--batch-size', '32', '--workers', '2']
    entry = [sys.executable, '-B', '-u', str(REPO / 'run_scene.py')]
    result = []
    for phase in ('calibrate', 'refit', 'test'):
        for name in NAMES:
            stage = name + '_' + phase
            if phase == 'test':
                argv = entry + ['evaluate', '--checkpoint', str(root / (name + '_refit') / 'last.pth'), '--split', 'test']
            else:
                argv = entry + ['train', '--phase', phase, '--seed', '43', '--model', 'swin' if name == 'swin' else 'gavit', '--pretrained-path', str(PRE)]
                if name != 'swin':
                    argv += ['--grouping', 'attentive_spatial' if name == 'fixed' else 'soft_prototype']
                if phase == 'refit':
                    argv += ['--calibration', str(root / (name + '_calibrate'))]
            result.append((stage, argv + ['--output', str(root / stage)] + common))
    return result


def check_stage(root, stage):
    name, phase = stage.split('_')
    dest = root / stage
    require((dest / 'run.exit').read_text().strip() == 'exit_code=0', 'Stage exit missing/nonzero')
    r = read(dest / 'run.json')
    require(r['status'] == 'complete', 'Stage incomplete')
    cal_path = root / (name + '_calibrate')
    cal = read(cal_path / 'run.json')
    if phase == 'test':
        refit = read(root / (name + '_refit') / 'run.json')
        m = read(dest / 'metrics.json')
        require(r['samples'] == m['samples'] == 25200 and r['split'] == m['split'] == 'test', 'Incomplete test')
        require(m['training_phase'] == 'refit' and m['fit_scope'] == 'train_plus_val', 'Wrong fit scope')
        require(m['manifest_sha256'] == MANIFEST_SHA and m['checkpoint_sha256'] == r['checkpoint_sha256'] == refit['checkpoint_sha256'], 'Wrong test weight')
        require(all(r['evaluation'][k] == cal[k] for k in ('code', 'hardware')), 'Test environment changed')
        require(all(math.isfinite(m[k]) and 0 <= m[k] <= 100 for k in ('oa_percent', 'macro_f1_percent')), 'Invalid metrics')
        require(len(m['confusion_matrix']) == 45 and sum(sum(row) for row in m['confusion_matrix']) == 25200, 'Incomplete confusion matrix')
        for file, count in (('predictions.csv', 25200), ('per_class.csv', 45)):
            with (dest / file).open() as f:
                require(sum(1 for _ in csv.reader(f)) - 1 == count, 'Incomplete ' + file)
        return m
    require(r['phase'] == phase and r['seed'] == 43 and r['model'] == ('swin' if name == 'swin' else 'gavit'), 'Wrong phase/seed/model')
    require(r['manifest_sha256'] == MANIFEST_SHA and r['pretrained_sha256'] == PRE_SHA, 'Changed inputs')
    require(r['reload_max_logit_diff'] <= 1e-5, 'Checkpoint reload mismatch')
    checkpoint = dest / ('best.pth' if phase == 'calibrate' else 'last.pth')
    require(digest(checkpoint) == r['checkpoint_sha256'], 'Checkpoint bytes changed')
    reference = read(root / 'swin_calibrate' / 'run.json')
    require(all(r[k] == reference[k] for k in ('training', 'code', 'hardware')), 'Shared recipe/environment changed')
    if name != 'swin':
        require(r['architecture']['grouping'] == ('attentive_spatial' if name == 'fixed' else 'soft_prototype'), 'Wrong grouping')
    if phase == 'calibrate':
        require(r['final_epoch'] == 30 and 1 <= r['best_epoch'] <= 30 and r['counts'] == {'train': 5040, 'val': 1260}, 'Incomplete calibration')
    else:
        require(r['counts'] == {'train': 6300, 'val': 0} and r['final_epoch'] == cal['best_epoch'] and r['best_epoch'] is None, 'Wrong refit epochs/counts')
        require(r['refit_source'] == {'run_json_sha256': digest(cal_path / 'run.json'), 'best_checkpoint_sha256': cal['checkpoint_sha256']}, 'Wrong calibration source')
        require(r['architecture'] == cal['architecture'] and r['graph_capture'] == cal['graph_capture'], 'Architecture changed')
    history = read(dest / 'history.json')
    require([row['epoch'] for row in history] == list(range(1, r['final_epoch'] + 1)), 'Incomplete training history')
    require(all(math.isfinite(row['train_loss']) for row in history), 'Nonfinite loss history')
    require(all((dest / file).is_file() for file in ('last.pth', 'last.train_state.pth')), 'Missing final checkpoint/state')
    if phase == 'calibrate':
        require(all(math.isfinite(row['validation']['oa_percent']) for row in history), 'Nonfinite validation history')
        require(max(history, key=lambda row: row['validation']['oa_percent'])['epoch'] == r['best_epoch'], 'Wrong validation selection/tie rule')
    return r


def run_jobs(root, scheduled, before, validate):
    results = {}
    for stage, argv in scheduled:
        before()
        with (root / (stage + '.process.log')).open('xb') as log:
            print('START', stage, flush=True)
            started = time.monotonic()
            rc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'PYTHONDONTWRITEBYTECODE': '1'}).returncode
        (root / (stage + '.process.exit')).write_text('exit_code=' + str(rc) + '\n')
        (root / (stage + '.process.json')).write_text(json.dumps(dict(argv=argv, exit_code=rc,
            elapsed_seconds=time.monotonic() - started, timing_scope='whole subprocess including entry preflight/IO'), indent=2) + '\n')
        if rc:
            raise subprocess.CalledProcessError(rc, argv)
        results[stage] = validate(stage)
        print('STAGE_COMPLETE', stage, 'wall_seconds', time.monotonic() - started, flush=True)
    return results


def main(root):
    # Parent stays stdlib-only; the released entry owns each fresh CUDA process.
    with (root / 'started.claim').open('x') as f:
        f.write(str(os.getpid()) + '\n')
    rc = 1
    started = time.monotonic()
    try:
        ready = preflight()
        (root / 'preflight.json').write_text(json.dumps(ready, indent=2) + '\n')
        schedule = jobs(root)
        (root / 'commands.json').write_text(json.dumps(schedule, indent=2) + '\n')
        def before():
            identity()
            idle_gpu()
            require(output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']) == ready['gpu'], 'Physical GPU changed')
        reports = run_jobs(root, schedule, before, lambda stage: check_stage(root, stage))
        summary = dict(status='complete', seed=43, elapsed_seconds=time.monotonic() - started,
                       selected_epochs={n: reports[n + '_calibrate']['best_epoch'] for n in NAMES},
                       tests={n: reports[n + '_test'] for n in NAMES})
        (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        print('NWPU43_THREE_MODELS_COMPLETE', flush=True)
        rc = 0
    except BaseException:
        traceback.print_exc()
    finally:
        (root / 'run.exit').write_text('exit_code=' + str(rc) + '\n')
    return rc


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    require(not sys.flags.optimize, 'Disable PYTHONOPTIMIZE')
    if sys.argv[1] == 'preflight':
        print(json.dumps(preflight(), indent=2))
    else:
        raise SystemExit(main(Path(sys.argv[1]).resolve()))
