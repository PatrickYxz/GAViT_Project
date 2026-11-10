"""E06: frozen AID seed42 checkpoints, synchronous FP32 model-only inference.

Run this standalone file outside the clean training checkout. No data decoding,
training, checkpoint writes, external packages, or network access are required.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

STAGES = [('swin', 1), ('gavit', 1), ('gavit', 32), ('swin', 32)]
MANIFEST_SHA = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
PRETRAINED_SHA = 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
WEIGHTS = {
    'swin': 'fc04ddd58e233bc7291b6a5e54a2c3e9efc45c0584a709aa28a0eba496178018',
    'gavit': 'a023563e7c4566436660a2d9bc778a9adb3e9b5a64f39f821ce81e909c913c7a',
}
PARAMETERS = {'swin': 27543960, 'gavit': 31375385}
PROTOCOL = {
    'name': 'aid-e06-model-only-v1', 'input_seed': 20260922,
    'shape_without_batch': [3, 224, 224], 'dtype': 'float32',
    'input': 'fixed synthetic RGB uniform[0,1], then ImageNet normalization',
    'input_location_during_timing': 'cuda:0', 'amp': False, 'tf32': False,
    'cudnn_benchmark': False, 'cudnn_deterministic': True, 'torch_threads': 8,
    'warmup_forwards': 20, 'repeats': 5, 'steps_per_repeat': 50,
    'timer': 'perf_counter, each forward followed by CUDA synchronize',
    'scope': 'full model including Python graph construction and device execution',
    'excludes': ['file IO', 'decode', 'resize', 'normalization', 'host-to-device transfer',
                 'model loading', 'warmup', 'argmax', 'metrics', 'instrumentation hooks'],
    'memory_scope': 'steady-state model + input + forward output/temporaries; after warmup',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def measure(forward, synchronize, clock, steps, warmup=0):
    """A returned forward output stays alive through the timed synchronization."""
    require(steps > 0 and warmup >= 0, 'Invalid measurement count')
    for _ in range(warmup):
        output = forward()
        synchronize()
        del output
    samples = []
    for _ in range(steps):
        synchronize()
        start = clock()
        output = forward()
        synchronize()
        samples.append((clock() - start) * 1000)
        del output
    return samples


def aggregate(repeats, batch):
    require(batch > 0 and bool(repeats) and all(repeats), 'Empty timing samples')
    flat = [x for row in repeats for x in row]
    require(all(math.isfinite(x) and x > 0 for x in flat), 'Nonpositive/nonfinite timing')
    means = [statistics.mean(row) for row in repeats]
    mean = statistics.mean(flat)
    return {'batch_ms_mean': mean, 'batch_ms_median': statistics.median(flat),
            'batch_ms_p95': sorted(flat)[math.ceil(.95 * len(flat)) - 1],
            'amortized_ms_per_image': mean / batch,
            'images_per_second': batch * len(flat) * 1000 / sum(flat),
            'repeat_batch_ms_means': means,
            'repeat_mean_sample_sd_ms': statistics.stdev(means) if len(means) > 1 else None,
            'repeat_mean_min_ms': min(means), 'repeat_mean_max_ms': max(means)}


def check_process_rows(rows, own_pid):
    for row in rows.splitlines():
        row = row.strip()
        if row and (not row.isdigit() or int(row) != own_pid):
            raise RuntimeError('GPU has another compute process, or PID query is unavailable: ' + row)


def gpu_snapshot():
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True)
    check_process_rows(processes, os.getpid())
    fields = 'uuid,name,driver_version,memory.total,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw'
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=' + fields,
        '--format=csv,noheader,nounits'], text=True)
    rows = list(csv.reader(raw.strip().splitlines(), skipinitialspace=True))
    require(len(rows) == 1 and len(rows[0]) == len(fields.split(',')), 'Require one physical GPU')
    return dict(zip(fields.split(','), rows[0]))


def preflight(args):
    repo, work = Path(args.repo).resolve(), Path(args.work).resolve()
    require(repo not in Path(__file__).resolve().parents, 'Keep probe outside training checkout')
    sys.path.insert(0, str(repo))
    from scene_classification import runner as r
    from scene_classification.modeling import architecture
    require(Path(r.__file__).resolve() == repo / 'scene_classification/runner.py', 'Wrong imported repository')
    code = r._code_identity()
    require(code['git'] == {'commit': 'deb2b45', 'dirty': False}, 'Require clean deb2b45; do not edit training repo')
    manifest_path = work / 'aid_protocol_20260920_03.json'
    require(sha256(manifest_path) == MANIFEST_SHA, 'Frozen manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    require(manifest['dataset'] == 'AID' and len(manifest['class_names']) == 30, 'Wrong dataset')
    metas = {}
    for model in ('swin', 'gavit'):
        checkpoint = work / f'aid_{model}_refit42_20260920_01/last.pth'
        meta = r._checkpoint_metadata(checkpoint, manifest_path, manifest)
        source = json.loads((checkpoint.parent / 'run.json').read_text())
        r.validate_source_run(source, meta, (checkpoint.parent / 'run.exit').read_text())
        expected_training = r.training_identity(r.RunConfig(model, '', '', ''))
        require(meta['model'] == model and meta['checkpoint_sha256'] == WEIGHTS[model]
                and meta['execution']['phase'] == 'refit' and meta['training'] == expected_training
                and meta['pretrained']['sha256'] == PRETRAINED_SHA
                and meta['data']['fit_scope'] == 'train_plus_val'
                and meta['final_epoch'] == {'swin': 21, 'gavit': 20}[model]
                and source['counts'] == {'train': 5000, 'val': 0}
                and meta['architecture'] == architecture(model, 30)
                and meta['code'] == code, 'Frozen seed42 refit/source/code mismatch: ' + model)
        metas[model] = (checkpoint, meta)
    return r, code, metas


def worker(args):
    before = gpu_snapshot()
    r, code, metas = preflight(args)
    import torch
    from scene_classification.modeling import capture_graph_once
    from utils import set_seed
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Require one CUDA device')
    torch.set_num_threads(PROTOCOL['torch_threads'])
    set_seed(PROTOCOL['input_seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    checkpoint, meta = metas[args.model]
    model = r._restore(checkpoint, meta, device)
    params = sum(p.numel() for p in model.parameters())
    require(params == PARAMETERS[args.model], 'Parameter count mismatch')
    generator = torch.Generator(device='cpu').manual_seed(PROTOCOL['input_seed'])
    x = torch.rand(args.batch, 3, 224, 224, generator=generator)
    x = (x - torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    input_sha = hashlib.sha256(x.numpy().tobytes()).hexdigest()
    x = x.to(device)
    graph, handle = capture_graph_once(model)
    with torch.inference_mode():
        y = model(x)
        require(tuple(y.shape) == (args.batch, 30) and bool(torch.isfinite(y).all()), 'Invalid logits')
        del y
        if handle:
            handle.remove()
        require(args.model == 'swin' or graph.get('actual_forward_checked'), 'Graph capture missing')
        torch.cuda.synchronize()
        # Discard the load/capture allocator cache; prime only real inference.
        torch.cuda.empty_cache()
        for _ in range(PROTOCOL['warmup_forwards']):
            model(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = {'allocated_bytes': torch.cuda.memory_allocated(), 'reserved_bytes': torch.cuda.memory_reserved()}
        repeats, telemetry = [], [before]
        print(f'FORWARD PASS: {args.model}; batch={args.batch}; graph={graph}; GPU={before["name"]}', flush=True)
        for index in range(PROTOCOL['repeats']):
            telemetry.append(gpu_snapshot())
            row = measure(lambda: model(x), torch.cuda.synchronize, time.perf_counter, PROTOCOL['steps_per_repeat'])
            repeats.append(row)
            print(f'{args.model} b{args.batch}: repeat {index + 1}/{PROTOCOL["repeats"]}; mean_ms={statistics.mean(row):.4f}', flush=True)
        allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
        telemetry.append(gpu_snapshot())
        y = model(x)
        require(bool(torch.isfinite(y).all()), 'Nonfinite final logits')
        del y
    require(r._code_identity() == code, 'Code changed during measurement')
    require(all(t['uuid'] == before['uuid'] and t['driver_version'] == before['driver_version'] for t in telemetry), 'GPU changed')
    report = {'status': 'complete', 'model': args.model, 'batch_size': args.batch,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'parameters': params, 'architecture': meta['architecture'], 'graph_capture': graph,
        'common_identity': {'manifest_sha256': MANIFEST_SHA, 'pretrained_sha256': PRETRAINED_SHA, 'code': code},
        'hardware': r._hardware(device), 'gpu_uuid': before['uuid'], 'driver_version': before['driver_version'], 'telemetry': telemetry,
        'protocol': PROTOCOL, 'input_sha256': input_sha, 'raw_batch_ms': repeats,
        'timing': aggregate(repeats, args.batch), 'memory_baseline': baseline,
        'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
        'probe_sha256': sha256(__file__)}
    require(report['checkpoint_sha256'] == WEIGHTS[args.model], 'Checkpoint changed during measurement')
    write_json(args.result, report)


def run_stages(out, command):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for model, batch in STAGES:
        tag = f'{model}_b{batch}'
        print(f'STAGE {tag}', flush=True)
        result = subprocess.run([*command, '--model', model, '--batch', str(batch),
                                 '--result', str(out / (tag + '.json'))], check=False)
        (out / (tag + '.exit')).write_text(f'exit_code={result.returncode}\n')
        result.check_returncode()


def validate_pair(reports):
    require(len(reports) == 4 and {(r['model'], r['batch_size']) for r in reports} == set(STAGES), 'Missing/duplicate stages')
    for r in reports:
        require(r['status'] == 'complete', 'Incomplete stage')
        for key in ('hardware', 'gpu_uuid', 'driver_version', 'common_identity', 'protocol', 'probe_sha256'):
            require(r[key] == reports[0][key], 'Pair mismatch: ' + key)
        peers = [p for p in reports if p['batch_size'] == r['batch_size']]
        require(all(p['input_sha256'] == r['input_sha256'] for p in peers), 'Different paired inputs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='/home/featurize/work/GAViT_AID_20260920')
    parser.add_argument('--work', default='/home/featurize/work')
    parser.add_argument('--output')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', choices=('swin', 'gavit'))
    parser.add_argument('--batch', type=int, choices=(1, 32))
    parser.add_argument('--result')
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    require(args.output, '--output is required')
    out = Path(args.output).resolve()
    require(Path(args.repo).resolve() not in out.parents and out != Path(args.repo).resolve(), 'Output must be outside training repo')
    out.mkdir(parents=True, exist_ok=True)
    # The paste launcher owns this directory; an exclusive marker prevents reuse.
    (out / 'started').open('x').close()
    start = time.perf_counter()
    try:
        import fcntl
        with (Path(args.work) / '.aid_efficiency.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            gpu_snapshot()
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--repo', args.repo, '--work', args.work]
            run_stages(out / 'stages', command)
            reports = [json.loads((out / f'stages/{model}_b{batch}.json').read_text()) for model, batch in STAGES]
            validate_pair(reports)
            table = [{k: r[k] for k in ('model', 'batch_size', 'parameters', 'timing', 'peak_allocated_bytes', 'peak_reserved_bytes')} for r in reports]
            summary = {'status': 'complete', 'scope': 'model-only synthetic input; not dataset throughput',
                       'protocol': PROTOCOL, 'hardware': reports[0]['hardware'], 'results': table,
                       'common_identity': reports[0]['common_identity'], 'gpu_uuid': reports[0]['gpu_uuid'],
                       'driver_version': reports[0]['driver_version'], 'checkpoint_sha256': WEIGHTS,
                       'graph_capture': {r['model']: r['graph_capture'] for r in reports},
                       'stage_exit_codes': {f'{m}_b{b}': 0 for m, b in STAGES},
                       'elapsed_seconds': time.perf_counter() - start, 'probe_sha256': sha256(__file__)}
            write_json(out / 'summary.json', summary)
            (out / 'run.exit').write_text('exit_code=0\n')
            print(json.dumps(summary, indent=2), flush=True)
            print('AID EFFICIENCY COMPLETE: ' + str(out), flush=True)
    except BaseException as exc:
        write_json(out / 'failure.json', {'status': 'failed', 'error': repr(exc), 'elapsed_seconds': time.perf_counter() - start})
        (out / 'run.exit').write_text('exit_code=1\n')
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
