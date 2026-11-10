"""Frozen CPU-only 90-case GAT neighbor deletion. Never reads images or trains."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import time

WORK = Path('/home/featurize/work')
MANIFEST_SHA = 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
CHECKPOINT_SHA = '674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12'
RUN_SHA = '9c1fa474d3f2e79203a1c245acfd15692a9a6e766acb71ef2566baff3d1bfd33'
FEEDBACK_SHA = '3501624ee57594859637fa88a25082c1f9be284f77f5f5180161ff6f2d4cce0b'
FEEDBACK_BYTES = 9306340
GRAPH_SHA = '895825a417f627594fda8407cf1e980f8bcef398201c233e1c20d900ec9046da'
POOL_TOOL_SHA = '3a144266ef35b3df3a60cf3e8abd614f5186467510f1bb857c3019eac5b4967a'
FEEDBACK_TOOL_SHA = '4cd5175261d271638a00e4a4c733f329ecde1c07b646b901c960f0181237eea3'
SCOPE = ('Selected 90 validation cases only; self-only is an inference intervention in a jointly '
         'trained model, not a trained no-graph baseline or population/stability evidence. '
         'Neighbor deletion changes GAT attention normalization; backbone and pooling still mix information.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def close(actual, expected, name):
    import torch
    try:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    except AssertionError as exc:
        raise ValueError('Replay mismatch: ' + name) from exc
    return float((actual - expected).abs().max())


def tensor(value, shape, name, dtype=None):
    import torch
    dtype = torch.float32 if dtype is None else dtype
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape
            and value.dtype == dtype and value.device.type == 'cpu'
            and bool(torch.isfinite(value).all()), 'Invalid tensor: ' + name)


def load_inputs(args):
    import torch
    print('CHECK sources: persistent caches and checkpoint; CPU only', flush=True)
    reports = {}
    for name, folder in [('pooling', args.pooling), ('feedback', args.feedback), ('calibration', args.calibration)]:
        require((folder/'run.exit').read_text().strip() == 'exit_code=0', name + ' did not complete')
        reports[name] = json.loads((folder/('run.json' if name == 'calibration' else 'summary.json')).read_text())
        require(reports[name]['status'] == 'complete', name + ' source incomplete')
    selected_path = args.pooling/'selected_cases.pt'
    record = reports['pooling']['cache_artifacts']['selected_cases.pt']
    files = {str(selected_path): record['sha256'], str(args.feedback/'soft_cache.pt'): FEEDBACK_SHA,
             str(args.calibration/'best.pth'): CHECKPOINT_SHA, str(args.calibration/'run.json'): RUN_SHA,
             str(args.repo/'models/graph_reasoning.py'): GRAPH_SHA}
    require(selected_path.stat().st_size == record['bytes'], 'Selected cache size mismatch')
    require((args.feedback/'soft_cache.pt').stat().st_size == FEEDBACK_BYTES, 'Feedback cache size mismatch')
    for path, expected in files.items():
        require(sha(path) == expected, 'Source hash mismatch: ' + path)
        print('HASH PASS ' + Path(path).name, flush=True)
    selected = torch.load(selected_path, map_location='cpu', weights_only=True)
    feedback = torch.load(args.feedback/'soft_cache.pt', map_location='cpu', weights_only=True)
    pi, fi = selected['identity'], feedback['identity']
    require(pi == json.loads((args.pooling/'identity.json').read_text()), 'Pooling embedded identity differs')
    require(json.loads(json.dumps(fi)) == json.loads((args.feedback/'identity.json').read_text()), 'Feedback embedded identity differs')
    for ident in (pi, fi):
        for key, value in dict(dataset='NWPU-RESISC45', seed=42, split='val', source_phase='calibrate',
                               samples=1260, manifest_sha256=MANIFEST_SHA).items():
            require(ident[key] == value, 'Invalid cache identity: ' + key)
    source = pi['source']
    require(source == fi['sources']['soft'] and source['checkpoint_sha256'] == CHECKPOINT_SHA
            and source['run_sha256'] == RUN_SHA and source['best_epoch'] == 28, 'Different soft checkpoint source')
    require(pi['model_code'] == fi['source_code'] == reports['calibration']['code'], 'Different source model code')
    require(pi['tool_sha256'] == POOL_TOOL_SHA and fi['tool_sha256'] == FEEDBACK_TOOL_SHA, 'Different cache writer')
    require(feedback['model'] == 'soft', 'Wrong feedback cache model')
    cases, index = validate_cases(selected['cases'], feedback)
    identity = dict(dataset='NWPU-RESISC45', seed=42, split='val', source_phase='calibrate', samples=90,
                    selection='val_first_two_lexicographic_paths_per_class_v1', source=source,
                    manifest_sha256=MANIFEST_SHA, model_code=pi['model_code'], files_sha256=files,
                    selected_cache_hash_basis='server original pooling summary (not independently preknown)',
                    conditions=['original_graph', 'self_loops_only'], scope=SCOPE,
                    tool_sha256=sha(__file__))
    return cases, feedback, index, reports['calibration'], identity


def validate_cases(cases, feedback):
    import torch
    paths, labels, names = feedback['paths'], feedback['labels'], feedback['class_names']
    require(len(paths) == len(set(paths)) == len(labels) == 1260, 'Wrong full-val paths/count')
    require(len(names) == len(set(names)) == 45 and Counter(labels) == {i: 28 for i in range(45)}, 'Wrong class support')
    for path, label in zip(paths, labels):
        p = PurePosixPath(path)
        require(type(label) is int and not p.is_absolute() and '..' not in p.parts and '\\' not in path
                and len(p.parts) == 2 and p.parts[0] == names[label], 'Invalid cached path/label')
    for key, shape in [('mean_tokens', (1260, 768)), ('mean_feedback', (1260, 768)), ('normal_logits', (1260, 45))]:
        tensor(feedback[key], shape, key)
    require(int((feedback['normal_logits'].argmax(-1) == torch.tensor(labels)).sum()) == 1181,
            'Cached full-val normal OA differs')
    index = {p: i for i, p in enumerate(paths)}
    expected = {p for label in range(45) for p in sorted(p for p, y in zip(paths, labels) if y == label)[:2]}
    require(len(cases) == 90 and len({c['path'] for c in cases}) == 90
            and {c['path'] for c in cases} == expected, 'Different fixed 90-case selection')
    for case in cases:
        require(case['label'] == labels[index[case['path']]], 'Selected label differs')
        t = case['tensors']['normal']
        for key, shape in [('S', (49, 16)), ('W', (49, 16)), ('regions', (16, 768)), ('pooled', (768,)), ('logits', (45,))]:
            tensor(t[key], shape, key)
        tensor(t['edges'], (2, 80), 'edges', torch.int64)
        require(bool((t['S'] >= 0).all()) and bool((t['W'] >= 0).all()), 'Negative assignment')
        close(t['S'].sum(-1), torch.ones(49), 'S row normalization')
        close(t['W'].sum(0), torch.ones(16), 'W column normalization')
        e = t['edges']; src, dst = e
        require(int(e.min()) >= 0 and int(e.max()) < 16 and torch.unique(e, dim=1).shape[1] == 80, 'Invalid graph indices')
        d = (src//4-dst//4).abs() + (src%4-dst%4).abs()
        require(bool((d[:48] == 1).all()) and bool((d[48:] > 1).all())
                and bool((torch.bincount(dst[48:], minlength=16) == 2).all()), 'Invalid spatial/feature edges')
        close(t['logits'], feedback['normal_logits'][index[case['path']]], 'independent cached logits')
    return sorted(cases, key=lambda c: c['path']), index


def load_modules(args, report, cached_head):
    import torch
    from torch import nn
    a = report['architecture']
    for key, value in dict(num_classes=45, num_regions=16, gat_hidden=256, gat_heads=4, gat_layers=2,
                           dropout=.1, grouping='soft_prototype', edge_type='sparse_hybrid', integration='token_feedback').items():
        require(a[key] == value, 'Unexpected architecture: ' + key)
    spec = importlib.util.spec_from_file_location('frozen_neighbor_graph', args.repo/'models/graph_reasoning.py')
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    graph = mod.GraphReasoning(768, 256, 4, 2, .1)
    feedback = nn.Sequential(nn.Linear(1024, 768), nn.LayerNorm(768))
    head = nn.Sequential(nn.LayerNorm(768), nn.Dropout(.1), nn.Linear(768, 45))
    state = torch.load(args.calibration/'best.pth', map_location='cpu', weights_only=True)
    require(not any(k.startswith('token_refinement.') for k in state), 'FFN checkpoint not allowed')
    for prefix, module in [('graph_reasoning.', graph), ('feedback_proj.', feedback), ('classifier.', head)]:
        module.load_state_dict({k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}, strict=True)
        module.eval().requires_grad_(False)
    require(cached_head.keys() == head.state_dict().keys()
            and all(torch.equal(v, cached_head[k]) for k, v in head.state_dict().items()), 'Cached head differs from checkpoint')
    mask = state['region_grouping.candidate_mask']
    tensor(mask, (49, 16), 'candidate mask', torch.bool)
    return graph, feedback, head, mask


def audited_graph(graph, regions, edges):
    """Observe actual internal GAT edges; cross-check against untouched forward."""
    import torch
    require(not any(m.training for m in graph.modules()), 'Graph must be eval')
    require(regions.device.type == edges.device.type == 'cpu', 'CPU only')
    counts = []
    with torch.inference_mode():
        expected = graph(regions, edges)
        x = graph.input_proj(regions)
        n = len(regions)
        loops = torch.arange(n).repeat(2, 1)
        wanted = torch.cat((edges, loops), dim=1)
        for gat, norm in zip(graph.gat_layers, graph.norms):
            require(gat.add_self_loops, 'Original GAT self-loops must stay enabled')
            update, (actual, alpha) = gat(x, edges, return_attention_weights=True)
            require(torch.equal(torch.sort(actual[0]*n+actual[1]).values,
                                torch.sort(wanted[0]*n+wanted[1]).values), 'Internal GAT edge set mismatch')
            require(bool(torch.isfinite(alpha).all()), 'Nonfinite attention')
            counts.append(actual.shape[1])
            x = graph.dropout(graph.act(norm(update + x)))
        close(x, expected, 'observed vs original GraphReasoning forward')
    return x, counts


def replay_case(graph, feedback, head, case, mean_tokens, cached_logits, cached_feedback):
    import torch
    require(not any(m.training for module in (graph, feedback, head) for m in module.modules()), 'Modules must be eval')
    errors = {}
    with torch.inference_mode():
        z, edges_normal = audited_graph(graph, case['regions'], case['edges'])
        mean_f = (case['S'] @ feedback(z)).mean(0)
        pooled = mean_tokens + mean_f
        normal = head(pooled)
        errors['independent_feedback'] = close(mean_f, cached_feedback, 'independent mean feedback')
        errors['pooled'] = close(pooled, case['pooled'], 'original pooled')
        errors['pooling_logits'] = close(normal, case['logits'], 'original pooling logits')
        errors['feedback_logits'] = close(normal, cached_logits, 'original independent logits')
        require(int(normal.argmax()) == int(case['logits'].argmax()) == int(cached_logits.argmax()), 'Original argmax changed')
        empty = case['edges'][:, :0]
        z_self, edges_self = audited_graph(graph, case['regions'], empty)
        self_f = (case['S'] @ feedback(z_self)).mean(0)
        self_logits = head(mean_tokens + self_f)
        restored = head(mean_tokens + (case['S'] @ feedback(graph(case['regions'], case['edges']))).mean(0))
        errors['restoration'] = close(restored, normal, 'restored original logits')
        require(int(restored.argmax()) == int(normal.argmax()), 'Restoration argmax changed')
        tensor(self_logits, tuple(normal.shape), 'self-only logits')
    return dict(normal_logits=normal, self_only_logits=self_logits, errors=errors,
                mean_feedback_delta_l2=float((self_f-mean_f).norm()),
                internal_edges=dict(normal=edges_normal, self_only=edges_self))


def compare(normal, self_only, labels):
    import torch
    y = torch.tensor(labels); a = normal.argmax(-1) == y; b = self_only.argmax(-1) == y
    metrics = {}
    for name, logits in [('normal', normal), ('self_only', self_only)]:
        n = logits.shape[1]; prediction = logits.argmax(-1)
        cm = torch.bincount(y*n+prediction, minlength=n*n).reshape(n, n)
        tp = cm.diag().float(); denom = cm.sum(0)+cm.sum(1)
        f1 = torch.where(denom > 0, 2*tp/denom, 0.)
        metrics[name] = dict(samples=len(y), correct=int((prediction == y).sum()),
                             oa_percent=float((prediction == y).double().mean()*100),
                             macro_f1_percent=float(f1.double().mean()*100),
                             cross_entropy=float(torch.nn.functional.cross_entropy(logits, y)))
    return dict(**metrics, paired=dict(both_correct=int((a & b).sum()), self_only_corrects=int((~a & b).sum()),
                                      self_only_breaks=int((a & ~b).sum()), both_wrong=int((~a & ~b).sum()),
                                      net_correct=int(b.sum()-a.sum())))


def run(args, out):
    import torch
    import torch_geometric
    require(str(torch.__version__).split('+')[0] == '2.2.2' and torch_geometric.__version__ == '2.8.0', 'Use existing torch2.2.2/PyG2.8.0 environment')
    torch.set_num_threads(args.threads)
    cases, cache, index, report, identity = load_inputs(args)
    graph, feedback, head, mask = load_modules(args, report, cache['classifier_state'])
    modules = (graph, feedback, head)
    before = [{k: v.clone() for k, v in m.state_dict().items()} for m in modules]
    identity['hardware'] = dict(device='cpu', threads=args.threads, python=platform.python_version(),
                                torch=str(torch.__version__), pyg=torch_geometric.__version__, precision='float32_no_amp')
    save_json(out/'identity.json', identity)
    print('SOURCE PASS: 90 fixed val cases; independent mean_tokens; CPU only', flush=True)
    results, rows = [], []
    for i, c in enumerate(cases):
        t = c['tensors']['normal']; j = index[c['path']]
        require(not bool(t['S'].masked_select(~mask).any())
                and not bool(t['W'].masked_select(~mask).any()), 'Assignment outside frozen candidates')
        r = replay_case(graph, feedback, head, t, cache['mean_tokens'][j], cache['normal_logits'][j], cache['mean_feedback'][j])
        require(r['internal_edges'] == dict(normal=[96, 96], self_only=[16, 16]), 'Unexpected internal graph counts')
        results.append(r)
        rows.append(dict(path=c['path'], label=c['label'], normal=int(r['normal_logits'].argmax()),
                         self_only=int(r['self_only_logits'].argmax()), mean_feedback_delta_l2=r['mean_feedback_delta_l2']))
        if (i+1) % 30 == 0:
            print(f'REPLAY {i+1}/90: normal/self-only/restoration PASS', flush=True)
    require(all(all(torch.equal(v, state[k]) for k, v in m.state_dict().items()) for m, state in zip(modules, before)), 'Model state changed')
    require(all(p.grad is None for m in modules for p in m.parameters()), 'Unexpected gradients')
    normal = torch.stack([r['normal_logits'] for r in results]); self_only = torch.stack([r['self_only_logits'] for r in results])
    summary = compare(normal, self_only, [c['label'] for c in cases])
    require(summary['normal']['correct'] == 85, 'Normal selected-case correct count differs')
    with (out/'predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    torch.save(dict(identity=identity, paths=[c['path'] for c in cases], labels=[c['label'] for c in cases],
                    normal_logits=normal, self_only_logits=self_only, class_names=cache['class_names']), out/'predictions.pt')
    summary.update(status='complete', samples=90, split='val', seed=42, scope=SCOPE,
                   all_checks_passed=True, internal_edges_per_layer=results[0]['internal_edges'],
                   max_errors={k: max(r['errors'][k] for r in results) for k in results[0]['errors']},
                   model_state_unchanged=True, predictions_sha256=sha(out/'predictions.pt'))
    for key in ('normal', 'self_only', 'paired'):
        print(key.upper() + ' ' + json.dumps(summary[key]), flush=True)
    return summary


def execute(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        print('START CPU neighbor probe: 90 cached validation cases; loading runtime', flush=True)
        summary = run(args, out)
        summary['elapsed_seconds'] = time.perf_counter()-started
        save_json(out/'summary.json', summary)
        (out/'run.exit').write_text('exit_code=0\n')
        print(f'NWPU NEIGHBOR PROBE COMPLETE {out}\nexit_code=0', flush=True)
        return summary
    except BaseException as exc:
        save_json(out/'summary.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter()-started))
        (out/'run.exit').write_text('exit_code=1\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooling', type=Path, default=WORK/'nwpu_pooling42_20261008_01')
    p.add_argument('--feedback', type=Path, default=WORK/'nwpu_feedback42_20261002_01')
    p.add_argument('--calibration', type=Path, default=WORK/'nwpu_ablation42_20260928_01/soft')
    p.add_argument('--repo', type=Path, default=WORK/'GAViT_SOFT_20260927')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, choices=range(1, 9), default=8)
    execute(p.parse_args())


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
