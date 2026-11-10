"""CPU-only weight/geometry analysis of the frozen 90-case NWPU capture.

No model, image, checkpoint, training, or network access. The captured tensors
do not include region features; weight overlap is not feature collapse.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import time

import torch


MANIFEST_SHA = 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
CHECKPOINTS = {
    'fixed': '1766fe7c330eb37fb350f0348bcd255bafc451b793411a927548ee005a140f67',
    'soft': '674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12'}
RUN_HASHES = {
    'fixed': '4d10169e3618252f079bce85094750eb0477e6caa8ad9a37f13394eecfbb81d8',
    'soft': '9c1fa474d3f2e79203a1c245acfd15692a9a6e766acb71ef2566baff3d1bfd33'}
SELECTION = 'val_first_two_lexicographic_paths_per_class_v1'
ERROR_KEYS = ('capture_max_logit_diff', 'reconstruction_max_logit_diff',
              'restoration_max_logit_diff', 'cached_probability_max_error',
              'S_row_error', 'W_column_error')
BUCKETS = {(True, True): 'both_correct', (False, True): 'soft_corrects',
           (True, False): 'soft_breaks', (False, False): 'both_wrong'}
SCOPE = ('Selected calibration validation cases only; no population or causal claims. '
         'Weight overlap is not semantic feature collapse: raw tokens and region features were not cached. '
         'Uniform-local is a geometry reference, not a trained or evaluated model. '
         'Checkpoint hashes and forward checks are recorded provenance, not new checkpoint/forward verification.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def geometry():
    i = torch.arange(49)
    anchors = (i // 7 * 4 // 7) * 4 + i % 7 * 4 // 7
    hard = torch.nn.functional.one_hot(anchors, 16).bool()
    k = torch.arange(16)
    mask = ((anchors[:, None] // 4 - k // 4).abs() <= 1) & ((anchors[:, None] % 4 - k % 4).abs() <= 1)
    # Compute coordinates directly in float64, not rounded float32 intermediates.
    xy = (torch.stack((i % 7, i // 7), 1).double() + .5) / 7
    centers = hard.double().T @ xy / hard.sum(0)[:, None]
    neighbors = [(a, b) for a in range(16) for b in range(a + 1, 16)
                 if abs(a // 4 - b // 4) + abs(a % 4 - b % 4) == 1]
    return anchors, hard, mask, xy, centers, neighbors


def validate_tensors(t, name):
    anchors, hard, mask, *_ = geometry()
    allowed = hard if name == 'fixed' else mask
    shapes = dict(S=(49, 16), W=(49, 16), candidate_mask=(49, 16),
                  anchor_assignment=(49,), edge_index=(2, 80), logits=(45,))
    for key, shape in shapes.items():
        require(isinstance(t.get(key), torch.Tensor) and tuple(t[key].shape) == shape, f'{name}: bad {key} shape')
        require(t[key].device.type == 'cpu' and bool(torch.isfinite(t[key]).all()), f'{name}: invalid {key}')
    require(t['candidate_mask'].dtype == torch.bool and torch.equal(t['candidate_mask'], allowed), 'candidate mask changed')
    require(t['anchor_assignment'].dtype == torch.int64 and torch.equal(t['anchor_assignment'], anchors), 'anchors changed')
    s, w = t['S'], t['W']
    require(s.is_floating_point() and w.is_floating_point(), 'non-floating weights')
    require(bool((s >= 0).all()) and bool((w >= 0).all()), 'negative weights')
    errors = dict(S_row_error=float((s.double().sum(1)-1).abs().max()),
                  W_column_error=float((w.double().sum(0)-1).abs().max()))
    require(max(errors.values()) <= 1e-5, 'normalization mismatch')
    require(not bool(s[~allowed].any()) and not bool(w[~allowed].any()), 'weights outside candidates')
    require(bool((s.sum(0) > 0).all()), 'zero S region mass')
    if name == 'fixed':
        require(torch.equal(s, hard.to(s.dtype)), 'fixed S is not the frozen hard assignment')
    e = t['edge_index']
    require(e.dtype == torch.int64 and int(e.min()) >= 0 and int(e.max()) < 16
            and torch.unique(e, dim=1).shape[1] == 80, 'invalid graph indices/duplicates')
    source, target = e
    d = (source // 4 - target // 4).abs() + (source % 4 - target % 4).abs()
    require(bool((d[:48] == 1).all()) and bool((d[48:] > 1).all())
            and bool((torch.bincount(target[48:], minlength=16) == 2).all()), 'invalid graph structure')
    return errors


def uniform_reference():
    anchors, _, mask, *_ = geometry()
    s = mask.double() / mask.sum(1, keepdim=True)
    return dict(S=s, W=s/s.sum(0, keepdim=True), candidate_mask=mask, anchor_assignment=anchors)


def measure(t):
    _, hard, _, xy, centers, neighbors = geometry()
    s, w = t['S'].double(), t['W'].double()
    mass = s.sum(0)
    # Remove <=1e-5 floating summation error after validation, not bad weights.
    w = w / w.sum(0, keepdim=True)
    u = s / mass[None]
    counts = t['candidate_mask'].sum(1)
    entropy = -(torch.where(s > 0, s * s.clamp_min(1e-300).log(), 0)).sum(1)
    entropy = torch.where(counts > 1, entropy/counts.double().clamp_min(2).log(), 0)
    support = t['candidate_mask'].sum(0)
    effective = 1 / w.square().sum(0)
    overlap = torch.minimum(w.T[:, None, :], w.T[None, :, :]).sum(-1)
    neighbor_values = torch.tensor([float(overlap[a, b]) for a, b in neighbors], dtype=torch.float64)
    reference_mass = uniform_reference()['S'].sum(0)
    scalars = dict(S_entropy_normalized_mean=float(entropy.mean()), S_mass_min=float(mass.min()),
                   S_mass_max=float(mass.max()), W_neighbor_overlap_mean=float(neighbor_values.mean()),
                   W_all_pair_overlap_max=float(overlap[torch.triu(torch.ones(16, 16, dtype=torch.bool), 1)].max()),
                   W_effective_tokens_mean=float(effective.mean()),
                   W_effective_support_fraction_mean=float((effective/support).mean()),
                   W_outside_original_region_mean=float((w * ~hard).sum(0).mean()))
    regions = dict(S_mass=mass.tolist(), S_mass_over_uniform_local=(mass/reference_mass).tolist(),
                   W_effective_tokens=effective.tolist(), candidate_support=support.tolist(),
                   W_effective_support_fraction=(effective/support).tolist())
    for name, weights in [('S', u), ('W', w)]:
        mu = weights.T @ xy
        drift_sq = (mu - centers).square().sum(1)
        radius_sq = (weights * (xy[:, None, :] - mu[None]).square().sum(-1)).sum(0)
        anchor_sq = (weights * (xy[:, None, :] - centers[None]).square().sum(-1)).sum(0)
        require(bool(torch.allclose(anchor_sq, radius_sq + drift_sq, atol=1e-12, rtol=1e-12)), 'geometry identity failed')
        drift, radius = drift_sq.sqrt(), radius_sq.sqrt()
        scalars.update({name+'_centroid_drift_mean': float(drift.mean()),
                        name+'_centroid_drift_max': float(drift.max()), name+'_center_rms_mean': float(radius.mean())})
        regions.update({name+'_centroid': mu.tolist(), name+'_centroid_drift': drift.tolist(),
                        name+'_center_rms': radius.tolist(), name+'_anchor_rms': anchor_sq.sqrt().tolist()})
    return dict(scalars=scalars, regions=regions, token_entropy_normalized=entropy.tolist(),
                neighbor_pairs=[list(p) for p in neighbors], neighbor_W_overlap=neighbor_values.tolist())


def aggregate(rows):
    if not rows:
        return {}
    result = {}
    for key in rows[0]:
        v = torch.tensor([r[key] for r in rows], dtype=torch.float64)
        result[key] = dict(mean=float(v.mean()), median=float(v.quantile(.5)), p90=float(v.quantile(.9)))
    return result


def difference(a, b):
    return {k: a[k]-b[k] for k in a}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def analyze(source, out):
    inventory = {}
    def read(relative):
        raw = (source/relative).read_bytes()
        inventory[relative] = hashlib.sha256(raw).hexdigest()
        return raw
    def read_json(relative):
        return json.loads(read(relative))
    require(read('run.exit').decode().strip() == 'exit_code=0', 'source exit is not zero')
    summary, identity, selection = (read_json(n+'.json') for n in ('summary', 'identity', 'selection'))
    for k, v in dict(status='complete', samples=90, classes=45, split='val', source_phase='calibrate',
                     selection=SELECTION, all_capture_checks_passed=True, spatial_edges=48, feature_edges=32).items():
        require(summary.get(k) == v, f'source summary mismatch: {k}')
    require(identity['manifest_sha256'] == MANIFEST_SHA and identity['selection'] == selection['selection'] == SELECTION,
            'manifest/selection identity mismatch')
    sources = identity['checkpoint_sources']
    require(sources == selection['checkpoint_sources'] and set(sources) == {'fixed', 'soft'}, 'checkpoint sources mismatch')
    for n, src in sources.items():
        require(src['checkpoint_sha256'] == CHECKPOINTS[n] and src['run_sha256'] == RUN_HASHES[n]
                and src['best_epoch'] == (29 if n == 'fixed' else 28), f'{n}: wrong recorded calibration identity')
        digest = src['cached_predictions_sha256']
        require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'invalid recorded predictions hash')
    rows = selection['records']
    require(len(rows) == 90 and Counter(r['label'] for r in rows) == {i: 2 for i in range(45)}, 'wrong class counts')
    paths = [r['path'] for r in rows]
    require(len(set(paths)) == 90 and identity['sample_ids'] == paths, 'duplicate or mismatched sample IDs')
    require(rows == sorted(rows, key=lambda r: (r['label'], r['path'])), 'selection order changed')
    for r in rows:
        p = PurePosixPath(r['path'])
        require(r['split'] == 'val' and bool(r['path']) and not p.is_absolute() and '..' not in p.parts
                and '\\' not in r['path'] and str(p) == r['path'], 'unsafe/non-val sample path')
    print('SOURCE PASS: 90 cached validation cases; CPU only', flush=True)
    ref = measure(uniform_reference())
    results, checks, measured_errors = [], [], []
    for i, row in enumerate(rows):
        record = read_json(f'cases/{i:03d}.json')
        require(record['path'] == row['path'] and record['label'] == row['label'], 'case path/label mismatch')
        case = dict(path=row['path'], label=row['label'], predictions={}, metrics={})
        for n in ('fixed', 'soft'):
            payload = torch.load(io.BytesIO(read(f'cases/{i:03d}_{n}.pt')), map_location='cpu', weights_only=True)
            require(payload['path'] == row['path'] and payload['label'] == row['label']
                    and payload['source'] == sources[n] and payload['cached_check_status'] == 'passed', f'{i}/{n}: cache identity/status mismatch')
            check = payload['checks']
            require(check == record['checks'][n] and check['spatial_edges'] == 48
                    and check['feature_edges'] == 32 and check['gat_self_loops_not_drawn'] is True, 'recorded checks mismatch')
            for k in ERROR_KEYS:
                require(math.isfinite(check[k]) and check[k] >= 0, 'nonfinite/negative recorded error')
            # Capture used mixed absolute/relative logit tolerance. Do not invent
            # a stricter absolute-only forward test from its saved maximum.
            for k in ('S_row_error', 'W_column_error', 'cached_probability_max_error'):
                require(check[k] <= 1e-5, 'recorded normalization/probability check failed')
            errors = validate_tensors(payload['tensors'], n)
            measured_errors.append(errors)
            checks.append(check)
            pred = int(payload['tensors']['logits'].argmax())
            require(pred == record['predictions'][n], 'cached argmax mismatch')
            case['predictions'][n] = pred
            case['metrics'][n] = measure(payload['tensors'])
        case['bucket'] = BUCKETS[case['predictions']['fixed'] == row['label'], case['predictions']['soft'] == row['label']]
        results.append(case)
        if (i + 1) % 30 == 0:
            print(f'CACHE {i+1}/90: PASS', flush=True)
    paired = {b: sum(r['bucket'] == b for r in results) for b in BUCKETS.values()}
    correct = {n: sum(r['predictions'][n] == r['label'] for r in results) for n in ('fixed', 'soft')}
    require(paired == summary['paired'] and correct == summary['correct'], 'aggregate predictions differ from source summary')
    require({k: max(c[k] for c in checks) for k in ERROR_KEYS} == summary['max_errors'], 'recorded maximum errors mismatch')
    stats = {n: aggregate([r['metrics'][n]['scalars'] for r in results]) for n in ('fixed', 'soft')}
    stats['uniform_local'] = aggregate([ref['scalars']])
    for n, reference in [('soft_minus_fixed', None), ('soft_minus_uniform_local', ref['scalars'])]:
        stats[n] = aggregate([difference(r['metrics']['soft']['scalars'],
                                        r['metrics']['fixed']['scalars'] if reference is None else reference) for r in results])
    buckets = {}
    for b, count in paired.items():
        buckets[b] = dict(samples=count, soft_minus_fixed=aggregate([
            difference(r['metrics']['soft']['scalars'], r['metrics']['fixed']['scalars']) for r in results if r['bucket'] == b]))
    write_json(out/'per_case.json', dict(cases=results, uniform_local=ref, coordinate_units='input_width_height', scope=SCOPE))
    report = dict(status='complete', samples=90, classes=45, split='val', source_phase='calibrate',
                  source=str(source), recorded_checkpoint_sources=sources, manifest_sha256=MANIFEST_SHA,
                  selection=SELECTION, correct=correct, paired=paired, statistics=stats, bucket_statistics=buckets,
                  recomputed_max_errors={k: max(e[k] for e in measured_errors) for k in measured_errors[0]},
                  recorded_max_errors=summary['max_errors'], coordinate_units='input_width_height',
                  source_files_sha256=inventory, scope=SCOPE, torch_version=str(torch.__version__))
    for n in ('fixed', 'soft', 'uniform_local'):
        compact = {k: round(stats[n][k]['mean'], 6) for k in
                   ('S_entropy_normalized_mean', 'W_neighbor_overlap_mean', 'W_effective_tokens_mean',
                    'S_centroid_drift_mean', 'W_centroid_drift_mean', 'W_center_rms_mean')}
        print(n.upper(), json.dumps(compact), flush=True)
    print('PAIRED', json.dumps(paired), flush=True)
    print('SCOPE: weight/geometry only; selected val cases; no causal or feature-collapse conclusion.', flush=True)
    return report


def execute(source, output):
    source, out = Path(source).resolve(), Path(output).resolve()
    require(not out.is_relative_to(source), 'output must be outside the read-only source tree')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        torch.set_num_threads(1)
        report = analyze(source, out)
        report['elapsed_seconds'] = time.perf_counter() - start
        # Launcher injects the hash after verifying the embedded source bytes.
        report['tool_sha256'] = globals().get('VERIFIED_TOOL_SHA256')
        write_json(out/'summary.json', report)
        (out/'run.exit').write_text('exit_code=0\n')
        print(f'ASSIGNMENT CACHE COMPLETE {out}\nexit_code=0', flush=True)
        return report
    except BaseException as exc:
        write_json(out/'summary.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                           elapsed_seconds=time.perf_counter()-start))
        (out/'run.exit').write_text('exit_code=1\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.source, args.output)


if __name__ == '__main__':
    main()
