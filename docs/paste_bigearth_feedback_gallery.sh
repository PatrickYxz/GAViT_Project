python - <<PY
from pathlib import Path
import hashlib
import subprocess
import sys
import tempfile

work = Path(tempfile.mkdtemp(prefix="pair44_feedback_", dir="/home/featurize/work"))
script = work / "diagnose_bigearth_feedback.py"
source = r'''#!/usr/bin/env python3
"""Fixed-case GAViT feedback intervention. Never trains or edits model code."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys
import tempfile
import time
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def gallery_path(source, name):
    source = Path(source).resolve()
    result = (source / name).resolve()
    if Path(name).is_absolute() or not result.is_relative_to(source):
        raise ValueError('Gallery image must be inside the source diagnosis directory')
    return result


def select_cases(path):
    """Keep the first preselected image per class/outcome/true-label stratum."""
    selected, seen = {}, set()
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            c, label, index = int(row['class_index']), int(row['label']), int(row['row_index'])
            key = (c, row['bucket'], label)
            if c not in (11, 14, 16) or label not in (0, 1) or key in seen:
                continue
            if row['bucket'] not in ('both_correct', 'both_wrong', 'swin_only_correct', 'gavit_only_correct'):
                raise ValueError('Unknown case bucket')
            seen.add(key)
            item = selected.setdefault(index, dict(row_index=index, image=row['image'], strata=[]))
            if item['image'] != row['image']:
                raise ValueError('Inconsistent image identity')
            item['strata'].append(dict(class_index=c, label=label, bucket=row['bucket']))
    if not selected:
        raise ValueError('No source cases')
    return [selected[i] for i in sorted(selected)]


def validate_topology(assignments, edges):
    """Validate captured graph input; never rebuild feature neighbors."""
    if len(assignments) != 49 or any(type(a) is not int or not 0 <= a < 16 for a in assignments):
        raise ValueError('Expected 49 token assignments to 16 regions')
    counts = [assignments.count(k) for k in range(16)]
    if min(counts) == 0 or len(edges) != 80 or len({tuple(e) for e in edges}) != 80:
        raise ValueError('Empty region, duplicate edge or wrong edge budget')
    kinds, spatial, incoming = [], set(), [0] * 16
    for source, target in edges:
        if not (0 <= source < 16 and 0 <= target < 16) or source == target:
            raise ValueError('Invalid graph input edge')
        a, b = divmod(source, 4); y, x = divmod(target, 4)
        if abs(a-y) + abs(b-x) == 1:
            kinds.append('spatial'); spatial.add((source, target))
        else:
            kinds.append('feature'); incoming[target] += 1
    if len(spatial) != 48 or incoming != [2] * 16:
        raise ValueError('Captured graph does not have 48 spatial and 2 incoming feature edges per region')
    return counts, kinds


def graph_svg(assignments, edges, kinds, norms):
    centers = []
    palette = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
               '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#fffac8','#800000','#aaffc3']
    parts = ['<svg viewBox="0 0 224 224" xmlns="http://www.w3.org/2000/svg">',
             '<defs><marker id="arrow" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="#222"/></marker></defs>']
    for i, region in enumerate(assignments):
        y, x = divmod(i, 7)
        parts.append(f'<rect class="token" x="{x*32}" y="{y*32}" width="32" height="32" fill="{palette[region]}" fill-opacity="0.25" stroke="white" stroke-width="0.4"/>')
    for k in range(16):
        points = [(i % 7 * 32 + 16, i // 7 * 32 + 16) for i, a in enumerate(assignments) if a == k]
        centers.append((sum(x for x, _ in points)/len(points), sum(y for _, y in points)/len(points)))
    for (source, target), kind in zip(edges, kinds):
        x1,y1 = centers[source]; x2,y2 = centers[target]
        # Keep arrow tips outside the node labels; opposite edges bend apart.
        dx, dy = x2-x1, y2-y1
        scale = 8 / math.hypot(dx, dy)
        x1,y1,x2,y2 = x1+dx*scale,y1+dy*scale,x2-dx*scale,y2-dy*scale
        mx,my = (x1+x2)/2 + (y2-y1)*.06, (y1+y2)/2 - (x2-x1)*.06
        color = '#1976d2' if kind == 'spatial' else '#ed6c02'
        parts.append(f'<path class="edge {kind}" data-source="{source}" data-target="{target}" d="M{x1},{y1} Q{mx},{my} {x2},{y2}" fill="none" stroke="{color}" stroke-width="0.7" opacity="0.75" marker-end="url(#arrow)"/>')
    for k, (x,y) in enumerate(centers):
        parts.append(f'<g><title>region {k}; feedback L2 {float(norms[k]):.6f}</title><circle cx="{x}" cy="{y}" r="7" fill="white" stroke="#333"/><text x="{x}" y="{y+2.5}" text-anchor="middle" font-size="7">{k}</text></g>')
    return ''.join(parts) + '</svg>'


def summarize(records):
    summary = {}
    for row in records:
        for case in row['strata']:
            c, y = case['class_index'], case['label']
            normal, zero = row['normal_scores'][c], row['zero_scores'][c]
            n_ok, z_ok = (normal >= .5) == bool(y), (zero >= .5) == bool(y)
            item = summary.setdefault(str(c), dict(n=0, feedback_corrects=0, feedback_breaks=0,
                                                   unchanged_correct=0, unchanged_wrong=0))
            item['n'] += 1
            key = ('feedback_corrects' if n_ok and not z_ok else 'feedback_breaks' if z_ok and not n_ok
                   else 'unchanged_correct' if n_ok else 'unchanged_wrong')
            item[key] += 1
    return summary


def probe(model, images):
    """Observe a real forward; then zero only feedback_proj output and restore."""
    import torch
    if any(m.training for m in model.modules()):
        raise ValueError('All modules must be in eval mode')
    captured, handles = {}, []
    def output_hook(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().clone()
        return hook
    def grouping_hook(module, inputs, output):
        captured['regions'], captured['assignments'] = [x.detach().clone() for x in output]
    def graph_input(module, inputs):
        for name, value in zip(('nodes', 'edge_index', 'cosine_or_unit_values'), inputs):
            captured[name] = value.detach().clone()
    with torch.inference_mode():
        normal = model(images)
        try:
            handles.append(model.backbone.register_forward_hook(output_hook('tokens')))
            handles.append(model.region_grouping.register_forward_hook(grouping_hook))
            handles.append(model.graph_reasoning.register_forward_pre_hook(graph_input))
            handles.append(model.graph_reasoning.register_forward_hook(output_hook('refined')))
            handles.append(model.feedback_proj.register_forward_hook(output_hook('feedback')))
            observed = model(images)
        finally:
            for handle in handles:
                handle.remove()
        zero_handle = model.feedback_proj.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
        try:
            zero = model(images)
        finally:
            zero_handle.remove()
        restored = model(images)
        token_feedback = captured['feedback'].gather(1, captured['assignments'].unsqueeze(-1).expand(-1,-1,captured['feedback'].shape[-1]))
        reconstruction = model.classifier((captured['tokens'] + token_feedback).mean(1))
        zero_reference = model.classifier(captured['tokens'].mean(1))
        comparisons = dict(capture=(normal, observed), reconstruction=(normal, reconstruction),
                           zero_branch=(zero, zero_reference), restoration=(normal, restored))
        errors = {}
        for name, (left, right) in comparisons.items():
            if left.shape != (1, 19) or not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f'Invalid logits: {name}')
            torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5, msg=name)
            errors[name] = float((left-right).abs().max().item())
        torch.testing.assert_close(captured['nodes'], captured['regions'].reshape(16,-1))
    return normal, zero, captured, errors


def prepare(args):
    run = json.loads((args.source / 'run.json').read_text())
    if run.get('status') != 'complete' or not run.get('scope', '').startswith('validation'):
        raise ValueError('Source must be a completed validation diagnostic')
    for name, expected in run['source_sha256'].items():
        if digest(args.repo / name) != expected:
            raise ValueError(f'Model/data source changed since cached predictions: {name}')
    for name, expected in run['diagnostic_sha256'].items():
        if digest(args.pair_tools / name) != expected:
            raise ValueError(f'Paired diagnostic source changed: {name}')
    selected = select_cases(args.source / 'cases.csv')
    by_id = {r['row_index']: r for r in selected}
    found = set()
    with (args.source / 'predictions.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            i = int(row['row_index'])
            if i not in by_id:
                continue
            if i in found:
                raise ValueError('Duplicate selected row in predictions')
            found.add(i); item = by_id[i]
            item.update(patch_path=row['patch_path'], labels=[int(row[f'label_{c}']) for c in range(19)],
                        cached_gavit_scores=[float(row[f'gavit_score_{c}']) for c in range(19)])
            if any(v not in (0, 1) for v in item['labels']) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in item['cached_gavit_scores']):
                raise ValueError('Invalid cached labels or probabilities')
            for case in item['strata']:
                if item['labels'][case['class_index']] != case['label']:
                    raise ValueError('Case labels do not match prediction cache')
            if args.input_mode == 'gallery':
                image = gallery_path(args.source, item['image'])
                item['input_sha256'] = digest(image)
            else:
                patch = Path(item['patch_path'])
                merged = next((patch / (patch.name + ext) for ext in ('.png','.jpg','.jpeg','.PNG','.JPG')
                               if (patch / (patch.name + ext)).is_file()), None)
                paths = [merged] if merged else [patch / f'{patch.name}_{b}.tif' for b in ('B04','B03','B02')]
                item['input_sha256'] = {p.name: digest(p) for p in paths}
    if found != set(by_id):
        raise ValueError('Selected cases missing from prediction cache')
    return run, selected


def run_diagnostic(args, out, source_run, selected):
    sys.path[:0] = [str(args.repo), str(args.pair_tools)]
    import torch
    from PIL import Image
    from torchvision import transforms
    from diagnose_bigearth_pair import build_model
    from models.bigearth_dataset import read_rgb_patch
    from utils import set_seed
    from importlib.metadata import version
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no inference performed')
    set_seed(42)
    checkpoint = Path(source_run['checkpoints']['gavit']['path'])
    model, identity = build_model('gavit', checkpoint, 'cuda')
    architecture = identity['metadata']['architecture']
    if identity['sha256'] != source_run['checkpoints']['gavit']['sha256']:
        raise ValueError('GAViT checkpoint differs from source diagnostic')
    if any(architecture.get(k) != v for k,v in dict(num_regions=16, grouping='attentive_spatial', edge_type='sparse_hybrid', integration='token_feedback').items()):
        raise ValueError('This probe only supports the approved K16 attentive sparse-hybrid token-feedback model')
    info = dict(checkpoint=identity, device=torch.cuda.get_device_name(0), torch=torch.__version__,
                cuda=torch.version.cuda, python=sys.version,
                packages={p:version(p) for p in ('torchvision','timm','torch-geometric','numpy','Pillow')},
                gat_internal_self_loops=[bool(g.add_self_loops) for g in model.graph_reasoning.gat_layers])
    save_json(out / 'loaded_model.json', info)
    print(f'DEVICE {info["device"]}; CASES {len(selected)}; INPUT {args.input_mode}', flush=True)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    records, cards = [], []
    torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    for position, item in enumerate(selected, 1):
        if args.input_mode == 'gallery':
            with Image.open(gallery_path(args.source, item['image'])) as im:
                picture = im.convert('RGB')
        else:
            picture = Image.fromarray(read_rgb_patch(item['patch_path']))
        picture = transforms.Resize((224,224))(picture)
        images = transform(picture).unsqueeze(0).to('cuda')
        normal, zero, captured, errors = probe(model, images)
        assignment = captured['assignments'][0].cpu().tolist()
        edges = captured['edge_index'].t().cpu().tolist()
        counts, kinds = validate_topology(assignment, edges)
        normal_scores, zero_scores = torch.sigmoid(normal)[0].cpu().tolist(), torch.sigmoid(zero)[0].cpu().tolist()
        cache_error = max(abs(a-b) for a,b in zip(normal_scores, item['cached_gavit_scores']))
        if args.input_mode == 'original' and cache_error > 1e-4:
            raise ValueError(f'Original-image prediction differs from full-val cache: {cache_error}')
        norms = captured['feedback'][0].norm(dim=-1).cpu().tolist()
        row = dict(item, normal_scores=normal_scores, zero_scores=zero_scores,
                   logit_delta=(normal-zero)[0].cpu().tolist(), gate_max_abs_errors=errors,
                   cached_prediction_max_abs_error=cache_error, assignments=assignment,
                   region_token_counts=counts, edges=edges, edge_kinds=kinds, feedback_L2=norms)
        records.append(row)
        stem = f'row_{item["row_index"]}'
        picture.save(out / f'{stem}.png')
        svg = graph_svg(assignment, edges, kinds, norms)
        (out / f'{stem}.svg').write_text(svg)
        torch.save({k:v.detach().cpu() for k,v in captured.items()}, out / f'{stem}.pt')
        cells = []
        for case in item['strata']:
            c = case['class_index']
            cells.append(f'class {c}, y={case["label"]}: cached={item["cached_gavit_scores"][c]:.4f}, normal={normal_scores[c]:.4f}, zero={zero_scores[c]:.4f}, delta_logit={row["logit_delta"][c]:+.4f}')
        detail = html.escape('\n'.join(cells))
        cards.append(f'<article><h3>{stem}</h3><div class="scene"><img src="{stem}.png"><object type="image/svg+xml" data="{stem}.svg"></object></div><pre>{detail}</pre></article>')
        save_json(out / f'{stem}.json', row)
        print(f'CASE {position}/{len(selected)} {stem}: GATES PASS; 48 spatial + 32 feature; ' + ' / '.join(cells), flush=True)
    summary = summarize(records)
    save_json(out / 'summary.json', dict(scope='selected_validation_case_intervention_not_population_metrics', input_mode=args.input_mode, classes=summary))
    page = '<!doctype html><meta charset="utf-8"><title>GAViT feedback probe</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{padding:16px;border:1px solid #ddd;margin:16px 0}.scene{position:relative;width:448px;height:448px}.scene img,.scene object{position:absolute;width:100%;height:100%}.scene object{pointer-events:none}pre{white-space:pre-wrap}</style>'
    page += '<h1>GAViT feedback probe</h1><p>Blue: 48 spatial input edges. Orange: 32 feature input edges. Arrows: source to target. GAT internal self-loops are recorded in loaded_model.json, not drawn. Cosine selection values are not GAT attention.</p>'
    page += f'<p>Input: {html.escape(args.input_mode)}. Gallery JPEG inputs differ from original RGB evaluation. Selected cases are not representative; no population AP/F1 is reported. Zero feedback is an intervention on a jointly trained model, not a trained no-GNN baseline.</p>'
    (out / 'cases.html').write_text(page + ''.join(cards))
    report = ['# GAViT fixed-case feedback probe', f'Input: {args.input_mode}; unique images: {len(records)}.',
              'All capture/reconstruction/zero-branch/restoration gates passed.',
              f'Max probability difference from original cached predictions: {max(r["cached_prediction_max_abs_error"] for r in records):.6f}.',
              'Counts describe selected class/label/outcome cases only; they are not population metrics.',
              'Gallery JPEG is a different input from original RGB. Zero feedback is not a trained baseline.',
              'Class 11: grassland; 14: beaches; 16: coastal wetlands. Threshold: 0.5.',
              '| Class | N | Feedback corrects | Feedback breaks | Both correct | Both wrong |', '|---|---:|---:|---:|---:|---:|']
    for c, r in sorted(summary.items(), key=lambda p:int(p[0])):
        report.append(f'| {c} | {r["n"]} | {r["feedback_corrects"]} | {r["feedback_breaks"]} | {r["unchanged_correct"]} | {r["unchanged_wrong"]} |')
    report += ['', 'Per-case JSON/PT, input PNG and captured topology SVG are retained. Open cases.html for inspection.']
    (out / 'report.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report), flush=True)
    return dict(images=len(records), probe_seconds=time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/home/featurize/work/pair44_fullval_8Tvk9d/diagnosis'))
    parser.add_argument('--repo', type=Path, default=Path('/home/featurize/work/GAViT_Project'))
    parser.add_argument('--pair-tools', type=Path, default=Path('/home/featurize/work/gavit-pair-diagnostic-20260915/tools'))
    parser.add_argument('--work', type=Path, default=Path('/home/featurize/work'))
    parser.add_argument('--out', type=Path)
    parser.add_argument('--input-mode', choices=['gallery','original'], default='gallery')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.out:
        args.out = args.out.resolve(); args.out.mkdir(parents=True, exist_ok=False)
    else:
        args.out = Path(tempfile.mkdtemp(prefix='pair44_feedback_', dir=args.work)).resolve()
    print(f'OUTPUT {args.out}', flush=True)
    try:
        source_run, selected = prepare(args)
        info = dict(started_at=datetime.now(timezone.utc).isoformat(), input_mode=args.input_mode,
                    scope='selected_validation_case_intervention_not_population_metrics',
                    diagnostic_sha256=digest(__file__),
                    source_run_sha256=digest(args.source / 'run.json'), source_cases_sha256=digest(args.source / 'cases.csv'),
                    source_predictions_sha256=digest(args.source / 'predictions.csv'), source_run=source_run,
                    selection='first existing case per class/outcome/true-label stratum; no selection by intervention effect',
                    cases=selected, argv=sys.argv)
        save_json(args.out / 'preflight.json', info)
        if not args.prepare_only:
            info['resources'] = run_diagnostic(args, args.out, source_run, selected)
        info.update(status='prepared_only' if args.prepare_only else 'complete', finished_at=datetime.now(timezone.utc).isoformat())
        save_json(args.out / 'run.json', info)
        print(f'FEEDBACK {info["status"].upper()}: {args.out}', flush=True)
        return 0
    except Exception as exc:
        save_json(args.out / 'failure.json', dict(error=str(exc), type=type(exc).__name__))
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
'''
script.write_text(source)
assert hashlib.sha256(script.read_bytes()).hexdigest() == "30990e5ad2678bc08d7717de94a1afa20c590001d9e4ea84810fb1d8096f403f", "Pasted code changed; do not run"
worker = work / "run_worker.py"
worker.write_text(chr(10).join([
    "from pathlib import Path",
    "import subprocess, sys",
    "rc = subprocess.call(sys.argv[1:])",
    "Path(__file__).with_name('diagnostic.exit').write_text(str(rc) + chr(10))",
    "print('DIAGNOSTIC_EXIT', rc, flush=True)",
    "raise SystemExit(rc)",
]))
command = [sys.executable, "-u", str(worker), sys.executable, "-u", str(script), "--out", str(work / "diagnosis"), "--input-mode", "gallery"]
with (work / "diagnostic.log").open("x") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
print("PID:", process.pid)
print("Output:", work)
print("Run this line to see progress:")
print("tail -f", work / "diagnostic.log")
print("Completion requires FEEDBACK COMPLETE and DIAGNOSTIC_EXIT 0.")
PY
