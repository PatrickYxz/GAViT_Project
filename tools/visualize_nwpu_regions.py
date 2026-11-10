"""Observe frozen NWPU calibration models on 2 lexicographic val images/class.

Standalone diagnostic: no training, graph reconstruction, or test-set selection.
Images may be read directly from the persistent ZIP, without extracting it.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import zipfile


MANIFEST_SHA = 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
CHECKPOINTS = {
    'fixed': '1766fe7c330eb37fb350f0348bcd255bafc451b793411a927548ee005a140f67',
    'soft': '674da671e1df1618add93cd18e791e0a81fdfb77ce0dc13478eb963bfa29cc12'}
RUN_HASHES = {
    'fixed': '4d10169e3618252f079bce85094750eb0477e6caa8ad9a37f13394eecfbb81d8',
    'soft': '9c1fa474d3f2e79203a1c245acfd15692a9a6e766acb71ef2566baff3d1bfd33'}
PALETTE = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
           '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
           '#9a6324', '#fffac8', '#800000', '#aaffc3']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_relative(path):
    p = PurePosixPath(path)
    if not path or p.is_absolute() or '..' in p.parts or '\\' in path or str(p) != path:
        raise ValueError(f'Unsafe image path: {path}')
    return p


def select_cases(manifest):
    names, rows = manifest['class_names'], manifest['records']
    paths = [r['path'] for r in rows]
    if len(paths) != len(set(paths)):
        raise ValueError('duplicate image paths')
    selected = []
    for label, name in enumerate(names):
        candidates = sorted((r for r in rows if r['split'] == 'val' and r['label'] == label),
                            key=lambda r: r['path'])
        if len(candidates) < 2:
            raise ValueError(f'Need at least two validation images for {name}')
        for row in candidates[:2]:
            if safe_relative(row['path']).parts[0] != name:
                raise ValueError('Image class/path mismatch')
        selected.extend(candidates[:2])
    return selected


class ImageSource:
    def __init__(self, root, archive):
        self.root, self.archive, self.zip = Path(root), Path(archive), None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.zip is not None:
            self.zip.close()

    def read(self, row):
        from PIL import Image
        relative = safe_relative(row['path'])
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError('Image path escapes data root')
        if path.exists():
            raw, location = path.read_bytes(), 'original_directory'
        else:
            if self.zip is None:
                self.zip = zipfile.ZipFile(self.archive)
            member = 'NWPU-RESISC45/' + str(relative)
            if self.zip.namelist().count(member) != 1:
                raise ValueError(f'Missing or duplicate ZIP member: {member}')
            raw, location = self.zip.read(member), 'zip'
        if hashlib.sha256(raw).hexdigest() != row['sha256']:
            raise ValueError(f'File hash mismatch: {relative}')
        with Image.open(io.BytesIO(raw)) as source:
            source.load()
            if source.mode != 'RGB' or source.size != (256, 256):
                raise ValueError(f'Not an original NWPU RGB image: {relative}')
            image = source.copy()
        pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        if pixels != row['pixel_sha256']:
            raise ValueError(f'Pixel hash mismatch: {relative}')
        return image, dict(path=str(relative), source=location, sha256=row['sha256'], pixel_sha256=pixels)


def validate_sources(args):
    if sha(args.manifest) != MANIFEST_SHA:
        raise ValueError('Frozen manifest hash mismatch')
    manifest = json.loads(args.manifest.read_text())
    if (manifest['dataset'] != 'NWPU-RESISC45' or len(manifest['class_names']) != 45
            or Counter(r['split'] for r in manifest['records']) != dict(train=5040, val=1260, test=25200)):
        raise ValueError('Unexpected NWPU manifest')
    selected = select_cases(manifest)
    commit = subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != COMMIT:
        raise ValueError('Frozen model repository commit mismatch')
    reports, cached, identities = {}, {}, {}
    val_paths = {r['path'] for r in manifest['records'] if r['split'] == 'val'}
    labels = {r['path']: r['label'] for r in manifest['records'] if r['split'] == 'val'}
    for name, grouping in [('fixed', 'attentive_spatial'), ('soft', 'soft_prototype')]:
        folder = args.calibration / name
        report_path, ckpt = folder / 'run.json', folder / 'best.pth'
        if sha(report_path) != RUN_HASHES[name] or sha(ckpt) != CHECKPOINTS[name]:
            raise ValueError(f'Frozen {name} calibration source hash mismatch')
        report = json.loads(report_path.read_text())
        if (report['status'] != 'complete' or report['phase'] != 'calibrate' or report['seed'] != 42
                or report['model'] != 'gavit' or report['dataset'] != 'NWPU-RESISC45'
                or report['architecture']['grouping'] != grouping
                or report['manifest_sha256'] != MANIFEST_SHA
                or report['checkpoint_sha256'] != CHECKPOINTS[name]
                or (folder / 'run.exit').read_text().strip() != 'exit_code=0'):
            raise ValueError(f'Invalid calibration identity: {name}')
        for relative, expected in report['code']['files_sha256'].items():
            safe_relative(relative)
            if sha(args.repo / relative) != expected:
                raise ValueError(f'Model source changed: {relative}')
        prediction_path = folder / 'validation_predictions.csv'
        with prediction_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        table = {r['path']: r for r in rows}
        if (len(rows) != len(table) or table.keys() != val_paths
                or any(int(r['label']) != labels[r['path']] for r in rows)):
            raise ValueError(f'Cached validation path/label mismatch: {name}')
        reports[name], cached[name] = report, table
        identities[name] = dict(run_sha256=RUN_HASHES[name], checkpoint_sha256=CHECKPOINTS[name],
                                cached_predictions_sha256=sha(prediction_path), best_epoch=report['best_epoch'])
    for key in ('training', 'code', 'manifest_sha256', 'pretrained_sha256'):
        if reports['fixed'][key] != reports['soft'][key]:
            raise ValueError(f'Paired calibration identity differs: {key}')
    return manifest, selected, reports, cached, identities


def validate_tensors(t):
    import torch
    s, w, edges = t['S'], t['W'], t['edge_index']
    if (s.shape != (49, 16) or w.shape != (49, 16)
            or not all(bool(torch.isfinite(x).all()) for x in t.values())
            or bool((s < 0).any()) or bool((w < 0).any())):
        raise ValueError('Invalid capture tensors')
    s_error, w_error = float((s.sum(-1) - 1).abs().max()), float((w.sum(0) - 1).abs().max())
    if max(s_error, w_error) > 1e-5:
        raise ValueError('Assignment normalization mismatch')
    if bool(s.masked_select(~t['candidate_mask']).any()) or bool(w.masked_select(~t['candidate_mask']).any()):
        raise ValueError('Assignment outside allowed candidates')
    if (edges.shape != (2, 80) or edges.dtype != torch.int64
            or int(edges.min()) < 0 or int(edges.max()) >= 16
            or torch.unique(edges, dim=1).shape[1] != 80):
        raise ValueError('Invalid captured edge count/indices')
    source, target = edges
    distance = (source // 4 - target // 4).abs() + (source % 4 - target % 4).abs()
    if (not bool((distance[:48] == 1).all()) or not bool((distance[48:] > 1).all())
            or not bool((torch.bincount(target[48:], minlength=16) == 2).all())):
        raise ValueError('Invalid captured spatial/feature edges')
    return dict(S_row_error=s_error, W_column_error=w_error, spatial_edges=48, feature_edges=32,
                gat_self_loops_not_drawn=True)


def capture(model, image):
    import torch
    if image.shape[0] != 1 or any(m.training for m in model.modules()):
        raise ValueError('Capture requires one image and eval model')
    soft = model.grouping == 'soft_prototype'
    group = model.region_grouping
    if model.grouping not in ('soft_prototype', 'attentive_spatial') or model.edge_type != 'sparse_hybrid':
        raise ValueError('Unsupported grouping/graph')
    tensors, handles = {}, []
    previous = getattr(group, 'assignment_observer', None)
    def save(key, value):
        tensors[key] = value.detach().clone()
    def observer(values):
        for key in ('S', 'W'):
            save(key, values[key])
    def grouping_hook(_m, _inputs, outputs):
        save('regions', outputs[0])
        if not soft:
            save('S', torch.nn.functional.one_hot(outputs[1], 16).to(outputs[0].dtype))
    def graph_hook(_m, inputs):
        save('nodes', inputs[0])
        save('edge_index', inputs[1])
    devices = [image.device.index or 0] if image.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        normal = model(image)
        try:
            if soft:
                group.assignment_observer = observer
            handles.append(model.backbone.register_forward_hook(lambda m, a, o: save('tokens', o)))
            handles.append(group.register_forward_hook(grouping_hook))
            handles.append(group.attn.register_forward_hook(lambda m, a, o: save('scores', o)))
            handles.append(model.graph_reasoning.register_forward_pre_hook(graph_hook))
            handles.append(model.feedback_proj.register_forward_hook(lambda m, a, o: save('feedback', o)))
            observed = model(image)
        finally:
            for handle in handles:
                handle.remove()
            if soft:
                group.assignment_observer = previous
        if not soft:
            tensors['W'] = tensors['scores'].expand(-1, -1, 16).masked_fill(group.membership[None] == 0, -torch.inf).softmax(1)
        tokens, s, w = tensors['tokens'], tensors['S'], tensors['W']
        pooled_regions = torch.einsum('bnk,bnd->bkd', w, tokens)
        torch.testing.assert_close(pooled_regions, tensors['regions'], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tensors['nodes'], tensors['regions'].reshape(16, -1), atol=0, rtol=0)
        reconstructed = model.classifier((tokens + torch.einsum('bnk,bkd->bnd', s, tensors['feedback'])).mean(1))
        restored = model(image)
        errors = {}
        for key, value in [('capture', observed), ('reconstruction', reconstructed), ('restoration', restored)]:
            torch.testing.assert_close(normal, value, atol=1e-5, rtol=1e-5)
            errors[key + '_max_logit_diff'] = float((normal - value).abs().max())
        result = dict(S=s[0], W=w[0], edge_index=tensors['edge_index'], logits=normal[0],
                      anchor_assignment=group.assignment,
                      candidate_mask=group.candidate_mask if soft else group.membership.bool())
        result = {k: v.detach().cpu().clone() for k, v in result.items()}
        report = validate_tensors(result)
        return result, {**report, **errors}


def check_cached(logits, cached, label):
    import torch
    if int(cached['label']) != label:
        raise ValueError('Cached label mismatch')
    reference = torch.tensor([float(cached[f'p_{i}']) for i in range(len(logits))])
    error = float((logits.softmax(0) - reference).abs().max())
    if not math.isfinite(error) or error > 1e-5 or int(logits.argmax()) != int(cached['prediction']):
        raise ValueError(f'Prediction differs from cached validation: max_probability_error={error}')
    return error


def png_uri(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()


def resized_input(image):
    from PIL import Image
    # Match the PIL bilinear Resize used by image_transform(False).
    return image.resize((224, 224), resample=Image.Resampling.BILINEAR)


def text(x, y, value, size=12):
    return f'<text x="{x}" y="{y}" font-size="{size}">{html.escape(str(value))}</text>'


def grid(values, color, opacity):
    return ''.join(f'<rect x="{i % 7 * 32}" y="{i // 7 * 32}" width="32" height="32" '
                   f'fill="{color(v)}" fill-opacity="{opacity}" stroke="white" stroke-width="0.4"/>'
                   for i, v in enumerate(values))


def graph_panel(tensors):
    anchors = tensors['anchor_assignment'].tolist()
    centers = []
    for region in range(16):
        indices = [i for i, a in enumerate(anchors) if a == region]
        centers.append((sum(i % 7 * 32 + 16 for i in indices) / len(indices),
                        sum(i // 7 * 32 + 16 for i in indices) / len(indices)))
    parts = [grid(tensors['S'].argmax(-1).tolist(), lambda k: PALETTE[k], .40)]
    for i, (source, target) in enumerate(tensors['edge_index'].t().tolist()):
        kind = 'spatial' if i < 48 else 'feature'
        color = '#0066bb' if i < 48 else '#df6400'
        x1, y1 = centers[source]; x2, y2 = centers[target]
        length = math.hypot(x2 - x1, y2 - y1)
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        parts.append(f'<line class="edge {kind}" data-source="{source}" data-target="{target}" '
                     f'x1="{x1 + 6*ux}" y1="{y1 + 6*uy}" x2="{x2 - 8*ux}" y2="{y2 - 8*uy}" '
                     f'stroke="{color}" stroke-width="1.1" marker-end="url(#{kind})"/>')
    for k, (x, y) in enumerate(centers):
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="white" stroke="#333"/>')
        parts.append(f'<text x="{x}" y="{y+2.5}" text-anchor="middle" font-size="7">{k}</text>')
    return ''.join(parts)


def entropy(tensors):
    import torch
    s = tensors['S']
    h = -(s * s.clamp_min(1e-30).log()).sum(-1)
    maximum = tensors['candidate_mask'].sum(-1).float().log()
    return torch.where(maximum > 0, h / maximum.clamp_min(1e-10), torch.zeros_like(h)).clamp(0, 1)


def heat_color(value):
    # Sequential white-to-purple, fixed [0,1] scale (no per-image autoscaling).
    return '#%02x%02x%02x' % tuple(round(255 + float(value) * (v - 255)) for v in (70, 30, 140))


def case_svg(image, fixed, soft, path, truth, fixed_prediction, soft_prediction):
    uri = png_uri(resized_input(image))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="410" viewBox="0 0 1024 410">',
             '<rect width="1024" height="410" fill="white"/><g font-family="sans-serif" fill="#202830">',
             '<defs>']
    for kind, color in [('spatial', '#0066bb'), ('feature', '#df6400')]:
        parts.append(f'<marker id="{kind}" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">'
                     f'<path d="M0,0 L5,2.5 L0,5" fill="{color}"/></marker>')
    parts += ['</defs>', text(16, 24, path, 17), text(16, 46, 'Validation case | label: ' + truth)]
    headings = ['Original model input', 'Fixed partition + graph', 'Soft argmax + graph', 'Soft assignment entropy']
    for i, heading in enumerate(headings):
        x = 16 + 256 * i
        parts.append(text(x, 72, heading, 14))
        subtitle = ['RGB / 224 x 224', 'Prediction: ' + fixed_prediction,
                    'Prediction: ' + soft_prediction, '0: concentrated; 1: diffuse'][i]
        parts.append(text(x, 91, subtitle, 11))
        parts.append(f'<g transform="translate({x},104)"><image width="224" height="224" href="{uri}"/>')
        if i in (1, 2):
            parts.append(graph_panel(fixed if i == 1 else soft))
        elif i == 3:
            parts.append(grid(entropy(soft).tolist(), heat_color, .88))
        parts.append('</g>')
    parts += [text(16, 350, 'Blue: 48 spatial edges; orange: 32 feature edges. Arrows: source -> target; not GAT attention.'),
              text(16, 370, 'Nodes use fixed anchor positions. GAT self-loops omitted. Soft argmax is not a pixel segmentation.'),
              text(16, 390, 'Entropy = H(S) / log(number of local candidates); assignment ambiguity, not prediction uncertainty.'),
              '</g></svg>']
    return ''.join(parts)


def membership_svg(image, soft):
    uri = png_uri(resized_input(image))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="356" viewBox="0 0 1024 356">',
             '<rect width="1024" height="356" fill="white"/><g font-family="sans-serif">',
             text(12, 24, 'Full soft membership S: 16 regions, shared white (0) to purple (1) scale', 16)]
    for region in range(16):
        x, y = 12 + region % 8 * 127, 48 + region // 8 * 144
        parts.append(text(x, y - 5, f'Region {region}'))
        parts.append(f'<g transform="translate({x},{y}) scale(.5)"><image width="224" height="224" href="{uri}"/>')
        parts.append(grid(soft['S'][:, region].tolist(), heat_color, .88) + '</g>')
    parts += [text(12, 344, 'S sums to 1 over regions per token. Pooling weights W sum over tokens and are saved separately in the PT file.'), '</g></svg>']
    return ''.join(parts)


def gallery(records):
    cards = []
    for row in records:
        svg = base64.b64encode(row['svg'].encode()).decode()
        maps = base64.b64encode(row['membership_svg'].encode()).decode()
        cards.append(f'<article><img alt="{html.escape(row["path"], quote=True)}" src="data:image/svg+xml;base64,{svg}">'
                     f'<details><summary>All 16 soft memberships</summary><img src="data:image/svg+xml;base64,{maps}" alt="Soft memberships"></details></article>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>NWPU region comparison</title>'
            '<style>body{font:16px system-ui;margin:24px auto;max-width:1100px;background:#edf0f4;color:#202830}'
            'article{background:white;margin:24px 0;padding:12px;border-radius:8px}img{width:100%;height:auto}'
            'summary{padding:12px;cursor:pointer}p{line-height:1.6}</style><h1>NWPU: fixed and soft regions</h1>'
            '<p>90 validation images: first two paths in every class. Seed42 calibration-best models. '
            'All cases retained, regardless of prediction. No test selection, retraining or causal intervention. '
            'Colors show region IDs, not semantic labels; anchor adjacency is a prior, not a claim of learned boundaries.</p>'
            + ''.join(cards) + '</html>')


def run(args, out):
    manifest, selected, reports, cached, identities = validate_sources(args)
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from scene_classification.modeling import build_model, image_transform
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no automatic CPU fallback')
    hardware = dict(device=str(device), name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                    python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                    amp=False, matmul_tf32=False, cudnn_tf32=False)
    print('SOURCE PASS: calibration best, fixed/soft seed42; 90 val images', flush=True)
    print('DEVICE ' + json.dumps(hardware), flush=True)
    names, tf = manifest['class_names'], image_transform(False)
    (out / 'cases').mkdir()
    write_json(out / 'selection.json', dict(records=selected, checkpoint_sources=identities,
                                            selection='val_first_two_lexicographic_paths_per_class_v1'))
    images, input_info = [], []
    with ImageSource(args.data_root, args.archive) as source:
        for i, row in enumerate(selected):
            image, info = source.read(row)
            # Persist verified pixels before any forward/cache gate can fail.
            image.save(out / 'cases' / f'{i:03d}_original.png')
            images.append(image)
            input_info.append(info)
    inputs = [tf(im).unsqueeze(0) for im in images]
    identity = dict(manifest_sha256=MANIFEST_SHA, checkpoint_sources=identities, model_code=reports['fixed']['code'],
                    tool_sha256=sha(__file__), selection='val_first_two_lexicographic_paths_per_class_v1',
                    sample_ids=[r['path'] for r in selected], inputs=input_info, hardware=hardware,
                    input_tensor_sha256=[hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in inputs])
    write_json(out / 'identity.json', identity)
    for name in ('fixed', 'soft'):
        model = build_model('gavit', 45, architecture_config=reports[name]['architecture']).to(device).eval()
        checkpoint = args.calibration / name / 'best.pth'
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        for i, (row, x) in enumerate(zip(selected, inputs)):
            tensors, checks = capture(model, x.to(device))
            path = out / 'cases' / f'{i:03d}_{name}.pt'
            payload = dict(path=row['path'], label=row['label'], source=identities[name],
                           tensors=tensors, checks=checks, cached_check_status='pending')
            torch.save(payload, path)
            try:
                checks['cached_probability_max_error'] = check_cached(tensors['logits'], cached[name][row['path']], row['label'])
            except Exception as exc:
                payload.update(cached_check_status='failed', error=f'{type(exc).__name__}: {exc}')
                torch.save(payload, path)
                raise
            payload['cached_check_status'] = 'passed'
            torch.save(payload, path)
            print(f'CAPTURE {name} {i+1}/90 {row["path"]}: PASS', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    records, checks = [], []
    for i, (row, image) in enumerate(zip(selected, images)):
        pair = {name: torch.load(out / 'cases' / f'{i:03d}_{name}.pt', map_location='cpu', weights_only=True)
                for name in ('fixed', 'soft')}
        fixed, soft = (pair[n]['tensors'] for n in ('fixed', 'soft'))
        predictions = {name: int(pair[name]['tensors']['logits'].argmax()) for name in pair}
        svg = case_svg(image, fixed, soft, row['path'], names[row['label']], names[predictions['fixed']], names[predictions['soft']])
        maps = membership_svg(image, soft)
        (out / 'cases' / f'{i:03d}.svg').write_text(svg)
        (out / 'cases' / f'{i:03d}_memberships.svg').write_text(maps)
        record = dict(path=row['path'], label=row['label'], predictions=predictions,
                      checks={name: pair[name]['checks'] for name in pair})
        write_json(out / 'cases' / f'{i:03d}.json', record)
        records.append(dict(**record, svg=svg, membership_svg=maps))
        checks.extend(record['checks'].values())
    (out / 'cases.html').write_text(gallery(records))
    correct = {name: sum(r['predictions'][name] == r['label'] for r in records) for name in ('fixed', 'soft')}
    paired = Counter((r['predictions']['fixed'] == r['label'], r['predictions']['soft'] == r['label']) for r in records)
    report = dict(status='complete', samples=90, classes=45, split='val', source_phase='calibrate',
                  selection=identity['selection'], correct=correct,
                  paired=dict(both_correct=paired[True, True], soft_corrects=paired[False, True],
                              soft_breaks=paired[True, False], both_wrong=paired[False, False]),
                  max_errors={key: max(c[key] for c in checks) for key in
                              ('capture_max_logit_diff', 'reconstruction_max_logit_diff', 'restoration_max_logit_diff',
                               'cached_probability_max_error', 'S_row_error', 'W_column_error')},
                  all_capture_checks_passed=True, spatial_edges=48, feature_edges=32,
                  input_sources=dict(Counter(r['source'] for r in input_info)),
                  scope='Fixed selected validation cases only; not population metrics or causal evidence',
                  html_bytes=(out / 'cases.html').stat().st_size)
    # Durable original pixels and raw tensors permit later rendering without data restoration.
    with zipfile.ZipFile(out / 'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [out / 'identity.json', out / 'selection.json', *sorted((out / 'cases').iterdir())]:
            archive.write(path, path.relative_to(out))
    report['evidence_zip_sha256'] = sha(out / 'evidence.zip')
    report['evidence_zip_bytes'] = (out / 'evidence.zip').stat().st_size
    return report


def execute(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    try:
        report = run(args, out)
        report['elapsed_seconds'] = time.perf_counter() - start
        write_json(out / 'summary.json', report)
        (out / 'run.exit').write_text('exit_code=0\n')
        print('NWPU REGIONS COMPLETE ' + str(out), flush=True)
        return report
    except BaseException as exc:
        write_json(out / 'summary.json', dict(status='failed', error=f'{type(exc).__name__}: {exc}',
                                             elapsed_seconds=time.perf_counter() - start))
        (out / 'run.exit').write_text('exit_code=1\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path('/home/featurize/work/GAViT_SOFT_20260927'))
    p.add_argument('--calibration', type=Path, default=Path('/home/featurize/work/nwpu_ablation42_20260928_01'))
    p.add_argument('--manifest', type=Path, default=Path('/home/featurize/work/nwpu_protocol_20260927_02.json'))
    p.add_argument('--data-root', type=Path, default=Path('/home/featurize/data/NWPU-RESISC45'))
    p.add_argument('--archive', type=Path, default=Path('/home/featurize/work/NWPU-RESISC45_a826b44.zip'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    execute(p.parse_args())


if __name__ == '__main__':
    main()
