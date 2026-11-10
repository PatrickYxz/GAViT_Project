"""Observe trained BEN K9/K16 on four archived original RGB inputs; never train.

Imports needed for tensor inference are deliberately deferred until the artifact
gate has passed. Installation, validation and rendering also run without torch.
"""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def partition(k):
    require(type(k) is int and k in (9,16), 'Expected K9 or K16')
    side = math.isqrt(k)
    return [(r*side//7)*side + c*side//7 for r in range(7) for c in range(7)]


def validate_topology(assignments, edges, k):
    require(assignments == partition(k), 'Captured fixed membership differs from position rule')
    side = math.isqrt(k)
    require(len(edges) == (42 if k == 9 else 80), 'Wrong pre-GAT edge budget')
    require(all(len(e) == 2 and all(type(v) is int and 0 <= v < k for v in e)
                and e[0] != e[1] for e in edges), 'Invalid edge endpoint')
    pairs = [tuple(e) for e in edges]
    require(len(set(pairs)) == len(pairs), 'Duplicate pre-GAT edge')
    distance = lambda s,t: abs(s//side-t//side) + abs(s%side-t%side)
    expected = {(s,t) for s in range(k) for t in range(k) if distance(s,t) == 1}
    spatial = {(s,t) for s,t in pairs if distance(s,t) == 1}
    kinds = ['spatial' if distance(s,t) == 1 else 'feature' for s,t in pairs]
    require(spatial == expected, 'Missing four-neighbor input edges')
    require(all(sum(t == q and kind == 'feature' for (s,t),kind in zip(pairs,kinds)) == 2
                for q in range(k)), 'Expected two incoming feature edges per query')
    return kinds


def validate_pooling(assignments, weights, k):
    require(assignments == partition(k) and len(weights) == 49, 'Wrong pooling dimensions')
    require(all(len(row) == k for row in weights), 'Wrong pooling columns')
    require(all(isinstance(v,(int,float)) and math.isfinite(v) and 0 <= v <= 1
                and (assignments[i] == r or v == 0)
                for i,row in enumerate(weights) for r,v in enumerate(row)), 'Invalid pooling membership or mass')
    require(all(abs(sum(row[r] for row in weights)-1) <= 1e-5 for r in range(k)), 'Non-unit pooling mass')


def compare_predictions(labels, p9, p16, classes):
    require(len(classes) == len(set(classes)) == len(labels) == len(p9) == len(p16) == 19,
            'Expected all 19 classes')
    require(all(v in (0,1) for v in labels), 'Labels must be binary')
    require(all(isinstance(p,(int,float)) and math.isfinite(p) and 0 <= p <= 1 for p in p9+p16),
            'Invalid class probabilities')
    result = []
    for i,(label,a,b) in enumerate(zip(labels,p9,p16)):
        aok, bok = (a >= .5) == bool(label), (b >= .5) == bool(label)
        outcome = ('both_correct' if aok and bok else 'both_wrong' if not aok and not bok
                   else 'k9_only_correct' if aok else 'k16_only_correct')
        result.append(dict(class_index=i, name=classes[i], label=label, p9=a, p16=b,
                           delta_probability=b-a, outcome=outcome))
    return result


def validate_summary(s, k9_sha, k16_sha, source_sha):
    require(s['status'] == 'complete' and s['dataset'] == 'BigEarthNet-19' and s['seed'] == 42,
            'K9 queue summary is incomplete or wrong seed')
    stages = s['stages']
    for name in ('smoke','train','test'):
        row = stages[name]
        require(row['status'] == 'complete' and row['seed'] == 42
                and row['source']['snapshot_sha256'] == source_sha, 'Wrong stage identity: '+name)
    require(stages['smoke']['final_epoch'] == 1 and
            math.isfinite(stages['smoke']['reload_max_logit_diff']) and
            0 <= stages['smoke']['reload_max_logit_diff'] <= 1e-5, 'Smoke reload gate failed')
    require(stages['train']['final_epoch'] == 30 and stages['train']['counts'] == dict(train=237871,val=122342),
            'Incomplete formal training')
    for name in ('train','test'):
        require(stages[name]['checkpoint_sha256'] == k9_sha, 'Formal best/test weights disagree')
    require(stages['test']['metrics']['checkpoint_sha256'] == k9_sha and
            stages['test']['metrics']['samples'] == 119825, 'Test metrics identity differs')
    require(s['control']['checkpoint_sha256'] == k16_sha, 'Historical K16 reference differs')


def install(payload, target):
    files = payload['files']
    require(set(files) == set(payload['sha256']) == {'ben_grid_pair_visual.py','config.json'},
            'Unexpected release paths')
    for name,text in files.items():
        require(sha_bytes(text.encode()) == payload['sha256'][name], 'Modified release: '+name)
    compile(files['ben_grid_pair_visual.py'], 'ben_grid_pair_visual.py', 'exec')
    json.loads(files['config.json'])
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    for name,text in files.items():
        (target/name).write_text(text)
    save(target/'release.json', payload['sha256'])
    return target


def svg_overlay(record, png, offset, prefix):
    k = record['num_regions']; a = record['assignments']
    kinds = validate_topology(a,record['edges'],k)
    require(kinds == record['edge_kinds'], 'Rendered edge kinds differ from captured edges')
    validate_pooling(a,record['pooling_weights'],k)
    colors = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#46f0f0','#f032e6',
              '#bcf60c','#fabebe','#008080','#e6beff','#9a6324','#fffac8','#800000','#aaffc3']
    parts = [f'<g transform="translate({offset},40)"><image width="224" height="224" href="{png}"/>']
    centers = []
    for r in range(k):
        cells = [i for i,v in enumerate(a) if v == r]
        xs,ys = [i%7*32 for i in cells],[i//7*32 for i in cells]
        x,y = min(xs),min(ys); w,h = max(xs)-x+32,max(ys)-y+32
        centers.append((x+w/2,y+h/2))
        parts.append(f'<rect class="regions" x="{x}" y="{y}" width="{w}" height="{h}" '
                     f'fill="{colors[r]}" fill-opacity=".12" stroke="white" stroke-width=".7"/>')
    for i,region in enumerate(a):
        mass = record['pooling_weights'][i][region]
        ratio = mass*a.count(region)
        color = '#dc2626' if ratio >= 1 else '#2563eb'
        alpha = min(.75, abs(ratio-1)*.35)
        parts.append(f'<rect class="pooling" x="{i%7*32}" y="{i//7*32}" width="32" height="32" '
                     f'fill="{color}" fill-opacity="{alpha:.5f}" stroke="white" stroke-width=".3">'
                     f'<title>token {i}; region {region}; pooling mass {mass:.6f}; relative to uniform {ratio:.3f}</title></rect>')
    for (s,t),kind in zip(record['edges'],kinds):
        x1,y1 = centers[s]; x2,y2 = centers[t]
        dx,dy = x2-x1,y2-y1; scale = 7/math.hypot(dx,dy)
        x1,y1,x2,y2 = x1+dx*scale,y1+dy*scale,x2-dx*scale,y2-dy*scale
        mx,my = (x1+x2)/2+(y2-y1)*.06,(y1+y2)/2-(x2-x1)*.06
        color = '#2563eb' if kind == 'spatial' else '#f97316'
        parts.append(f'<path class="{kind}" data-k="{k}" data-source="{s}" data-target="{t}" '
                     f'd="M{x1:.3f},{y1:.3f} Q{mx:.3f},{my:.3f} {x2:.3f},{y2:.3f}" fill="none" '
                     f'stroke="{color}" stroke-width=".9" opacity=".85" marker-end="url(#{prefix}_{kind})">'
                     f'<title>{kind}: region {s} to query {t}</title></path>')
    for r,(x,y) in enumerate(centers):
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#fff" stroke="#374151" stroke-width=".6"/>'
                     f'<text x="{x}" y="{y+2.5}" font-size="7" text-anchor="middle">{r}</text>')
    parts.append('</g>')
    return ''.join(parts)


def render_svg(case, png_bytes, fixture=False):
    png = 'data:image/png;base64,'+base64.b64encode(png_bytes).decode()
    prefix = f'row{case["row_index"]}'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="290" viewBox="0 0 760 290">',
             '<style>.spatial,.pooling{display:none}text{font-family:Arial,sans-serif;fill:#15232b}</style>',
             '<rect width="760" height="290" fill="white"/><defs>']
    for kind,color in (('spatial','#2563eb'),('feature','#f97316')):
        parts.append(f'<marker id="{prefix}_{kind}" markerWidth="5" markerHeight="5" refX="4.5" '
                     f'refY="2.5" orient="auto"><path d="M0,0 L5,2.5 L0,5" fill="{color}"/></marker>')
    parts.append('</defs>')
    for x,title in ((24,'RGB input'),(268,'3 x 3 / K=9'),(512,'4 x 4 / K=16')):
        parts.append(f'<text x="{x+112}" y="25" text-anchor="middle" font-size="13" font-weight="bold">{title}</text>')
    parts.append(f'<image x="24" y="40" width="224" height="224" href="{png}"/>')
    parts.append(svg_overlay(case['models']['9'],png,268,prefix))
    parts.append(svg_overlay(case['models']['16'],png,512,prefix))
    caption = 'FIXTURE: synthetic edges and scores; layout QA only' if fixture else 'Captured feature edges; arrows: neighbor to query. Spatial edges and pooling available in HTML.'
    parts.append(f'<text x="24" y="283" font-size="9">{html.escape(caption)}</text></svg>')
    return ''.join(parts)


def render_html(records, classes, pngs, evidence, fixture=False):
    cards = []
    names = {'k16_only_correct':'4x4 corrects', 'k9_only_correct':'4x4 breaks',
             'both_correct':'both correct', 'both_wrong':'both wrong'}
    for case,png in zip(records,pngs):
        rows = compare_predictions(case['labels'],case['models']['9']['probabilities'],
                                   case['models']['16']['probabilities'],classes)
        truth = ', '.join(classes[i] for i,v in enumerate(case['labels']) if v)
        table = ''.join(f'<tr class="{r["outcome"]}"><td>{html.escape(r["name"])}</td><td>{r["label"]}</td>'
                        f'<td>{r["p9"]*100:.2f}%</td><td>{r["p16"]*100:.2f}%</td>'
                        f'<td>{r["delta_probability"]*100:+.2f} pp</td><td>{names[r["outcome"]]}</td></tr>' for r in rows)
        flips = [r for r in rows if r['outcome'] in ('k16_only_correct','k9_only_correct')]
        changed = '; '.join(f'{r["name"]}: {names[r["outcome"]]}' for r in flips) or 'No label crosses the 0.5 threshold.'
        cards.append(f'<article><h2>Validation row {case["row_index"]}</h2><p><b>Labels:</b> {html.escape(truth)}</p>'
                     + render_svg(case,png,fixture) + f'<p>{html.escape(changed)}</p><details><summary>All 19 class probabilities</summary>'
                     '<table><thead><tr><th>Class</th><th>Label</th><th>3x3</th><th>4x4</th><th>4x4 minus 3x3</th><th>At 0.5</th></tr></thead>'
                     f'<tbody>{table}</tbody></table></details></article>')
    data = json.dumps(dict(evidence=evidence,classes=classes,cases=records),ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    title = 'LAYOUT FIXTURE — no model results' if fixture else 'BigEarthNet: trained 3x3 / 4x4, seed42'
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>BEN grid comparison</title>
<style>body{font:15px/1.5 system-ui;color:#15232b;max-width:1120px;margin:30px auto;padding:0 18px;background:#f6f8fa}
h1{font-size:26px}h2{font-size:19px}article{background:white;border:1px solid #d9e1e7;border-radius:9px;padding:18px;margin:20px 0}
svg{display:block;width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:left}
.controls{position:sticky;top:0;background:#f6f8faee;padding:12px;border:1px solid #d9e1e7;z-index:1}.controls label{margin-right:18px;white-space:nowrap}
tr.k16_only_correct{background:#e4f5e9}tr.k9_only_correct{background:#fff0ed}summary{cursor:pointer;font-weight:600}p{overflow-wrap:anywhere}
@media print{.controls{display:none}article{break-inside:avoid}body{margin:0;background:white}details{display:block}}
</style><h1>'''+html.escape(title)+'''</h1><p>All four archived original-input validation cases are retained. This historical diagnostic subset does not estimate population accuracy, AP or F1. Both models are evaluated on the identical lossless Resize224 RGB input.</p>
<p>Orange: captured non-spatial cosine top-2 feature edges. Blue: four-neighbor spatial edges. Arrows show neighbor → query; GAT internal self-loops are not drawn. Edge colors are types, not semantic labels or GAT attention. Region colors identify fixed spatial bins.</p>
<div class="controls"><label><input type="checkbox" checked data-layer="regions">Region coverage</label><label><input type="checkbox" checked data-layer="feature">Feature edges</label><label><input type="checkbox" data-layer="spatial">Spatial edges</label><label><input type="checkbox" data-layer="pooling">Pooling weights</label></div>
<p>Pooling overlay: red = above uniform, blue = below uniform within that region; transparent = uniform. These weights summarize tokens within a region, not class attribution or directly comparable total mass across different K. Hover for exact values.</p>'''+''.join(cards)+'''<details><summary>Provenance and completion scope</summary><pre>'''+html.escape(json.dumps(evidence,indent=2,ensure_ascii=False))+'''</pre></details>
<script type="application/json" id="capture-data">'''+data+'''</script><script>
for (const control of document.querySelectorAll('[data-layer]')) {
 const update=()=>{for(const node of document.querySelectorAll('svg .'+control.dataset.layer)){node.style.display=control.checked?'':'none';if(control.checked)node.style.display='inline';}};
 control.addEventListener('change',update);update();
}
</script></html>'''


def preflight(root, cfg):
    training = Path(cfg['training_root']); code = training/'code'
    expected = cfg['snapshot']
    require(read(code/'snapshot.json') == expected, 'Frozen source manifest changed')
    for name,sha in expected['files'].items():
        require(digest(code/name) == sha, 'Frozen source changed: '+name)
    require(digest(code/'legacy_graph.py') == expected['original_files']['models/graph_construction.py'],
            'Legacy K16 reference graph changed')
    summary = read(training/'summary.json')
    validate_summary(summary,cfg['checkpoints']['9']['sha256'],cfg['checkpoints']['16']['sha256'],expected['snapshot_sha256'])
    for name in ('run.exit','smoke/run.exit','train/run.exit','test/run.exit'):
        require((training/name).read_text().strip() == 'exit_code=0', 'Nonzero or missing training exit: '+name)
    require(read(training/'test/metrics.json') == cfg['reported_metrics'], 'Reported test metrics changed')
    require(summary['stages']['test']['metrics'] == cfg['reported_metrics'], 'Summary/test metrics disagree')
    identities = {}
    for k,item in cfg['checkpoints'].items():
        path = Path(item['path']); require(digest(path) == item['sha256'], 'Weights changed: K'+k)
        meta = read(path.with_suffix('.meta.json'))
        require(meta['checkpoint_sha256'] == item['sha256'] and meta['architecture'] == dict(cfg['architecture'],num_regions=int(k))
                and meta['training']['seed'] == 42 and meta['training']['epochs'] == 30, 'Wrong trained model: K'+k)
        if k == '9':
            require(meta == summary['stages']['train']['metadata'] and meta['code'] == expected, 'K9 metadata/source mismatch')
        else:
            require(meta == summary['control']['metadata'] and meta['best']['epoch'] == 13, 'K16 reference identity mismatch')
        identities[k] = dict(path=str(path),sha256=item['sha256'],metadata=meta)
    require(digest(cfg['val_csv']) == cfg['val_csv_sha256'], 'Validation split bytes differ')
    selected = {r['row_index']:r for r in cfg['cases']}
    require(list(selected) == [97389,14680,28450,4439] and len(selected) == 4, 'Fixed four-case selection changed')
    with Path(cfg['val_csv']).open(newline='') as f:
        count = 0
        for count,row in enumerate(csv.DictReader(f),1):
            if count-1 in selected:
                item = selected[count-1]
                labels = [float(row[f'label_{i}']) for i in range(19)]
                require(labels == item['labels'] and row['patch_path'] == item['patch_path'], 'Validation row/label identity changed')
        require(count == 122342, 'Wrong validation count')
    from PIL import Image
    (root/'inputs').mkdir()
    for item in cfg['cases']:
        path = Path(item['png_path']); require(digest(path) == item['png_sha256'], 'Archived input missing/changed: '+str(path))
        with Image.open(path) as im:
            require(im.size == (224,224) and im.mode == 'RGB', 'Expected original-input Resize224 RGB PNG')
        shutil.copyfile(path,root/'inputs'/path.name)
    (root/'source_reports').mkdir()
    for name in ('summary.json','preflight.json','train/run.json','test/run.json','test/metrics.json','run.exit'):
        shutil.copyfile(training/name,root/'source_reports'/name.replace('/','_'))
    save(root/'source_reports/loaded_checkpoints.json',identities)
    save(root/'artifact_gate.json',dict(status='pass',checkpoint_identities=identities,
         source_snapshot=expected,summary=summary,input_cases=cfg['cases'],selection=cfg['selection']))
    print('ARTIFACT_GATE_PASS: completed30 / exact two weights / frozen source / 4 original PNGs / val labels',flush=True)
    return identities


def capture(model, images):
    import torch
    require(all(not m.training for m in model.modules()), 'Capture needs eval mode')
    captured,handles = {},[]
    def out(name):
        def hook(module,inputs,output):
            captured[name] = output.detach().clone()
        return hook
    def grouping(module,inputs,output):
        captured['regions'],captured['assignments'] = [v.detach().clone() for v in output]
    def graph_input(module,inputs):
        for name,value in zip(('nodes','edge_index','cosine_or_unit_values'),inputs):
            captured[name] = value.detach().clone()
    with torch.inference_mode():
        normal = model(images)
        try:
            handles.append(model.backbone.register_forward_hook(out('tokens')))
            handles.append(model.region_grouping.attn.register_forward_hook(out('pooling_scores')))
            handles.append(model.region_grouping.register_forward_hook(grouping))
            handles.append(model.graph_reasoning.register_forward_pre_hook(graph_input))
            handles.append(model.graph_reasoning.register_forward_hook(out('refined')))
            handles.append(model.feedback_proj.register_forward_hook(out('feedback')))
            observed = model(images)
        finally:
            for handle in handles:
                handle.remove()
        restored = model(images)
        mask = model.region_grouping.membership.unsqueeze(0)
        scores = captured['pooling_scores'].expand(-1,-1,model.num_regions).masked_fill(mask == 0,float('-inf'))
        weights = torch.softmax(scores,dim=1).masked_fill(mask == 0,0.)
        pooled = torch.einsum('bnk,bnd->bkd',weights,captured['tokens'])
        feedback = captured['feedback'].gather(1,captured['assignments'].unsqueeze(-1).expand(-1,-1,768))
        reconstruction = model.classifier((captured['tokens']+feedback).mean(1))
        errors = {}
        for name,left,right in (('capture',normal,observed),('restoration',normal,restored),
                                ('pooling',captured['regions'],pooled),('feedback',observed,reconstruction)):
            require(bool(torch.isfinite(left).all() and torch.isfinite(right).all()), 'Nonfinite '+name)
            errors[name] = float((left-right).abs().max())
            require(0 <= errors[name] <= 1e-5, 'Capture/reconstruction changed computation: '+name)
        torch.testing.assert_close(captured['nodes'],captured['regions'].reshape(model.num_regions,-1),atol=0,rtol=0)
        require(tuple(normal.shape) == (1,19), 'Wrong output dimensions')
        captured['pooling_weights'] = weights
        captured['logits'] = observed
        captured['input'] = images.detach().clone()
    return observed,captured,errors


def run(root, device='cpu'):
    root = Path(root).resolve(); cfg = read(root/'config.json')
    release = read(root/'release.json')
    require(all(digest(root/name) == sha for name,sha in release.items()), 'Installed release changed')
    started = time.monotonic(); identities = preflight(root,cfg)
    code = Path(cfg['training_root'])/'code'
    sys.path.insert(0,str(code))
    import importlib.metadata
    import importlib.util
    import numpy as np
    import torch
    from PIL import Image
    from models.gavit import GAViT
    from models import gavit
    from models.bigearth_dataset import CLASSES_19
    require(CLASSES_19 == cfg['classes'], 'Frozen class order differs from configured labels')
    torch.set_num_threads(8)
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    if device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA requested but unavailable')
        require(torch.cuda.device_count() == 1, 'Select one visible CUDA device')
        gpu = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    else:
        gpu = None
    hardware = dict(device=device,gpu_identity=gpu,python=sys.version,packages={
        name:importlib.metadata.version(name) for name in ('torch','torchvision','timm','torch-geometric','numpy','Pillow')})
    precision = dict(dtype='torch.float32',amp=False,matmul_tf32=False,cudnn_tf32=True,threads=8)
    print('DEVICE '+json.dumps(hardware)+'; 4 inputs x 2 trained models; inference only',flush=True)
    save(root/'runtime.json',dict(hardware=hardware,precision=precision))
    inputs=[]; pngs=[]
    for item in cfg['cases']:
        path = root/'inputs'/Path(item['png_path']).name
        pngs.append(path.read_bytes())
        with Image.open(path) as im:
            array = np.array(im,dtype=np.uint8,copy=True)
        x = torch.from_numpy(array).permute(2,0,1).contiguous().to(dtype=torch.float32).div(255)
        mean = torch.tensor([.485,.456,.406]).view(3,1,1)
        std = torch.tensor([.229,.224,.225]).view(3,1,1)
        inputs.append(((x-mean)/std).unsqueeze(0).to(device))
    cases=[dict(row_index=item['row_index'],labels=item['labels'],patch_path=item['patch_path'],
                input_png_sha256=item['png_sha256'],input_tensor_sha256=sha_bytes(x.cpu().numpy().tobytes()),models={})
           for item,x in zip(cfg['cases'],inputs)]
    (root/'captures').mkdir(); (root/'figures').mkdir()
    for k in (9,16):
        model = GAViT(**dict(cfg['architecture'],num_regions=k),pretrained=False).to(device).eval()
        state = torch.load(cfg['checkpoints'][str(k)]['path'],map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True); del state
        require(sum(p.numel() for p in model.parameters()) == 31366926, 'Unexpected parameter count')
        # Check the historical K16 function using the actual loaded control.
        legacy_error = None
        if k == 16:
            spec = importlib.util.spec_from_file_location('legacy_reference',code/'legacy_graph.py')
            legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
            with torch.inference_mode():
                a = model(inputs[0]); current = gavit.build_sparse_hybrid_graph
                try:
                    gavit.build_sparse_hybrid_graph = legacy.build_sparse_hybrid_graph
                    b = model(inputs[0])
                finally:
                    gavit.build_sparse_hybrid_graph = current
            legacy_error = float((a-b).abs().max())
            require(math.isfinite(legacy_error) and legacy_error <= 1e-5, 'Loaded K16 differs from legacy graph function')
        for case,x in zip(cases,inputs):
            before = sha_bytes(x.cpu().numpy().tobytes())
            logits,tensors,errors = capture(model,x)
            require(before == sha_bytes(x.cpu().numpy().tobytes()) == case['input_tensor_sha256'], 'Input tensor changed')
            assignment = tensors['assignments'][0].cpu().tolist()
            edges = tensors['edge_index'].t().cpu().tolist()
            kinds = validate_topology(assignment,edges,k)
            weights = tensors['pooling_weights'][0].cpu().tolist(); validate_pooling(assignment,weights,k)
            probabilities = torch.sigmoid(logits)[0].cpu().tolist()
            require(all(math.isfinite(v) for v in probabilities), 'Nonfinite predictions')
            tensor_path = root/'captures'/f'row_{case["row_index"]}_K{k}.pt'
            torch.save({name:value.detach().cpu() for name,value in tensors.items()},tensor_path)
            record = dict(num_regions=k,assignments=assignment,edges=edges,edge_kinds=kinds,
                          pooling_weights=weights,probabilities=probabilities,gate_max_abs_errors=errors,
                          feedback_l2=tensors['feedback'][0].norm(dim=-1).cpu().tolist(),
                          checkpoint_sha256=identities[str(k)]['sha256'],tensor_path=str(tensor_path.relative_to(root)),
                          tensor_sha256=digest(tensor_path),legacy_k16_max_logit_diff=legacy_error,
                          gat_internal_self_loops=[bool(layer.add_self_loops) for layer in model.graph_reasoning.gat_layers])
            case['models'][str(k)] = record
            save(root/'captures'/f'row_{case["row_index"]}_K{k}.json',record)
            print(f'CAPTURE_PASS row={case["row_index"]} K={k}; {kinds.count("spatial")} spatial + {kinds.count("feature")} feature; gates={errors}',flush=True)
        del model
    for case in cases:
        case['comparison'] = compare_predictions(case['labels'],case['models']['9']['probabilities'],
                                                case['models']['16']['probabilities'],cfg['classes'])
    evidence = dict(status='complete',scope='four_fixed_historical_validation_cases_not_population_metrics',
                    dataset='BigEarthNet-19',seed=42,hardware=hardware,precision=precision,
                    selection=cfg['selection'],input_mode='lossless_original_reader_resize224_RGB_PNG',
                    source_snapshot=cfg['snapshot'],checkpoints=identities,
                    elapsed_seconds=time.monotonic()-started,summary_archived='source_reports/summary.json',
                    no_new_training=True,no_full_test_rerun=True,
                    limitations=['Single seed with historical K16 training batch; visual evidence is not a causal edge ablation.',
                                 'All four historical diagnostic cases retained; not a representative BEN sample.'])
    save(root/'captures.json',dict(evidence=evidence,classes=cfg['classes'],cases=cases))
    for case,png in zip(cases,pngs):
        (root/'figures'/f'row_{case["row_index"]}.svg').write_text(render_svg(case,png))
    (root/'comparison.html').write_text(render_html(cases,cfg['classes'],pngs,evidence))
    file_hashes={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()
                 and p.name not in ('run.log','run.exit')}
    save(root/'summary.json',dict(status='complete',dataset='BigEarthNet-19',seed=42,samples=4,models=[9,16],
        paired_capture_checks_passed=True,source_summary=evidence['summary_archived'],elapsed_seconds=time.monotonic()-started,
        checkpoint_sha256={k:v['sha256'] for k,v in identities.items()},
        html='comparison.html',raw_data='captures.json',files_sha256=file_hashes))
    print('BEN_GRID_PAIR_VISUAL_COMPLETE '+str(root),flush=True)


def main():
    sys.dont_write_bytecode = True
    root = Path(sys.argv[1]).resolve()
    device = sys.argv[2] if len(sys.argv) > 2 else 'cpu'
    require(device in ('cpu','cuda'), 'Device must be cpu or cuda')
    rc = 1
    class Tee:
        def __init__(self,stream,log): self.stream,self.log=stream,log
        def write(self,s): self.stream.write(s); self.log.write(s); self.log.flush()
        def flush(self): self.stream.flush(); self.log.flush()
    with (root/'run.log').open('x') as log:
        stdout,stderr = sys.stdout,sys.stderr
        sys.stdout,sys.stderr = Tee(stdout,log),Tee(stderr,log)
        try:
            run(root,device)
            (root/'run.exit').write_text('exit_code=0\n')
            archive=root/'ben_grid_pair_visual.zip'
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
                for path in sorted(root.rglob('*')):
                    if path.is_file() and path != archive:
                        z.write(path,path.relative_to(root))
            print('RETURN_HTML '+str(root/'comparison.html'),flush=True)
            print('RETURN_AUDIT_ZIP '+str(archive),flush=True)
            rc=0
        except BaseException:
            traceback.print_exc()
            (root/'run.exit').write_text('exit_code=1\n')
        finally:
            sys.stdout,sys.stderr=stdout,stderr
    sys.exit(rc)


if __name__ == '__main__':
    main()
