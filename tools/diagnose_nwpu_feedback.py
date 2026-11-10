"""Frozen NWPU calibration-val feedback sensitivity; no training or test use."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ALPHAS = (0., .25, .5, .75, 1.)
HELPER_SHA = '35ed63ce1ceda120fe9b5e78b157032c5e89d6b351f0caf67e05eef19d4080ba'
SCOPE = ('Frozen-weight validation sensitivity only. Alpha zero is not a trained no-graph '
         'baseline or Swin. No automatic alpha selection, retraining, or test evaluation.')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_helper(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HELPER_SHA:
        raise ValueError('Frozen region helper hash mismatch')
    spec = importlib.util.spec_from_file_location('nwpu_frozen_region_helper', path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    return base


def select_validation(manifest):
    rows, names = manifest['records'], manifest['class_names']
    if len(names) != 45 or len(set(names)) != 45 or len({r['path'] for r in rows}) != len(rows):
        raise ValueError('Wrong classes or duplicate paths')
    selected = sorted((r for r in rows if r['split'] == 'val'), key=lambda r:r['path'])
    if len(selected) != 1260 or Counter(r['label'] for r in selected) != {i:28 for i in range(45)}:
        raise ValueError('Expected all 1260 validation images, 28 per class')
    if any(r['path'].split('/')[0] != names[r['label']] for r in selected):
        raise ValueError('Validation class/path mismatch')
    return selected


def capture_batch(model, images):
    """One unchanged model forward; replay only the classifier for fixed alphas."""
    import torch
    if any(m.training for m in model.modules()):
        raise ValueError('Capture requires eval model')
    if model.grouping not in ('attentive_spatial','soft_prototype') or model.integration != 'token_feedback':
        raise ValueError('Unsupported model grouping/integration')
    tensors, handles = {}, []
    def keep(key, value):
        tensors[key] = value.detach().clone()
    devices = [images.device.index or 0] if images.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        try:
            handles.append(model.backbone.register_forward_hook(lambda m,a,o: keep('mean_tokens',o.mean(1))))
            handles.append(model.region_grouping.register_forward_hook(lambda m,a,o: keep('assignments',o[1])))
            handles.append(model.feedback_proj.register_forward_hook(lambda m,a,o: keep('regions_feedback',o)))
            handles.append(model.classifier.register_forward_pre_hook(lambda m,a: keep('original_pooled',a[0])))
            normal = model(images)
        finally:
            for h in handles:
                h.remove()
        s, f = tensors['assignments'], tensors['regions_feedback']
        if model.grouping == 'soft_prototype':
            token_feedback = torch.einsum('bnk,bkd->bnd',s,f)
        else:
            token_feedback = f.gather(1,s.unsqueeze(-1).expand(-1,-1,f.shape[-1]))
        mean_feedback = token_feedback.mean(1)
        x = tensors['mean_tokens']
        torch.testing.assert_close(x+mean_feedback,tensors['original_pooled'],atol=1e-5,rtol=1e-5)
        logits = torch.stack([model.classifier(x+alpha*mean_feedback) for alpha in ALPHAS])
        if not all(torch.isfinite(v).all() for v in (x,mean_feedback,logits,normal)):
            raise ValueError('Nonfinite captured values')
        torch.testing.assert_close(logits[-1],normal,atol=1e-5,rtol=1e-5)
        if not torch.equal(logits[-1].argmax(-1),normal.argmax(-1)):
            raise ValueError('Reconstruction changes prediction')
        result = dict(mean_tokens=x,mean_feedback=mean_feedback,normal_logits=normal,scaled_logits=logits)
        checks = dict(reconstruction_max_logit_diff=float((logits[-1]-normal).abs().max()),
                      pooled_max_abs_diff=float((x+mean_feedback-tensors['original_pooled']).abs().max()))
        return {k:v.detach().cpu().clone() for k,v in result.items()}, checks


def check_cached_batch(base, logits, rows, cached):
    if len(logits) != len(rows):
        raise ValueError('Incomplete prediction batch')
    return max(base.check_cached(z,cached[r['path']],r['label']) for r,z in zip(rows,logits))


def paired_counts(labels, original, candidate):
    if not labels or len(labels) != len(original) or len(labels) != len(candidate):
        raise ValueError('Wrong paired prediction lengths')
    c=Counter((a==y,b==y) for y,a,b in zip(labels,original,candidate))
    return dict(both_correct=c[True,True],corrects=c[False,True],breaks=c[True,False],
                both_wrong=c[False,False],net_correct=c[False,True]-c[True,False])


def score_logits(logits, labels, names):
    import torch
    from scene_classification.metrics import classification_metrics
    if logits.shape != (len(labels),len(names)) or not torch.isfinite(logits).all():
        raise ValueError('Wrong shape or nonfinite logits')
    predictions=logits.argmax(-1).tolist()
    m=classification_metrics(labels,predictions,names)
    m['cross_entropy']=float(torch.nn.functional.cross_entropy(logits,torch.tensor(labels)))
    m['correct']=sum(a==b for a,b in zip(labels,predictions))
    return m


def analyze_cache(cache):
    """CPU-only summaries; no data, backbone, graph, or weight reconstruction."""
    labels=cache['labels']; original=cache['normal_logits'].argmax(-1).tolist()
    results=[]
    for i,alpha in enumerate(ALPHAS):
        logits=cache['scaled_logits'][i]
        m=score_logits(logits,labels,cache['class_names'])
        m.update(alpha=alpha,paired_vs_original=paired_counts(labels,original,logits.argmax(-1).tolist()))
        results.append(m)
    original_metrics=score_logits(cache['normal_logits'],labels,cache['class_names'])
    for m in results:
        m['delta_oa_pp']=m['oa_percent']-original_metrics['oa_percent']
        m['delta_macro_f1_pp']=m['macro_f1_percent']-original_metrics['macro_f1_percent']
        m['delta_cross_entropy']=m['cross_entropy']-original_metrics['cross_entropy']
    return results


def run(args, out):
    print('PREFLIGHT: verifying frozen sources and all validation identities',flush=True)
    base=load_helper(args.helper)
    manifest,_,reports,cached,identities=base.validate_sources(args)
    selected=select_validation(manifest)
    sys.path.insert(0,str(args.repo.resolve()))
    import torch
    from scene_classification.modeling import build_model,image_transform
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; no automatic CPU fallback')
    hardware=dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                  python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                  amp=False,matmul_tf32=False,cudnn_tf32=False,threads=8)
    identity=dict(dataset='NWPU-RESISC45',split='val',source_phase='calibrate',seed=42,
                  samples=1260,alphas=ALPHAS,manifest_sha256=base.MANIFEST_SHA,source_code=reports['fixed']['code'],
                  sources=identities,hardware=hardware,tool_sha256=base.sha(__file__),helper_sha256=HELPER_SHA,
                  batch_size=32,selection='all_frozen_validation_paths',scope=SCOPE)
    save_json(out/'identity.json',identity)
    print('SOURCE PASS: fixed/soft calibration best, all 1260 val images',flush=True)
    print('DEVICE '+json.dumps(hardware),flush=True)
    names=manifest['class_names']; labels=[r['label'] for r in selected]
    tf=image_transform(False)
    curves, maxima, artifacts, timing = {},{},{},{}
    first_input_info=None
    for name in ('fixed','soft'):
        started=time.perf_counter()
        print('LOAD '+name+': reading trained checkpoint',flush=True)
        model=build_model('gavit',45,architecture_config=reports[name]['architecture'])
        model.load_state_dict(torch.load(args.calibration/name/'best.pth',map_location='cpu',weights_only=True),strict=True)
        model=model.to(device).eval()
        pieces=[];checks=[];input_info=[]
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        with base.ImageSource(args.data_root,args.archive) as reader:
            for start in range(0,len(selected),32):
                rows=selected[start:start+32]; batch=[]
                for row in rows:
                    image,info=reader.read(row); x=tf(image)
                    info['input_tensor_sha256']=hashlib.sha256(x.numpy().tobytes()).hexdigest()
                    input_info.append(info);batch.append(x)
                captured,checked=capture_batch(model,torch.stack(batch).to(device))
                checked['cached_probability_max_error']=check_cached_batch(base,captured['normal_logits'],rows,cached[name])
                pieces.append(captured);checks.append(checked)
                if start==0 or (start//32+1)%10==0 or start+32>=len(selected):
                    print(f'CAPTURE {name}: {min(start+32,len(selected))}/1260; cached and reconstruction PASS',flush=True)
        if first_input_info is None:
            first_input_info=input_info
            save_json(out/'inputs.json',input_info)
        elif input_info != first_input_info:
            raise ValueError('Models used different input identities')
        capture={k:torch.cat([p[k] for p in pieces],dim=1 if k=='scaled_logits' else 0) for k in pieces[0]}
        if capture['normal_logits'].shape!=(1260,45) or capture['scaled_logits'].shape!=(5,1260,45):
            raise ValueError('Incomplete validation capture')
        capture.update(labels=labels,paths=[r['path'] for r in selected],class_names=names,alphas=list(ALPHAS),
                       model=name,identity=identity,classifier_state={k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()})
        path=out/(name+'_cache.pt');torch.save(capture,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True)
        head=copy.deepcopy(model.classifier).cpu().eval()
        head.load_state_dict(loaded['classifier_state'],strict=True)
        with torch.inference_mode():
            replay=torch.stack([head(loaded['mean_tokens']+a*loaded['mean_feedback']) for a in ALPHAS])
        torch.testing.assert_close(replay,loaded['scaled_logits'],atol=1e-5,rtol=1e-5)
        if not torch.equal(replay.argmax(-1),loaded['scaled_logits'].argmax(-1)):
            raise ValueError('CPU cache replay changes prediction')
        curves[name]=analyze_cache(loaded)
        if abs(curves[name][-1]['oa_percent']-reports[name]['best_val_oa_percent'])>1e-9:
            raise ValueError('Alpha1 OA differs from calibration best')
        maxima[name]={k:max(c[k] for c in checks) for k in checks[0]}
        maxima[name]['cpu_replay_max_logit_diff']=float((replay-loaded['scaled_logits']).abs().max())
        artifacts[name]=dict(cache_sha256=base.sha(path),cache_bytes=path.stat().st_size)
        with (out/(name+'_predictions.csv')).open('w',newline='') as stream:
            fields=['path','label']+[f'prediction_alpha_{a:g}' for a in ALPHAS]
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            predictions=loaded['scaled_logits'].argmax(-1).tolist()
            for i,row in enumerate(selected):
                writer.writerow(dict(path=row['path'],label=row['label'],**{f'prediction_alpha_{a:g}':predictions[j][i] for j,a in enumerate(ALPHAS)}))
        timing[name]=dict(elapsed_seconds=time.perf_counter()-started,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
        print('MODEL COMPLETE '+name,flush=True)
        del model,head,capture,loaded,pieces
        if device.type=='cuda':torch.cuda.empty_cache()
    save_json(out/'metrics.json',curves)
    compact={name:[{k:r[k] for k in ('alpha','correct','oa_percent','macro_f1_percent','cross_entropy',
                                   'delta_oa_pp','delta_macro_f1_pp','delta_cross_entropy','paired_vs_original')} for r in rows]
             for name,rows in curves.items()}
    lines=['# NWPU calibration-val feedback sensitivity','',SCOPE,'',
           '| Model | Alpha | OA % | macro-F1 % | CE | Corrects | Breaks |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,rows in compact.items():
        for r in rows:
            p=r['paired_vs_original']
            lines.append(f"| {name} | {r['alpha']:g} | {r['oa_percent']:.6f} | {r['macro_f1_percent']:.6f} | {r['cross_entropy']:.6f} | {p['corrects']} | {p['breaks']} |")
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return dict(status='complete',dataset='NWPU-RESISC45',split='val',source_phase='calibrate',samples=1260,
                seed=42,alphas=ALPHAS,curves=compact,max_errors=maxima,cache_artifacts=artifacts,
                model_timing=timing,input_sources=dict(Counter(r['source'] for r in first_input_info)),scope=SCOPE)


def execute(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        report=run(args,out)
        report['elapsed_seconds']=time.perf_counter()-started
        save_json(out/'summary.json',report)
        (out/'run.exit').write_text('exit_code=0\n')
        print('NWPU FEEDBACK SENSITIVITY COMPLETE '+str(out),flush=True)
        return report
    except BaseException as exc:
        save_json(out/'summary.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started))
        (out/'run.exit').write_text('exit_code=1\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--helper',type=Path,default=Path('/home/featurize/work/nwpu_region_tool_20260929_35ed63ce1ced.py'))
    p.add_argument('--repo',type=Path,default=Path('/home/featurize/work/GAViT_SOFT_20260927'))
    p.add_argument('--calibration',type=Path,default=Path('/home/featurize/work/nwpu_ablation42_20260928_01'))
    p.add_argument('--manifest',type=Path,default=Path('/home/featurize/work/nwpu_protocol_20260927_02.json'))
    p.add_argument('--data-root',type=Path,default=Path('/home/featurize/data/NWPU-RESISC45'))
    p.add_argument('--archive',type=Path,default=Path('/home/featurize/work/NWPU-RESISC45_a826b44.zip'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    execute(p.parse_args())


if __name__=='__main__':
    sys.dont_write_bytecode=True
    main()
