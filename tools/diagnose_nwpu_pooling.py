"""Frozen NWPU soft checkpoint: intervene only on final region pooling W.

Normal W=softmax_token(logS+scores); intervention U=S/sum_token(S).
S and prototype attention stay unchanged. Feature edges are recomputed from
the resulting regions. This is an inference intervention, not a trained model.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

LEGACY_SHA = '4cd5175261d271638a00e4a4c733f329ecde1c07b646b901c960f0181237eea3'
SCOPE = ('Frozen-weight final-pooling intervention on all 1260 calibration-val images. '
         'Not a trained baseline or stability result; no test evaluation or automatic training. '
         'S and prototype attention stay unchanged; changed feature edges are downstream effects. '
         'U-centroid equals normalized-S centroid by construction, not evidence of performance gain.')


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_legacy(path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != LEGACY_SHA:
        raise ValueError('Frozen diagnostic helper hash mismatch')
    spec = importlib.util.spec_from_file_location('nwpu_frozen_feedback_helper', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_capture(t, group):
    import torch
    s, w, regions, e = t['S'], t['W'], t['regions'], t['edges']
    b = len(s)
    if s.shape != (b,49,16) or w.shape != s.shape or regions.shape != (b,16,768) or e.shape != (2,80*b):
        raise ValueError('Unexpected captured tensor shape')
    if not all(bool(torch.isfinite(v).all()) for v in t.values()):
        raise ValueError('Nonfinite capture')
    if (bool((s<0).any()) or bool((w<0).any())
            or float((s.sum(-1)-1).abs().max())>1e-5 or float((w.sum(1)-1).abs().max())>1e-5
            or bool(s.masked_select(~group.candidate_mask[None]).any())
            or bool(w.masked_select(~group.candidate_mask[None]).any())):
        raise ValueError('Invalid membership/pooling weights')
    for i in range(b):
        local = e[:,80*i:80*(i+1)]-16*i
        src,dst = local
        distance = (src//4-dst//4).abs()+(src%4-dst%4).abs()
        if (e.dtype != torch.int64 or int(local.min())<0 or int(local.max())>=16
                or torch.unique(local,dim=1).shape[1]!=80 or not bool((distance[:48]==1).all())
                or not bool((distance[48:]>1).all()) or not bool((torch.bincount(dst[48:],minlength=16)==2).all())):
            raise ValueError('Invalid actual graph')


def capture_pair(model, images):
    import torch
    if (any(m.training for m in model.modules()) or model.grouping!='soft_prototype'
            or model.edge_type!='sparse_hybrid' or model.integration!='token_feedback'
            or getattr(model,'token_refinement',None) is not None):
        raise ValueError('Requires original eval soft-prototype sparse-hybrid model without FFN')
    group = model.region_grouping
    def one_forward(intervene):
        tensors, handles = {}, []
        previous = group.assignment_observer
        def keep(key, value): tensors[key]=value.detach().clone()
        def observer(values):
            keep('S',values['S']); keep('original_W',values['W'])
        def grouping_hook(module,args,outputs):
            regions,s = outputs
            torch.testing.assert_close(s,tensors['S'],atol=0,rtol=0)
            torch.testing.assert_close(regions,torch.einsum('bnk,bnd->bkd',tensors['original_W'],args[0]),atol=1e-5,rtol=1e-5)
            mass=s.sum(1,keepdim=True)
            if not bool((mass>0).all()): raise ValueError('Zero membership mass')
            w=s/mass if intervene else tensors['original_W']
            used=torch.einsum('bnk,bnd->bkd',w,args[0]) if intervene else regions
            keep('W',w); keep('regions',used)
            return used,s
        def graph_hook(module,args):
            torch.testing.assert_close(args[0],tensors['regions'].reshape(-1,768),atol=0,rtol=0)
            keep('edges',args[1])
        try:
            group.assignment_observer=observer
            handles.append(group.register_forward_hook(grouping_hook))
            handles.append(model.graph_reasoning.register_forward_pre_hook(graph_hook))
            handles.append(model.classifier.register_forward_pre_hook(lambda m,a:keep('pooled',a[0])))
            keep('logits',model(images))
        finally:
            for h in handles: h.remove()
            group.assignment_observer=previous
        tensors.pop('original_W')
        check_capture(tensors,group)
        return tensors
    devices=[images.device.index or 0] if images.is_cuda else []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        normal=one_forward(False)
        candidate=one_forward(True)
        restored=one_forward(False)
        torch.testing.assert_close(normal['S'],candidate['S'],atol=0,rtol=0)
        torch.testing.assert_close(normal['S'],restored['S'],atol=0,rtol=0)
        torch.testing.assert_close(normal['logits'],restored['logits'],atol=1e-5,rtol=1e-5)
        if not torch.equal(normal['logits'].argmax(-1),restored['logits'].argmax(-1)):
            raise ValueError('Restoration changes predictions')
        checks=dict(S_max_abs_diff=float((normal['S']-candidate['S']).abs().max()),
                    restoration_max_logit_diff=float((normal['logits']-restored['logits']).abs().max()),
                    U_column_error=float((candidate['W'].sum(1)-1).abs().max()))
        result={name:{k:v.detach().cpu().clone() for k,v in t.items()}
                for name,t in [('normal',normal),('membership_only',candidate)]}
        return result,checks


def region_diagnostics(capture):
    import torch
    result=[]
    for i in range(len(capture['normal']['S'])):
        row={}
        edges=[]
        for name in ('normal','membership_only'):
            region=capture[name]['regions'][i].double()
            norms=region.norm(dim=-1)
            valid=norms>1e-12
            z=region/norms.clamp_min(1e-12)[:,None]
            mask=torch.triu(torch.ones(16,16,dtype=torch.bool),1) & valid[:,None] & valid[None]
            values=(z@z.T)[mask]
            row[name+'_region_cosine_mean']=float(values.mean()) if len(values) else None
            row[name+'_zero_norm_regions']=int((~valid).sum())
            e=capture[name]['edges'][:,80*i+48:80*(i+1)]-16*i
            edges.append(set(map(tuple,e.T.tolist())))
        row['common_feature_edges']=len(edges[0]&edges[1])
        result.append(row)
    return result


def run(args,out):
    print('PREFLIGHT: frozen code, checkpoints, split and validation sources',flush=True)
    legacy=load_legacy(args.legacy)
    base=legacy.load_helper(args.helper)
    manifest,selected90,reports,cached,sources=base.validate_sources(args)
    selected=legacy.select_validation(manifest)
    selected_paths={r['path'] for r in selected90}
    if len(selected_paths)!=90: raise ValueError('Expected 90 frozen feature cases')
    sys.path.insert(0,str(args.repo.resolve()))
    import torch
    from scene_classification.modeling import build_model,image_transform
    torch.set_num_threads(8); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no CPU fallback')
    hardware=dict(device=str(device),name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
                  python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                  amp=False,matmul_tf32=False,cudnn_tf32=False,threads=8)
    identity=dict(dataset='NWPU-RESISC45',seed=42,split='val',source_phase='calibrate',samples=1260,
                  manifest_sha256=base.MANIFEST_SHA,source=sources['soft'],model_code=reports['soft']['code'],
                  modes=['normal','membership_only'],hardware=hardware,batch_size=32,scope=SCOPE,
                  tool_sha256=base.sha(__file__),legacy_sha256=LEGACY_SHA,helper_sha256=legacy.HELPER_SHA)
    save_json(out/'identity.json',identity)
    print('SOURCE PASS: soft calibration best; all 1260 validation images',flush=True)
    print('DEVICE '+json.dumps(hardware),flush=True)
    print('LOAD soft checkpoint',flush=True)
    model=build_model('gavit',45,architecture_config=reports['soft']['architecture'])
    model.load_state_dict(torch.load(args.calibration/'soft'/'best.pth',map_location='cpu',weights_only=True),strict=True)
    model=model.to(device).eval()
    tf=image_transform(False)
    pieces=[]; checks=[]; inputs=[]; cases=[]; diagnostics=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with base.ImageSource(args.data_root,args.archive) as reader:
        for start in range(0,len(selected),32):
            rows=selected[start:start+32]; images=[]
            for row in rows:
                im,info=reader.read(row); x=tf(im)
                info['input_tensor_sha256']=hashlib.sha256(x.numpy().tobytes()).hexdigest()
                inputs.append(info); images.append(x)
            capture,check=capture_pair(model,torch.stack(images).to(device))
            check['cached_probability_max_error']=legacy.check_cached_batch(base,capture['normal']['logits'],rows,cached['soft'])
            checks.append(check)
            pieces.append({name+'_'+k:capture[name][k] for name in ('normal','membership_only') for k in ('pooled','logits')})
            for i,(row,diag) in enumerate(zip(rows,region_diagnostics(capture))):
                diagnostics.append(dict(path=row['path'],label=row['label'],**diag))
                if row['path'] in selected_paths:
                    pair={}
                    for name,t in capture.items():
                        pair[name]={k:(v[:,80*i:80*(i+1)]-16*i if k=='edges' else v[i]).clone() for k,v in t.items()}
                    cases.append(dict(path=row['path'],label=row['label'],tensors=pair))
            if start==0 or (start//32+1)%10==0 or start+32>=len(selected):
                print(f'PAIR {min(start+32,len(selected))}/1260: cached/S/graph/restoration PASS',flush=True)
    save_json(out/'inputs.json',inputs)
    save_json(out/'region_diagnostics.json',diagnostics)
    cache={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
    cache.update(labels=[r['label'] for r in selected],paths=[r['path'] for r in selected],
                 class_names=manifest['class_names'],identity=identity,
                 classifier_state={k:v.detach().cpu().clone() for k,v in model.classifier.state_dict().items()})
    torch.save(cache,out/'prediction_cache.pt')
    if len(cases)!=90: raise ValueError('Incomplete selected feature capture')
    torch.save(dict(identity=identity,cases=cases),out/'selected_cases.pt')
    loaded=torch.load(out/'prediction_cache.pt',map_location='cpu',weights_only=True)
    head=copy.deepcopy(model.classifier).cpu().eval()
    head.load_state_dict(loaded['classifier_state'],strict=True)
    metrics={}; replay_error=0.
    with torch.inference_mode():
        for name in ('normal','membership_only'):
            z=loaded[name+'_logits']; replay=head(loaded[name+'_pooled'])
            if z.shape!=(1260,45): raise ValueError('Incomplete logits')
            torch.testing.assert_close(replay,z,atol=1e-5,rtol=1e-5)
            if not torch.equal(replay.argmax(-1),z.argmax(-1)): raise ValueError('CPU replay changes predictions')
            replay_error=max(replay_error,float((replay-z).abs().max()))
            metrics[name]=legacy.score_logits(z,loaded['labels'],loaded['class_names'])
    if abs(metrics['normal']['oa_percent']-reports['soft']['best_val_oa_percent'])>1e-9:
        raise ValueError('Normal validation OA differs from frozen calibration')
    paired=legacy.paired_counts(loaded['labels'],loaded['normal_logits'].argmax(-1).tolist(),
                                loaded['membership_only_logits'].argmax(-1).tolist())
    save_json(out/'metrics.json',metrics)
    compact={n:{k:r[k] for k in ('samples','correct','oa_percent','macro_f1_percent','cross_entropy')} for n,r in metrics.items()}
    maxima={k:max(c[k] for c in checks) for k in checks[0]}
    maxima['cpu_replay_max_logit_diff']=replay_error
    feature_summary={}
    for key in ('normal_region_cosine_mean','membership_only_region_cosine_mean','common_feature_edges'):
        v=[r[key] for r in diagnostics if r[key] is not None]
        feature_summary[key]=dict(valid_images=len(v),mean=sum(v)/len(v) if v else None)
    for name in ('normal','membership_only'): print(name.upper(),json.dumps(compact[name]),flush=True)
    print('PAIRED',json.dumps(paired),flush=True)
    print('CHECKS',json.dumps(maxima),flush=True)
    print('REGIONS',json.dumps(feature_summary),flush=True)
    return dict(status='complete',samples=1260,dataset='NWPU-RESISC45',seed=42,split='val',source_phase='calibrate',
                metrics=compact,paired=paired,max_errors=maxima,region_summary=feature_summary,
                input_sources=dict(Counter(r['source'] for r in inputs)),feature_cases=90,
                cache_artifacts={n:dict(sha256=base.sha(out/n),bytes=(out/n).stat().st_size)
                                 for n in ('prediction_cache.pt','selected_cases.pt')},
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,scope=SCOPE)


def execute(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    try:
        report=run(args,out)
        report['elapsed_seconds']=time.perf_counter()-started
        save_json(out/'summary.json',report)
        (out/'run.exit').write_text('exit_code=0\n')
        print(f'NWPU POOLING PROBE COMPLETE {out}\nexit_code=0',flush=True)
        return report
    except BaseException as exc:
        save_json(out/'summary.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started))
        (out/'run.exit').write_text('exit_code=1\n')
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy',type=Path,default=Path('/home/featurize/work/nwpu_feedback_tool_20261002_4cd5175261d2.py'))
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
