"""External, fail-closed soft45 queue around the frozen ed480e6 entries."""
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import resource
import runpy
import sys
import time
import traceback

REPO = Path('/home/featurize/work/GAViT_SOFT_20260927')
COMMIT = 'ed480e60fea03ed786fdba70c144de5e815c34f5'
OLD_TOOL = Path('/home/featurize/work/bigearth_seed45_20260922_01/run_seed45.py')
OLD_SHA = '8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369'
PREFLIGHT_SHA = '08ef296aa7e34174630d2e77b45c767b05742dfba012bcf5aeaf845e12a5fa03'
TRAINING = dict(seed=45,epochs=30,batch_size=32,lr=.0003,weight_decay=.0001,
                optimizer='AdamW',scheduler='CosineAnnealingLR',loss='BCEWithLogitsLoss',f1_threshold=.5)
ARCH = dict(num_classes=19,num_regions=16,knn_k=5,gat_hidden=256,gat_heads=4,gat_layers=2,
            dropout=.1,grouping='soft_prototype',edge_type='sparse_hybrid',integration='token_feedback',
            soft_assignment=dict(assignment_version='local_prototype_cosine_v1',assignment_dim=128,
                assignment_radius=1,assignment_metric='chebyshev',assignment_temperature=.1,
                assignment_iterations=1,assignment_projection='shared_ln_linear_no_bias',
                assignment_init='xavier_uniform',assignment_norm_eps=1e-6,assignment_ln_eps=1e-5,
                prototype_pooling='fixed_region_attention',region_pooling='attentive_log_membership',
                feedback_mapping='soft_membership',graph_node_geometry='anchor_index_prior'))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def load_base():
    require(hashlib.sha256(OLD_TOOL.read_bytes()).hexdigest() == OLD_SHA, 'Original runner changed')
    spec = importlib.util.spec_from_file_location('frozen_base45', OLD_TOOL)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.REPO, b.COMMIT = REPO, COMMIT
    return b


def entry_args(root, stage, b):
    require(stage in ('gavit_train','gavit_test'), 'Only soft45 train/test are authorized')
    argv = b.entry_args(root, stage)
    if stage == 'gavit_train':
        argv[argv.index('--grouping')+1] = 'soft_prototype'
    return argv


def check_metadata(meta):
    require(meta['model']=='gavit' and meta['dataset']=='BigEarthNet-19', 'Wrong dataset/model')
    require(meta['architecture']==ARCH and meta['training']==TRAINING, 'Changed soft architecture or recipe')
    require(meta['execution']['run_stage']=='formal', 'Expected formal checkpoint')
    require(meta['git']=={'commit':COMMIT[:7],'dirty':False}, 'Wrong checkpoint code identity')


def check_training(meta, history, last_epoch):
    check_metadata(meta)
    require(last_epoch==30 and [r['epoch'] for r in history]==list(range(1,31)), 'Incomplete 30-epoch training')
    require(all(all(math.isfinite(v) for v in r.values()) for r in history), 'Nonfinite history')
    best = max(history, key=lambda r:r['val_mAP_percent'])
    require(meta['best']['metric']=='val_mAP' and meta['best']['epoch']==best['epoch']
            and abs(meta['best']['value']-best['val_mAP_percent'])<=.000051, 'Best checkpoint disagrees with history')


def check_report(r, stage, ready, b):
    require(r['status']=='complete' and r['stage']==stage and r['model']=='gavit' and r['seed']==45, 'Incomplete/wrong stage')
    require(r['commit']==COMMIT and r['architecture']==ARCH, 'Wrong code/architecture')
    require(r['pretrained_sha256']==b.PRETRAINED_SHA and r['splits']=={k:list(v) for k,v in b.FROZEN.items()}, 'Changed data/pretraining')
    require(r['hardware']==ready['hardware'] and r['precision']==ready['precision'], 'Queue environment changed')
    if stage=='gavit_train':
        require(r['final_epoch']==30 and r['parameters']==31466766, 'Incomplete training or wrong parameter count')
    else:
        m=r['metrics']
        require(m['samples']==119825 and m['f1_threshold']==.5 and m['checkpoint_sha256']==r['checkpoint_sha256'], 'Wrong test identity')
        ap=list(m['per_class_ap_percent'].values())
        require(len(ap)==19 and all(math.isfinite(v) and 0<=v<=100 for v in ap), 'Invalid class AP')
        require(abs(sum(ap)/19-m['mAP_percent'])<1e-8, 'mAP does not match 19 classes')
        require(all(math.isfinite(m[k]) and 0<=m[k]<=100 for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')), 'Invalid metrics')


def check_test_arrays(ns, np, count=119825):
    p,y=ns['probs_np'],ns['labels_np']
    require(p.shape==y.shape==(count,19) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Incomplete/nonfinite probabilities')
    require(np.isin(y,[0,1]).all() and (y.sum(0)>0).all() and np.array_equal(y,ns['test_labels']), 'Wrong labels or missing class')
    require(len(ns['test_patches'])==len(set(ns['test_patches']))==count, 'Duplicate/missing test paths')
    require(len(ns['ap_per_class'])==len(set(ns['CLASSES_19']))==19, 'Missing AP classes')


def install_history_recorder(utils, dest, b):
    original=utils.save_epoch_artifacts
    history=[]
    previous=[time.monotonic()]

    def record(*args, **kwargs):
        caller=inspect.currentframe().f_back.f_globals
        result=original(*args, **kwargs)
        now=time.monotonic()
        row=dict(epoch=kwargs['epoch'],train_loss=float(caller['avg_loss']),
                 val_mAP_percent=float(kwargs['metric_value']),val_macro_f1_percent=float(caller['val_f1']),
                 next_epoch_lr=float(args[1].param_groups[0]['lr']),epoch_wall_seconds=now-previous[0])
        require(all(math.isfinite(v) for v in row.values()), 'Nonfinite epoch result')
        history.append(row)
        b.save(dest/'history.json',history)
        previous[0]=now
        print('EPOCH RESOURCE '+json.dumps(row),flush=True)
        return result

    utils.save_epoch_artifacts=record
    return history, original


def check_probe(directory, torch, mask, sample_ids, best_epoch, final_epoch=30):
    ident=read(directory/'probe_manifest.json')
    require(ident['split']=='val' and ident['sample_ids']==sample_ids and len(sample_ids)==8
            and ident['soft_assignment']==ARCH['soft_assignment'], 'Wrong fixed validation probe')
    rows=[json.loads(s) for s in (directory/'assignment_diagnostics.jsonl').read_text().splitlines()]
    require([r['epoch'] for r in rows]==list(range(final_epoch+1))
            and all(r['sample_ids']==sample_ids and r['split']=='val' for r in rows), 'Incomplete probe history')
    errors={'S_row':0.,'W_column':0.}
    for name,epoch in (('epoch_000.pt',0),('best.pt',best_epoch),('final.pt',final_epoch)):
        saved=torch.load(directory/name,map_location='cpu',weights_only=True)
        require(saved['identity']==ident and saved['epoch']==epoch, 'Wrong probe snapshot')
        values=saved['tensors']
        for key,axis,label in (('S',-1,'S_row'),('W',1,'W_column')):
            t=values[key]
            require(tuple(t.shape)==(8,49,16) and torch.isfinite(t).all() and (t>=0).all()
                    and torch.count_nonzero(t[:,~mask])==0, 'Invalid probe assignment')
            errors[label]=max(errors[label],float((t.sum(axis)-1).abs().max()))
        e=values['edge_index']
        require(tuple(e.shape)==(2,640), 'Wrong captured edge count')
        for i in range(8):
            edges=e[:,i*80:(i+1)*80]-i*16
            require(bool(((edges>=0)&(edges<16)).all()), 'Cross-image edge')
            pairs=[tuple(v) for v in edges.t().tolist()]
            require(len(set(pairs))==80 and all(s!=t for s,t in pairs), 'Duplicate/self edge')
            distances=[abs(s//4-t//4)+abs(s%4-t%4) for s,t in pairs]
            require(all(d==1 for d in distances[:48]) and all(d>1 for d in distances[48:]), 'Wrong spatial/feature edges')
            require(all(sum(t==q for s,t in pairs[48:])==2 for q in range(16)), 'Wrong feature degree')
    require(max(errors.values())<1e-5, 'Probe normalization failed')
    return dict(normalization_errors=errors,actual_forward_checked=True,images_checked=8,
                snapshots_checked=3,edges_per_image=80,spatial_edges=48,feature_edges=32)


def environment(torch, b):
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'One CUDA GPU required; no CPU fallback')
    torch.set_num_threads(8)
    hardware=dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                  threads=torch.get_num_threads(),python=sys.version,
                  packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},
                  gpu_identity=b.output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader']))
    precision=dict(dtype=str(torch.get_default_dtype()),matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                   cudnn_tf32=torch.backends.cudnn.allow_tf32,amp=False)
    return hardware,precision


def worker(root, stage):
    b=load_base()
    os.chdir(REPO)
    sys.path.insert(0,str(REPO))
    b.identity()
    import torch
    import numpy as np
    hardware,precision=environment(torch,b)
    ready=read(root/'preflight.json')
    require(hardware==ready['hardware'] and precision==ready['precision'], 'Environment changed since preflight')
    print('DEVICE',hardware,'PRECISION',precision,flush=True)
    dest=root/stage
    argv=entry_args(root,stage,b)
    ckpt=root/'gavit_train/gavit.pth'
    if stage=='gavit_test':
        meta=read(ckpt.with_suffix('.meta.json'))
        check_metadata(meta)
        require(b.digest(ckpt)==meta['checkpoint_sha256']==read(root/'gavit_train/run.json')['checkpoint_sha256'], 'Test weight differs from trained best')
    r=dict(stage=stage,status='running',model='gavit',seed=45,commit=COMMIT,architecture=ARCH,
           pretrained_sha256=b.PRETRAINED_SHA,splits=b.FROZEN,hardware=hardware,precision=precision,
           argv=argv,launcher_sha256=b.digest(__file__))
    b.save(dest/'run.json',r)
    torch.cuda.reset_peak_memory_stats()
    started=time.monotonic()
    history=None
    if stage=='gavit_train':
        import utils
        history,original=install_history_recorder(utils,dest,b)
    sys.argv=argv
    try:
        ns=runpy.run_path(argv[0],run_name='__main__')
    finally:
        if history is not None:
            utils.save_epoch_artifacts=original
    meta=read(ckpt.with_suffix('.meta.json'))
    check_metadata(meta)
    sha=b.digest(ckpt)
    require(sha==meta['checkpoint_sha256'], 'Checkpoint bytes changed')
    if stage=='gavit_train':
        state=torch.load(ns['TRAIN_STATE_PATH'],map_location='cpu',weights_only=True)
        check_training(meta,history,state['epoch'])
        del state
        require(len(ns['train_set'])==237871 and len(ns['val_set'])==122342, 'Wrong train/val counts')
        probe=check_probe(ckpt.with_suffix('.assignment_probe'),torch,
                          ns['model'].region_grouping.candidate_mask.cpu(),ns['val_patches'][:8],meta['best']['epoch'])
        r.update(final_epoch=30,best=meta['best'],parameters=sum(p.numel() for p in ns['model'].parameters()),
                 metadata=meta,counts={'train':237871,'val':122342},graph_capture=probe,
                 probe_summary=ns['probe'].summary(),
                 history_timing_scope='checkpoint-to-checkpoint; first includes initialization; later include previous probe; entry SOFT EPOCH excludes probe')
    else:
        check_test_arrays(ns,np)
        metrics=dict(samples=119825,mAP_percent=float(ns['mAP']),macro_f1_percent=float(ns['macro_f1']),
                     micro_f1_percent=float(ns['micro_f1']),f1_threshold=.5,checkpoint_sha256=sha,
                     per_class_ap_percent=dict(zip(ns['CLASSES_19'],[float(x)*100 for x in ns['ap_per_class']])))
        b.save(dest/'metrics.json',metrics)
        np.savez_compressed(dest/'predictions.npz',paths=np.asarray(ns['test_patches']),
                            labels=ns['labels_np'].astype(np.uint8),probabilities=ns['probs_np'])
        r['metrics']=metrics
    elapsed=time.monotonic()-started
    if stage=='gavit_train':
        r['effective_train_images_per_second']=237871*30/elapsed
        r['throughput_scope']='training images counted; wall time also includes validation, initialization, probes, checkpoint IO and artifact checks; not pure model throughput'
    r.update(status='complete',checkpoint=str(ckpt),checkpoint_sha256=sha,
             elapsed_seconds=elapsed,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),main_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             resource_scope='whole stage including checkpoint/probe artifact checks; main RSS excludes loader workers')
    # JSON normalizes tuple values for the shared validator and on-disk report.
    r=json.loads(json.dumps(r,allow_nan=False))
    check_report(r,stage,ready,b)
    b.save(dest/'run.json',r)
    print('STAGE COMPLETE '+stage,flush=True)


def run_stages(root,jobs,b,ready):
    reports={}
    for stage,argv in jobs:
        b.idle_gpu()
        b.run_jobs(root,[(stage,argv)],b.identity)
        r=read(root/stage/'run.json')
        check_report(r,stage,ready,b)
        if stage=='gavit_test':
            require(r['checkpoint_sha256']==reports['gavit_train']['checkpoint_sha256'], 'Test report used a different checkpoint')
        reports[stage]=r
    return reports


def claim_run(root):
    # Reject duplicate invocations before they can write any exit/summary marker.
    with (root/'started.claim').open('x') as f:
        f.write(str(os.getpid())+'\n')


def main(root):
    import fcntl
    b=load_base()
    with (root.parent/'.bigearth_soft45.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),str(root)]
        # Parent stays stdlib-only: child preflight may initialize CUDA and exits first.
        b.run_jobs(root,[('preflight',command+['preflight'])],b.idle_gpu)
        ready=read(root/'preflight.json')
        require(ready==read(root/'approved_preflight.json'), 'Preflight changed since user return; inspect before training')
        reports=run_stages(root,[(s,command+[s]) for s in ('gavit_train','gavit_test')],b,ready)
        b.save(root/'summary.json',dict(status='complete',dataset='BigEarthNet-19',seed=45,
               elapsed_seconds=time.monotonic()-started,controls=ready['controls'],
               physical_gpu_identity_changed=ready['physical_gpu_identity_changed'],
               reference_gpu_identity=ready['reference_gpu_identity'],stages=reports))
        print('BIGEARTH SOFT45 COMPLETE '+str(root),flush=True)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    require(not sys.flags.optimize, 'Disable PYTHONOPTIMIZE')
    root=Path(sys.argv[1]).resolve()
    if len(sys.argv)==3:
        if sys.argv[2]=='preflight':
            script=(root/'preflight.py').read_bytes()
            require(hashlib.sha256(script).hexdigest()==PREFLIGHT_SHA,'Preflight script changed')
            ns={'__name__':'soft45_preflight','__file__':str(root/'preflight.py')}
            exec(compile(script,ns['__file__'],'exec'),ns)
            ns['b'].save(root/'preflight.json',ns['report'])
        else:
            worker(root,sys.argv[2])
    else:
        claim_run(root)
        rc=1
        try:
            main(root)
            rc=0
        except BaseException:
            traceback.print_exc()
        finally:
            # Preserve an existing exit marker if someone invokes an old queue directly.
            if not (root/'run.exit').exists():
                (root/'run.exit').write_text(f'exit_code={rc}\n')
        sys.exit(rc)
