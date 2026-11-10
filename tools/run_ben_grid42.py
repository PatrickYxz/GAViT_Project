"""One authorized BEN fixed-K9 seed42 run: gate, smoke, full30, best test.

Runs outside existing checkouts. Historical K16 is retained as a separately
identified reference, with hardware/software-batch limitations reported.
"""
import csv
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
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ben_grid42_patch import BASE, digest, topology, verify_snapshot

WORK=Path('/home/featurize/work')
RUN=WORK/'ben_grid3_seed42_20261018_01'
SPLITS=WORK/'GAViT_Project/bigearth_files/splits'
PRETRAINED=SPLITS.parent/'model.safetensors'
BASE_HELPER=WORK/'bigearth_seed45_20260922_01/run_seed45.py'
BASE_HELPER_SHA='8ac0083f41ccfb38555ed428d99971d075cb1d7ec3e413b4fa56548dcc1d0369'
CONTROL_STEM='best_bigearth_gavit_sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654'
CONTROL_SHA='9b9acb76d93fa8799e979a227ce1d2496573da937b9329ab0ce3ca1c3f29859e'
CONTROL_LOG='sparse_hybrid_4n_top2_cd39be9_formal_seed42_20260814_134654_test_rtx3060_20260817_114017.log'
CONTROL_LOG_SHA='e2105d38c87720b344d3da92c84ac67e73734e717367c99d703dd800d0e48184'


def require(ok,message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def architecture():
    return dict(num_classes=19,num_regions=9,knn_k=5,gat_hidden=256,gat_heads=4,gat_layers=2,
                dropout=.1,grouping='attentive_spatial',edge_type='sparse_hybrid',integration='token_feedback')


def recipe(epochs):
    return dict(seed=42,epochs=epochs,batch_size=32,lr=.0003,weight_decay=.0001,
                optimizer='AdamW',scheduler='CosineAnnealingLR',loss='BCEWithLogitsLoss',f1_threshold=.5)


def entry_args(root,stage):
    root=Path(root)
    if stage not in ('smoke','train','test'):
        raise ValueError('Unknown stage')
    if stage=='test':
        return [str(root/'code/test_bigearth.py'),'--model','gavit','--data_dir',str(SPLITS),
                '--ckpt',str(root/'train/gavit.pth'),'--batch_size','32','--seed','42']
    return [str(root/'code/train_bigearth.py'),'--model','gavit','--data_dir',str(root/'smoke_splits' if stage=='smoke' else SPLITS),
            '--epochs','1' if stage=='smoke' else '30','--batch_size','32','--lr','0.0003','--seed','42',
            '--num_regions','9','--grouping','attentive_spatial','--edge_type','sparse_hybrid','--knn_k','5',
            '--gat_hidden','256','--gat_heads','4','--gat_layers','2','--integration','token_feedback','--dropout','0.1',
            '--run_stage','smoke' if stage=='smoke' else 'formal','--run_tag',root.name+'_'+stage,
            '--pretrained_path',str(PRETRAINED),'--checkpoint_path',str(root/stage/'gavit.pth')]


def claim_run(root):
    with (Path(root)/'started.claim').open('x') as f:
        f.write(str(os.getpid())+'\n')


def load_base():
    require(digest(BASE_HELPER)==BASE_HELPER_SHA,'Existing data/queue helper changed; preserve and inspect')
    spec=importlib.util.spec_from_file_location('grid42_existing_data_gate',BASE_HELPER)
    b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
    return b


def identity(root):
    root=Path(root);release=read(root/'release.json')
    for name,expected in release['tools'].items():
        require(digest(root/name)==expected,'Experiment helper changed: '+name)
    verify_snapshot(root/'code',release['snapshot'])
    b=load_base()
    require(digest(PRETRAINED)==b.PRETRAINED_SHA,'Pretrained bytes changed')
    for split,(_,sha) in b.FROZEN.items():
        require(digest(SPLITS/(split+'.csv'))==sha,'Split bytes changed: '+split)
    return b,release


def environment():
    import torch
    require(not sys.flags.optimize,'Disable PYTHONOPTIMIZE')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required')
    torch.set_num_threads(8)
    require(torch.get_default_dtype()==torch.float32,'FP32 dtype required')
    # Explicitly fix and record precision instead of relying on new defaults.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    hardware=dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                  threads=torch.get_num_threads(),python=sys.version,
                  packages={n:importlib.metadata.version(n) for n in ('torch','torchvision','timm','torch-geometric','numpy','scikit-learn','pillow')},
                  gpu_identity=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip())
    gpu=list(csv.reader(hardware['gpu_identity'].splitlines(),skipinitialspace=True))
    require(len(gpu)==1 and len(gpu[0])==3 and gpu[0][0].startswith('GPU-') and
            hardware['gpu']==gpu[0][1]=='NVIDIA GeForce RTX 4090','Expected one RTX4090 with UUID')
    require(hardware['packages']['torch'].split('+')[0]=='2.2.2','Expected validated torch2.2.2 runtime')
    precision=dict(dtype='torch.float32',matmul_tf32=False,cudnn_tf32=True,amp=False)
    return hardware,precision


def control_record():
    ckpt=WORK/'GAViT_Project/checkpoints'/(CONTROL_STEM+'.pth')
    log=WORK/'GAViT_Project/logs'/CONTROL_LOG
    require(digest(ckpt)==CONTROL_SHA and digest(log)==CONTROL_LOG_SHA,'Historical K16 control bytes changed')
    meta=read(ckpt.with_suffix('.meta.json'))
    require(meta['checkpoint_sha256']==CONTROL_SHA and meta['architecture']=={**architecture(),'num_regions':16}
            and meta['training']==recipe(30) and meta['best']['epoch']==13,'Historical K16 identity mismatch')
    return dict(checkpoint=str(ckpt),checkpoint_sha256=CONTROL_SHA,test_log=str(log),test_log_sha256=CONTROL_LOG_SHA,
                metadata=meta,metrics_printed=dict(mAP_percent=72.,macro_f1_percent=66.,micro_f1_percent=76.7),
                limitations='Historical4090 training/3060 test; new K9 uses current4090. Same seed and frozen base do not remove physical-GPU/software-batch confounding. Historical mAP printed to one decimal.')


def existing_k9(work,own_root):
    found=[]
    for parent in Path(work).iterdir():
        if not parent.is_dir() or parent.resolve()==Path(own_root).resolve():
            continue
        if not any(word in parent.name.lower() for word in ('ben','bigearth','gavit')):
            continue
        for folder,dirs,files in os.walk(parent,followlinks=False):
            folder=Path(folder)
            dirs[:]=[d for d in dirs if d not in ('.git','.local','.cache','.venv','venv','data','datasets',
                         'bigearth_files','node_modules','__pycache__','.worktrees','code')]
            if len(folder.relative_to(parent).parts)>=4:
                dirs[:]=[]
            for name in files:
                if not name.endswith('.meta.json'):
                    continue
                path=folder/name
                try:
                    meta=read(path)
                except (OSError,ValueError):
                    if 'k9' in name.lower():
                        found.append(dict(path=str(path),status='unreadable K9 metadata'))
                    continue
                if meta.get('dataset')=='BigEarthNet-19' and meta.get('architecture',{}).get('num_regions')==9 and meta.get('training',{}).get('seed')==42:
                    found.append(dict(path=str(path),status=meta.get('execution',{}).get('run_stage','unknown')))
    return found


def check_training(meta,history,last_epoch,epochs,snapshot_sha):
    require(meta['model']=='gavit' and meta['dataset']=='BigEarthNet-19' and
            meta['architecture']==architecture() and meta['training']==recipe(epochs),'Wrong model, architecture or recipe')
    require(meta['graph_topology']==topology(9) and meta['code']['snapshot_sha256']==snapshot_sha,'Wrong graph/code identity')
    require(meta['execution']['run_stage']==('smoke' if epochs==1 else 'formal'),'Wrong run stage')
    require(last_epoch==epochs and [r['epoch'] for r in history]==list(range(1,epochs+1)),'Incomplete training')
    require(all(all(math.isfinite(v) for v in row.values()) for row in history),'Nonfinite training history')
    best=max(history,key=lambda row:row['val_mAP_percent'])
    require(meta['best']['metric']=='val_mAP' and meta['best']['epoch']==best['epoch'] and
            abs(meta['best']['value']-best['val_mAP_percent'])<=.000051,'Best checkpoint differs from validation history')


def check_metrics(m,sha):
    require(m['samples']==119825 and m['f1_threshold']==.5 and m['checkpoint_sha256']==sha,'Wrong test identity')
    ap=list(m['per_class_ap_percent'].values())
    require(len(ap)==19 and all(math.isfinite(v) and 0<=v<=100 for v in ap),'Invalid per-class AP')
    require(abs(sum(ap)/19-m['mAP_percent'])<1e-8,'mAP inconsistent with classes')
    require(all(math.isfinite(m[k]) and 0<=m[k]<=100 for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')),'Invalid test metrics')


def run_stage(root,stage,command):
    root=Path(root);dest=root/stage;dest.mkdir(exist_ok=False)
    save(dest/'command.json',command)
    print('START '+stage,flush=True)
    with (dest/'run.log').open('wb') as out:
        proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (dest/'run.pid').write_text(str(proc.pid)+'\n')
        while True:
            chunk=proc.stdout.read1(65536)
            if not chunk:
                break
            out.write(chunk);out.flush()
            sys.stdout.buffer.write(chunk);sys.stdout.buffer.flush()
        rc=proc.wait()
        proc.stdout.close()
    (dest/'run.exit').write_text(f'exit_code={rc}\n')
    if rc:
        print('FAILED '+stage+'; inspect '+str(dest/'run.log'),flush=True)
        raise subprocess.CalledProcessError(rc,command)
    print('STAGE_COMPLETE '+stage,flush=True)


def preflight(root):
    b,release=identity(root)
    inventory=existing_k9(WORK,root)
    save(root/'existing_k9_inventory.json',inventory)
    require(not inventory,'Existing K9 seed42 artifacts require inspection; do not retrain: '+json.dumps(inventory))
    hardware,precision=environment()
    control=control_record()
    reference=read(WORK/'bigearth_soft42_20261013_01/preflight.json')
    require(all(hardware[k]==reference['hardware'][k] for k in ('cuda','cudnn','python','packages')),
            'Runtime differs from validated soft42; inspect before training')
    require(precision==reference['precision'],'Precision differs from recorded FP32 protocol')
    # A full split/RGB existence scan precedes even the one-epoch smoke.
    seen=set();scans={};smoke=root/'smoke_splits';smoke.mkdir(exist_ok=False)
    for name,(count,sha) in b.FROZEN.items():
        report=b.scan_split(SPLITS/(name+'.csv'),count,sha,seen)
        if name!='test':
            with (smoke/(name+'.csv')).open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['patch_path']+[f'label_{i}' for i in range(19)])
                writer.writerows(report['examples'][:96 if name=='train' else 64])
        report.pop('examples');scans[name]=report
    sys.path.insert(0,str(root/'code'))
    from ben_grid42_tensor_checks import run_checks
    checks=run_checks(root/'code')
    require(checks['status']=='pass','K9/K16 tensor checks failed')
    save(root/'preflight.json',dict(status='BEN_GRID3_42_PREFLIGHT_PASS',seed=42,hardware=hardware,precision=precision,
         splits=scans,control=control,tensor_checks=checks,source=release['snapshot'],training_started=False,
         smoke_split_sha256={s:digest(smoke/(s+'.csv')) for s in ('train','val')}))
    print('BEN_GRID3_42_PREFLIGHT_PASS',flush=True)


def worker(root,stage):
    b,release=identity(root)
    os.chdir(root/'code');sys.path.insert(0,str(root/'code'))
    import torch
    import numpy as np
    hardware,precision=environment();ready=read(root/'preflight.json')
    require(hardware==ready['hardware'] and precision==ready['precision'],'GPU/software changed since preflight')
    require(ready['source']==release['snapshot'],'Source changed since preflight')
    for s,sha in ready['smoke_split_sha256'].items():
        require(digest(root/'smoke_splits'/(s+'.csv'))==sha,'Smoke split changed')
    from ben_grid42_tensor_checks import install_capture
    captured=install_capture(9)
    dest=root/stage;history=[];previous=[time.monotonic()]
    import utils
    original=utils.save_epoch_artifacts
    def record(*args,**kwargs):
        ns=inspect.currentframe().f_back.f_globals
        args[4]['code']=release['snapshot']
        result=original(*args,**kwargs)
        now=time.monotonic()
        row=dict(epoch=kwargs['epoch'],train_loss=float(ns['avg_loss']),val_mAP_percent=float(kwargs['metric_value']),
                 val_macro_f1_percent=float(ns['val_f1']),next_epoch_lr=float(args[1].param_groups[0]['lr']),
                 epoch_wall_seconds=now-previous[0])
        require(all(math.isfinite(v) for v in row.values()),'Nonfinite epoch')
        history.append(row);save(dest/'history.json',history);previous[0]=now
        print('EPOCH_RESOURCE '+json.dumps(row),flush=True)
        return result
    if stage!='test':
        utils.save_epoch_artifacts=record
    else:
        previous_report=read(root/'train/run.json')
        require(digest(root/'train/gavit.pth')==previous_report['checkpoint_sha256'],'Test weights changed')
    argv=entry_args(root,stage);save(dest/'run.json',dict(status='running',stage=stage,argv=argv))
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();sys.argv=argv
    print('DEVICE '+json.dumps(hardware)+' PRECISION '+json.dumps(precision),flush=True)
    try:
        ns=runpy.run_path(argv[0],run_name='__main__')
    finally:
        utils.save_epoch_artifacts=original
    ckpt=root/('train' if stage=='test' else stage)/'gavit.pth'
    meta=read(ckpt.with_suffix('.meta.json'));sha=digest(ckpt)
    require(meta['checkpoint_sha256']==sha and captured.get('actual_forward_checked'),'Weight hash or graph capture failed')
    require(meta['architecture']==architecture() and meta['code']==release['snapshot'],'Wrong saved architecture or source')
    report=dict(status='complete',stage=stage,model='gavit',dataset='BigEarthNet-19',seed=42,argv=argv,
                base_commit=BASE,source=release['snapshot'],hardware=hardware,precision=precision,
                checkpoint=str(ckpt),checkpoint_sha256=sha,metadata=meta,graph_capture=captured)
    if stage!='test':
        epochs=1 if stage=='smoke' else 30
        state=torch.load(ns['TRAIN_STATE_PATH'],map_location='cpu',weights_only=True)
        check_training(meta,history,state['epoch'],epochs,release['snapshot']['snapshot_sha256']);del state
        counts={'train':len(ns['train_set']),'val':len(ns['val_set'])}
        require(counts==({'train':96,'val':64} if stage=='smoke' else {'train':237871,'val':122342}),'Wrong sample counts')
        require(ns['first_step_logged'] and ns['DEVICE']=='cuda','First optimizer batch not verified')
        require(ns['total_params']==31366926,'Unexpected parameter count')
        if stage=='smoke':
            x=ns['val_set'][0][0].unsqueeze(0).to('cuda');model=ns['model'].eval()
            from models.gavit import GAViT
            fresh=GAViT(**meta['architecture'],pretrained=False).to('cuda').eval()
            fresh.load_state_dict(torch.load(ckpt,map_location='cpu',weights_only=True),strict=True)
            with torch.no_grad():
                diff=float((model(x)-fresh(x)).abs().max())
            require(diff<=1e-5,'Smoke checkpoint reload mismatch');report['reload_max_logit_diff']=diff
        report.update(final_epoch=epochs,best=meta['best'],counts=counts,parameters=ns['total_params'])
    else:
        prob,y=ns['probs_np'],ns['labels_np']
        require(prob.shape==y.shape==(119825,19) and np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all(),'Invalid test arrays')
        require(np.isin(y,[0,1]).all() and (y.sum(0)>0).all() and np.array_equal(y,ns['test_labels']),'Wrong test labels')
        require(len(ns['test_patches'])==len(set(ns['test_patches']))==119825,'Wrong test paths')
        metrics=dict(samples=119825,mAP_percent=float(ns['mAP']),macro_f1_percent=float(ns['macro_f1']),
                     micro_f1_percent=float(ns['micro_f1']),f1_threshold=.5,checkpoint_sha256=sha,
                     per_class_ap_percent=dict(zip(ns['CLASSES_19'],[float(v)*100 for v in ns['ap_per_class']])))
        check_metrics(metrics,sha);save(dest/'metrics.json',metrics)
        np.savez_compressed(dest/'predictions.npz',paths=np.asarray(ns['test_patches']),labels=y.astype(np.uint8),probabilities=prob)
        report['metrics']=metrics
    torch.cuda.synchronize()
    report.update(elapsed_seconds=time.monotonic()-started,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),main_process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  resource_scope='entire entry and checkpoint checks; CPU RSS excludes loader workers')
    if stage=='train':
        report['effective_training_images_per_second']=237871*30/report['elapsed_seconds']
        report['throughput_scope']='training image count divided by full stage wall time including validation and IO'
    save(dest/'run.json',report);print('STAGE_COMPLETE '+stage,flush=True)


def main(root):
    import fcntl
    with (root.parent/'.ben_grid3_seed42.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        b,_=identity(root)
        command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),str(root)]
        started=time.monotonic();reports={}
        for stage in ('preflight','smoke','train','test'):
            identity(root);b.idle_gpu()
            run_stage(root,stage,command+[stage])
            if stage=='preflight':
                require(read(root/'preflight.json')['status']=='BEN_GRID3_42_PREFLIGHT_PASS','Preflight not passed')
            else:
                reports[stage]=read(root/stage/'run.json')
                require(reports[stage]['status']=='complete','Incomplete stage')
                require(reports[stage]['hardware']==read(root/'preflight.json')['hardware'],'Stage GPU changed')
                if stage=='smoke':
                    require(reports[stage]['reload_max_logit_diff']<=1e-5,'Smoke gate failed')
                    with (root/'experiment_diary.md').open('x') as diary:
                        diary.write('# BEN K9 seed42 experiment record\n\n'
                            'Preflight and independent 96/64 one-epoch smoke completed before formal training.\n'
                            'Formal30 plus selected-best full test were authorized; historical K16 remains a limited reference.\n\n'
                            '```json\n'+json.dumps(dict(preflight=read(root/'preflight.json'),smoke=reports[stage]),indent=2)+'\n```\n')
        ready=read(root/'preflight.json')
        require(reports['train']['checkpoint_sha256']==reports['test']['checkpoint_sha256'],'Test not paired to trained best')
        delta=reports['test']['metrics']['mAP_percent']-ready['control']['metrics_printed']['mAP_percent']
        save(root/'summary.json',dict(status='complete',dataset='BigEarthNet-19',seed=42,
             elapsed_seconds=time.monotonic()-started,stages=reports,control=ready['control'],
             delta_k9_minus_historical_k16_map_pp=delta,
             comparison_scope='single-seed historical-reference comparison; control rounded to0.1pp; GPU/software batches differ'))
        with (root/'experiment_diary.md').open('a') as diary:
            diary.write('\nFormal training and selected-best test complete.\n\n```json\n'+
                        json.dumps(read(root/'summary.json'),indent=2)+'\n```\n')
        print('BEN_GRID3_SEED42_COMPLETE '+str(root),flush=True)


if __name__=='__main__':
    sys.dont_write_bytecode=True
    root=Path(sys.argv[1]).resolve()
    if len(sys.argv)==3:
        if sys.argv[2]=='preflight':
            preflight(root)
        else:
            require(sys.argv[2] in ('smoke','train','test'),'Unknown stage');worker(root,sys.argv[2])
    else:
        claim_run(root)
        rc=1
        try:
            main(root);rc=0
        except BaseException:
            traceback.print_exc()
        finally:
            with (root/'run.exit').open('x') as f:
                f.write(f'exit_code={rc}\n')
        sys.exit(rc)
