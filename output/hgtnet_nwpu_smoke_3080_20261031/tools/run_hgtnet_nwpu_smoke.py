"""Offline HGTNet NWPU360/90 smoke; inspect/status never import ML libraries."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from external_baselines.hgtnet import nwpu_data as DATA, resources as RES
from external_baselines.hgtnet.config import load_config, config_sha256

SCHEMA='hgtnet-nwpu-3080-smoke-v1'
CONFIG_SHA='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25'
WEIGHTS_SHA='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
WEIGHTS_BYTES=351654207


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def _write(path,value,replace=False):
    path=Path(path)
    if replace:
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        os.replace(temporary,path)
    else:
        with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)


def _source_identity():
    files=sorted((ROOT/'external_baselines/hgtnet').glob('*.py'))
    files += [ROOT/'external_baselines/__init__.py',ROOT/'tools/run_dfagcn_nwpu_smoke.py',
              Path(__file__).resolve(),ROOT/'configs/hgtnet_independent_v0.json']
    return {str(path.relative_to(ROOT)):_sha(path) for path in files}


def run_command(args,uuid):
    entry=os.environ.get('HGTNET_DELIVERY_ENTRY')
    command=[sys.executable,entry] if entry else [sys.executable,str(Path(__file__).resolve())]
    command += ['run','--gpu-uuid',uuid]
    for name in ['manifest','data_root','archive','weights','config','output','scratch_root']:
        command += ['--'+name.replace('_','-'),str(getattr(args,name))]
    return shlex.join(command)


def inspect_resources(args):
    blockers=[]
    report={'schema':SCHEMA,'action':'inspect','ready':False,'formal_training_started':False,
            'manifest':str(args.manifest),'weights':str(args.weights),'output':str(args.output),
            'scratch_root':str(args.scratch_root),'requested_gpu_uuid':args.gpu_uuid,'blockers':blockers,
            'readiness_scope':'stdlib preflight only; run verifies real imports/CUDA and staged ZIP/image bytes'}
    try:
        cfg=load_config(args.config)
        if config_sha256(cfg)!=CONFIG_SHA:raise ValueError('frozen HGTNet candidate config changed')
        report['config_sha256']=CONFIG_SHA
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('config');report['config_error']=str(exc)
    try:
        inputs=DATA.prepare_inputs(args.manifest)
        report.update(manifest_sha256=DATA.MANIFEST_SHA,counts=inputs['counts'])
        try:
            DATA.validate_selected(args.data_root,inputs['selected']['pool'])
            report['data_source']='verified_images'
        except (OSError,ValueError) as exc:
            report['data_error']=str(exc)
            if not args.archive.is_file() or args.archive.is_symlink() or args.archive.stat().st_size<=0:
                raise ValueError('selected images unavailable and persistent NWPU ZIP not present')
            report.update(data_source='persistent_zip_pending_full_hash_and_selected_recovery',
                          archive_sha_verified=False,archive_expected_sha256=DATA.ARCHIVE_SHA,
                          archive_bytes=args.archive.stat().st_size)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('manifest' if 'inputs' not in locals() else 'data');report['input_error']=str(exc)
    try:
        if args.weights.is_symlink() or args.weights.stat().st_size!=WEIGHTS_BYTES or _sha(args.weights)!=WEIGHTS_SHA:
            raise ValueError('complete official Swin-B candidate bytes/SHA required')
        report.update(weights_sha256=WEIGHTS_SHA,weights_bytes=WEIGHTS_BYTES)
    except (OSError,ValueError) as exc:blockers.append('weights');report['weights_error']=str(exc)
    report['dependencies']=RES.dependency_inventory()
    blockers.extend(RES.dependency_blockers(report['dependencies']))
    report['nvidia']=RES.inventory_gpu()
    try:report['selected_gpu']=RES.select_gpu(args.gpu_uuid,report['nvidia'])
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    try:RES.require_new_paths(args.output,args.scratch_root)
    except (ValueError,OSError,RuntimeError) as exc:blockers.append('output_or_scratch');report['path_error']=str(exc)
    report['disk']=RES.capacity(args.output,args.scratch_root)
    blockers.extend(report['disk']['blockers'])
    report['ready']=not blockers
    if report['ready']:report['run_command']=run_command(args,report['selected_gpu']['uuid'])
    return report


def _runtime_and_torch(args,gpu):
    loaded=sys.modules.get('torch')
    if loaded is not None and loaded.cuda.is_initialized():raise ValueError('CUDA selection requires a fresh process')
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu_uuid
    import torch
    import torchvision
    import numpy
    from PIL import __version__ as pillow_version
    if (not torch.cuda.is_available() or torch.cuda.device_count()!=1
            or torch.cuda.get_device_name(0)!=RES.TARGET_NAME
            or torch.cuda.get_device_properties(0).total_memory//(1024**2)!=RES.TARGET_MEMORY_MIB):
        raise ValueError('Actual CUDA-visible device differs from selected10GiB RTX3080')
    torch.set_num_threads(8);torch.set_default_dtype(torch.float32)
    random.seed(42);numpy.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':sys.version,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
             'numpy':numpy.__version__,'pillow':pillow_version,'cuda':torch.version.cuda,
             'cudnn':torch.backends.cudnn.version(),'gpu':gpu,'physical_gpu_uuid':args.gpu_uuid,
             'device':'cuda:0','cpu_threads':torch.get_num_threads(),'amp':False,'dtype':'float32','seed':42,
             'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
             'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
             'pid':os.getpid(),'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']}
    RES.check_runtime(runtime,args.gpu_uuid)
    return torch,runtime


class _Tee:
    def __init__(self,original,log):self.original,self.log=original,log
    def write(self,value):self.original.write(value);self.log.write(value);self.flush()
    def flush(self):self.original.flush();self.log.flush()


def _observed_first_step(output):
    try:
        value=json.loads((Path(output)/'first_step.json').read_text())
        return value.get('optimizer_step_succeeded') is True and value.get('optimizer_steps')==1
    except (OSError,ValueError,TypeError):return False


def run_smoke(args):
    if not args.gpu_uuid:raise ValueError('run requires an explicit current physical --gpu-uuid from inspect')
    preflight=inspect_resources(args)
    if not preflight['ready']:raise ValueError('preflight blockers: '+', '.join(preflight['blockers']))
    args.output.mkdir(parents=True,exist_ok=False)
    _write(args.output/'preflight.json',preflight)
    _write(args.output/'command.json',{'argv':list(sys.argv),'cwd':os.getcwd(),'pid':os.getpid()})
    result={'schema':SCHEMA,'completed':False,'training_started':False,'formal_training_started':False,
            'action':'run','input_scope':'real_nwpu','counts':{'train':360,'val':90,'test':0}}
    started=time.perf_counter()
    with (args.output/'run.log').open('x') as log,contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
        try:
            sources=_source_identity();_write(args.output/'source_identity.json',sources)
            torch,runtime=_runtime_and_torch(args,preflight['selected_gpu'])
            _write(args.output/'runtime.json',runtime)
            print('HGTNET_SMOKE_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
            cfg=load_config(args.config);inputs=DATA.prepare_inputs(args.manifest)
            _write(args.output/'selected_inputs.json',inputs)
            args.scratch_root.mkdir(parents=True,exist_ok=False)
            data_root=args.data_root
            used_archive=preflight['data_source']!='verified_images'
            if used_archive:
                data_root=DATA.recover_selected(args.archive,inputs['selected']['pool'],args.scratch_root/'images')
            DATA.validate_selected(data_root,inputs['selected']['pool'])
            from torch.utils.data import DataLoader
            from external_baselines.hgtnet.model import HGTNet,parameter_counts
            from external_baselines.hgtnet.pretrained import load_pretrained_file
            from external_baselines.hgtnet.nwpu_engine import run_epoch_smoke
            train=DATA.make_dataset(data_root,inputs['selected']['train'])
            val=DATA.make_dataset(data_root,inputs['selected']['val'])
            train_loader=DataLoader(train,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(42),num_workers=0,pin_memory=False)
            val_loader=DataLoader(val,batch_size=2,shuffle=False,num_workers=0,pin_memory=False)
            torch.cuda.reset_peak_memory_stats(0)
            model=HGTNet(cfg)
            pretraining=load_pretrained_file(model,args.weights,WEIGHTS_SHA);model.pretraining_loaded=True
            _write(args.output/'pretraining.json',pretraining)
            binding={'schema':'hgtnet-nwpu-smoke-binding-v1','input_scope':'real_nwpu','config':cfg,
                     'config_sha256':CONFIG_SHA,'counts':result['counts'],'class_names':inputs['class_names'],
                     'manifest_sha256':DATA.MANIFEST_SHA,'source_identity':sources,'pretraining':pretraining,
                     'runtime':runtime,'val_paths':[r['path'] for r in inputs['selected']['val']],
                     'val_labels':[r['label'] for r in inputs['selected']['val']]}
            _write(args.output/'binding.json',binding)
            def progress(value):
                _write(args.output/'progress.json',dict(value,pid=os.getpid()),replace=True)
                print('HGTNET_SMOKE_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
            result.update(run_epoch_smoke(model,train_loader,val_loader,args.output,binding=binding,
                          device='cuda:0',model_factory=lambda:HGTNet(cfg),on_progress=progress))
            result.update(schema=SCHEMA,runtime=runtime,parameters=parameter_counts(model),manifest_sha256=DATA.MANIFEST_SHA,
                          config_sha256=CONFIG_SHA,weights_sha256=WEIGHTS_SHA,source_identity=sources,
                          physical_gpu_uuid=args.gpu_uuid,selected_pool_count=450,
                          data_source='persistent_zip_verified_selected_recovery' if used_archive else 'verified_images',
                          archive_sha_verified=used_archive,archive_sha256=DATA.ARCHIVE_SHA if used_archive else None,
                          data_root_used=str(data_root),
                          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                          cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                          memory_scope='whole smoke CUDA allocator including update diagnostics and checkpoint reload; not whole process')
            if (_source_identity()!=sources or _sha(args.manifest)!=DATA.MANIFEST_SHA
                    or _sha(args.weights)!=WEIGHTS_SHA or config_sha256(load_config(args.config))!=CONFIG_SHA):
                raise ValueError('Inputs or implementation changed during smoke')
            DATA.validate_selected(data_root,inputs['selected']['pool'])
            result.update(completed=True,wall_seconds=time.perf_counter()-started,
                          timing_scope='after stdlib preflight; includes actual imports, staged recovery, initialization, epoch, validation and reload')
            _write(args.output/'summary.json',result)
            (args.output/'run.exit').write_text('0\n')
            progress({'stage':'complete','completed':True,'training_started':True})
            print('HGTNET_NWPU_SMOKE_COMPLETE '+str(args.output/'summary.json'),flush=True)
            return 0
        except BaseException as exc:
            result.update(completed=False,training_started=_observed_first_step(args.output),
                          error=type(exc).__name__+': '+str(exc),wall_seconds=time.perf_counter()-started)
            traceback.print_exc()
            if not (args.output/'summary.json').exists():_write(args.output/'summary.json',result)
            (args.output/'run.exit').write_text('1\n')
            _write(args.output/'progress.json',{'stage':'failed','completed':False,
                   'training_started':result['training_started'],'error':result['error']},replace=True)
            return 1


def status(output):
    output=Path(output)
    if not output.exists():return {'schema':SCHEMA,'state':'not_started','output':str(output),'completed':False}
    report={'schema':SCHEMA,'state':'incomplete','output':str(output),'completed':False}
    for filename,key in [('progress.json','progress'),('summary.json','summary')]:
        try:report[key]=json.loads((output/filename).read_text())
        except (OSError,ValueError):pass
    try:report['exit_code']=int((output/'run.exit').read_text().strip())
    except (OSError,ValueError):report['exit_code']=None
    summary=report.get('summary',{})
    if summary:
        report.pop('summary')
        for key in ['training_started','formal_training_started','counts','best_epoch','final_epoch','optimizer_steps',
                    'weights_sha256','config_sha256','manifest_sha256','physical_gpu_uuid','cuda_peak_allocated_bytes',
                    'cuda_peak_reserved_bytes','wall_seconds','error']:
            if key in summary:report[key]=summary[key]
        if 'val_metrics' in summary:report['smoke_val']={k:summary['val_metrics'][k] for k in ['accuracy_percent','macro_f1_percent']}
        if 'checkpoint' in summary:report['checkpoint']={k:summary['checkpoint'][k] for k in ['sha256','reload_max_abs_error']}
        if summary.get('completed') is True and report['exit_code']==0:report.update(state='complete',completed=True)
    if report['exit_code'] is not None and report['exit_code']!=0:report['state']='failed'
    return report


def parser():
    result=argparse.ArgumentParser(description=__doc__);sub=result.add_subparsers(dest='action',required=True)
    for action in ['inspect','run','status']:
        command=sub.add_parser(action)
        command.add_argument('--output',type=Path,default=ROOT/'runs/nwpu_smoke_seed42_01')
        if action=='status':continue
        command.add_argument('--gpu-uuid')
        command.add_argument('--manifest',type=Path,default=DATA.MANIFEST)
        command.add_argument('--data-root',type=Path,default=DATA.DATA_ROOT)
        command.add_argument('--archive',type=Path,default=DATA.ARCHIVE)
        command.add_argument('--weights',type=Path,default=ROOT.parent/'swin_b-68c6b09e.pth')
        command.add_argument('--config',type=Path,default=ROOT/'configs/hgtnet_independent_v0.json')
        command.add_argument('--scratch-root',type=Path,default=Path('/tmp/hgtnet_nwpu_smoke_3080_20261031_seed42_01'))
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='status':print(json.dumps(status(args.output),indent=2,sort_keys=True));return 0
    if args.action=='inspect':print(json.dumps(inspect_resources(args),indent=2,sort_keys=True));return 0
    return run_smoke(args)


if __name__=='__main__':raise SystemExit(main())
