"""Frozen AID240/60 one-epoch smoke on an idle RTX3060; stdlib inspect/status."""
from __future__ import annotations

import argparse
from collections import Counter
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path,PurePosixPath
import platform
import random
import re
import shlex
import sys
import time
import traceback


ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from external_baselines.hgtnet import aid_data as DATA, resources as RES
from external_baselines.hgtnet.config import load_config,config_sha256
open_regular_readonly=DATA.inspector.open_regular_readonly

# Keep this exact source set aligned with the isolated delivery whitelist.
SOURCE_FILES=(
    'external_baselines/__init__.py','external_baselines/hgtnet/__init__.py',
    *('external_baselines/hgtnet/'+name+'.py' for name in (
        'aid_data','aid_engine','aid_entry','config','model','attention','graph',
        'pretrained','smoke','checkpoint','nwpu_engine','resources')),
    'configs/hgtnet_aid_independent_v0.json','tools/inspect_hgtnet_aid_inputs.py',
    'tools/package_hgtnet_aid_smoke.py',
)

SCHEMA='hgtnet-aid-smoke-v1'
CONFIG_SHA='f313abb893b29d674ace281d8d2905670ee65db131f4240640079123e783589b'
WEIGHTS_SHA='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
WEIGHTS_BYTES=351654207
COUNTS={'train':240,'val':60,'test':0}
TARGET_NAME='NVIDIA GeForce RTX 3060'
TARGET_MEMORY_MIB=12288
PROTOCOL={'seed':42,'epochs':1,'batch_size':2,'optimizer':'Adam','lr':1e-4,
          'loss':'CrossEntropyLoss','augmentation':False,'amp':False,'tf32':False,
          'preprocessing':'RGB_resize224_bilinear_ImageNet_normalize'}
COMPLETION_CHECKS=dict(source_identity=True,config=True,manifest=True,weights=True,selected_images=True,
                       selected_image_count=300,test_image_bytes_read=0,checkpoint_sha256=True,checkpoint_reload=True)
GRADIENT_GROUPS={'patch_embed','transformer','dual_attention','graph','head'}


def _sha(path):
    with open_regular_readonly(path) as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def _write(path,value,*,replace=False):
    path=Path(path)
    raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if not replace:
        with path.open('x') as stream:stream.write(raw)
        return
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x') as stream:stream.write(raw)
    os.replace(temporary,path)


def _read_json(path):
    with open_regular_readonly(path) as stream:
        return json.load(stream,object_pairs_hook=DATA.inspector._unique_json_object,
                         parse_constant=DATA.inspector._reject_json_constant)


def _source_identity():
    identity={name:_sha(ROOT/name) for name in SOURCE_FILES}
    metadata=ROOT/'delivery_manifest.json'
    if metadata.exists() or metadata.is_symlink():
        manifest=_read_json(metadata)
        expected={name:manifest['files'][name]['sha256'] for name in SOURCE_FILES}
        if manifest.get('schema')!='hgtnet-aid-delivery-v1' or identity!=expected:
            raise ValueError('installed immutable source identity changed')
    return identity


def lock_path(output):
    path=Path(output)
    return path.with_name(path.name+'.lock')


def _new_output(output,*,check_lock=True):
    output=Path(output).absolute()
    for current in (output,*output.parents):
        if current.is_symlink():raise ValueError('output contains symbolic link: '+str(current))
    if output.exists():raise FileExistsError('output already exists: '+str(output))
    if check_lock and (lock_path(output).exists() or lock_path(output).is_symlink()):
        raise ValueError('output is locked: '+str(lock_path(output)))
    for folder in ('external_baselines','tools','configs'):
        protected=ROOT/folder
        if output==protected or protected in output.parents:
            raise ValueError('output cannot be an immutable source directory')
    return output


@contextlib.contextmanager
def _run_lock(output):
    output=_new_output(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=lock_path(output)
    with lock.open('x') as stream:
        json.dump({'pid':os.getpid(),'output':str(output)},stream)
    try:
        _new_output(output,check_lock=False)
        yield
    finally:lock.unlink()


def select_gpu(requested_uuid,inventory):
    if (not isinstance(inventory,dict) or inventory.get('error')
            or not isinstance(inventory.get('gpus'),list) or not isinstance(inventory.get('processes'),list)):
        raise ValueError('working NVIDIA GPU and process queries are required')
    gpus,processes=inventory['gpus'],inventory['processes']
    for item in gpus:RES._validate_gpu(item)
    for item in processes:RES._validate_process(item)
    if len({item['uuid'] for item in gpus})!=len(gpus):raise ValueError('duplicate physical GPU UUID')
    if requested_uuid is not None and not RES._uuid(requested_uuid):raise ValueError('invalid requested GPU UUID')
    busy={item['gpu_uuid'] for item in processes}
    candidates=[item for item in gpus if item['name']==TARGET_NAME
                and item['memory_total_mib']==TARGET_MEMORY_MIB and item['memory_free_mib']>0
                and item['uuid'] not in busy and (requested_uuid is None or item['uuid']==requested_uuid)]
    if len(candidates)!=1:raise ValueError('exactly one selected idle RTX3060 12288MiB is required')
    return dict(candidates[0])


def _dependency_blockers(inventory):
    blockers=[]
    versions={'torch':{'2.2.2','2.2.2+cu121'},'torchvision':{'0.17.2','0.17.2+cu121'},
              'numpy':{'1.26.4'},'Pillow':{'10.3.0'}}
    python=inventory.get('python',{})
    if python.get('located') is not True or not re.fullmatch(r'3\.11\.\d+',python.get('version') or ''):
        blockers.append('version:python')
    for name,accepted in versions.items():
        item=inventory.get(name,{})
        if item.get('located') is not True:blockers.append('dependency:'+name)
        elif item.get('version') not in accepted:blockers.append('version:'+name)
    return blockers


def run_command(args,uuid):
    delivered=os.environ.get('HGTNET_AID_DELIVERY_ENTRY')
    command=[sys.executable,str(Path(delivered).absolute()) if delivered else str(Path(__file__).resolve()),
             'run','--gpu-uuid',uuid]
    for name in ('manifest','data_root','weights','config','output'):
        command+=['--'+name.replace('_','-'),str(Path(getattr(args,name)).absolute())]
    return shlex.join(command)


def inspect_resources(args):
    blockers=[]
    report=dict(schema=SCHEMA,action='inspect',ready=False,training_started=False,
                formal_training_started=False,input_scope='real_aid',counts=dict(COUNTS),
                target_gpu='3060',requested_gpu_uuid=args.gpu_uuid,
                manifest=str(args.manifest),data_root=str(args.data_root),weights=str(args.weights),
                output=str(args.output),blockers=blockers,
                readiness_scope='stdlib metadata and byte preflight; run separately verifies actual imports/CUDA')
    for label,check in [
        ('source',lambda:report.update(source_identity=_source_identity())),
        ('config',lambda:_check_config(args.config)),
        ('output',lambda:_new_output(args.output)),
    ]:
        try:check()
        except (OSError,ValueError,KeyError,RuntimeError) as exc:
            blockers.append(label);report[label+'_error']=str(exc)
    try:
        inputs=DATA.prepare_inputs(args.manifest)
        if inputs['counts']!=COUNTS:raise ValueError('AID selected counts changed')
        report.update(manifest_sha256=inputs['manifest_sha256'],selected_pool_count=len(inputs['selected']['pool']))
        report['selected_images']=DATA.validate_selected(args.data_root,inputs['selected']['pool'])
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('inputs');report['input_error']=str(exc)
    try:
        with open_regular_readonly(args.weights) as stream:
            size=os.fstat(stream.fileno()).st_size
            digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if size!=WEIGHTS_BYTES or digest!=WEIGHTS_SHA:raise ValueError('complete candidate weights size/SHA mismatch')
        report.update(weights_sha256=digest,weights_bytes=size)
    except (OSError,ValueError) as exc:
        blockers.append('weights');report['weights_error']=str(exc)
    report['dependencies']=RES.dependency_inventory()
    blockers.extend(_dependency_blockers(report['dependencies']))
    report['nvidia']=RES.inventory_gpu()
    try:report['selected_gpu']=select_gpu(args.gpu_uuid,report['nvidia'])
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    try:
        report['disk']=RES._capacity_path(args.output,RES.WORK_BYTES)
        if report['disk']['free_bytes']<RES.WORK_BYTES:blockers.append('disk_capacity')
    except (OSError,ValueError) as exc:blockers.append('disk_capacity');report['disk_error']=str(exc)
    report['ready']=not blockers
    if report['ready']:
        report['config_sha256']=CONFIG_SHA
        report['run_command']=run_command(args,report['selected_gpu']['uuid'])
    return report


def _check_config(path):
    with open_regular_readonly(path) as stream:config=json.load(stream)
    if config_sha256(config)!=CONFIG_SHA:raise ValueError('frozen 30-class AID configuration changed')
    return config


def _check_cuda_device(probe,gpu):
    def reject(reason):
        raise ValueError(reason+'; CUDA probe='+repr(probe)+'; selected NVML GPU='+repr(gpu))
    if not isinstance(probe,dict) or not isinstance(gpu,dict):reject('CUDA/NVML mappings required')
    if probe.get('available') is not True:reject('CUDA is not available')
    if type(probe.get('device_count')) is not int or probe['device_count']!=1:reject('exactly one CUDA-visible device required')
    if probe.get('name')!=gpu.get('name'):reject('CUDA-visible name differs from selected GPU')
    nominal,actual=gpu.get('memory_total_mib'),probe.get('total_memory_bytes')
    if type(nominal) is not int or nominal<=0 or type(actual) is not int or actual<=0:reject('positive integer capacities required')
    nominal_bytes=nominal*1024**2
    if not nominal_bytes-1024**3<=actual<=nominal_bytes:reject('CUDA capacity outside NVML minus at most1GiB')
    return dict(probe,nvml_total_mib=nominal,cuda_total_mib=actual//1024**2,
                interface_gap_bytes=nominal_bytes-actual,interface_gap_mib=(nominal_bytes-actual)/1024**2,
                max_interface_gap_mib=1024)


def check_runtime(runtime,uuid):
    if not re.fullmatch(r'3\.11\.\d+',runtime.get('python') or ''):raise ValueError('actual runtime Python3.11 required')
    fixed=dict(torch='2.2.2+cu121',torchvision='0.17.2+cu121',numpy='1.26.4',pillow='10.3.0',
               cuda='12.1',cudnn=8902,physical_gpu_uuid=uuid,cuda_visible_devices=uuid,device='cuda:0',
               cpu_threads=8,amp=False,dtype='float32',seed=42,matmul_tf32=False,cudnn_tf32=False,
               cudnn_benchmark=False,cudnn_deterministic=True)
    for name,expected in fixed.items():
        actual=runtime.get(name)
        if type(actual) is not type(expected) or actual!=expected:raise ValueError('actual runtime mismatch: '+name)
    selected=runtime.get('gpu',{})
    if (selected.get('uuid')!=uuid or selected.get('name')!=TARGET_NAME
            or type(selected.get('memory_total_mib')) is not int or selected['memory_total_mib']!=TARGET_MEMORY_MIB):
        raise ValueError('actual runtime physical GPU identity mismatch')
    return runtime


def _runtime_and_torch(args,gpu):
    if 'torch' in sys.modules:raise ValueError('fresh process required for CUDA selection before Torch import')
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    import torch
    import torchvision
    import numpy
    from PIL import __version__ as pillow_version
    probe=dict(available=torch.cuda.is_available(),device_count=torch.cuda.device_count(),
               name=None,total_memory_bytes=None)
    if probe['available'] and probe['device_count']==1:
        probe.update(name=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    print('HGTNET_AID_CUDA_DEVICE_PROBE '+json.dumps(probe,sort_keys=True),flush=True)
    cuda_device=_check_cuda_device(probe,gpu)
    torch.cuda.set_device(0)
    torch.set_num_threads(8);torch.set_default_dtype(torch.float32)
    random.seed(42);numpy.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                 numpy=numpy.__version__,pillow=pillow_version,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                 gpu=gpu,physical_gpu_uuid=gpu['uuid'],device='cuda:0',cpu_threads=torch.get_num_threads(),
                 amp=False,dtype='float32',seed=42,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32,cudnn_benchmark=torch.backends.cudnn.benchmark,
                 cudnn_deterministic=torch.backends.cudnn.deterministic,pid=os.getpid(),
                 cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device=cuda_device)
    check_runtime(runtime,gpu['uuid'])
    return torch,runtime


class _Tee:
    def __init__(self,original,log):self.original,self.log=original,log
    def write(self,value):self.original.write(value);self.log.write(value);self.flush()
    def flush(self):self.original.flush();self.log.flush()


def _observed_training(output):
    steps=0;started=False
    for name in ('optimizer_progress.json','first_step.json','progress.json'):
        try:value=_read_json(Path(output)/name)
        except (OSError,ValueError):continue
        if not isinstance(value,dict):continue
        actual=value.get('optimizer_steps')
        if type(actual) is int and actual>steps:steps=actual
        started=started or value.get('training_started') is True or (type(actual) is int and actual>0)
    return dict(training_started=started,optimizer_steps=steps)


def _checkpoint_ok(output,checkpoint):
    if not isinstance(checkpoint,dict):return False
    error=checkpoint.get('reload_max_abs_error')
    if type(error) not in (int,float) or not math.isfinite(error) or not 0<=error<=1e-6:return False
    if checkpoint.get('reload_tolerance')!=1e-6:return False
    if (type(checkpoint.get('best_epoch')) is not int or checkpoint['best_epoch']!=1
            or type(checkpoint.get('final_epoch')) is not int or checkpoint['final_epoch']!=1
            or checkpoint.get('synthetic_inputs') is not False):return False
    path=(Path(output)/'smoke_epoch1.pt').absolute()
    if checkpoint.get('path')!=str(path) or type(checkpoint.get('bytes')) is not int or checkpoint['bytes']<=0:return False
    try:
        with open_regular_readonly(path) as stream:
            return (os.fstat(stream.fileno()).st_size==checkpoint['bytes']
                    and hashlib.file_digest(stream,'sha256').hexdigest()==checkpoint.get('sha256'))
    except (OSError,ValueError):return False


def _same_json(actual,expected):
    return (json.dumps(actual,sort_keys=True,separators=(',',':'),allow_nan=False)
            ==json.dumps(expected,sort_keys=True,separators=(',',':'),allow_nan=False))


def _completion_evidence_ok(output,summary):
    """Check recorded completion consistently using JSON and byte hashes only.

    Checkpoint tensor/schema/reload validation is performed by the engine during
    run. Status verifies its recorded receipt and immutable bytes without loading
    pickle or importing ML packages.
    """
    try:
        if (summary.get('schema')!=SCHEMA or summary.get('input_scope')!='real_aid'
                or summary.get('completed') is not True or summary.get('training_started') is not True
                or summary.get('formal_training_started') is not False or not _same_json(summary.get('counts'),COUNTS)
                or any(type(summary.get(key)) is not int or summary[key]!=expected
                       for key,expected in [('optimizer_steps',120),('best_epoch',1),('final_epoch',1),('selected_pool_count',300)])
                or summary.get('config_sha256')!=CONFIG_SHA or summary.get('manifest_sha256')!=DATA.MANIFEST_SHA
                or summary.get('weights_sha256')!=WEIGHTS_SHA or not _same_json(summary.get('protocol'),PROTOCOL)
                or not _same_json(summary.get('completion_checks'),COMPLETION_CHECKS)
                or summary.get('data_source')!='verified_images' or not _checkpoint_ok(output,summary.get('checkpoint'))):
            return False
        runtime=summary['runtime'];uuid=summary['physical_gpu_uuid']
        if not RES._uuid(uuid):return False
        check_runtime(runtime,uuid);_check_cuda_device(runtime['cuda_device'],runtime['gpu'])
        output=Path(output)
        if not _same_json(_read_json(output/'runtime.json'),runtime):return False
        sources=_source_identity()
        if summary.get('source_identity')!=sources or _read_json(output/'source_identity.json')!=sources:return False
        binding=_read_json(output/'binding.json')
        if (binding['schema']!='hgtnet-aid-smoke-binding-v1' or binding['input_scope']!='real_aid'
                or not _same_json(binding['runtime'],runtime) or binding['source_identity']!=sources
                or not _same_json(binding['counts'],COUNTS)
                or binding['config_sha256']!=CONFIG_SHA or config_sha256(binding['config'])!=CONFIG_SHA
                or binding['manifest_sha256']!=DATA.MANIFEST_SHA or binding['weights_sha256']!=WEIGHTS_SHA):return False
        names=binding['class_names']
        if (not isinstance(names,list) or len(names)!=30 or len(set(names))!=30
                or any(not isinstance(name,str) or not name for name in names) or names!=sorted(names)
                or {re.sub('[^a-z]','',name.lower()) for name in names}!=DATA.inspector.AID_CLASSES):return False
        all_paths=[]
        for split,count,per_class in [('train',240,8),('val',60,2)]:
            paths,labels=binding[split+'_paths'],binding[split+'_labels']
            if (not isinstance(paths,list) or not isinstance(labels,list) or len(paths)!=count or len(labels)!=count
                    or any(type(label) is not int or not 0<=label<30 for label in labels)
                    or Counter(labels)!={label:per_class for label in range(30)}):return False
            for path,label in zip(paths,labels):
                parsed=PurePosixPath(path)
                if (not isinstance(path,str) or not path or '\\' in path or '\x00' in path or parsed.is_absolute()
                        or '..' in parsed.parts or parsed.as_posix()!=path or len(parsed.parts)<2
                        or parsed.parts[0]!=names[label]):return False
            all_paths.extend(paths)
        if len(set(all_paths))!=300:return False
        history=summary['history']
        if (not isinstance(history,dict) or not _same_json(_read_json(output/'history.json'),[history])
                or type(history.get('epoch')) is not int or history['epoch']!=1
                or type(history.get('optimizer_steps')) is not int or history['optimizer_steps']!=120):return False
        for key in ('train_loss','val_loss','train_seconds','validation_seconds','train_images_per_second','train_accuracy_percent'):
            value=history[key]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0:return False
        if (history['train_seconds']<=0 or history['validation_seconds']<=0 or history['train_images_per_second']<=0
                or history['train_accuracy_percent']>100
                or abs(history['train_images_per_second']-240/history['train_seconds'])>1e-6):return False
        first=_read_json(output/'first_step.json');gradients=first['gradients']
        if (first['stage']!='first_optimizer_step' or first['training_started'] is not True
                or first['optimizer_step_succeeded'] is not True or first['diagnostic_passed'] is not True
                or type(first['optimizer_steps']) is not int or first['optimizer_steps']!=1
                or type(first['samples_seen']) is not int or first['samples_seen']!=2 or first['device']!='cuda:0'
                or not _same_json(first['updated_groups'],{name:True for name in GRADIENT_GROUPS})
                or not _same_json(gradients,summary['first_batch_gradients']) or gradients['passed'] is not True
                or gradients['missing']!=[] or gradients['nonfinite']!=[] or set(gradients['groups'])!=GRADIENT_GROUPS):return False
        for key in ('loss','elapsed_seconds'):
            if type(first[key]) not in (int,float) or not math.isfinite(first[key]) or first[key]<0:return False
        for value in gradients['groups'].values():
            if (type(value['tensors']) is not int or value['tensors']<=0
                    or type(value['l2']) not in (int,float) or not math.isfinite(value['l2']) or value['l2']<=0):return False
        updates=_read_json(output/'optimizer_progress.json')
        if (updates['training_started'] is not True or type(updates['optimizer_steps']) is not int
                or updates['optimizer_steps']!=120 or type(updates['samples_seen']) is not int or updates['samples_seen']!=240):return False
        predictions=_read_json(output/'val_predictions.json')
        if (predictions['scope']!='smoke_validation_only' or predictions['paths']!=binding['val_paths']
                or not _same_json(predictions['labels'],binding['val_labels']) or not isinstance(predictions['predictions'],list)
                or len(predictions['predictions'])!=60
                or any(type(label) is not int or not 0<=label<30 for label in predictions['predictions'])):return False
        cm=[[0]*30 for _ in range(30)]
        for label,predicted in zip(predictions['labels'],predictions['predictions']):cm[label][predicted]+=1
        per_class=[];f1s=[]
        for i,name in enumerate(names):
            tp=cm[i][i];support=sum(cm[i]);predicted=sum(row[i] for row in cm)
            f1=2*tp/(support+predicted) if support+predicted else 0.0
            f1s.append(f1)
            per_class.append({'class':name,'support':support,'correct':tp,'accuracy_percent':100*tp/support,
                              'f1_percent':100*f1})
        metrics=dict(scope='smoke_validation_only',accuracy_percent=100*sum(cm[i][i] for i in range(30))/60,
                     macro_f1_percent=100*sum(f1s)/30,per_class=per_class,confusion_matrix=cm)
        return _same_json(predictions['metrics'],metrics) and _same_json(summary['val_metrics'],metrics)
    except (OSError,ValueError,KeyError,TypeError,AttributeError,OverflowError,ZeroDivisionError):
        return False


def _end_checks(args,sources,inputs,result):
    """Return success receipt only after every real end-of-run check succeeds."""
    if (_source_identity()!=sources or _sha(args.manifest)!=DATA.MANIFEST_SHA
            or _sha(args.weights)!=WEIGHTS_SHA or config_sha256(_check_config(args.config))!=CONFIG_SHA):
        raise ValueError('inputs or implementation changed during smoke')
    selected=DATA.validate_selected(args.data_root,inputs['selected']['pool'])
    if (selected.get('verified_count')!=300 or selected.get('checked_count')!=300
            or selected.get('test_image_bytes_read')!=0 or not _checkpoint_ok(args.output,result.get('checkpoint'))):
        raise ValueError('end-of-run selected-image/checkpoint evidence incomplete')
    return dict(COMPLETION_CHECKS)


def run_smoke(args):
    preflight=inspect_resources(args)
    if not preflight['ready']:raise ValueError('preflight blockers: '+', '.join(preflight['blockers']))
    selected=preflight['selected_gpu'];args.gpu_uuid=selected['uuid']
    with _run_lock(args.output):
        args.output.mkdir(exist_ok=False)
        _write(args.output/'preflight.json',preflight)
        _write(args.output/'command.json',dict(argv=list(sys.argv),cwd=os.getcwd(),pid=os.getpid()))
        result=dict(schema=SCHEMA,action='run',completed=False,training_started=False,
                    formal_training_started=False,optimizer_steps=0,input_scope='real_aid',
                    target_gpu='3060',counts=dict(COUNTS))
        started=time.perf_counter()
        with (args.output/'run.log').open('x') as log,contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
            try:
                sources=_source_identity()
                if 'source_identity' in preflight and sources!=preflight['source_identity']:
                    raise ValueError('source changed after preflight')
                result['source_identity']=sources
                _write(args.output/'source_identity.json',sources)
                torch,runtime=_runtime_and_torch(args,selected)
                _write(args.output/'runtime.json',runtime)
                print('HGTNET_AID_SMOKE_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
                cfg=_check_config(args.config);inputs=DATA.prepare_inputs(args.manifest)
                DATA.validate_selected(args.data_root,inputs['selected']['pool'])
                _write(args.output/'selected_inputs.json',inputs)
                from torch.utils.data import DataLoader
                from external_baselines.hgtnet.model import HGTNet,parameter_counts
                from external_baselines.hgtnet.pretrained import load_pretrained_file
                from external_baselines.hgtnet.aid_engine import run_epoch_smoke
                train=DATA.make_dataset(args.data_root,inputs['selected']['train'])
                val=DATA.make_dataset(args.data_root,inputs['selected']['val'])
                train_loader=DataLoader(train,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(42),
                                        num_workers=0,pin_memory=False,drop_last=False)
                val_loader=DataLoader(val,batch_size=2,shuffle=False,num_workers=0,pin_memory=False,drop_last=False)
                torch.cuda.reset_peak_memory_stats(0)
                model=HGTNet(cfg)
                pretraining=load_pretrained_file(model,args.weights,WEIGHTS_SHA);model.pretraining_loaded=True
                _write(args.output/'pretraining.json',pretraining)
                binding=dict(schema='hgtnet-aid-smoke-binding-v1',input_scope='real_aid',config=cfg,
                             config_sha256=CONFIG_SHA,class_names=inputs['class_names'],counts=dict(COUNTS),
                             manifest_sha256=DATA.MANIFEST_SHA,weights_sha256=WEIGHTS_SHA,
                             source_identity=sources,pretraining=pretraining,runtime=runtime)
                for split in ('train','val'):
                    binding[split+'_paths']=[row['path'] for row in inputs['selected'][split]]
                    binding[split+'_labels']=[row['label'] for row in inputs['selected'][split]]
                _write(args.output/'binding.json',binding)
                def progress(value):
                    _write(args.output/'progress.json',dict(value,pid=os.getpid()),replace=True)
                    print('HGTNET_AID_SMOKE_PROGRESS '+json.dumps(value,sort_keys=True),flush=True)
                result.update(run_epoch_smoke(model,train_loader,val_loader,args.output,binding=binding,
                              device='cuda:0',model_factory=lambda:HGTNet(cfg),on_progress=progress))
                if (result.get('completed') is not True or result.get('optimizer_steps')!=120
                        or result.get('counts')!=COUNTS or result.get('formal_training_started') is not False
                        or not _checkpoint_ok(args.output,result.get('checkpoint'))):
                    raise ValueError('engine completion/checkpoint evidence incomplete')
                completion_checks=_end_checks(args,sources,inputs,result)
                result.update(schema=SCHEMA,completed=True,runtime=runtime,parameters=parameter_counts(model),
                              config_sha256=CONFIG_SHA,manifest_sha256=DATA.MANIFEST_SHA,weights_sha256=WEIGHTS_SHA,
                              source_identity=sources,physical_gpu_uuid=selected['uuid'],selected_pool_count=300,
                              completion_checks=completion_checks,
                              data_source='verified_images',data_root_used=str(args.data_root),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),wall_seconds=time.perf_counter()-started,
                              memory_scope='whole smoke CUDA allocator including update diagnostics and checkpoint reload',
                              timing_scope='after stdlib preflight; imports, initialization, epoch, validation and reload')
                if not _completion_evidence_ok(args.output,result):
                    raise ValueError('complete AID evidence is missing or inconsistent')
                _write(args.output/'summary.json',result)
                with (args.output/'run.exit').open('x') as stream:stream.write('0\n')
                progress(dict(stage='complete',completed=True,training_started=True,optimizer_steps=120))
                print('HGTNET_AID_SMOKE_COMPLETE '+str(args.output/'summary.json'),flush=True)
                return 0
            except BaseException as exc:
                observed=_observed_training(args.output)
                actual=getattr(exc,'optimizer_steps',0)
                if type(actual) is int:observed['optimizer_steps']=max(actual,observed['optimizer_steps'],result.get('optimizer_steps',0))
                observed['training_started']=(observed['training_started'] or observed['optimizer_steps']>0
                                              or result.get('training_started') is True or getattr(exc,'training_started',False) is True)
                result.update(observed,schema=SCHEMA,completed=False,formal_training_started=False,error=type(exc).__name__+': '+str(exc),
                              wall_seconds=time.perf_counter()-started)
                traceback.print_exc()
                _write(args.output/'summary.json',result,replace=(args.output/'summary.json').exists())
                with (args.output/'run.exit').open('w') as stream:stream.write('1\n')
                _write(args.output/'progress.json',dict(stage='failed',completed=False,error=result['error'],**observed),replace=True)
                return 1


def status(output):
    output=Path(output)
    report=dict(schema=SCHEMA,action='status',state='not_started',output=str(output),completed=False,
                formal_training_started=False,counts=dict(COUNTS))
    if not output.exists() and not output.is_symlink():return report
    report['state']='incomplete'
    for filename,key in [('progress.json','progress'),('summary.json','summary')]:
        try:
            value=_read_json(output/filename)
            if isinstance(value,dict):report[key]=value
        except (OSError,ValueError):pass
    try:
        with open_regular_readonly(output/'run.exit') as stream:report['exit_code']=int(stream.read().strip())
    except (OSError,ValueError):report['exit_code']=None
    summary=report.pop('summary',{})
    for key in ('training_started','formal_training_started','counts','optimizer_steps','best_epoch','final_epoch',
                'weights_sha256','config_sha256','manifest_sha256','physical_gpu_uuid','wall_seconds','error',
                'cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'):
        if key in summary:report[key]=summary[key]
    observed=_observed_training(output)
    summary_steps=summary.get('optimizer_steps',0)
    if type(summary_steps) is not int or summary_steps<0:summary_steps=0
    report['optimizer_steps']=max(summary_steps,observed['optimizer_steps'])
    report['training_started']=summary.get('training_started') is True or observed['training_started']
    if isinstance(summary.get('val_metrics'),dict):
        report['smoke_val']={key:summary['val_metrics'].get(key) for key in ('accuracy_percent','macro_f1_percent')}
    checkpoint=summary.get('checkpoint')
    if isinstance(checkpoint,dict):
        report['checkpoint']={key:checkpoint.get(key) for key in ('sha256','reload_max_abs_error')}
    try:
        source_ok=summary.get('source_identity')==_source_identity() if summary else None
    except (OSError,ValueError,KeyError):source_ok=False
    report['source_identity_matches']=source_ok
    evidence_ok=_completion_evidence_ok(output,summary) if summary else False
    report['completion_evidence_valid']=evidence_ok
    if (summary.get('schema')==SCHEMA and summary.get('completed') is True and report['exit_code']==0
            and report.get('progress',{}).get('stage')=='complete'
            and report['progress'].get('completed') is True and summary.get('counts')==COUNTS
            and summary.get('optimizer_steps')==120 and summary.get('formal_training_started') is False
            and report['progress'].get('training_started') is True and report['progress'].get('optimizer_steps')==120
            and source_ok and evidence_ok):
        report.update(state='complete',completed=True)
    if summary and source_ok is False:report['state']='source_changed'
    if report['exit_code'] is not None and report['exit_code']!=0:report['state']='failed'
    return report


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    sub=result.add_subparsers(dest='action',required=True)
    for action in ('inspect','run','status'):
        command=sub.add_parser(action)
        command.add_argument('--output',type=Path,default=ROOT/'runs/aid_smoke_seed42_01')
        if action=='status':continue
        command.add_argument('--gpu-uuid')
        command.add_argument('--manifest',type=Path,default=DATA.MANIFEST)
        command.add_argument('--data-root',type=Path,default=DATA.DATA_ROOT)
        command.add_argument('--weights',type=Path,default=Path('/home/featurize/work/swin_b-68c6b09e.pth'))
        command.add_argument('--config',type=Path,default=ROOT/'configs/hgtnet_aid_independent_v0.json')
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='status':print(json.dumps(status(args.output),indent=2,sort_keys=True));return 0
    if args.action=='inspect':print(json.dumps(inspect_resources(args),indent=2,sort_keys=True));return 0
    return run_smoke(args)


if __name__=='__main__':raise SystemExit(main())
