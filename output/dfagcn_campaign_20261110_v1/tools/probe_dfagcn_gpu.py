"""One disposable DFAGCN batch on the explicitly selected 10GiB RTX3080.

This reuses only frozen features from the completed4090 smoke. It does not
continue its learned checkpoint, select hyperparameters or run any dataset test.
``inspect`` is stdlib-only; ``run`` performs VGG inference and one Adam step.
"""
from __future__ import annotations

from collections import Counter
import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from numbers import Integral
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time
import traceback


ROOT=Path(__file__).resolve().parents[1]
SCHEMA='dfagcn-3080-one-batch-probe-v1'
TARGET_UUID='GPU-70be4364-0641-e89d-828e-78727c6001ec'
TARGET_NAME='NVIDIA GeForce RTX 3080'
TARGET_MEMORY_MIB=10240
MANIFEST=Path('/home/featurize/work/nwpu_protocol_20260927_02.json')
MANIFEST_SHA='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
OLD_SMOKE=Path('/home/featurize/work/dfagcn_nwpu_smoke_20261029/runs/nwpu_smoke_01')
WEIGHTS_SHA='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
PREPROCESSING={'resize_mode':'bilinear_stretch','channels':'RGB','mean':[.485,.456,.406],
               'std':[.229,.224,.225],'fc_hw':224,'conv_hw':448,'augmentation':False,
               'weights_kind':'torchvision_imagenet1k_v1'}


def sha256_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def load_manifest(path,*,expected_sha256=MANIFEST_SHA):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('frozen manifest SHA256 mismatch')
    value=json.loads(raw)
    names,rows=value.get('class_names'),value.get('records')
    counts={'train':5040,'val':1260,'test':25200}
    if (value.get('dataset')!='NWPU-RESISC45' or value.get('schema_version') not in (1,2)
            or not isinstance(names,list) or len(names)!=45 or len(set(names))!=45
            or names!=sorted(names) or not isinstance(rows,list) or len(rows)!=31500
            or value.get('counts')!=counts):
        raise ValueError('NWPU manifest schema/class counts changed')
    seen=set();per_class=Counter()
    for row in rows:
        name,label,split=row['path'],row['label'],row['split']
        path=PurePosixPath(name)
        if (type(label) is not int or not 0<=label<45 or split not in counts or name in seen
                or path.is_absolute() or '..' in path.parts or path.as_posix()!=name
                or '\\' in name or len(path.parts)!=2 or path.parts[0]!=names[label]
                or not re.fullmatch('[0-9a-f]{64}',row['sha256'])):
            raise ValueError('invalid frozen manifest record')
        seen.add(name);per_class[(label,split)]+=1
    if any(per_class[(label,split)]!=number for label in range(45)
           for split,number in [('train',112),('val',28),('test',560)]):
        raise ValueError('NWPU per-class split identity changed')
    return value


def select_pool(manifest):
    result=[]
    for split,limit in [('train',8),('val',2)]:
        counts=Counter()
        for row in sorted(manifest['records'],key=lambda r:r['path']):
            if row['split']==split and counts[row['label']]<limit:
                result.append(row);counts[row['label']]+=1
        if any(counts[label]!=limit for label in range(45)):
            raise ValueError('prior smoke subset has missing classes')
    result.sort(key=lambda r:r['path'])
    if len(result)!=450 or len({r['path'] for r in result})!=450:
        raise ValueError('prior frozen pool must contain450 distinct train/val samples')
    return result


def expected_cache_identity(manifest,weights_sha):
    from external_baselines.dfagcn.identity import build_cache_identity
    source=_digest({name:sha256_file(ROOT/'external_baselines/dfagcn'/name)
                    for name in ('features.py','extraction.py','cache.py','identity.py')})
    return build_cache_identity(manifest_sha256=MANIFEST_SHA,class_names=manifest['class_names'],
        samples=[{'path':row['path'],'sha256':row['sha256']} for row in select_pool(manifest)],
        weights_sha256=weights_sha,preprocessing=PREPROCESSING,
        feature_spec={'pooling_hw':[7,7],'conv_activation':'pre_relu','fc_activation':'post_relu'},
        source_sha256=source,stage='frozen_pretrained')


def inspect_cache(path,expected_identity):
    from external_baselines.dfagcn.identity import cache_key
    path=Path(path)
    raw=(path/'index.json').read_bytes();index=json.loads(raw)
    if (not isinstance(index,dict) or set(index)!={'schema','identity','key','arrays'}
            or index['schema']!='dfagcn-pooled-cache-v1'
            or index['key']!=cache_key(expected_identity)
            or cache_key(index['identity'])!=cache_key(expected_identity)):
        raise ValueError('old frozen cache does not match expected sample order/source')
    widths={'conv3':256,'conv4':512,'conv5':512,'fc':4096}
    if set(index['arrays'])!=set(widths):raise ValueError('frozen cache branches changed')
    for name,width in widths.items():
        item=index['arrays'][name]
        shape=[450,width] if name=='fc' else [450,49,width]
        array=path/(name+'.npy')
        if (item.get('file')!=name+'.npy' or item.get('dtype')!='float32' or item.get('shape')!=shape
                or not re.fullmatch('[0-9a-f]{64}',item.get('sha256',''))
                or not array.is_file() or array.is_symlink()):
            raise ValueError('frozen cache array header/file mismatch: '+name)
    return {'samples':450,'index_sha256':hashlib.sha256(raw).hexdigest(),
            'identity_key':index['key'],'array_bytes_verified':False,
            'scope':'header/file presence checked; numerical byte hashes checked by run'}


def read_weights_sha(metadata,explicit):
    expected=None
    if metadata is not None and Path(metadata).is_file():
        expected=json.loads(Path(metadata).read_text()).get('sha256')
    if explicit is not None:
        if expected is not None and explicit!=expected:raise ValueError('weight metadata/explicit SHA conflict')
        expected=explicit
    if expected!=WEIGHTS_SHA:
        raise ValueError('the complete identified official VGG16 SHA256 is required')
    return expected


def gpu_inventory():
    result={'gpus':[],'processes':[],'error':None}
    try:
        query=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.free,driver_version',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=False)
        if query.returncode:raise ValueError(query.stderr.strip() or 'NVIDIA GPU query failed')
        for row in csv.reader(query.stdout.splitlines()):
            if len(row)!=5:raise ValueError('unexpected NVIDIA GPU query columns')
            result['gpus'].append({'name':row[0].strip(),'uuid':row[1].strip(),
                'memory_total_mib':int(row[2].strip()),'memory_free_mib':int(row[3].strip()),
                'driver_version':row[4].strip()})
        query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=False)
        if query.returncode:raise ValueError(query.stderr.strip() or 'NVIDIA process query failed')
        for row in csv.reader(query.stdout.splitlines()):
            if len(row)!=4:raise ValueError('unexpected NVIDIA compute-process query columns')
            result['processes'].append({'pid':int(row[0].strip()),'gpu_uuid':row[1].strip(),
                'name':row[2].strip(),'used_memory_mib':row[3].strip()})
    except (OSError,ValueError,subprocess.SubprocessError) as exc:result['error']=str(exc)
    return result


def require_gpu(uuid,inventory):
    if not isinstance(uuid,str) or not uuid.startswith('GPU-') or inventory.get('error'):
        raise ValueError('working NVIDIA query and explicit physical UUID required')
    matching=[gpu for gpu in inventory['gpus'] if gpu['uuid']==uuid]
    if (len(matching)!=1 or matching[0]['name']!=TARGET_NAME
            or matching[0]['memory_total_mib']!=TARGET_MEMORY_MIB):
        raise ValueError('requested UUID does not identify the selected10GiB RTX3080')
    if not 0<matching[0]['memory_free_mib']<=TARGET_MEMORY_MIB:
        raise ValueError('requested GPU has no valid available memory')
    if any(row.get('gpu_uuid')==uuid for row in inventory['processes']):
        raise ValueError('requested GPU already has a compute process')
    return matching[0]


def select_cuda_uuid(uuid):
    existing=sys.modules.get('torch')
    if existing is not None and existing.cuda.is_initialized():
        raise ValueError('GPU selection requires a fresh CUDA process')
    os.environ['CUDA_VISIBLE_DEVICES']=uuid


def inspect_resources(args):
    blockers=[]
    report={'schema':SCHEMA,'action':'inspect','formal_training_started':False,
        'requested_gpu_uuid':args.gpu_uuid,'manifest':str(args.manifest),'cache':str(args.cache),
        'weights':str(args.weights),'output':str(args.output),'blockers':blockers}
    manifest=None;expected=None
    try:
        manifest=load_manifest(args.manifest)
        report['manifest_sha256']=MANIFEST_SHA
        report['frozen_pool_samples']=len(select_pool(manifest))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('manifest');report['manifest_error']=str(exc)
    try:
        expected=read_weights_sha(args.weights_metadata,args.weights_sha256)
        if Path(args.weights).name!='vgg16-397923af.pth' or sha256_file(args.weights)!=expected:
            raise ValueError('VGG16 file bytes/name differ from delivery identity')
        report['weights_sha256']=expected
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('weights');report['weights_error']=str(exc)
    try:
        if manifest is None or expected is None:raise ValueError('manifest/weight identity required before cache check')
        report['cache_header']=inspect_cache(args.cache,expected_cache_identity(manifest,expected))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('frozen_cache');report['cache_error']=str(exc)
    try:
        recipe=json.loads(Path(args.recipe).read_text())
        required={'batch_size':100,'seed':42,'pooling_hw':[7,7],'gcn_dims':[256,256],
                  'fc_dims':[1024,512],'knn_k':10,'metric':'euclidean','dropout':0.5}
        if any(recipe.get(k)!=v for k,v in required.items()):raise ValueError('formal batch/architecture recipe changed')
        report['recipe_sha256']=sha256_file(args.recipe)
        report['graph_batch_samples']=100
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('recipe');report['recipe_error']=str(exc)
    report['dependencies']={}
    for name,distribution in [('torch','torch'),('torchvision','torchvision'),('numpy','numpy'),('PIL','Pillow')]:
        try:
            located=importlib.util.find_spec(name) is not None
            version=importlib.metadata.version(distribution) if located else None
        except (ImportError,ValueError,importlib.metadata.PackageNotFoundError):located=False;version=None
        report['dependencies'][name]={'located':located,'installed_version':version,'import_tested':False}
        if not located:blockers.append('dependency:'+name)
    if Path(args.output).exists() or Path(args.output).is_symlink():blockers.append('output_already_exists')
    inventory=gpu_inventory();report['nvidia']=inventory
    try:report['selected_gpu']=require_gpu(args.gpu_uuid,inventory)
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    report['ready']=not blockers
    report['readiness_scope']='inventory only; actual imports/VGG/Adam batch checked by run'
    if report['ready']:
        entry=os.environ.get('DFAGCN_DELIVERY_ENTRY',str(Path(__file__).resolve()))
        command=[sys.executable,entry,'run','--gpu-uuid',args.gpu_uuid,
            '--manifest',str(args.manifest),'--cache',str(args.cache),'--weights',str(args.weights),
            '--weights-sha256',expected,'--recipe',str(args.recipe),'--output',str(args.output)]
        if Path(args.weights_metadata).is_file():command+=['--weights-metadata',str(args.weights_metadata)]
        report['run_command']=shlex.join(command)
    return report


def check_graph_batch(cache,labels,recipe,device):
    """One real Adam batch on a new model; never writes a checkpoint/cache."""
    import torch
    from torch.nn import functional as F
    from external_baselines.dfagcn.config import validate_recipe
    from external_baselines.dfagcn.model import DFAGCNBranches
    recipe=validate_recipe(recipe)
    count=recipe['batch_size']
    if len(cache)<count or len(labels)!=len(cache):raise ValueError('cache/labels lack a complete declared batch')
    if any(isinstance(x,bool) or not isinstance(x,Integral) or not 0<=x<45 for x in labels):
        raise ValueError('labels must be45-class integer indices')
    torch.manual_seed(recipe['seed'])
    model=DFAGCNBranches(recipe,45).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=recipe['learning_rate'],weight_decay=recipe['weight_decay'])
    indices=torch.randperm(len(cache),generator=torch.Generator().manual_seed(recipe['seed']))[:count].tolist()
    samples=[cache[index] for index in indices]
    features={name:torch.stack([row[name] for row in samples]).to(device)
              for name in ('conv3','conv4','conv5','fc')}
    target=torch.tensor([int(labels[index]) for index in indices],dtype=torch.long,device=device)
    before=next(model.parameters()).detach().clone()
    if torch.device(device).type=='cuda':torch.cuda.synchronize()
    started=time.monotonic()
    model.train();optimizer.zero_grad(set_to_none=True)
    logits=model(features)
    loss=sum(F.cross_entropy(value,target) for value in logits.values())
    if not torch.isfinite(loss):raise ValueError('nonfinite disposable batch loss')
    loss.backward()
    gradients=all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    if not gradients:raise ValueError('nonfinite/missing disposable model gradient')
    optimizer.step()
    finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    changed=not torch.equal(before,next(model.parameters()).detach())
    if not finite or not changed:raise ValueError('disposable Adam step failed to update finite parameters')
    if torch.device(device).type=='cuda':torch.cuda.synchronize()
    return {'batch_samples':count,'device':str(device),'classes':45,'loss':float(loss.detach()),
        'gradients_finite':gradients,'parameters_finite':finite,'disposable_parameters_changed':changed,
        'optimizer_steps':1,'elapsed_seconds':time.monotonic()-started,'formal_training_started':False}


def _runtime_and_torch(args,gpu):
    select_cuda_uuid(args.gpu_uuid)
    import torch
    import torchvision
    import numpy
    from PIL import __version__ as pillow_version
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!=TARGET_NAME:
        raise ValueError('CUDA-visible physical device does not match selected RTX3080')
    torch.set_num_threads(args.cpu_threads);torch.set_default_dtype(torch.float32)
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':sys.version,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
        'numpy':numpy.__version__,'pillow':pillow_version,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':gpu,'physical_gpu_uuid':args.gpu_uuid,'device':'cuda:0','cpu_threads':args.cpu_threads,
        'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,
        'cudnn_deterministic':True,'dtype':'float32','seed':42}
    return torch,runtime


def _source_identity():
    files=sorted((ROOT/'external_baselines/dfagcn').glob('*.py'))
    files += [Path(__file__).resolve(),ROOT/'configs/dfagcn_independent_v1.json']
    return {str(path.relative_to(ROOT)):sha256_file(path) for path in files}


def run_probe(args):
    inventory=inspect_resources(args)
    if not inventory['ready']:raise ValueError('probe resources unavailable: '+', '.join(inventory['blockers']))
    gpu=require_gpu(args.gpu_uuid,inventory['nvidia'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    result={'schema':SCHEMA,'completed':False,'formal_training_started':False,
            'disposable_training_batches_completed':0,'action':'run','command':list(sys.argv)}
    _write(output/'preflight.json',inventory)
    try:
        sources=_source_identity();_write(output/'source_identity.json',sources)
        manifest=load_manifest(args.manifest);rows=select_pool(manifest)
        recipe=json.loads(Path(args.recipe).read_text())
        _write(output/'recipe.json',recipe)
        torch,runtime=_runtime_and_torch(args,gpu);_write(output/'runtime.json',runtime)
        print('DFAGCN_GPU_PROBE_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
        from external_baselines.dfagcn.cache import FeatureCache
        from external_baselines.dfagcn.extraction import load_vgg_adapter
        cache=FeatureCache(args.cache,expected_identity=expected_cache_identity(manifest,inventory['weights_sha256']))
        torch.cuda.reset_peak_memory_stats(0)
        vgg_started=time.monotonic()
        adapter=load_vgg_adapter(args.weights,inventory['weights_sha256'],device='cuda:0')
        with torch.no_grad():
            features=adapter(torch.zeros(1,3,224,224,device='cuda:0'),torch.zeros(1,3,448,448,device='cuda:0'))
        if not all(bool(torch.isfinite(value).all()) for value in features.values()):raise ValueError('nonfinite VGG CUDA forward')
        shapes={name:list(value.shape) for name,value in features.items()}
        if shapes!={'fc':[1,4096],'conv3':[1,256,112,112],'conv4':[1,512,56,56],'conv5':[1,512,28,28]}:
            raise ValueError('VGG CUDA output shapes differ from extraction contract')
        torch.cuda.synchronize()
        vgg={'batch_samples':1,'input':'normalized zero tensors for compatibility/memory only','shapes':shapes,
             'elapsed_seconds':time.monotonic()-vgg_started,'finite':True,
             'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
             'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        print('DFAGCN_GPU_PROBE_VGG_BATCH '+json.dumps(vgg,sort_keys=True),flush=True)
        del adapter,features
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        graph=check_graph_batch(cache,[row['label'] for row in rows],recipe,'cuda:0')
        graph.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        print('DFAGCN_GPU_PROBE_GRAPH_BATCH '+json.dumps(graph,sort_keys=True),flush=True)
        if sources!=_source_identity() or sha256_file(args.recipe)!=inventory['recipe_sha256']:
            raise ValueError('probe source/recipe changed during execution')
        result.update(completed=True,disposable_training_batches_completed=1,runtime=runtime,
            manifest_sha256=MANIFEST_SHA,weights_sha256=inventory['weights_sha256'],
            recipe_sha256=inventory['recipe_sha256'],frozen_cache_identity_key=cache.key,
            frozen_cache_content_sha256=cache.content_sha256,frozen_cache_samples=450,
            frozen_cache_scope='prior4090 frozen feature cache for this compatibility probe only; rebuild full formal pool on3080',
            vgg=vgg,graph=graph,wall_seconds=time.monotonic()-started,
            cuda_peak_allocated_bytes=max(vgg['cuda_peak_allocated_bytes'],graph['cuda_peak_allocated_bytes']),
            cuda_peak_reserved_bytes=max(vgg['cuda_peak_reserved_bytes'],graph['cuda_peak_reserved_bytes']),
            dataset_test_samples=0,formal_training_epochs=0,checkpoint_written=False,
            interpretation='One disposable batch GPU/memory check, not a repeated epoch or formal accuracy experiment.')
        _write(output/'summary.json',result);_write(output/'run.exit',{'exit_code':0})
        print('DFAGCN_GPU_PROBE_COMPLETE '+str(output/'summary.json'),flush=True)
        return result
    except BaseException as exc:
        result.update(error=str(exc),error_type=type(exc).__name__,wall_seconds=time.monotonic()-started)
        _write(output/'failure.json',result);_write(output/'run.exit',{'exit_code':1})
        (output/'traceback.log').write_text(traceback.format_exc())
        raise


def parser():
    root=argparse.ArgumentParser(description='Bounded10GiB RTX3080 VGG/one-Adam-batch compatibility check')
    commands=root.add_subparsers(dest='action',required=True)
    local=ROOT/'weights/vgg16-397923af.pth'
    default_weights=local if local.exists() else ROOT.parent/'vgg16-397923af.pth'
    for name in ('inspect','run'):
        command=commands.add_parser(name)
        command.add_argument('--gpu-uuid',required=name=='run',default=None if name=='run' else TARGET_UUID)
        command.add_argument('--manifest',type=Path,default=MANIFEST)
        command.add_argument('--cache',type=Path,default=OLD_SMOKE/'pool_cache')
        command.add_argument('--weights',type=Path,default=default_weights)
        command.add_argument('--weights-metadata',type=Path,default=ROOT/'bundled_weights_provenance.json')
        command.add_argument('--weights-sha256')
        command.add_argument('--recipe',type=Path,default=ROOT/'configs/dfagcn_independent_v1.json')
        command.add_argument('--output',type=Path,default=ROOT/'runs/gpu_probe_01')
        command.add_argument('--cpu-threads',type=int,default=8)
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.cpu_threads<1:raise ValueError('cpu threads must be positive')
    if args.action=='inspect':
        report=inspect_resources(args);print(json.dumps(report,indent=2,sort_keys=True))
        return 0 if report['ready'] else 2
    run_probe(args)
    return 0


if __name__=='__main__':
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    try:raise SystemExit(main())
    except (OSError,ValueError,TypeError,KeyError,RuntimeError) as exc:
        print('DFAGCN_GPU_PROBE_STOP '+str(exc),file=sys.stderr)
        raise SystemExit(2)
