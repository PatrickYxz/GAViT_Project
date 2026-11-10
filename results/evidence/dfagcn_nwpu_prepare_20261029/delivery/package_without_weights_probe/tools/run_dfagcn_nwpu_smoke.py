"""Portable real NWPU 360/90 engineering smoke, without online acquisition.

``inspect`` uses the standard library and never trains or restores data. ``run``
requires an explicit physical RTX 4090 UUID and checks resources again before
CUDA work. Only train/validation images are read; final test is never invoked.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=Path('/home/featurize/work/nwpu_protocol_20260927_02.json')
MANIFEST_SHA='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
DATA_ROOT=Path('/home/featurize/data/NWPU-RESISC45')
ARCHIVE=Path('/home/featurize/work/NWPU-RESISC45_a826b44.zip')
ARCHIVE_SHA='beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53'
COUNTS={'train':5040,'val':1260,'test':25200}
LIMITS={'train':8,'val':2}
SCHEMA='dfagcn-real-nwpu-smoke-v1'


def sha256_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def _relative(name):
    if not isinstance(name,str) or not name or '\\' in name:
        raise ValueError('invalid relative image path')
    path=PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or path.as_posix()!=name
            or not path.parts or ':' in path.parts[0] or name=='.'):
        raise ValueError('unsafe or non-normalized relative path')
    return path


def _sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ValueError('full lowercase SHA256 is required')
    return value


def load_manifest(path,*,expected_sha256=MANIFEST_SHA):
    """Check immutable manifest bytes and the real NWPU taxonomy/count shape."""
    path=Path(path)
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('frozen NWPU manifest SHA256 mismatch')
    value=json.loads(raw)
    names,rows=value.get('class_names'),value.get('records')
    if (value.get('dataset')!='NWPU-RESISC45' or value.get('schema_version') not in (1,2)
            or not isinstance(names,list) or len(names)!=45 or len(set(names))!=45
            or any(not isinstance(name,str) or not name for name in names) or names!=sorted(names)
            or not isinstance(rows,list) or len(rows)!=31500 or value.get('counts')!=COUNTS):
        raise ValueError('frozen NWPU class/count/schema identity mismatch')
    seen=set()
    per_class=Counter()
    for row in rows:
        path=_relative(row['path'])
        label=row['label']
        if (type(label) is not int or not 0<=label<45 or len(path.parts)!=2
                or path.parts[0]!=names[label] or row['split'] not in COUNTS or row['path'] in seen):
            raise ValueError('invalid or duplicate NWPU manifest record')
        _sha(row['sha256'])
        seen.add(row['path'])
        per_class[(label,row['split'])]+=1
    if any(per_class[(label,split)]!=count for label in range(45)
           for split,count in [('train',112),('val',28),('test',560)]):
        raise ValueError('NWPU per-class split counts changed')
    return value


def select_smoke(manifest):
    selected={}
    for split,limit in LIMITS.items():
        counters=Counter()
        rows=[]
        for row in sorted(manifest['records'],key=lambda r:r['path']):
            if row['split']==split and counters[row['label']]<limit:
                counters[row['label']]+=1
                rows.append(row)
        if len(rows)!=45*limit or any(counters[label]!=limit for label in range(45)):
            raise ValueError('smoke subset lacks a declared class or sample count')
        selected[split]=rows
    selected['pool']=sorted(selected['train']+selected['val'],key=lambda r:r['path'])
    if len({r['path'] for r in selected['pool']})!=450:
        raise ValueError('smoke train/validation overlap')
    return selected


def validate_selected(root,records):
    root=Path(root).resolve()
    for row in records:
        relative=_relative(row['path'])
        path=root/str(relative)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError('selected image missing or outside data root: '+row['path'])
        if sha256_file(path)!=row['sha256']:
            raise ValueError('selected image SHA256 mismatch: '+row['path'])
    return True


def recover_selected(archive,records,destination,*,expected_sha256=ARCHIVE_SHA):
    """Recover only selected members into a fresh directory, never extractall."""
    archive,destination=Path(archive),Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    wanted={str(_relative(row['path'])):row for row in records}
    if len(wanted)!=len(records):
        raise ValueError('duplicate selected image paths')
    if sha256_file(archive)!=expected_sha256:
        raise ValueError('persistent NWPU ZIP SHA256 mismatch')
    with zipfile.ZipFile(archive) as source:
        members={}
        seen=set()
        for info in source.infolist():
            if info.is_dir():
                _relative(info.filename.rstrip('/'))
                continue
            name=str(_relative(info.filename))
            if name in seen or ((info.external_attr>>16)&0o170000)==0o120000:
                raise ValueError('duplicate or symlink archive member')
            seen.add(name)
            relative=name[len('NWPU-RESISC45/'):] if name.startswith('NWPU-RESISC45/') else name
            if relative in wanted:
                if relative in members or info.flag_bits&1:
                    raise ValueError('ambiguous or encrypted selected ZIP member')
                members[relative]=info
        if set(members)!=set(wanted):
            raise ValueError('persistent ZIP lacks selected manifest images')
        destination.mkdir(parents=True,exist_ok=False)
        for relative,row in wanted.items():
            raw=source.read(members[relative])
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:
                raise ValueError('selected archive image SHA256 mismatch: '+relative)
            path=destination/relative
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:
                stream.write(raw)
    validate_selected(destination,records)
    _write(destination/'recovery.json',{'archive_sha256':expected_sha256,'selected_images':len(records),
                                        'complete':True,'test_images_extracted':0})
    return destination


def read_weights_sha(metadata,explicit):
    expected=None
    if metadata is not None and Path(metadata).is_file():
        expected=_sha(json.loads(Path(metadata).read_text())['sha256'])
    if explicit is not None:
        explicit=_sha(explicit)
        if expected is not None and explicit!=expected:
            raise ValueError('explicit VGG SHA256 differs from delivered metadata')
        expected=explicit
    if expected is None or not expected.startswith('397923af'):
        raise ValueError('full official VGG16 SHA256 identity is required')
    return expected


def gpu_inventory():
    result={'gpus':[],'processes':[],'error':None}
    try:
        command=['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader,nounits']
        query=subprocess.run(command,capture_output=True,text=True,timeout=15,check=False)
        if query.returncode:
            result['error']=query.stderr.strip() or 'nvidia-smi GPU query failed'
            return result
        for row in csv.reader(query.stdout.splitlines()):
            if len(row)!=3:raise ValueError('unexpected NVIDIA GPU query columns')
            result['gpus'].append({'name':row[0].strip(),'uuid':row[1].strip(),
                                   'memory_total_mib':int(row[2].strip())})
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory',
                                  '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=False)
        if processes.returncode:
            result['error']=processes.stderr.strip() or 'nvidia-smi process query failed'
            return result
        for row in csv.reader(processes.stdout.splitlines()):
            if not row or len(row)!=4:continue
            result['processes'].append({'pid':int(row[0].strip()),'gpu_uuid':row[1].strip(),
                'name':row[2].strip(),'used_memory_mib':row[3].strip()})
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        result['error']=str(exc)
    return result


def require_gpu(uuid,inventory):
    if not isinstance(uuid,str) or not uuid.startswith('GPU-') or inventory.get('error'):
        raise ValueError('a working NVIDIA query and explicit physical GPU UUID are required')
    matching=[gpu for gpu in inventory['gpus'] if gpu['uuid']==uuid]
    if len(matching)!=1 or 'RTX 4090' not in matching[0]['name']:
        raise ValueError('requested physical GPU is not the required RTX 4090')
    if any(row.get('gpu_uuid')==uuid for row in inventory['processes']):
        raise ValueError('requested RTX 4090 already has a compute process')
    return matching[0]


def _recipe(path):
    recipe=json.loads(Path(path).read_text())
    expected={'seed':42,'epochs':1,'batch_size':8,'pooling_hw':[7,7],
        'knn_k':10,'gcn_dims':[256,256],'fc_dims':[1024,512],'metric':'euclidean',
        'C_grid':[1.0],'dropout':0.5,'fusion':{'alpha':0.7,'beta':0.3,'gamma':0.4}}
    if any(recipe.get(key)!=value for key,value in expected.items()):
        raise ValueError('one-epoch independent NWPU smoke recipe changed')
    return recipe


def _source_files():
    files=sorted((ROOT/'external_baselines'/'dfagcn').glob('*.py'))
    files.extend([ROOT/'external_baselines/__init__.py',ROOT/'scene_classification/__init__.py',
                  ROOT/'scene_classification/data.py',ROOT/'scene_classification/metrics.py',
                  ROOT/'experiment_identity.py',ROOT/'grouping_identity.py',Path(__file__).resolve()])
    return files


def _source_identity():
    return {str(path.relative_to(ROOT)):sha256_file(path) for path in _source_files()}


def inspect_resources(args):
    blockers=[]
    report={'schema':SCHEMA,'action':'inspect','training_started':False,
            'manifest':str(args.manifest),'data_root':str(args.data_root),'archive':str(args.archive),
            'weights':str(args.weights),'output':str(args.output),'blockers':blockers}
    selected=None
    try:
        manifest=load_manifest(args.manifest)
        selected=select_smoke(manifest)
        report['manifest_sha256']=MANIFEST_SHA
        report['counts']=manifest['counts']
        report['selected_counts']={key:len(rows) for key,rows in selected.items()}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('manifest');report['manifest_error']=str(exc)
    if selected is not None:
        try:
            validate_selected(args.data_root,selected['pool'])
            report['data_source']='existing_selected_images_verified'
        except (OSError,ValueError) as exc:
            report['selected_data_error']=str(exc)
            if Path(args.archive).is_file():
                report['data_source']='persistent_zip_pending_full_hash_and_selected_recovery'
                report['archive_expected_sha256']=ARCHIVE_SHA
                report['archive_bytes']=Path(args.archive).stat().st_size
                report['archive_sha_verified']=False
            else:
                report['data_source']='unavailable';blockers.append('selected_images_or_archive')
    try:
        expected=read_weights_sha(args.weights_metadata,args.weights_sha256)
        if Path(args.weights).name!='vgg16-397923af.pth':
            raise ValueError('unexpected VGG16 checkpoint filename')
        if sha256_file(args.weights)!=expected:
            raise ValueError('VGG16 file SHA256 differs from delivered full digest')
        report['weights_sha256']=expected
        report['weights_bytes']=Path(args.weights).stat().st_size
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('weights');report['weights_error']=str(exc)
    report['dependencies']={}
    for name,distribution in [('torch','torch'),('torchvision','torchvision'),('numpy','numpy'),('PIL','Pillow')]:
        try:
            found=importlib.util.find_spec(name) is not None
            version=importlib.metadata.version(distribution) if found else None
        except (ImportError,ValueError,importlib.metadata.PackageNotFoundError):
            found=False;version=None
        report['dependencies'][name]={'located':found,'installed_version':version,'import_tested':False}
        if not found:blockers.append('dependency:'+name)
    report['liblinear']={}
    for name in ('train','predict'):
        path=Path(getattr(args,name+'_binary'))
        available=path.is_file() and os.access(path,os.X_OK)
        report['liblinear'][name]={'path':str(path),'executable':available,
                                    'sha256':sha256_file(path) if available else None}
        if not available:blockers.append('liblinear:'+name)
    try:
        recipe=_recipe(args.recipe)
        report['recipe_sha256']=sha256_file(args.recipe)
        report['recipe_epochs']=recipe['epochs']
    except (OSError,ValueError,KeyError,TypeError) as exc:
        blockers.append('recipe');report['recipe_error']=str(exc)
    if Path(args.output).exists() or Path(args.output).is_symlink():
        blockers.append('output_already_exists')
    inventory=gpu_inventory()
    report['nvidia']=inventory
    candidates=[]
    for gpu in inventory['gpus']:
        try:candidates.append(require_gpu(gpu['uuid'],inventory))
        except ValueError:pass
    if not candidates:blockers.append('idle_rtx4090')
    report['ready']=not blockers
    report['readiness_scope']='resource inventory only; ZIP recovery, imports and CUDA execution checked by run'
    if report['ready']:
        entry=os.environ.get('DFAGCN_DELIVERY_ENTRY',str(Path(__file__).resolve()))
        command=[sys.executable,entry,'run','--gpu-uuid',candidates[0]['uuid'],
                 '--manifest',str(args.manifest),'--data-root',str(args.data_root),'--archive',str(args.archive),
                 '--weights',str(args.weights),'--weights-sha256',report['weights_sha256'],
                 '--train-binary',str(args.train_binary),'--predict-binary',str(args.predict_binary),
                 '--recipe',str(args.recipe),'--output',str(args.output)]
        if Path(args.weights_metadata).is_file():command+=['--weights-metadata',str(args.weights_metadata)]
        report['run_command']=shlex.join(command)
    return report


def _write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def _verified_imports_and_device(args,gpu):
    # No module imported above this point initializes CUDA; pick by physical UUID.
    os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    import torch
    import torchvision
    import numpy
    from PIL import __version__ as pillow_version
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise ValueError('selected physical UUID did not expose exactly one CUDA device')
    if 'RTX 4090' not in torch.cuda.get_device_name(0):
        raise ValueError('CUDA-visible device differs from required RTX 4090')
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.reset_peak_memory_stats(0)
    runtime={'python':sys.version,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
             'numpy':numpy.__version__,'pillow':pillow_version,'cuda':torch.version.cuda,
             'cudnn':torch.backends.cudnn.version(),'device':'cuda:0','gpu':gpu,
             'cpu_threads':args.cpu_threads,'amp':False,'matmul_tf32':False,'cudnn_tf32':False,
             'cudnn_benchmark':False,'cudnn_deterministic':True,'seed':42}
    return torch,runtime


def run_smoke(args):
    report=inspect_resources(args)
    if not report['ready']:
        raise ValueError('preflight resources unavailable: '+', '.join(report['blockers']))
    gpu=require_gpu(args.gpu_uuid,report['nvidia'])
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    state={'schema':SCHEMA,'completed':False,'training_started':False,'action':'run',
           'command':list(sys.argv),'preflight':report,'physical_gpu_uuid':gpu['uuid']}
    _write(output/'preflight.json',report)
    try:
        sources=_source_identity()
        _write(output/'source_identity.json',sources)
        for relative in sources:
            target=output/'source'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/relative,target)
        recipe=_recipe(args.recipe)
        _write(output/'recipe.json',recipe)
        manifest=load_manifest(args.manifest)
        selected=select_smoke(manifest)
        try:
            validate_selected(args.data_root,selected['pool'])
            data_root=Path(args.data_root)
            data_mode='existing_selected_images_verified'
        except (OSError,ValueError):
            print('DFAGCN_NWPU_RECOVER_SELECTED images=450 test_images=0',flush=True)
            data_root=recover_selected(args.archive,selected['pool'],output/'images')
            data_mode='persistent_zip_selected_recovery'
        _write(output/'selected_rows.json',selected)
        torch,runtime=_verified_imports_and_device(args,gpu)
        _write(output/'runtime.json',runtime)
        print('DFAGCN_NWPU_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
        from external_baselines.dfagcn.extraction import load_vgg_adapter,extract_cache,PREPROCESSING
        from external_baselines.dfagcn.identity import build_cache_identity
        from external_baselines.dfagcn.cache import write_feature_cache
        from external_baselines.dfagcn.pipeline import feature_source_sha256,source_sha256,calibrate,refit
        weight_started=time.monotonic()
        adapter=load_vgg_adapter(args.weights,report['weights_sha256'],device='cuda:0')
        torch.cuda.synchronize()
        weights_load_seconds=time.monotonic()-weight_started
        first_batch={}
        def observe(module,inputs,outputs):
            if first_batch:return
            torch.cuda.synchronize()
            first_batch.update(samples=inputs[0].shape[0],device=str(outputs['fc'].device),
                               fc_shape=list(outputs['fc'].shape),conv3_shape=list(outputs['conv3'].shape),
                               cuda_allocated_bytes=torch.cuda.memory_allocated(0),pid=os.getpid())
            print('DFAGCN_NWPU_VGG_BATCH '+json.dumps(first_batch,sort_keys=True),flush=True)
        handle=adapter.register_forward_hook(observe)
        def identity(rows):
            return build_cache_identity(manifest_sha256=MANIFEST_SHA,class_names=manifest['class_names'],
                samples=[{'path':row['path'],'sha256':row['sha256']} for row in rows],
                weights_sha256=report['weights_sha256'],preprocessing=dict(PREPROCESSING,weights_kind='torchvision_imagenet1k_v1'),
                feature_spec={'pooling_hw':recipe['pooling_hw'],'conv_activation':'pre_relu','fc_activation':'post_relu'},
                source_sha256=feature_source_sha256(),stage='frozen_pretrained')
        extract_started=time.monotonic()
        try:
            pool=extract_cache(adapter,data_root,selected['pool'],identity(selected['pool']),output/'pool_cache',
                pool_hw=recipe['pooling_hw'],batch_size=args.extract_batch_size,device='cuda:0')
        finally:
            handle.remove()
        torch.cuda.synchronize()
        extract_seconds=time.monotonic()-extract_started
        if not first_batch or first_batch['device']!='cuda:0' or first_batch['cuda_allocated_bytes']<=0:
            raise ValueError('real CUDA VGG batch was not observed')
        del adapter
        lookup={row['path']:index for index,row in enumerate(selected['pool'])}
        for split in ('train','val'):
            rows=selected[split]
            def batches():
                for start in range(0,len(rows),recipe['batch_size']):
                    samples=[pool[lookup[row['path']]] for row in rows[start:start+recipe['batch_size']]]
                    yield {name:torch.stack([sample[name] for sample in samples])
                           for name in ('conv3','conv4','conv5','fc')}
            write_feature_cache(output/(split+'_cache'),identity(rows),batches())
        state['training_started']=True
        selection=calibrate(args.manifest,output/'train_cache',output/'val_cache',recipe,output/'calibration',
            train_binary=args.train_binary,predict_binary=args.predict_binary,mode='engineering',device='cuda:0',
            train_per_class=8,val_per_class=2)
        fitted=refit(args.manifest,output/'pool_cache',output/'calibration/selection.json',output/'refit',
            train_binary=args.train_binary,predict_binary=args.predict_binary,device='cuda:0')
        torch.cuda.synchronize()
        if (fitted['fit_scope']!='train_plus_val_subset' or fitted['fit_samples']!=450
                or fitted['epoch']!=1 or fitted['reload_max_abs_error']>1e-6):
            raise ValueError('refit smoke identity/reload result is invalid')
        for phase in ('calibration','refit'):
            history=json.loads((output/phase/'history.json').read_text())
            if len(history)!=1 or history[0]['epoch']!=1:
                raise ValueError('one-epoch smoke history incomplete')
        if sources!=_source_identity() or sha256_file(args.manifest)!=MANIFEST_SHA:
            raise ValueError('source or frozen manifest changed during smoke')
        state.update(completed=True,counts={'train':360,'val':90,'pool':450,'test':0},
            selected_epoch=selection['selected_epoch'],selected_C=selection['selected_C'],
            data_mode=data_mode,manifest_sha256=MANIFEST_SHA,weights_sha256=report['weights_sha256'],
            recipe_sha256=sha256_file(args.recipe),pipeline_source_sha256=source_sha256(),
            runtime=runtime,first_vgg_batch=first_batch,weights_load_seconds=weights_load_seconds,
            pooled_feature_extraction_seconds=extract_seconds,feature_images_per_second=450/extract_seconds,
            calibration_seconds=selection['elapsed_seconds'],refit_seconds=fitted['elapsed_seconds'],
            wall_seconds=time.monotonic()-started,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0),reload_max_abs_error=fitted['reload_max_abs_error'],
            validation_metrics=selection['scores'][0]['metrics'],refit_fit_metrics=fitted['fit_metrics'],
            interpretation='Real NWPU train/validation engineering smoke; no test result or formal score.',
            native_binaries={name:sha256_file(getattr(args,name+'_binary')) for name in ('train','predict')})
        _write(output/'summary.json',state)
        _write(output/'run.exit',{'exit_code':0})
        print('DFAGCN_NWPU_SMOKE_COMPLETE '+str(output/'summary.json'),flush=True)
        return state
    except BaseException as exc:
        state.update(error=str(exc),error_type=type(exc).__name__,wall_seconds=time.monotonic()-started)
        _write(output/'failure.json',state)
        (output/'traceback.log').write_text(traceback.format_exc())
        _write(output/'run.exit',{'exit_code':1})
        raise


def parser():
    result=argparse.ArgumentParser(description='Real NWPU360/90 inspect/run; no downloads or final test')
    commands=result.add_subparsers(dest='action',required=True)
    local_weights=ROOT/'weights/vgg16-397923af.pth'
    default_weights=local_weights if local_weights.exists() else ROOT.parent/'vgg16-397923af.pth'
    for mode in ('inspect','run'):
        command=commands.add_parser(mode)
        command.add_argument('--manifest',type=Path,default=MANIFEST)
        command.add_argument('--data-root',type=Path,default=DATA_ROOT)
        command.add_argument('--archive',type=Path,default=ARCHIVE)
        command.add_argument('--weights',type=Path,default=default_weights)
        command.add_argument('--weights-metadata',type=Path,default=ROOT/'bundled_weights_provenance.json')
        command.add_argument('--weights-sha256')
        command.add_argument('--recipe',type=Path,default=ROOT/'configs/dfagcn_nwpu_smoke_v1.json')
        command.add_argument('--train-binary',type=Path,default=ROOT/'liblinear/bin/train')
        command.add_argument('--predict-binary',type=Path,default=ROOT/'liblinear/bin/predict')
        command.add_argument('--output',type=Path,default=ROOT/'runs/nwpu_smoke_01')
        command.add_argument('--cpu-threads',type=int,default=8)
        command.add_argument('--extract-batch-size',type=int,default=1)
        if mode=='run':command.add_argument('--gpu-uuid',required=True)
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if args.cpu_threads<1 or args.extract_batch_size<1:
        raise ValueError('thread and extraction batch counts must be positive')
    if args.action=='inspect':
        report=inspect_resources(args)
        print(json.dumps(report,indent=2,sort_keys=True))
        return 0 if report['ready'] else 2
    run_smoke(args)
    return 0


if __name__=='__main__':
    # Standalone invocation must locate the extracted repository without cwd assumptions.
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    try:
        raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,zipfile.BadZipFile) as exc:
        print('DFAGCN_NWPU_STOP '+str(exc),file=sys.stderr)
        raise SystemExit(2)
