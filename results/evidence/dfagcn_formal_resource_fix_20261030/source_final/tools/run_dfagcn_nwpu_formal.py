"""Fixed seed42 NWPU formal stages on the identified RTX3080.

The algorithms live in external_baselines.dfagcn. This wrapper adds resource
gates, isolated artifacts, readable progress and compact completion evidence.
Inspection/status use only the standard library. Partial runs are preserved;
this entry does not resume epochs or overwrite an existing experiment.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
import traceback
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='dfagcn-nwpu-formal-3080-v1'
RECIPE_SHA='86faee50060a74119250a538cd96df90e65864bf319ee4befb01dddd6effca92'
MANIFEST_SHA='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
DISK_GATE_BYTES=40*2**30
FEATURE_CACHE_BYTES=10102579200
SVM_TEXT_UPPER_BYTES=15434779500
PERSISTENT_GATE_BYTES=SVM_TEXT_UPPER_BYTES+2**30
VERSIONS={'torch':'2.2.2+cu121','torchvision':'0.17.2+cu121','numpy':'1.26.4','Pillow':'10.3.0'}


def _helper(name):
    spec=importlib.util.spec_from_file_location('_dfagcn_formal_'+name,ROOT/'tools'/name)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


GPU=_helper('probe_dfagcn_gpu.py')
SMOKE=_helper('run_dfagcn_nwpu_smoke.py')


def _write(path,value,*,replace=False):
    path=Path(path)
    if replace:
        temporary=path.with_name(path.name+'.tmp')
        temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        temporary.replace(path)
    else:GPU._write(path,value)


def recipe_bytes(path):
    if GPU.sha256_file(path)!=RECIPE_SHA:raise ValueError('complete formal recipe byte SHA256 changed')
    return json.loads(Path(path).read_text())


def select_full(manifest):
    result={split:sorted([r for r in manifest['records'] if r['split']==split],key=lambda r:r['path'])
            for split in ('train','val','test')}
    result['pool']=sorted(result['train']+result['val'],key=lambda r:r['path'])
    if {key:len(rows) for key,rows in result.items()}!={'train':5040,'val':1260,'pool':6300,'test':25200}:
        raise ValueError('formal run requires complete5040/1260/6300/25200 records')
    return result


def dependency_inventory():
    result={'python':platform.python_version()}
    for name in VERSIONS:
        try:
            located=importlib.util.find_spec('PIL' if name=='Pillow' else name) is not None
            version=importlib.metadata.version(name) if located else None
        except (ImportError,ValueError,importlib.metadata.PackageNotFoundError):located=False;version=None
        result[name]={'located':located,'version':version,'import_tested':False}
    return result


def _disk(path,*,minimum_gate_bytes=DISK_GATE_BYTES):
    candidate=Path(path).absolute()
    while not candidate.exists():candidate=candidate.parent
    usage=shutil.disk_usage(candidate)
    return {'checked_path':str(candidate),'free_bytes':usage.free,'minimum_gate_bytes':minimum_gate_bytes,
            'device_id':candidate.stat().st_dev,
            'feature_cache_bytes':FEATURE_CACHE_BYTES,
            'scope':'capacity inventory of the actual destination filesystem; not measured cost'}


def _version_matches(name,value):
    wanted=VERSIONS[name]
    allowed=(wanted,wanted.split('+',1)[0]) if name in ('torch','torchvision') else (wanted,)
    return value in allowed


def _select_gpu(requested,inventory):
    if requested is not None:return GPU.require_gpu(requested,inventory)
    eligible=[]
    for gpu in inventory.get('gpus',[]):
        try:eligible.append(GPU.require_gpu(gpu['uuid'],inventory))
        except ValueError:pass
    if len(eligible)!=1:
        raise ValueError('exactly one idle10GiB RTX3080 is required, or specify --gpu-uuid explicitly')
    return eligible[0]


def inspect_resources(args):
    blockers=[]
    report={'schema':SCHEMA,'action':'inspect','formal_training_started':False,'output':str(args.output),
            'requested_gpu_uuid':args.gpu_uuid,'scratch_root':str(args.scratch_root),'blockers':blockers}
    try:
        manifest=SMOKE.load_manifest(args.manifest,expected_sha256=MANIFEST_SHA)
        selected=select_full(manifest)
        report.update(manifest_sha256=MANIFEST_SHA,counts={k:len(v) for k,v in selected.items()})
        try:
            SMOKE.validate_selected(args.data_root,manifest['records'])
            report['data_source']='existing31500_images_verified'
        except (OSError,ValueError) as exc:
            report['data_error']=str(exc)
            if not Path(args.archive).is_file():raise ValueError('complete source images or persistent ZIP required')
            report.update(data_source='persistent_zip_pending_full_hash_and_staged_recovery',
                archive_expected_sha256=SMOKE.ARCHIVE_SHA,archive_bytes=Path(args.archive).stat().st_size,
                archive_sha_verified=False)
    except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('data_manifest');report['manifest_error']=str(exc)
    try:
        expected=GPU.read_weights_sha(args.weights_metadata,args.weights_sha256)
        if Path(args.weights).name!='vgg16-397923af.pth' or GPU.sha256_file(args.weights)!=expected:
            raise ValueError('complete official pretrained VGG file identity changed')
        report.update(weights_sha256=expected,weights_bytes=Path(args.weights).stat().st_size)
    except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('weights');report['weights_error']=str(exc)
    try:recipe_bytes(args.recipe);report['recipe_sha256']=RECIPE_SHA
    except (OSError,ValueError,TypeError) as exc:blockers.append('recipe');report['recipe_error']=str(exc)
    versions=dependency_inventory();report['dependencies']=versions
    if versions.get('python')!='3.11.8':blockers.append('python_version')
    for name,wanted in VERSIONS.items():
        item=versions.get(name,{})
        if not item.get('located') or not _version_matches(name,item.get('version')):blockers.append('version:'+name)
    report['version_scope']='distribution metadata may omit CUDA suffix; run still checks actual imported CUDA12.1/cuDNN8902 runtime'
    report['native_binaries']={}
    for name in ('train','predict'):
        path=Path(getattr(args,name+'_binary'))
        if path.is_file() and not path.is_symlink() and os.access(path,os.X_OK):
            report['native_binaries'][name]={'path':str(path),'sha256':GPU.sha256_file(path)}
        else:blockers.append('binary:'+name)
    try:
        report['disk']=_disk(args.scratch_root)
        report['persistent_disk']=_disk(args.output.parent,minimum_gate_bytes=PERSISTENT_GATE_BYTES)
        report['persistent_disk']['svm_text_upper_bytes']=SVM_TEXT_UPPER_BYTES
        report['persistent_disk']['artifact_and_margin_bytes']=2**30
        if report['disk']['free_bytes']<DISK_GATE_BYTES:blockers.append('disk_capacity')
        if report['persistent_disk']['free_bytes']<PERSISTENT_GATE_BYTES:blockers.append('persistent_disk_capacity')
        if report['disk']['device_id']==report['persistent_disk']['device_id']:
            report['combined_filesystem_gate_bytes']=DISK_GATE_BYTES+PERSISTENT_GATE_BYTES
            if min(report['disk']['free_bytes'],report['persistent_disk']['free_bytes'])<DISK_GATE_BYTES+PERSISTENT_GATE_BYTES:
                blockers.append('combined_disk_capacity')
    except OSError as exc:blockers.append('disk');report['disk_error']=str(exc)
    if args.cpu_threads!=8:blockers.append('cpu_threads_must_be8')
    if Path(args.output).exists() or Path(args.output).is_symlink():blockers.append('output_already_exists')
    if Path(args.scratch_root).exists() or Path(args.scratch_root).is_symlink():blockers.append('scratch_already_exists')
    output_path,scratch_path=Path(args.output).resolve(),Path(args.scratch_root).resolve()
    if output_path.is_relative_to(scratch_path) or scratch_path.is_relative_to(output_path):
        blockers.append('overlapping_output_and_scratch')
    inventory=GPU.gpu_inventory();report['nvidia']=inventory
    try:
        report['selected_gpu']=_select_gpu(args.gpu_uuid,inventory)
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    report['ready']=not blockers
    report['readiness_scope']='stdlib inventory; run verifies actual imports, CUDA/software and staged ZIP hashes; no repeated smoke'
    if report['ready']:
        entry=os.environ.get('DFAGCN_DELIVERY_ENTRY',str(Path(__file__).resolve()))
        command=[sys.executable,'-u',entry,'run','--gpu-uuid',report['selected_gpu']['uuid']]
        for name in ('manifest','data_root','archive','weights','weights_metadata','recipe','train_binary',
                     'predict_binary','output','background_log','scratch_root'):
            command+=['--'+name.replace('_','-'),str(getattr(args,name))]
        command+=['--weights-sha256',report['weights_sha256']]
        report['run_command']=shlex.join(command)
        if not Path(args.background_log).exists() and Path(args.background_log).parent.is_dir():
            report['nohup_command']='( set -C; exec nohup '+shlex.join(command)+' > '+shlex.quote(str(args.background_log))+' 2>&1 < /dev/null ) &'
        else:report['background_log_unavailable']='existing log or missing parent; use foreground run_command or a fresh --background-log path'
    return report


def require_runtime(runtime,*,expected_gpu_uuid):
    expected={'numpy':VERSIONS['numpy'],
        'pillow':VERSIONS['Pillow'],'cuda':'12.1','cudnn':8902,'device':'cuda:0',
        'physical_gpu_uuid':expected_gpu_uuid,'cpu_threads':8,'seed':42,'dtype':'float32',
        'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
    if (not runtime.get('python','').startswith('3.11.8 ') or any(runtime.get(k)!=v for k,v in expected.items())
            or any(not _version_matches(name,runtime.get(name)) for name in ('torch','torchvision'))):
        raise ValueError('actual CUDA/software/FP32 runtime differs from fixed formal recipe')


def recover_test_images(archive,records,destination,*,expected_sha256=SMOKE.ARCHIVE_SHA):
    """Verified fresh test recovery; unlike the smoke helper its test count is real."""
    archive,destination=Path(archive),Path(destination)
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    if not records or any(r.get('split')!='test' for r in records):raise ValueError('test recovery requires only test records')
    wanted={str(SMOKE._relative(r['path'])):r for r in records}
    if len(wanted)!=len(records):raise ValueError('duplicate selected test paths')
    if GPU.sha256_file(archive)!=expected_sha256:raise ValueError('persistent NWPU ZIP SHA256 mismatch')
    with zipfile.ZipFile(archive) as source:
        members={};seen=set()
        for info in source.infolist():
            name=str(SMOKE._relative(info.filename.rstrip('/') if info.is_dir() else info.filename))
            if info.is_dir():continue
            if name in seen or ((info.external_attr>>16)&0o170000)==0o120000:raise ValueError('duplicate/symlink ZIP member')
            seen.add(name)
            relative=name[len('NWPU-RESISC45/'):] if name.startswith('NWPU-RESISC45/') else name
            if relative in wanted:
                if relative in members or info.flag_bits&1:raise ValueError('ambiguous/encrypted selected ZIP member')
                members[relative]=info
        if set(members)!=set(wanted):raise ValueError('persistent ZIP lacks selected test images')
        if _disk(destination.parent,minimum_gate_bytes=0)['free_bytes']<sum(x.file_size for x in members.values()):
            raise ValueError('insufficient space for selected test image bytes')
        destination.mkdir(parents=True,exist_ok=False)
        for name,row in wanted.items():
            raw=source.read(members[name])
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('selected test member SHA256 mismatch')
            path=destination/name;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('xb') as stream:stream.write(raw)
    SMOKE.validate_selected(destination,records)
    _write(destination/'recovery.json',{'archive_sha256':expected_sha256,'selected_images':len(records),
        'test_images_extracted':len(records),'complete':True})
    return destination


def derive_split_caches(pool,selected,identity,output):
    import torch
    from external_baselines.dfagcn.cache import write_feature_cache
    lookup={row['path']:i for i,row in enumerate(selected['pool'])}
    for split in ('train','val'):
        rows=selected[split]
        def batches():
            for start in range(0,len(rows),100):
                samples=[pool[lookup[r['path']]] for r in rows[start:start+100]]
                yield {name:torch.stack([s[name] for s in samples]) for name in ('conv3','conv4','conv5','fc')}
        write_feature_cache(Path(output)/(split+'_cache'),identity(rows),batches())


def _source_identity():
    files=list((ROOT/'external_baselines/dfagcn').glob('*.py'))
    files += [Path(__file__).resolve(),ROOT/'tools/probe_dfagcn_gpu.py',ROOT/'tools/run_dfagcn_nwpu_smoke.py',
        ROOT/'configs/dfagcn_independent_v1.json',ROOT/'scene_classification/data.py',ROOT/'scene_classification/metrics.py']
    return {str(p.relative_to(ROOT)):GPU.sha256_file(p) for p in sorted(files)}


def _unchanged(args,preflight,sources):
    recipe_bytes(args.recipe)
    if sources!=_source_identity() or GPU.sha256_file(args.manifest)!=MANIFEST_SHA:
        raise ValueError('formal source or frozen manifest changed during run')
    if GPU.sha256_file(args.weights)!=preflight['weights_sha256']:raise ValueError('pretrained input bytes changed')
    for name,record in preflight['native_binaries'].items():
        if GPU.sha256_file(getattr(args,name+'_binary'))!=record['sha256']:raise ValueError('native binary changed')


def execute_pipeline(args,output,preflight):
    """CUDA boundary: unchanged full pipeline, deliberately no engineering smoke."""
    manifest=SMOKE.load_manifest(args.manifest,expected_sha256=MANIFEST_SHA);selected=select_full(manifest)
    recipe=recipe_bytes(args.recipe)
    torch,runtime=GPU._runtime_and_torch(args,preflight['selected_gpu'])
    require_runtime(runtime,expected_gpu_uuid=args.gpu_uuid)
    _write(output/'runtime.json',runtime)
    print('DFAGCN_FORMAL_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
    from external_baselines.dfagcn.cache import FeatureCache
    from external_baselines.dfagcn.extraction import load_vgg_adapter,extract_cache,PREPROCESSING
    from external_baselines.dfagcn.identity import build_cache_identity
    from external_baselines.dfagcn.model import DFAGCNBranches
    from external_baselines.dfagcn.pipeline import feature_source_sha256,source_sha256,calibrate,refit,evaluate
    model=DFAGCNBranches(recipe,45)
    parameters={'graph_including_three_ce_heads':sum(p.numel() for p in model.parameters()),
                'scope':'neural parameters only; frozen VGG and LIBLINEAR reported separately'}
    del model
    phases={};(output/'phases').mkdir();(output/'cache_manifests').mkdir()
    stage_identity={'manifest_sha256':MANIFEST_SHA,'recipe_sha256':RECIPE_SHA,
        'weights_sha256':preflight['weights_sha256'],'physical_gpu_uuid':args.gpu_uuid,
        'source_identity_sha256':GPU.sha256_file(output/'source_identity.json'),
        'runtime_sha256':GPU.sha256_file(output/'runtime.json')}
    def phase(name,work):
        _write(output/'progress.json',{'phase':name,'state':'running'},replace=True)
        print('DFAGCN_FORMAL_PHASE '+name+' started',flush=True)
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        started=time.monotonic();value=work();torch.cuda.synchronize()
        report={'elapsed_seconds':time.monotonic()-started,'completed':True,'identity':stage_identity,
                'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated(0),
                'cuda_peak_reserved_bytes':torch.cuda.max_memory_reserved(0)}
        _write(output/'phases'/(name+'.json'),report);phases[name]=report
        print('DFAGCN_FORMAL_PHASE '+name+' '+json.dumps(report,sort_keys=True),flush=True)
        return value
    def identity(rows):
        return build_cache_identity(manifest_sha256=MANIFEST_SHA,class_names=manifest['class_names'],
            samples=[{'path':r['path'],'sha256':r['sha256']} for r in rows],weights_sha256=preflight['weights_sha256'],
            preprocessing=dict(PREPROCESSING,weights_kind='torchvision_imagenet1k_v1'),
            feature_spec={'pooling_hw':recipe['pooling_hw'],'conv_activation':'pre_relu','fc_activation':'post_relu'},
            source_sha256=feature_source_sha256(),stage='frozen_pretrained')
    def extract(split):
        rows=selected[split]
        try:SMOKE.validate_selected(args.data_root,rows);data_root=args.data_root;mode='existing_images_verified'
        except (OSError,ValueError):
            recovery=SMOKE.recover_selected if split=='pool' else recover_test_images
            print(f'DFAGCN_FORMAL_RECOVERY split={split} images={len(rows)}',flush=True)
            data_root=recovery(args.archive,rows,args.scratch_root/'images'/split);mode='persistent_zip_verified_recovery'
        adapter=load_vgg_adapter(args.weights,preflight['weights_sha256'],device='cuda:0')
        if 'frozen_vgg_extractor' not in parameters:parameters['frozen_vgg_extractor']=sum(p.numel() for p in adapter.parameters())
        batches=0;started=time.monotonic()
        def observe(module,inputs,outputs):
            nonlocal batches
            batches+=inputs[0].shape[0]
            if batches==1 or batches%500==0 or batches==len(rows):
                torch.cuda.synchronize()
                print('DFAGCN_FORMAL_VGG_BATCH '+json.dumps({'split':split,'images_completed':batches,
                    'images_total':len(rows),'device':str(outputs['fc'].device),'batch_samples':inputs[0].shape[0],
                    'elapsed_seconds':time.monotonic()-started},sort_keys=True),flush=True)
        handle=adapter.register_forward_hook(observe)
        cache_path=args.scratch_root/(split+'_cache')
        try:cache=extract_cache(adapter,data_root,rows,identity(rows),cache_path,
                    pool_hw=recipe['pooling_hw'],batch_size=1,device='cuda:0')
        finally:handle.remove()
        torch.cuda.synchronize();elapsed=time.monotonic()-started
        if batches!=len(rows):raise ValueError('real VGG batches did not cover the complete declared split')
        shutil.copyfile(cache_path/'index.json',output/'cache_manifests'/(split+'.json'))
        _write(output/(split+'_extraction.json'),{'samples':len(cache),'cache_key':cache.key,
            'content_sha256':cache.content_sha256,'cache_directory':str(cache_path),
            'data_mode':mode,'vgg_extraction_seconds':elapsed,
            'feature_images_per_second':len(rows)/elapsed,
            'timing_scope':'VGG extraction, image reads/preprocessing and cache write/hash/validation; excludes ZIP recovery and weights load'})
        del adapter;torch.cuda.empty_cache()
        return cache
    pool=phase('pool_cache',lambda:extract('pool'))
    phase('split_caches',lambda:derive_split_caches(pool,selected,identity,args.scratch_root))
    for split in ('train','val'):
        shutil.copyfile(args.scratch_root/(split+'_cache/index.json'),output/'cache_manifests'/(split+'.json'))
    selection=phase('calibration',lambda:calibrate(args.manifest,args.scratch_root/'train_cache',args.scratch_root/'val_cache',recipe,
        output/'calibration',train_binary=args.train_binary,predict_binary=args.predict_binary,mode='formal',device='cuda:0'))
    if selection['mode']!='formal' or selection['fit_scope']!='train' or len(selection['scores'])!=9:
        raise ValueError('formal calibration did not complete the declared full protocol')
    fitted=phase('refit',lambda:refit(args.manifest,args.scratch_root/'pool_cache',output/'calibration/selection.json',
        output/'refit',train_binary=args.train_binary,predict_binary=args.predict_binary,device='cuda:0'))
    if fitted['fit_scope']!='train_plus_val' or fitted['fit_samples']!=6300 or not fitted['completed']:
        raise ValueError('fresh full6300 refit is required before test extraction')
    del pool
    test_cache=phase('test_cache',lambda:extract('test'))
    if len(test_cache)!=25200:raise ValueError('full25200 test cache required')
    tested=phase('test',lambda:evaluate(args.manifest,args.scratch_root/'test_cache',output/'refit',output/'test',
        predict_binary=args.predict_binary,device='cuda:0'))
    if not tested['completed'] or len(tested['sample_paths'])!=25200:raise ValueError('full final test incomplete')
    for relative in ('calibration/best.pt','calibration/final.pt','refit/final.pt','refit/model.pt',
                     'calibration/selection.json','refit/refit.json','refit/linear/model.txt','test/test.json'):
        if not (output/relative).is_file():raise ValueError('required formal artifact missing: '+relative)
    compact={key:tested['metrics'][key] for key in ('oa_percent','macro_f1_percent')}
    return {'counts':{k:len(v) for k,v in selected.items()},'runtime':runtime,'parameters':parameters,
        'scratch_root':str(args.scratch_root),'storage_scope':'regenerable cache/images on instance scratch; checkpoint/history/log/metrics and cache manifests persist in output',
        'pipeline_source_sha256':source_sha256(),'selected_epoch':selection['selected_epoch'],'selected_C':selection['selected_C'],
        'test_metrics':compact,'phase_seconds':{k:v['elapsed_seconds'] for k,v in phases.items()},'phases':phases,
        'cuda_peak_allocated_bytes':max(v['cuda_peak_allocated_bytes'] for v in phases.values()),
        'cuda_peak_reserved_bytes':max(v['cuda_peak_reserved_bytes'] for v in phases.values()),
        'pool_extraction':json.loads((output/'pool_extraction.json').read_text()),
        'test_extraction':json.loads((output/'test_extraction.json').read_text()),
        'cached_test_seconds':tested['cached_pipeline_elapsed_seconds'],
        'timing_scope':'cached test excludes VGG; test_extraction reports separate end-to-end extraction stage',
        'reload_max_abs_error':fitted['reload_max_abs_error'],
        'metrics_path':str(output/'test/test.json'),'selection_path':str(output/'calibration/selection.json'),
        'refit_path':str(output/'refit/refit.json')}


class _Tee:
    def __init__(self,original,log):self.original,self.log=original,log
    def write(self,text):self.original.write(text);self.log.write(text);return len(text)
    def flush(self):self.original.flush();self.log.flush()


def _observed_training_started(output):
    """Preserve batch evidence on failure without inferring execution from a call."""
    uncertain=False
    for line in (output/'run.log').read_text().splitlines():
        if not line.startswith('DFAGCN_TRAIN_BATCH '):continue
        uncertain=True
        try:
            batch=json.loads(line.split(' ',1)[1])
            count,loss=batch.get('batch_samples'),batch.get('loss')
            if (batch.get('phase') in ('calibration','refit') and batch.get('device')=='cuda:0'
                    and type(count) is int and count>0 and type(loss) in (int,float) and math.isfinite(loss)):
                return True
        except (ValueError,TypeError,AttributeError):pass
    progress=output/'progress.json'
    if progress.is_file():
        try:
            uncertain|=json.loads(progress.read_text()).get('phase') in ('calibration','refit','test_cache','test','complete')
        except (ValueError,TypeError,AttributeError):uncertain=True
    return None if uncertain else False


def run_formal(args):
    preflight=inspect_resources(args)
    if not preflight['ready']:raise ValueError('formal resources unavailable: '+', '.join(preflight['blockers']))
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();state={'schema':SCHEMA,'completed':False,'formal_training_started':False,
        'physical_gpu_uuid':args.gpu_uuid,'command':list(sys.argv)}
    with (output/'run.log').open('x',buffering=1) as log,redirect_stdout(_Tee(sys.stdout,log)),redirect_stderr(_Tee(sys.stderr,log)):
        try:
            args.scratch_root.mkdir(parents=True,exist_ok=False)
            _write(output/'preflight.json',preflight);_write(output/'command.json',{'argv':list(sys.argv)})
            _write(output/'storage.json',{'scratch_root':str(args.scratch_root),'persistent_output':str(output),
                'scratch_disk':preflight['disk'],'persistent_disk':preflight['persistent_disk'],
                'scratch_ephemeral':True,'scratch_contents':'frozen feature arrays and recovered input images; regenerate after instance removal',
                'persistent_contents':'source/recipe/runtime, logs/history/checkpoints, LIBLINEAR models/text/metadata, predictions/metrics, cache index manifests'})
            sources=_source_identity();_write(output/'source_identity.json',sources)
            for name in sources:
                destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,destination)
            _write(output/'recipe.json',recipe_bytes(args.recipe))
            result=execute_pipeline(args,output,preflight)
            _unchanged(args,preflight,sources)
            state.update(result,completed=True,formal_training_started=True,wall_seconds=time.monotonic()-started,
                manifest_sha256=MANIFEST_SHA,recipe_sha256=RECIPE_SHA,weights_sha256=preflight['weights_sha256'],
                native_binaries=preflight['native_binaries'],interpretation='Independent implementation, one seed42; not official-paper numerical equivalence or a stability claim.')
            _write(output/'summary.json',state);_write(output/'run.exit',{'exit_code':0})
            _write(output/'progress.json',{'phase':'complete','state':'complete'},replace=True)
            print('DFAGCN_FORMAL_COMPLETE '+str(output/'summary.json'),flush=True)
            return state
        except BaseException as exc:
            log.flush()
            state.update(error=str(exc),error_type=type(exc).__name__,wall_seconds=time.monotonic()-started,
                         formal_training_started=_observed_training_started(output))
            _write(output/'failure.json',state);_write(output/'run.exit',{'exit_code':1})
            (output/'traceback.log').write_text(traceback.format_exc())
            print('DFAGCN_FORMAL_FAILED '+str(output/'failure.json'),flush=True)
            raise


def status(output):
    output=Path(output)
    result={'schema':SCHEMA,'output':str(output),'state':'not_started'}
    if not output.exists():return result
    result.update(state='partial',log_path=str(output/'run.log'))
    if (output/'progress.json').is_file():result['progress']=json.loads((output/'progress.json').read_text())
    if (output/'failure.json').is_file():
        failed=json.loads((output/'failure.json').read_text());result.update(state='failed',error=failed.get('error'))
    if (output/'summary.json').is_file() and (output/'run.exit').is_file():
        summary=json.loads((output/'summary.json').read_text());exit_record=json.loads((output/'run.exit').read_text())
        if summary.get('completed') and exit_record.get('exit_code')==0:
            result.update(state='complete',selected_epoch=summary.get('selected_epoch'),selected_C=summary.get('selected_C'),
                test_metrics={k:summary.get('test_metrics',{}).get(k) for k in ('oa_percent','macro_f1_percent')},
                metrics_path=summary.get('metrics_path'),wall_seconds=summary.get('wall_seconds'))
    for phase in ('calibration','refit'):
        history=output/phase/'history.json'
        if history.is_file():
            rows=json.loads(history.read_text())
            if rows:result[phase+'_last_epoch']=rows[-1]
    return result


def parser():
    root=argparse.ArgumentParser(description='Fixed full NWPU seed42 pipeline on10GiB RTX3080')
    commands=root.add_subparsers(dest='action',required=True)
    local=ROOT/'weights/vgg16-397923af.pth'
    for name in ('inspect','run','status'):
        command=commands.add_parser(name)
        command.add_argument('--output',type=Path,default=ROOT/'runs/nwpu_seed42_01')
        if name=='status':continue
        command.add_argument('--gpu-uuid',required=name=='run',default=None)
        command.add_argument('--scratch-root',type=Path,default=Path('/tmp')/(ROOT.name+'_nwpu_seed42_01'))
        command.add_argument('--manifest',type=Path,default=SMOKE.MANIFEST)
        command.add_argument('--data-root',type=Path,default=SMOKE.DATA_ROOT)
        command.add_argument('--archive',type=Path,default=SMOKE.ARCHIVE)
        command.add_argument('--weights',type=Path,default=local if local.exists() else ROOT.parent/'vgg16-397923af.pth')
        command.add_argument('--weights-metadata',type=Path,default=ROOT/'bundled_weights_provenance.json')
        command.add_argument('--weights-sha256')
        command.add_argument('--recipe',type=Path,default=ROOT/'configs/dfagcn_independent_v1.json')
        binaries=ROOT/'liblinear/bin'
        previous=ROOT.parent/'dfagcn_nwpu_formal_3080_20261030/liblinear/bin'
        if not all((binaries/x).is_file() for x in ('train','predict')) and all((previous/x).is_file() for x in ('train','predict')):
            binaries=previous
        command.add_argument('--train-binary',type=Path,default=binaries/'train')
        command.add_argument('--predict-binary',type=Path,default=binaries/'predict')
        command.add_argument('--cpu-threads',type=int,default=8)
        command.add_argument('--background-log',type=Path,default=ROOT/'formal_seed42_01.log')
    return root


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='status':print(json.dumps(status(args.output),indent=2,sort_keys=True));return 0
    if args.action=='inspect':
        report=inspect_resources(args);print(json.dumps(report,indent=2,sort_keys=True));return 0 if report['ready'] else 2
    run_formal(args);return 0


if __name__=='__main__':
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
    try:raise SystemExit(main())
    except (OSError,ValueError,TypeError,KeyError,RuntimeError,zipfile.BadZipFile) as exc:
        print('DFAGCN_FORMAL_STOP '+str(exc),file=sys.stderr);raise SystemExit(2)
