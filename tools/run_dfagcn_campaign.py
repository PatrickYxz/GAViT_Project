"""Frozen independent DFAGCN campaign, seeds43--45; single explicit GPU.

No algorithm files are changed. All calibration/refit/test calls use the v1
pipeline. Only seed, physical GPU binding and disk destinations differ.
Inspect/status remain stdlib-only. Fresh run refuses partial or existing output.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
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
SCHEMA='dfagcn-campaign-20261110-v1'
SEEDS=(43,44,45)
WEIGHTS_SHA='397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0'
RECIPE_SHA='86faee50060a74119250a538cd96df90e65864bf319ee4befb01dddd6effca92'
MANIFEST_SHA='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
DISK_GATE_BYTES=60*2**30
FEATURE_CACHE_BYTES=10102579200
SVM_TEXT_UPPER_BYTES=15434779500
PERSISTENT_GATE_BYTES=2**30
FROZEN_SOURCE_SHA256={'external_baselines/dfagcn/__init__.py': 'b35023697642f9d2de385bfaf26ef7d40d1e028de629a5088d1c4275e1583f89', 'external_baselines/dfagcn/__main__.py': '935a1c1166b0c1ea35a82256345000bf2c73ded718d77773bc27a71ecce28f7d', 'external_baselines/dfagcn/cache.py': 'abe24274c615b303eef7000a533c3902e56a2fc9ee904fda795a0248197d6a80', 'external_baselines/dfagcn/cli.py': '89a4e4c340d70ae5b070e74b56618d2f657b62a2146c5c10e7fe142f0be850c3', 'external_baselines/dfagcn/config.py': '2d71ed07fca78bb6e1d8ba6bca2a537cf2e9eaeb134f6c7b8f91b08459111e60', 'external_baselines/dfagcn/extraction.py': '819d69decc9853c32df70bca71ee34603420bdae5d04ddd8c65c941b13607178', 'external_baselines/dfagcn/features.py': '2bdf43f7ff78504b9ee5823e00e48092bde890f460ab64c7337fafd94b9553f1', 'external_baselines/dfagcn/fusion.py': '22524aa98322256d01980ea5c5d6a87cbe05a782244bed11575e7a34d8df9329', 'external_baselines/dfagcn/graph.py': 'dcae20bd209b84a6b6dd2a43bc11081b43707073276e94287f9c5d1659a71b27', 'external_baselines/dfagcn/identity.py': '1c8feda534668f4161349a8bae21539c8c41ce9c117cad7dd9f879f5989e75a8', 'external_baselines/dfagcn/linear.py': '8644f8b3b311e73a6082555cc27399926c1dac1e8e440adc27cd35a2557fa941', 'external_baselines/dfagcn/model.py': 'f638c4a606ca392348cf6714856cbd01ddc1b81ef2cc63cb4275b6d5e4d681ff', 'external_baselines/dfagcn/pipeline.py': 'a7971e4314b25574dfd54f006eed569454cc4b67feab8aea6f831c846926a61a', 'external_baselines/dfagcn/smoke.py': '198a09ae8a12de96f7a801624a5cda940b81a4340e632b652c473fddc81be927', 'configs/dfagcn_independent_v1.json': '86faee50060a74119250a538cd96df90e65864bf319ee4befb01dddd6effca92', 'scene_classification/data.py': 'ebcf0dd7585a060465df45bdf4e44ec7f47edd95f1fa8a12cbdd5723b948fa69', 'scene_classification/metrics.py': '6ef64ee533d0ca5c2dadae066a6254dcd4d107a24709b77009910067f37320b3'}
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


def require_gpu(uuid,inventory):
    if not isinstance(uuid,str) or not uuid.startswith('GPU-') or inventory.get('error'):
        raise ValueError('working NVIDIA query and explicit physical UUID required')
    matching=[gpu for gpu in inventory.get('gpus',[]) if gpu['uuid']==uuid]
    if len(matching)!=1:raise ValueError('requested physical GPU UUID not uniquely present')
    gpu=matching[0]
    if not 0<gpu['memory_free_mib']<=gpu['memory_total_mib'] or gpu['memory_free_mib']<9000:
        raise ValueError('at least9000MiB free GPU memory required for unchanged batch100 recipe')
    if any(row.get('gpu_uuid')==uuid for row in inventory.get('processes',[])):
        raise ValueError('requested GPU already has a compute process')
    return gpu


def require_cuda_memory(nvml_mib,cuda_bytes):
    if type(cuda_bytes) is not int or cuda_bytes<=0 or abs(nvml_mib*2**20-cuda_bytes)>2**30:
        raise ValueError('NVML/CUDA total-memory gap exceeds1GiB')


def _select_gpu(requested,inventory):
    return require_gpu(requested,inventory)


def recipe_for_seed(path,seed):
    if type(seed) is not int or seed not in SEEDS:raise ValueError('campaign seeds frozen to43,44,45')
    result=recipe_bytes(path);result['seed']=seed
    return result


def source_parity(reference):
    reference=Path(reference)
    files=sorted((ROOT/'external_baselines/dfagcn').glob('*.py'))
    files += [ROOT/'configs/dfagcn_independent_v1.json',ROOT/'scene_classification/data.py',ROOT/'scene_classification/metrics.py']
    hashes={str(p.relative_to(ROOT)):GPU.sha256_file(p) for p in files}
    matched=all((reference/name).is_file() and GPU.sha256_file(reference/name)==digest for name,digest in hashes.items())
    return {'matched':matched,'reference':str(reference),'files':hashes}


def expected_cache_identity(manifest,rows):
    from external_baselines.dfagcn.identity import build_cache_identity
    names=('features.py','extraction.py','cache.py','identity.py')
    source=GPU._digest({name:GPU.sha256_file(ROOT/'external_baselines/dfagcn'/name) for name in names})
    return build_cache_identity(manifest_sha256=MANIFEST_SHA,class_names=manifest['class_names'],
        samples=[{'path':r['path'],'sha256':r['sha256']} for r in rows],weights_sha256=WEIGHTS_SHA,
        preprocessing=GPU.PREPROCESSING,feature_spec={'pooling_hw':[7,7],'conv_activation':'pre_relu','fc_activation':'post_relu'},
        source_sha256=source,stage='frozen_pretrained')


def inspect_frozen_cache(path,identities):
    from external_baselines.dfagcn.identity import cache_key
    path=Path(path)
    ready=json.loads((path/'cache.ready.json').read_text())
    if ready.get('schema')!=SCHEMA or not ready.get('completed') or set(ready.get('splits',{}))!=set(identities):
        raise ValueError('cache incomplete; preserve it and use a new cache path')
    verified={}
    for split,identity in identities.items():
        directory=path/(split+'_cache');raw=(directory/'index.json').read_bytes();index=json.loads(raw)
        if (index.get('schema')!='dfagcn-pooled-cache-v1' or index.get('identity')!=identity
                or index.get('key')!=cache_key(identity) or set(index.get('arrays',{}))!={'conv3','conv4','conv5','fc'}):
            raise ValueError('frozen cache exact identity mismatch: '+split)
        if ready['splits'][split]!=hashlib.sha256(raw).hexdigest():raise ValueError('cache ready receipt/index mismatch')
        for name,width in {'conv3':256,'conv4':512,'conv5':512,'fc':4096}.items():
            entry=index['arrays'][name];array=directory/(name+'.npy')
            shape=[len(identity['samples']),width] if name=='fc' else [len(identity['samples']),49,width]
            if (entry.get('file')!=name+'.npy' or entry.get('shape')!=shape or entry.get('dtype')!='float32'
                    or array.is_symlink() or not array.is_file() or GPU.sha256_file(array)!=entry.get('sha256')):
                raise ValueError('frozen cache numeric byte identity mismatch: '+split+'/'+name)
        verified[split]={'index_sha256':ready['splits'][split],'key':index['key'],'samples':len(identity['samples'])}
    return {'completed':True,'array_bytes_verified':True,'splits':verified,'seed_independent':True,
        'scope':'frozen pretrained VGG; no learned graph/fusion/model features reused'}


def require_local_storage(path):
    original=Path(path).absolute()
    if original==Path('/home/featurize/work') or Path('/home/featurize/work') in original.parents:
        raise ValueError('features and SVM scratch must be outside persistent work')
    path=original.resolve()
    if path==Path('/home/featurize/work') or Path('/home/featurize/work') in path.parents:
        raise ValueError('features and SVM scratch must be outside persistent work')
    mounts=Path('/proc/mounts')
    if mounts.is_file():
        matching=[]
        for line in mounts.read_text().splitlines():
            fields=line.split()
            if len(fields)>2:
                mount=Path(fields[1].replace('\\040',' '))
                if path==mount or mount in path.parents:matching.append((len(str(mount)),fields[2]))
        if matching and max(matching)[1] in {'nfs','nfs4','cifs','smb3','fuse.sshfs'}:
            raise ValueError('feature/SVM scratch filesystem must be instance local')
    return str(path)


@contextmanager
def scratch_linear_io(pipeline,output,scratch):
    """Original CLI functions and numerical arguments; relocate only disk I/O.

    Numeric .svm text remains ephemeral. Models, logs, prediction labels and
    original CLI metadata persist immediately after each call, also on failure.
    """
    output,scratch=Path(output).resolve(),Path(scratch).resolve()
    originals={name:getattr(pipeline,name) for name in ('train_linear','predict_linear')}
    def invoke(name,args,kwargs):
        position=2;destination=Path(args[position]).resolve()
        relative=destination.relative_to(output);local=scratch/'svm'/relative
        destination.mkdir(parents=True,exist_ok=False)
        forwarded=list(args);forwarded[position]=local
        try:
            result=originals[name](*forwarded,**kwargs)
        finally:
            receipts={}
            if local.exists():
                for file in sorted(local.iterdir()):
                    if not file.is_file() or file.is_symlink():raise ValueError('unexpected native scratch entry')
                    if file.suffix=='.svm':
                        receipts[file.name]={'path':str(file),'bytes':file.stat().st_size,'sha256':GPU.sha256_file(file),'ephemeral':True}
                    else:shutil.copyfile(file,destination/file.name)
            _write(destination/'scratch_receipt.json',{'schema':SCHEMA,'original_function':name,'local_output':str(local),
                'svm_files':receipts,'persistent_output':str(destination),'numerical_arguments_unchanged':True})
        return destination/Path(result).name if name=='train_linear' else result
    pipeline.train_linear=lambda *a,**kw:invoke('train_linear',a,kw)
    pipeline.predict_linear=lambda *a,**kw:invoke('predict_linear',a,kw)
    try:yield
    finally:
        for name,original in originals.items():setattr(pipeline,name,original)


def inspect_resources(args):
    blockers=[]
    report={'schema':SCHEMA,'action':'inspect','formal_training_started':False,'output':str(args.output),
            'seed':args.seed,'requested_gpu_uuid':args.gpu_uuid,'scratch_root':str(args.scratch_root),'blockers':blockers}
    try:
        changed=[name for name,digest in FROZEN_SOURCE_SHA256.items() if GPU.sha256_file(ROOT/name)!=digest]
        if changed:raise ValueError('frozen seed42 algorithm source changed: '+','.join(changed))
        report['algorithm_source_parity']={'matched':True,'files':FROZEN_SOURCE_SHA256}
    except (OSError,ValueError) as exc:blockers.append('algorithm_source');report['algorithm_error']=str(exc)
    manifest=None;selected={}
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
    try:recipe_for_seed(args.recipe,args.seed);report['recipe_sha256']=RECIPE_SHA
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
            with path.open('rb') as binary_stream:header=binary_stream.read(20)
            if platform.system()=='Linux' and (header[:4]!=b'\x7fELF' or len(header)<20
                    or platform.machine()=='x86_64' and int.from_bytes(header[18:20],'little')!=62):
                blockers.append('binary_platform:'+name)
        else:blockers.append('binary:'+name)
    try:
        cache_path=args.cache or args.scratch_root/'cache'
        require_local_storage(args.scratch_root);require_local_storage(cache_path)
        reuse=cache_path.exists()
        recovered_image_bytes=0
        if report.get('data_source')!='existing31500_images_verified' and not reuse:
            with zipfile.ZipFile(args.archive) as source:
                recovered_image_bytes=sum(info.file_size for info in source.infolist() if not info.is_dir())
        scratch_required=SVM_TEXT_UPPER_BYTES+recovered_image_bytes+2**30
        cache_required=(0 if reuse else FEATURE_CACHE_BYTES)+2**30
        report['disk']=_disk(args.scratch_root,minimum_gate_bytes=scratch_required)
        report['cache_disk']=_disk(cache_path,minimum_gate_bytes=cache_required)
        report['persistent_disk']=_disk(args.output.parent,minimum_gate_bytes=PERSISTENT_GATE_BYTES)
        report['storage_budget']={'svm_text_upper_bytes':SVM_TEXT_UPPER_BYTES,'recovered_image_bytes':recovered_image_bytes,
            'new_frozen_feature_bytes':0 if reuse else FEATURE_CACHE_BYTES,'margin_per_destination_bytes':2**30,
            'scope':'destination filesystem byte inventory; SVM text is conservative .17g upper bound; ZIP image bytes from actual central directory'}
        by_device={}
        for key in ('disk','cache_disk','persistent_disk'):
            info=report[key]
            entry=by_device.setdefault(info['device_id'],{'required':0,'free':info['free_bytes']})
            entry['required']+=info['minimum_gate_bytes'];entry['free']=min(entry['free'],info['free_bytes'])
        report['filesystem_capacity_gates']={str(k):v for k,v in by_device.items()}
        if any(v['free']<v['required'] for v in by_device.values()):blockers.append('disk_capacity')
    except (OSError,ValueError,zipfile.BadZipFile) as exc:blockers.append('disk');report['disk_error']=str(exc)
    if args.cpu_threads!=8:blockers.append('cpu_threads_must_be8')
    if Path(args.output).exists() or Path(args.output).is_symlink():blockers.append('output_already_exists')
    if args.cache is not None and (Path(args.cache).exists() or Path(args.cache).is_symlink()):
        try:
            if Path(args.cache).is_symlink():raise ValueError('cache root must not be symlink')
            if manifest is None:raise ValueError('frozen manifest required for cache identity')
            report['reused_cache']=inspect_frozen_cache(args.cache,{k:expected_cache_identity(manifest,v) for k,v in selected.items()})
        except (OSError,ValueError,KeyError,TypeError) as exc:blockers.append('cache');report['cache_error']=str(exc)
    if Path(args.scratch_root).exists() or Path(args.scratch_root).is_symlink():blockers.append('scratch_already_exists')
    output_path,scratch_path=Path(args.output).resolve(),Path(args.scratch_root).resolve()
    if output_path.is_relative_to(scratch_path) or scratch_path.is_relative_to(output_path):
        blockers.append('overlapping_output_and_scratch')
    cache_path=Path(args.cache or args.scratch_root/'cache').resolve()
    if output_path==cache_path or output_path in cache_path.parents or cache_path in output_path.parents:
        blockers.append('overlapping_output_and_cache')
    inventory=GPU.gpu_inventory();report['nvidia']=inventory
    try:
        report['selected_gpu']=_select_gpu(args.gpu_uuid,inventory)
    except ValueError as exc:blockers.append('target_gpu');report['gpu_error']=str(exc)
    report['ready']=not blockers
    report['readiness_scope']='stdlib inventory; run verifies actual imports, CUDA/software and staged ZIP hashes; no repeated smoke'
    if report['ready']:
        entry=os.environ.get('DFAGCN_CAMPAIGN_ENTRY',str(Path(__file__).resolve()))
        command=[sys.executable,'-u',entry,'run','--seed',str(args.seed),'--gpu-uuid',report['selected_gpu']['uuid']]
        if args.cache is not None:command+=['--cache',str(args.cache)]
        for name in ('manifest','data_root','archive','weights','weights_metadata','recipe','train_binary',
                     'predict_binary','output','background_log','scratch_root'):
            command+=['--'+name.replace('_','-'),str(getattr(args,name))]
        command+=['--weights-sha256',report['weights_sha256']]
        report['run_command']=shlex.join(command)
        if not Path(args.background_log).exists() and Path(args.background_log).parent.is_dir():
            report['nohup_command']='( set -C; exec nohup '+shlex.join(command)+' > '+shlex.quote(str(args.background_log))+' 2>&1 < /dev/null ) &'
        else:report['background_log_unavailable']='existing log or missing parent; use foreground run_command or a fresh --background-log path'
    return report


def require_runtime(runtime,*,expected_gpu_uuid,seed):
    expected={'numpy':VERSIONS['numpy'],
        'pillow':VERSIONS['Pillow'],'cuda':'12.1','cudnn':8902,'device':'cuda:0',
        'physical_gpu_uuid':expected_gpu_uuid,'cpu_threads':8,'seed':seed,'dtype':'float32',
        'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True}
    if (not runtime.get('python','').startswith('3.11.8 ') or any(runtime.get(k)!=v for k,v in expected.items())
            or any(not _version_matches(name,runtime.get(name)) for name in ('torch','torchvision'))):
        raise ValueError('actual CUDA/software/FP32 runtime differs from fixed formal recipe')


def runtime_and_torch(args,gpu):
    GPU.select_cuda_uuid(args.gpu_uuid)
    import torch
    import torchvision
    import numpy
    from PIL import __version__ as pillow_version
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!=gpu['name']:
        raise ValueError('CUDA-visible device name/count differs from selected physical UUID')
    memory=torch.cuda.get_device_properties(0).total_memory
    require_cuda_memory(gpu['memory_total_mib'],memory)
    # CUDA_VISIBLE_DEVICES UUID mapping is established before the first import/init.
    torch.set_num_threads(args.cpu_threads);torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':sys.version,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),
        'numpy':numpy.__version__,'pillow':pillow_version,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
        'gpu':gpu,'cuda_total_memory_bytes':memory,'nvml_cuda_memory_gap_bytes':abs(gpu['memory_total_mib']*2**20-memory),
        'physical_gpu_uuid':args.gpu_uuid,'device':'cuda:0','cpu_threads':args.cpu_threads,
        'amp':False,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False,
        'cudnn_deterministic':True,'dtype':'float32','seed':args.seed}
    require_runtime(runtime,expected_gpu_uuid=args.gpu_uuid,seed=args.seed)
    return torch,runtime


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
    recipe_for_seed(args.recipe,args.seed)
    if sources!=_source_identity() or GPU.sha256_file(args.manifest)!=MANIFEST_SHA:
        raise ValueError('formal source or frozen manifest changed during run')
    if GPU.sha256_file(args.weights)!=preflight['weights_sha256']:raise ValueError('pretrained input bytes changed')
    for name,record in preflight['native_binaries'].items():
        if GPU.sha256_file(getattr(args,name+'_binary'))!=record['sha256']:raise ValueError('native binary changed')


def execute_pipeline(args,output,preflight):
    """CUDA boundary: unchanged full pipeline, deliberately no engineering smoke."""
    manifest=SMOKE.load_manifest(args.manifest,expected_sha256=MANIFEST_SHA);selected=select_full(manifest)
    recipe=recipe_for_seed(args.recipe,args.seed)
    torch,runtime=runtime_and_torch(args,preflight['selected_gpu'])
    _write(output/'runtime.json',runtime)
    print('DFAGCN_FORMAL_DEVICE '+json.dumps(runtime,sort_keys=True),flush=True)
    from external_baselines.dfagcn.cache import FeatureCache
    from external_baselines.dfagcn.extraction import load_vgg_adapter,extract_cache,PREPROCESSING
    from external_baselines.dfagcn.model import DFAGCNBranches
    from external_baselines.dfagcn import pipeline
    from external_baselines.dfagcn.pipeline import source_sha256
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
    cache_root=args.cache or args.scratch_root/'cache'
    reuse=preflight.get('reused_cache') is not None
    if not reuse:cache_root.mkdir(parents=True,exist_ok=False)
    def identity(rows):
        return expected_cache_identity(manifest,rows)
    def extract(split):
        rows=selected[split]
        cache_path=cache_root/(split+'_cache')
        if reuse:
            cache=FeatureCache(cache_path,expected_identity=identity(rows))
            shutil.copyfile(cache_path/'index.json',output/'cache_manifests'/(split+'.json'))
            _write(output/(split+'_extraction.json'),{'samples':len(cache),'cache_key':cache.key,'content_sha256':cache.content_sha256,
                'cache_directory':str(cache_path),'data_mode':'verified_seed_independent_cache_reuse','vgg_extraction_seconds':None,
                'feature_images_per_second':None,'timing_scope':'reused immutable frozen features; no VGG extraction timed'})
            return cache
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
        cache_path=cache_root/(split+'_cache')
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
    if not reuse:phase('split_caches',lambda:derive_split_caches(pool,selected,identity,cache_root))
    for split in ('train','val'):
        shutil.copyfile(cache_root/(split+'_cache/index.json'),output/'cache_manifests'/(split+'.json'))
    with scratch_linear_io(pipeline,output,args.scratch_root):
        selection=phase('calibration',lambda:pipeline.calibrate(args.manifest,cache_root/'train_cache',cache_root/'val_cache',recipe,
            output/'calibration',train_binary=args.train_binary,predict_binary=args.predict_binary,mode='formal',device='cuda:0'))
        if selection['mode']!='formal' or selection['fit_scope']!='train' or len(selection['scores'])!=9:
            raise ValueError('formal calibration did not complete the declared full protocol')
        fitted=phase('refit',lambda:pipeline.refit(args.manifest,cache_root/'pool_cache',output/'calibration/selection.json',
            output/'refit',train_binary=args.train_binary,predict_binary=args.predict_binary,device='cuda:0'))
        if fitted['fit_scope']!='train_plus_val' or fitted['fit_samples']!=6300 or not fitted['completed']:
            raise ValueError('fresh full6300 refit is required before test extraction')
        del pool
        test_cache=phase('test_cache',lambda:extract('test'))
        if len(test_cache)!=25200:raise ValueError('full25200 test cache required')
        tested=phase('test',lambda:pipeline.evaluate(args.manifest,cache_root/'test_cache',output/'refit',output/'test',
            predict_binary=args.predict_binary,device='cuda:0'))
        if not tested['completed'] or len(tested['sample_paths'])!=25200:raise ValueError('full final test incomplete')
    for relative in ('calibration/best.pt','calibration/final.pt','refit/final.pt','refit/model.pt',
                     'calibration/selection.json','refit/refit.json','refit/linear/model.txt','test/test.json'):
        if not (output/relative).is_file():raise ValueError('required formal artifact missing: '+relative)
    if not reuse:
        _write(cache_root/'cache.ready.json',{'schema':SCHEMA,'completed':True,'seed_independent':True,
            'splits':{k:GPU.sha256_file(cache_root/(k+'_cache/index.json')) for k in selected}})
    _write(output/'frozen_cache_receipt.json',inspect_frozen_cache(cache_root,{k:identity(v) for k,v in selected.items()}))
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
        'physical_gpu_uuid':args.gpu_uuid,'seed':args.seed,'campaign_seeds':list(SEEDS),'command':list(sys.argv)}
    with (output/'run.log').open('x',buffering=1) as log,redirect_stdout(_Tee(sys.stdout,log)),redirect_stderr(_Tee(sys.stderr,log)):
        try:
            args.scratch_root.mkdir(parents=True,exist_ok=False)
            _write(output/'preflight.json',preflight);_write(output/'command.json',{'argv':list(sys.argv)})
            _write(output/'storage.json',{'scratch_root':str(args.scratch_root),'persistent_output':str(output),
                'scratch_disk':preflight['disk'],'persistent_disk':preflight['persistent_disk'],
                'scratch_ephemeral':True,'scratch_contents':'frozen feature arrays and recovered input images; regenerate after instance removal',
                'persistent_contents':'source/recipe/runtime, logs/history/checkpoints, LIBLINEAR models/metadata/text receipts, predictions/metrics, cache index manifests'})
            sources=_source_identity();_write(output/'source_identity.json',sources)
            for name in sources:
                destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/name,destination)
            _write(output/'recipe_base.json',recipe_bytes(args.recipe));_write(output/'recipe.json',recipe_for_seed(args.recipe,args.seed))
            result=execute_pipeline(args,output,preflight)
            _unchanged(args,preflight,sources)
            state.update(result,completed=True,formal_training_started=True,wall_seconds=time.monotonic()-started,
                manifest_sha256=MANIFEST_SHA,recipe_sha256=RECIPE_SHA,weights_sha256=preflight['weights_sha256'],
                native_binaries=preflight['native_binaries'],interpretation='Independent v1 recipe, one prespecified campaign seed; not official-paper numerical equivalence. Aggregate with unchanged seed42 only after identity audit.')
            _write(output/'summary.json',state)
            seal_completion(output)
            _write(output/'run.exit',{'exit_code':0})
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


def seal_completion(output):
    files={}
    for path in sorted(output.rglob('*')):
        if path.is_file() and not path.is_symlink() and path.name not in ('run.log','progress.json','completion.json','run.exit'):
            files[str(path.relative_to(output))]=GPU.sha256_file(path)
    _write(output/'completion.json',{'schema':SCHEMA,'completed':True,'files':files})


def verify_completion(output):
    output=Path(output)
    try:
        receipt=json.loads((output/'completion.json').read_text())
        summary=json.loads((output/'summary.json').read_text())
        exit_record=json.loads((output/'run.exit').read_text())
        required={'summary.json','source_identity.json','runtime.json','recipe.json','recipe_base.json',
            'calibration/best.pt','calibration/final.pt','calibration/selection.json','refit/model.pt',
            'refit/final.pt','refit/refit.json','refit/linear/model.txt','test/test.json','frozen_cache_receipt.json'}
        if (receipt.get('schema')!=SCHEMA or not receipt.get('completed') or not summary.get('completed')
                or summary.get('seed') not in SEEDS or exit_record.get('exit_code')!=0
                or not required.issubset(receipt.get('files',{}))):raise ValueError('incomplete completion record')
        for name,digest in receipt['files'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts or str(relative)!=name:raise ValueError('unsafe completion path')
            path=output/relative
            if not path.is_file() or path.is_symlink() or GPU.sha256_file(path)!=digest:raise ValueError('completion byte mismatch: '+name)
        sources=json.loads((output/'source_identity.json').read_text())
        for name,digest in sources.items():
            if (ROOT/name).is_symlink() or GPU.sha256_file(ROOT/name)!=digest:raise ValueError('current source identity mismatch: '+name)
        tested=json.loads((output/'test/test.json').read_text());selection=json.loads((output/'calibration/selection.json').read_text())
        refit=json.loads((output/'refit/refit.json').read_text())
        for record in (tested,selection,refit):
            declared=record.pop('record_sha256',None)
            if declared!=GPU._digest(record) or not record.get('completed'):raise ValueError('pipeline record seal mismatch')
        if (len(tested.get('sample_paths',[]))!=25200 or len(tested.get('predictions',[]))!=25200
                or len(set(tested['sample_paths']))!=25200 or refit.get('fit_samples')!=6300
                or refit.get('fit_scope')!='train_plus_val' or refit.get('reload_max_abs_error',1)>1e-6
                or selection.get('fit_scope')!='train' or len(selection.get('scores',[]))!=9
                or selection.get('selected_epoch')!=summary.get('selected_epoch')
                or selection.get('selected_C')!=summary.get('selected_C')):raise ValueError('full protocol completion mismatch')
        return True,None
    except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:return False,str(exc)


def status(output):
    output=Path(output);result={'schema':SCHEMA,'output':str(output),'state':'not_started','completion_evidence_valid':False}
    if not output.exists():return result
    result.update(state='partial',log_path=str(output/'run.log'))
    for name in ('progress','failure'):
        path=output/(name+'.json')
        if path.is_file():
            try:result[name]=json.loads(path.read_text())
            except ValueError:result[name]={'error':'invalid JSON'}
    if 'failure' in result:result.update(state='failed',error=result['failure'].get('error'))
    valid,error=verify_completion(output);result['completion_evidence_valid']=valid
    if valid:
        summary=json.loads((output/'summary.json').read_text())
        result.update(state='complete',seed=summary['seed'],selected_epoch=summary['selected_epoch'],selected_C=summary['selected_C'],
            test_metrics=summary.get('test_metrics'),wall_seconds=summary.get('wall_seconds'))
    elif (output/'summary.json').exists():result['completion_error']=error
    for phase in ('calibration','refit'):
        history=output/phase/'history.json'
        if history.is_file():
            try:
                rows=json.loads(history.read_text())
                if rows:result[phase+'_last_epoch']=rows[-1]
            except ValueError:result[phase+'_history_error']='invalid JSON'
    return result


def parser():
    root=argparse.ArgumentParser(description='Frozen DFAGCN full NWPU campaign: prespecified seeds43--45')
    commands=root.add_subparsers(dest='action',required=True)
    local=ROOT/'weights/vgg16-397923af.pth'
    for name in ('inspect','run','status'):
        command=commands.add_parser(name)
        command.add_argument('--output',type=Path,required=True)
        if name=='status':continue
        command.add_argument('--seed',type=int,choices=SEEDS,required=True)
        command.add_argument('--gpu-uuid',required=True)
        command.add_argument('--cache',type=Path,help='instance-local frozen cache root; fresh or fully sealed, never partial')
        command.add_argument('--scratch-root',type=Path,required=True)
        command.add_argument('--manifest',type=Path,default=Path('/home/featurize/work/nwpu_protocol_20260927_02.json'))
        command.add_argument('--data-root',type=Path,default=SMOKE.DATA_ROOT)
        command.add_argument('--archive',type=Path,default=Path('/home/featurize/work/NWPU-RESISC45_a826b44.zip'))
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
        command.add_argument('--background-log',type=Path,default=ROOT/'campaign_background.log')
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
