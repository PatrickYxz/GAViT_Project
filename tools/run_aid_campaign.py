"""Frozen one-job AID campaign commands; inspect/status need only Python stdlib."""
from __future__ import annotations
import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
WEIGHTS_SHA256 = 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
ARCHIVE_SHA256 = 'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7'
ARCHIVE = '/home/featurize/work/AID_visionflan_20260920.zip'
DATA_ROOT = '/home/featurize/data/AID/AID'
WEIGHTS = '/home/featurize/work/GAViT_Project/bigearth_files/model.safetensors'
ORIGINAL_MANIFEST = ROOT / 'results/evidence/hgtnet_aid_formal_complete_20261106/raw/aid_protocol_20260920_03.json'
DEFAULT_MANIFEST = ROOT / 'provenance/aid_protocol_20260920_03.json' if (ROOT / 'provenance/aid_protocol_20260920_03.json').is_file() else ORIGINAL_MANIFEST
COUNTS = {'train':4000, 'val':1000, 'test':5000}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    if not isinstance(name, str) or not name or '\\' in name or any(ord(c)<32 for c in name):
        raise ValueError('unsafe path: '+str(name))
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or ':' in path.parts[0] or path.as_posix()!=name:
        raise ValueError('unsafe path: '+name)
    return name


def safe_target(root, name):
    target = Path(root) / safe_relative(name)
    for parent in (target, *target.parents):
        if parent.is_symlink() and not (sys.platform=='darwin' and str(parent) in ('/var','/tmp')): raise ValueError('unsafe symbolic link: '+str(parent))
    return target


def validate_job(variant, seed):
    if type(seed) is not int or not ((variant=='soft' and seed in (42,43,44,45)) or (variant in ('fixed','swin') and seed==45)):
        raise ValueError('Job outside frozen campaign: soft42/43/44/45, fixed45, swin45 only')


def load_manifest(path):
    path = Path(path)
    if sha256_file(path)!=MANIFEST_SHA256: raise ValueError('Frozen manifest SHA256 mismatch')
    data = json.loads(path.read_text())
    records, names = data.get('records', []), data.get('class_names', [])
    counts = {split:sum(row.get('split')==split for row in records) for split in COUNTS}
    if data.get('dataset')!='AID' or len(names)!=30 or len(set(names))!=30 or data.get('counts')!=COUNTS or counts!=COUNTS or len(records)!=10000:
        raise ValueError('Frozen AID dataset/class/split counts mismatch')
    seen = set()
    for row in records:
        name = safe_relative(row['path']); label = row['label']
        if name in seen or type(label) is not int or not 0<=label<30 or PurePosixPath(name).parts[0]!=names[label]:
            raise ValueError('AID duplicate path or class label mismatch')
        if not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')): raise ValueError('Missing image SHA256')
        seen.add(name)
    return data


def verify_images(root, records):
    for row in records:
        path = safe_target(root, row['path'])
        if not path.is_file(): raise FileNotFoundError('Missing selected image: '+str(path))
        if sha256_file(path)!=row['sha256']: raise ValueError('Selected image SHA256 mismatch: '+row['path'])
    return {'selected_images':len(records), 'selected_image_bytes_verified':True}


def stage_images(archive, scratch_root, records, *, expected_archive_sha256=ARCHIVE_SHA256):
    """Recover only manifest-selected bytes into a new instance-local directory."""
    archive, scratch = Path(archive), Path(scratch_root).absolute()
    if scratch.is_relative_to(Path('/home/featurize/work')):
        raise ValueError('Raw images must stage on instance-local data, outside persistent work')
    safe_target(scratch.parent,scratch.name)
    if scratch.exists(): raise FileExistsError('Staging requires a new directory: '+str(scratch))
    if sha256_file(archive)!=expected_archive_sha256: raise ValueError('Full AID archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names)!=len(set(names)): raise ValueError('Duplicate ZIP members')
        for info in z.infolist():
            safe_relative(info.filename.rstrip('/') if info.is_dir() else info.filename)
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise ValueError('unsafe ZIP member type: '+info.filename)
        selected = {}
        by_tail={}
        for name in names:
            if name.endswith('/'):continue
            tail='/'.join(PurePosixPath(name).parts[-2:])
            by_tail.setdefault(tail,[]).append(name)
        for row in records:
            relative = safe_relative(row['path'])
            matches = by_tail.get(relative,[])
            if len(matches)!=1: raise ValueError('Missing or ambiguous selected archive image: '+relative)
            selected[relative] = matches[0]
        scratch.mkdir(parents=True,exist_ok=False)
        for row in records:
            path=safe_target(scratch,row['path']); path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with z.open(selected[row['path']]) as source, path.open('xb') as output:
                for chunk in iter(lambda: source.read(1024*1024), b''):
                    digest.update(chunk); output.write(chunk)
            if digest.hexdigest()!=row['sha256']: raise ValueError('Selected image SHA256 mismatch: '+row['path'])
    verify_images(scratch,records)
    return scratch


def _json_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')


def source_identity():
    manifest=ROOT/'delivery_manifest.json'
    if manifest.is_file():
        data=json.loads(manifest.read_text())
        description={k:v for k,v in data.items() if k!='bundle_id'}
        canonical=json.dumps(description,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if hashlib.sha256(canonical).hexdigest()!=data.get('bundle_id'): raise ValueError('Delivery manifest changed')
        for name,record in data['files'].items():
            p=safe_target(ROOT,name)
            if not p.is_file() or p.stat().st_size!=record['bytes'] or sha256_file(p)!=record['sha256']:
                raise ValueError('Immutable source changed: '+name)
        return {'base_commit':data['build_provenance']['git_base_commit'],
                'delivery':'verified_source_only_bundle', 'dirty':bool(data['build_provenance']['working_tree_dirty']),
                'source_state':data['build_provenance']['source_state'], 'bundle_id':data['bundle_id'],
                'files_sha256':{n:r['sha256'] for n,r in data['files'].items()}}
    result=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True)
    commit=result.stdout.strip()
    if not re.fullmatch('[0-9a-f]{40}',commit): raise ValueError('Full Git base commit required')
    paths=[ROOT/n for n in ('utils.py','experiment_identity.py','grouping_identity.py','soft_assignment_diagnostics.py','tools/run_aid_campaign.py','tools/package_aid_campaign.py')]
    paths+=sorted((ROOT/'models').glob('*.py'))+sorted((ROOT/'scene_classification').glob('*.py'))
    status=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    return {'base_commit':commit,'delivery':'local_source_snapshot_without_clean_git_assertion',
            'dirty':bool(status.strip()),'source_state':'uncommitted_source_snapshot',
            'files_sha256':{p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths}}


def nvidia_inventory():
    def query(option):
        completed=subprocess.run(['nvidia-smi',option,'--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
        return list(csv.reader(completed.stdout.splitlines(),skipinitialspace=True))
    gpus=[]
    for row in query('--query-gpu=index,uuid,name,memory.total,memory.free,driver_version'):
        if len(row)!=6: raise ValueError('Unexpected nvidia-smi GPU inventory')
        gpus.append({'index':int(row[0]),'uuid':row[1],'name':row[2], 'memory_total_mib':int(row[3]),'memory_free_mib':int(row[4]),'driver':row[5]})
    processes=[]
    for row in query('--query-compute-apps=pid,gpu_uuid,process_name,used_memory'):
        if len(row)!=4: raise ValueError('Unexpected nvidia-smi process inventory')
        processes.append({'pid':int(row[0]),'uuid':row[1],'name':row[2],'used_memory_mib':row[3]})
    return {'gpus':gpus,'processes':processes}


def select_gpu(inventory, requested_uuid):
    if not requested_uuid or not re.fullmatch(r'GPU-[0-9a-fA-F-]+',requested_uuid): raise ValueError('Explicit physical --gpu-uuid required')
    matches=[g for g in inventory['gpus'] if g['uuid'].lower()==requested_uuid.lower()]
    if len(matches)!=1: raise ValueError('Requested GPU UUID not uniquely present')
    gpu=matches[0]
    if any(p['uuid']==gpu['uuid'] and p['pid']!=os.getpid() for p in inventory['processes']):
        raise ValueError('Selected GPU has an existing compute process')
    if gpu['memory_free_mib']<gpu['memory_total_mib']-1024: raise ValueError('Selected GPU is not idle (more than 1024 MiB occupied)')
    return gpu


def dependency_inventory():
    versions={}
    for name in ('torch','torchvision','timm','torch-geometric','safetensors','numpy','Pillow','scikit-learn'):
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return {'python':sys.version,'packages':versions,'import_tested':False}


def _cuda_uuid():
    candidates=[]
    try:
        candidates += [line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libcudart.so' in line]
    except OSError: pass
    candidates += [ctypes.util.find_library('cudart'),'libcudart.so.12','libcudart.so']
    for name in dict.fromkeys(candidates):
        if not name: continue
        try:
            library=ctypes.CDLL(name); buffer=(ctypes.c_ubyte*16)()
            function=library.cudaDeviceGetUuid; function.argtypes=[ctypes.c_void_p,ctypes.c_int]; function.restype=ctypes.c_int
            if function(ctypes.byref(buffer),0)==0: return 'GPU-'+str(uuid.UUID(bytes=bytes(buffer)))
        except (OSError,AttributeError): continue
    raise ValueError('Cannot independently verify CUDA GPU UUID through libcudart')


def verify_cuda_identity(nvml,runtime):
    if str(runtime.get('uuid','')).lower()!=nvml['uuid'].lower(): raise ValueError('CUDA and NVML GPU UUID mismatch')
    if runtime.get('name','').strip()!=nvml['name'].strip(): raise ValueError('CUDA and NVML GPU name mismatch')
    delta=abs(runtime['memory_total_bytes']/2**20-nvml['memory_total_mib'])
    if delta>1024: raise ValueError('CUDA and NVML GPU capacity gap exceeds 1024 MiB')


def runtime_preflight(gpu):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise ValueError('Exactly one selected CUDA GPU must be visible')
    torch.cuda.init()
    prop=torch.cuda.get_device_properties(0)
    actual_uuid=getattr(prop,'uuid',None)
    runtime={'uuid':str(actual_uuid) if actual_uuid else _cuda_uuid(), 'name':prop.name,
             'memory_total_bytes':prop.total_memory,'torch':torch.__version__, 'cuda':torch.version.cuda,
             'import_tested':True, 'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'cuda_device_uuid':str(actual_uuid) if actual_uuid else _cuda_uuid(),
             'visible_cuda_devices':1,'device':'cuda:0',
             'verification_method':'torch_device_properties_uuid' if actual_uuid else 'cudaDeviceGetUuid'}
    verify_cuda_identity(gpu,runtime)
    from torchvision import transforms  # Exercise binary compatibility before output creation.
    import timm
    from torch_geometric.nn import GATConv
    runtime['timm']=timm.__version__
    return runtime


def status_report(output):
    output=Path(output)
    report={'output':str(output),'status':'incomplete','training_started':False,'import_tested':False}
    if not output.exists(): report['reason']='output_missing'; return report
    report['training_started']=(output/'calibration/first_update.json').is_file() or (output/'refit/first_update.json').is_file()
    try:
        completion=output/'completion.json'
        if completion.is_file():
            marker=output/'COMPLETED'
            if not marker.is_file() or marker.read_text()!=sha256_file(completion)+'\n':raise ValueError('Missing or changed COMPLETED receipt seal')
            data=json.loads(completion.read_text())
            if data.get('status')!='complete' or data.get('summary',{}).get('status')!='complete': raise ValueError('Completion summary is not complete')
            hashes=data.get('artifact_sha256',{})
            required={'job_identity.json','selection.json','run.exit','calibration/first_update.json',
                      'refit/first_update.json','calibration/completion.json','refit/completion.json'}
            if data.get('identity',{}).get('mode')=='formal':required.update(('metrics.json','predictions.csv','per_class.csv','confusion_matrix.csv','evaluation/latest.json'))
            if not isinstance(hashes,dict) or not required.issubset(hashes):raise ValueError('Missing required completion markers or artifact hashes')
            for name,expected in hashes.items():
                path=safe_target(output,name)
                if not path.is_file() or sha256_file(path)!=expected: raise ValueError('Completion artifact hash mismatch: '+name)
            if (output/'run.exit').read_text()!='exit_code=0\n': raise ValueError('Missing successful run.exit')
            identity=output/'job_identity.json'
            if identity.exists() and json.loads(identity.read_text())!=data.get('identity'): raise ValueError('Completion identity mismatch')
            report.update(status='complete',summary=data['summary'],completion_verified=True,training_started=True)
            return report
        import fcntl
        lock=output/'.job.lock'
        if lock.is_file():
            with lock.open('r') as stream:
                try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    report['status']='running'
                    report['training_started']=(output/'calibration/first_update.json').is_file() or (output/'refit/first_update.json').is_file()
                    report['reason']='active_job_lock_with_training_marker' if report['training_started'] else 'active_job_lock_waiting_first_update'
                    return report
        report['reason']='no_verified_completion_or_active_job_lock'
        if (output/'run.exit').exists(): report['run_exit']=(output/'run.exit').read_text().strip()
    except (ValueError,KeyError,TypeError,OSError) as exc:
        report['reason']=str(exc); report['completion_verified']=False
    return report


def inspect_report(args):
    report={'command':args.command,'import_tested':False,'training_started':False,'ready_for_runtime_gate':False,
            'variant':args.variant,'seed':args.seed,'output':str(args.output),'blockers':[],
            'counts':{**COUNTS,'pool':5000},'expected_manifest_sha256':MANIFEST_SHA256,
            'expected_weights_sha256':WEIGHTS_SHA256,'dependencies':dependency_inventory()}
    for name,version in report['dependencies']['packages'].items():
        if version is None:report['blockers'].append('Missing runtime dependency metadata: '+name)
    try: validate_job(args.variant,args.seed)
    except ValueError as exc: report['blockers'].append(str(exc))
    try: report['source_identity']=source_identity()
    except (ValueError,OSError,subprocess.SubprocessError) as exc: report['blockers'].append('source: '+str(exc))
    try:
        manifest=load_manifest(args.manifest)
        report['manifest']={'path':str(args.manifest),'sha256':sha256_file(args.manifest),'verified':True}
        if args.data_root.is_dir(): report['images']=verify_images(args.data_root,manifest['records'])
        elif args.scratch_root and args.scratch_root.is_dir(): report['images']=verify_images(args.scratch_root,manifest['records']); report['staged_data_root']=str(args.scratch_root)
        else:
            report['data_stage_required']=True
            if not args.archive.is_file(): report['blockers'].append('dataset missing and archive missing: '+str(args.archive))
            if not args.scratch_root: report['blockers'].append('dataset missing; explicit new --scratch-root required')
            elif args.scratch_root.is_relative_to(Path('/home/featurize/work')): report['blockers'].append('scratch root must be instance-local outside persistent work')
            if args.archive.is_file():
                actual=sha256_file(args.archive);report['archive_sha256']=actual
                if actual!=ARCHIVE_SHA256:report['blockers'].append('Full AID archive SHA256 mismatch')
    except (ValueError,KeyError,TypeError,OSError) as exc:report['blockers'].append('data: '+str(exc))
    try:
        actual=sha256_file(args.weights); report['weights']={'path':str(args.weights),'sha256':actual}
        if actual!=WEIGHTS_SHA256:report['blockers'].append('Frozen Swin-Tiny weights SHA256 mismatch')
    except OSError as exc: report['blockers'].append('weights: '+str(exc))
    try:
        report['nvidia']=nvidia_inventory()
        if args.gpu_uuid:report['selected_gpu']=select_gpu(report['nvidia'],args.gpu_uuid)
        else: report['blockers'].append('Explicit physical --gpu-uuid required before runtime gate')
    except (ValueError,OSError,subprocess.SubprocessError) as exc:report['blockers'].append('GPU inventory: '+str(exc))
    if args.command!='resume' and args.output.exists():report['blockers'].append('Output already exists; fresh runs never overwrite')
    if args.command=='resume' and not args.output.is_dir():report['blockers'].append('Resume output does not exist')
    report['ready_for_runtime_gate']=not report['blockers']
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','smoke','run','resume','status'])
    parser.add_argument('--variant',choices=['soft','fixed','swin'],default='soft')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    parser.add_argument('--data-root',type=Path,default=Path(DATA_ROOT))
    parser.add_argument('--weights',type=Path,default=Path(WEIGHTS))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--archive',type=Path,default=Path(ARCHIVE))
    parser.add_argument('--scratch-root',type=Path)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--cpu-threads',type=int,default=8)
    args=parser.parse_args(argv)
    mode='smoke' if args.command=='smoke' else 'formal'
    if args.output is None: args.output=Path('/home/featurize/work/aid_campaign_20261110')/(args.variant+'_seed'+str(args.seed)+'_'+mode+'_v1')
    for key in ('manifest','data_root','weights','output','archive','scratch_root'):
        if getattr(args,key) is not None:setattr(args,key,getattr(args,key).absolute())
    if args.command=='status': print(json.dumps(status_report(args.output),indent=2,sort_keys=True));return 0
    report=inspect_report(args)
    if args.command=='inspect': print(json.dumps(report,indent=2,sort_keys=True));return 0 if report['ready_for_runtime_gate'] else 2
    try:
        if args.batch_size!=32 or args.workers!=2 or args.cpu_threads!=8:raise ValueError('Frozen batch32/workers2/threads8 recipe required')
        if report['blockers']:raise ValueError('; '.join(report['blockers']))
        if sys.flags.optimize:raise ValueError('Disable PYTHONOPTIMIZE for experiment execution')
        if report.get('staged_data_root'):args.data_root=Path(report['staged_data_root'])
        elif report.get('data_stage_required'):
            data=load_manifest(args.manifest)
            args.data_root=stage_images(args.archive,args.scratch_root,data['records'])
        selected=select_gpu(nvidia_inventory(),args.gpu_uuid)
        os.environ['CUDA_VISIBLE_DEVICES']=selected['uuid']
        os.environ['OMP_NUM_THREADS']=str(args.cpu_threads);os.environ['MKL_NUM_THREADS']=str(args.cpu_threads)
        sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
        runtime=runtime_preflight(selected)
        report.update(runtime=runtime,import_tested=True,staged_data_root=str(args.data_root))
        evidence=args.output.parent/(args.output.name+'.preflight')/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex+'.json')
        _json_write(evidence,report)
        from scene_classification.aid_campaign_engine import run_job
        result=run_job({'variant':args.variant,'seed':args.seed,'manifest':str(args.manifest),
            'data_root':str(args.data_root),'weights':str(args.weights),'output':str(args.output),
            'mode':mode,'resume':args.command=='resume','device':'cuda:0','workers':args.workers,
            'batch_size':args.batch_size,'cpu_threads':args.cpu_threads,'gpu_uuid':selected['uuid'],
            'verified_gpu_identity':runtime,'source_identity':report['source_identity'],'expected_weights_sha256':WEIGHTS_SHA256})
        print(json.dumps(result,indent=2,sort_keys=True)); return 0
    except Exception as exc:
        state=status_report(args.output)
        print(json.dumps({'status':'refused_or_failed','error':str(exc),'output':str(args.output),
                          'import_tested':report.get('import_tested',False),
                          'training_started':state['training_started'],'job_status':state},indent=2),file=sys.stderr)
        return 2

if __name__=='__main__':raise SystemExit(main())
