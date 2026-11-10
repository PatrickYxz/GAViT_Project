"""Build and verify a source-only, self-extracting frozen AID campaign delivery."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import runpy
import stat
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='aid-campaign-delivery-20261110-v1'
MANIFEST='delivery_manifest.json'
RECEIPT='.delivery_receipt.json'
RUNNER='tools/run_aid_campaign.py'
WEIGHTS={'architecture':'swin_tiny_patch4_window7_224','filename':'model.safetensors',
         'sha256':'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36',
         'kind':'timm Swin-Tiny local pretrained state', 'weight_file_included':False}
AID_MANIFEST='results/evidence/hgtnet_aid_formal_complete_20261106/raw/aid_protocol_20260920_03.json'
AID_MANIFEST_SHA256='30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
RECEIPTS=tuple('.worktrees/sparse-hybrid-4n-top2/results/evidence/'+run+'/run.json' for run in
               ('aid_swin_calibrate42_20260920_01','aid_gavit_calibrate42_20260920_01'))


def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _json(value):return json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'


def _relative(name):
    if not isinstance(name,str) or not name or '\\' in name or any(ord(c)<32 for c in name):raise ValueError('unsafe archive path: '+str(name))
    path=PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or path.as_posix()!=name or ':' in path.parts[0]:raise ValueError('unsafe archive path: '+name)
    return name


def _target(root,name):
    path=Path(root)/_relative(name)
    for parent in (path,*path.parents):
        if parent.is_symlink() and not (sys.platform=='darwin' and str(parent) in ('/var','/tmp')):raise ValueError('unsafe symbolic link: '+str(parent))
    return path


def _source_names():
    names=['utils.py','experiment_identity.py','grouping_identity.py','soft_assignment_diagnostics.py',
           'tools/run_aid_campaign.py','tools/package_aid_campaign.py']
    names += [p.relative_to(ROOT).as_posix() for folder in ('models','scene_classification') for p in sorted((ROOT/folder).glob('*.py'))]
    if 'scene_classification/aid_campaign_engine.py' not in names:raise ValueError('Campaign engine is not ready to package')
    return sorted(names)


def build_bundle(output_path):
    """Freeze real source bytes and Git base; never assert a clean checkout."""
    output=Path(output_path).absolute();sidecar=output.with_suffix(output.suffix+'.manifest.json')
    for path in (output,sidecar):
        _target(path.parent,path.name)
        if path.exists():raise FileExistsError(path)
    files={name:_target(ROOT,name).read_bytes() for name in _source_names()}
    aid=_target(ROOT,AID_MANIFEST).read_bytes()
    if _sha(aid)!=AID_MANIFEST_SHA256:raise ValueError('Original AID manifest changed')
    files['provenance/aid_protocol_20260920_03.json']=aid
    receipts=[]
    for name in RECEIPTS:
        raw=(ROOT/name).read_bytes();data=json.loads(raw)
        if data.get('pretrained_sha256')!=WEIGHTS['sha256'] or data['architecture'].get('backbone',WEIGHTS['architecture'])!=WEIGHTS['architecture']:
            raise ValueError('Original AID Swin-Tiny weight receipt mismatch: '+name)
        receipts.append({'original_path':name,'sha256':_sha(raw),'pretrained_sha256':data['pretrained_sha256'],
                         'original_architecture':data['architecture']})
    files['provenance/weights_receipts.json']=_json(receipts)
    files['configs/aid_campaign_20261110_v1.json']=_json({'schema':SCHEMA,
        'allowed_jobs':{'soft':[42,43,44,45],'fixed':[45],'swin':[45]},'manifest_sha256':AID_MANIFEST_SHA256,
        'counts':{'train':4000,'val':1000,'pool':5000,'test':5000,'classes':30},'weights':WEIGHTS,
        'batch_size':32,'workers':2,'cpu_threads':8,'device':'cuda:0','gpu_model_fixed':False,
        'archive':{'path':'/home/featurize/work/AID_visionflan_20260920.zip','sha256':'cdb1af67657ef1ed42ed481bf009df689ddc9ea13dfb6eb36277882312703bf7'},
        'calibration_max_epochs':30,'refit':'fresh initialization from identical pretrained state; selected calibration epoch; cosine T_max30',
        'test':'one committed test transaction after refit','automatic_training':False})
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    if not re.fullmatch('[0-9a-f]{40}',revision):raise ValueError('Full Git base commit required')
    status=subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    bootstrap=Path(__file__).read_bytes()
    description={'schema':SCHEMA,'runner':RUNNER,'weights':WEIGHTS,
        'build_provenance':{'git_base_commit':revision,'source_state':'uncommitted_source_snapshot',
                            'working_tree_dirty':bool(status.strip()),'clean_git_asserted':False},
        'files':{name:{'sha256':_sha(raw),'bytes':len(raw)} for name,raw in sorted(files.items())},
        'bootstrap_sha256':_sha(bootstrap),'weight_file_included':False,'compiled_binaries_included':False,
        'dataset_images_included':False,'gpu_execution_verified':False}
    manifest={**description,'bundle_id':_sha(_canonical(description))}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream, zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,raw in [('__main__.py',bootstrap),(MANIFEST,_json(manifest)),*sorted(files.items())]:
            item=zipfile.ZipInfo(name,date_time=(2026,11,10,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED;item.external_attr=(stat.S_IFREG|0o644)<<16
            z.writestr(item,raw)
    record={'path':str(output),'sha256':_sha(output.read_bytes()),'bytes':output.stat().st_size,'bundle_id':manifest['bundle_id'],'manifest':manifest}
    with sidecar.open('xb') as stream:stream.write(_json(record))
    return record


def _read_bundle(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive members')
        for item in z.infolist():
            _relative(item.filename);mode=item.external_attr>>16
            if item.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):raise ValueError('Only regular source archive members allowed')
            if any(parent.as_posix() in names for parent in PurePosixPath(item.filename).parents if parent!=PurePosixPath('.')):raise ValueError('Archive path collision')
        raw=z.read(MANIFEST);manifest=json.loads(raw)
        if manifest.get('schema')!=SCHEMA or manifest.get('runner')!=RUNNER:raise ValueError('Unsupported delivery identity')
        description={k:v for k,v in manifest.items() if k!='bundle_id'}
        if manifest.get('bundle_id')!=_sha(_canonical(description)):raise ValueError('Delivery manifest changed')
        if manifest.get('weights')!=WEIGHTS:raise ValueError('Delivery weights changed')
        files=manifest.get('files')
        if not isinstance(files,dict) or not files or set(names)!=set(files)|{MANIFEST,'__main__.py'}:raise ValueError('Unexpected archive source set')
        required={'utils.py','experiment_identity.py','grouping_identity.py','soft_assignment_diagnostics.py',RUNNER,
                  'scene_classification/aid_campaign_engine.py','scene_classification/aid_campaign_state.py',
                  'provenance/aid_protocol_20260920_03.json','provenance/weights_receipts.json','configs/aid_campaign_20261110_v1.json'}
        if not required.issubset(files):raise ValueError('Incomplete campaign bundle')
        if _sha(z.read('__main__.py'))!=manifest['bootstrap_sha256']:raise ValueError('Bootstrap changed')
        payload={}
        for name,record in files.items():
            _relative(name)
            if not (name in required or name=='tools/package_aid_campaign.py' or (name.startswith(('models/','scene_classification/')) and name.endswith('.py') and len(PurePosixPath(name).parts)==2)):
                raise ValueError('Unexpected source scope: '+name)
            content=z.read(name)
            if record!={'sha256':_sha(content),'bytes':len(content)}:raise ValueError('Archive source changed: '+name)
            payload[name]=content
        if _sha(payload['provenance/aid_protocol_20260920_03.json'])!=AID_MANIFEST_SHA256:raise ValueError('AID manifest changed')
    return manifest,raw,payload


def install_bundle(path):
    path=Path(path).absolute();_target(path.parent,path.name)
    workspace=path.with_suffix('');_target(workspace.parent,workspace.name)
    manifest,raw,payload=_read_bundle(path)
    if workspace.exists():
        receipt=_target(workspace,RECEIPT);metadata=_target(workspace,MANIFEST)
        if not workspace.is_dir() or not receipt.is_file() or not metadata.is_file():raise ValueError('Existing workspace is foreign or incomplete; choose new package filename')
        if json.loads(receipt.read_text())!={'schema':SCHEMA,'bundle_id':manifest['bundle_id']} or metadata.read_bytes()!=raw:raise ValueError('Workspace delivery identity changed')
        for name,record in manifest['files'].items():
            p=_target(workspace,name)
            if not p.is_file() or p.stat().st_size!=record['bytes'] or _sha(p.read_bytes())!=record['sha256']:raise ValueError('Immutable source changed: '+name)
        for p in workspace.rglob('*'):
            relative=p.relative_to(workspace)
            if p.is_symlink():raise ValueError('Workspace contains symbolic link: '+str(relative))
            if p.is_file() and relative.as_posix() not in manifest['files'] and (p.suffix in ('.py','.pyc','.so','.pyd') or relative.parts[0] in ('models','scene_classification','tools','configs','provenance')):
                raise ValueError('Unexpected immutable workspace source: '+str(relative))
    else:
        workspace.mkdir(exist_ok=False)
        for name,content in payload.items():
            target=_target(workspace,name);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(content)
        with (workspace/MANIFEST).open('xb') as stream:stream.write(raw)
        with (workspace/RECEIPT).open('xb') as stream:stream.write(_json({'schema':SCHEMA,'bundle_id':manifest['bundle_id']}))
    return workspace,manifest


def _bundle_entry(arguments):
    if not arguments or arguments[0] not in ('inspect','status','smoke','run','resume','--help','-h'):raise ValueError('Explicit inspect/status/smoke/run/resume required; no automatic training')
    entry=Path(sys.argv[0]).absolute();workspace,manifest=install_bundle(entry)
    os.chdir(workspace);sys.path.insert(0,str(workspace));sys.dont_write_bytecode=True
    os.environ['AID_CAMPAIGN_DELIVERY_ENTRY']=str(entry)
    script=_target(workspace,manifest['runner']);sys.argv=[str(script),*arguments]
    runpy.run_path(str(script),run_name='__main__');return 0


def main(argv=None):
    arguments=sys.argv[1:] if argv is None else argv
    try:
        if zipfile.is_zipfile(sys.argv[0]):return _bundle_entry(arguments)
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
        args=parser.parse_args(arguments);print(json.dumps(build_bundle(args.output),indent=2,sort_keys=True));return 0
    except (ValueError,KeyError,TypeError,OSError,RuntimeError,zipfile.BadZipFile,subprocess.SubprocessError) as exc:
        print('AID campaign delivery failed: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
