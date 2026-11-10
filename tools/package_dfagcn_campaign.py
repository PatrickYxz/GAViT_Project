"""Immutable portable DFAGCN campaign zipapp; explicit native build, no installs.

Pinned LIBLINEAR source/provenance is bundled. Optional --linux-bin-dir includes
verified ELF executables; unavailable historical binaries are never invented.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='dfagcn-campaign-delivery-20261110-v1'
MANIFEST='delivery_manifest.json'
RECEIPT='.delivery_receipt.json'
RUNNER='tools/run_dfagcn_campaign.py'

def _old():
    spec=importlib.util.spec_from_file_location('_dfagcn_campaign_delivery_helper',ROOT/'tools/package_dfagcn_nwpu_smoke.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);return helper

def _sha(data):return hashlib.sha256(data).hexdigest()
def _file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def _write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
def _relative(name):
    from pathlib import PurePosixPath
    if (not isinstance(name,str) or not name or '\\' in name or ':' in name or any(ord(c)<32 for c in name)
            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts or PurePosixPath(name).as_posix()!=name):
        raise ValueError('unsafe delivery member: '+str(name))
    return name

def _target(root,name):
    path=root/_relative(name);current=path
    while current!=root:
        if current.is_symlink():raise ValueError('symbolic link in delivery path: '+name)
        current=current.parent
    return path

def build_bundle(output,*,linux_bin_dir=None):
    output=Path(output);sidecar=output.with_suffix(output.suffix+'.manifest.json')
    if output.exists() or sidecar.exists():raise FileExistsError('delivery output already exists')
    old=ROOT/'output/dfagcn_nwpu_formal_3080_20261030_fix1'
    spec=importlib.util.spec_from_file_location('_dfagcn_campaign_for_delivery',ROOT/RUNNER)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    parity=runner.source_parity(old)
    if not parity['matched']:raise ValueError('source differs from completed seed42 fix1')
    members=[*sorted('external_baselines/dfagcn/'+p.name for p in (ROOT/'external_baselines/dfagcn').glob('*.py')),
        'external_baselines/__init__.py','scene_classification/__init__.py','scene_classification/data.py','scene_classification/metrics.py',
        'experiment_identity.py','grouping_identity.py','configs/dfagcn_independent_v1.json','configs/dfagcn_nwpu_smoke_v1.json',
        RUNNER,'tools/probe_dfagcn_gpu.py','tools/run_dfagcn_nwpu_smoke.py','tools/package_dfagcn_nwpu_smoke.py']
    payload={name:_target(ROOT,name).read_bytes() for name in members}
    for name in ('COPYRIGHT','provenance.json','liblinear-v250-source.tar.gz'):
        payload['third_party/liblinear/'+name]=(old/'third_party/liblinear'/name).read_bytes()
    payload['bundled_weights_provenance.json']=(old/'bundled_weights_provenance.json').read_bytes()
    freeze={'schema':runner.SCHEMA,'frozen_before_new_test':True,'additional_seeds':[43,44,45],
        'selection_reason':'Consecutive extension of existing42; fixed before new results; old negative history retained.',
        'source_parity':parity,'base_recipe_sha256':runner.RECIPE_SHA,'manifest_sha256':runner.MANIFEST_SHA,
        'weights_sha256':runner.WEIGHTS_SHA,'archive_sha256':runner.SMOKE.ARCHIVE_SHA,
        'seed42_reference':{'oa_percent':86.53174603174604,'macro_f1_percent':86.45890874660422,'test_samples':25200,
            'scope':'existing independent v1 result; not official reproduction or unified training protocol'},
        'numerical_change':'seed only; hardware may differ and is recorded; FP32/TF32/optimizer/loss/epoch/C selection unchanged',
        'scratch_policy':'instance-local frozen features/recovered images/SVM text; persistent learned artifacts and receipts'}
    payload['campaign_freeze.json']=json.dumps(freeze,indent=2,sort_keys=True).encode()+b'\n'
    native={}
    if linux_bin_dir is not None:
        for name in ('train','predict'):
            source=Path(linux_bin_dir)/name
            if source.is_symlink() or not source.is_file():raise ValueError('regular native binary required')
            raw=source.read_bytes()
            if raw[:4]!=b'\x7fELF' or len(raw)<20 or int.from_bytes(raw[18:20],'little')!=62:
                raise ValueError('Linux x86_64 ELF binary required; Mac binaries cannot be bundled')
            payload['liblinear/bin/'+name]=raw;native[name]={'sha256':_sha(raw),'bytes':len(raw)}
        provenance=Path(linux_bin_dir).parent/'build.json'
        if not provenance.is_file():raise ValueError('native build.json provenance required beside bin')
        record=json.loads(provenance.read_text())
        if not record.get('completed') or record.get('platform')!={'system':'Linux','machine':'x86_64'}:
            raise ValueError('native build provenance must confirm completed Linux x86_64 build')
        for name,item in native.items():
            if record.get('binaries',{}).get(name,{}).get('sha256')!=item['sha256']:raise ValueError('native provenance byte mismatch')
        payload['liblinear/build.json']=provenance.read_bytes()
    bootstrap=Path(__file__).read_bytes()
    description={'schema':SCHEMA,'runner':RUNNER,'files':{name:{'sha256':_sha(raw),'bytes':len(raw)} for name,raw in sorted(payload.items())},
        'bootstrap_sha256':_sha(bootstrap),'compiled_binaries_included':bool(native),'native_binaries':native,'weight_file_included':False}
    manifest=dict(description,bundle_id=_sha(_canonical(description)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream,zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        archive.writestr('__main__.py',bootstrap);archive.writestr(MANIFEST,json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        for name,raw in sorted(payload.items()):archive.writestr(name,raw)
    result={'path':str(output.resolve()),'sha256':_file(output),'bytes':output.stat().st_size,'manifest':manifest}
    _write(sidecar,result);return result


def install_bundle(path):
    path=Path(path).resolve();workspace=path.with_suffix('')
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('duplicate bundle member')
        for name in names:_relative(name)
        raw=archive.read(MANIFEST);manifest=json.loads(raw)
        description={k:v for k,v in manifest.items() if k!='bundle_id'}
        if manifest.get('schema')!=SCHEMA or manifest.get('bundle_id')!=_sha(_canonical(description)):
            raise ValueError('delivery manifest changed')
        if set(names)!=set(manifest['files'])|{MANIFEST,'__main__.py'} or _sha(archive.read('__main__.py'))!=manifest['bootstrap_sha256']:
            raise ValueError('bundle membership/bootstrap changed')
        payload={name:archive.read(name) for name in manifest['files']}
        for name,data in payload.items():
            if manifest['files'][name]!={'sha256':_sha(data),'bytes':len(data)}:raise ValueError('bundle source changed: '+name)
    if workspace.is_symlink():raise ValueError('workspace symlink forbidden')
    if workspace.exists():
        if not workspace.is_dir() or not (workspace/RECEIPT).is_file():raise ValueError('existing incomplete/foreign workspace preserved')
        if json.loads((workspace/RECEIPT).read_text())!={'schema':SCHEMA,'bundle_id':manifest['bundle_id']} or (workspace/MANIFEST).read_bytes()!=raw:
            raise ValueError('delivery receipt/manifest changed')
        for name,item in manifest['files'].items():
            target=_target(workspace,name)
            if not target.is_file() or _file(target)!=item['sha256']:raise ValueError('immutable delivered source changed: '+name)
    else:
        workspace.mkdir(exist_ok=False)
        for name,data in payload.items():
            target=_target(workspace,name);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            if name in ('liblinear/bin/train','liblinear/bin/predict'):target.chmod(0o755)
        with (workspace/MANIFEST).open('xb') as stream:stream.write(raw)
        _write(workspace/RECEIPT,{'schema':SCHEMA,'bundle_id':manifest['bundle_id']})
    return workspace,manifest


def main(argv=None):
    arguments=sys.argv[1:] if argv is None else argv
    if zipfile.is_zipfile(sys.argv[0]):
        entry=Path(sys.argv[0]).resolve();workspace,manifest=install_bundle(entry)
        global ROOT
        ROOT=workspace
        if arguments==['build-linear']:
            print(json.dumps(_old().build_linear(workspace),indent=2,sort_keys=True));return 0
        if not arguments or arguments[0] not in ('inspect','run','status','--help','-h'):
            raise ValueError('use inspect, run, status or build-linear')
        os.chdir(workspace);sys.path.insert(0,str(workspace))
        os.environ['DFAGCN_CAMPAIGN_ENTRY']=str(entry)
        sys.argv=[str(workspace/RUNNER),*arguments];runpy.run_path(str(workspace/RUNNER),run_name='__main__');return 0
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--linux-bin-dir')
    args=parser.parse_args(arguments);print(json.dumps(build_bundle(args.output,linux_bin_dir=args.linux_bin_dir),indent=2,sort_keys=True));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,KeyError,TypeError,zipfile.BadZipFile) as exc:
        print('DFAGCN_CAMPAIGN_DELIVERY_STOP '+str(exc),file=sys.stderr);raise SystemExit(2)
