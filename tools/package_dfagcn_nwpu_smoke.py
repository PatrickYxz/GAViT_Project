"""Build a stdlib-only zipapp that installs a checked DFAGCN smoke workspace.

The zipapp carries source and the pinned official LIBLINEAR source archive.
It never downloads weights, installs Python packages, or starts training unless
the user invokes the runner's explicit ``run`` command.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile


SCHEMA = 'dfagcn-nwpu-delivery-v1'
MANIFEST = 'delivery_manifest.json'
RECEIPT = '.delivery_receipt.json'
RUNNER = 'tools/run_dfagcn_nwpu_smoke.py'
ARCHIVE = 'third_party/liblinear/liblinear-v250-source.tar.gz'
PROVENANCE = 'third_party/liblinear/provenance.json'


def _sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _sha_file(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            result.update(block)
    return result.hexdigest()


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _relative(name):
    if not isinstance(name,str) or not name or '\\' in name or any(ord(c)<32 for c in name):
        raise ValueError('unsafe archive path: '+str(name))
    path=PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or path.as_posix()!=name
            or not path.parts or ':' in path.parts[0]):
        raise ValueError('unsafe archive path: '+str(name))
    return name


def _entrypoint(name):
    name=_relative(name)
    if name in ('__main__.py',MANIFEST) or not name.endswith('.py'):
        raise ValueError('entrypoint must be a non-reserved Python source member')
    return name


def _target(root,name):
    path=root/_relative(name)
    current=path
    while current!=root:
        if current.is_symlink():
            raise ValueError('symbolic link is not an immutable source file: '+name)
        current=current.parent
    return path


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write('\n')


def _weights_metadata(path):
    value=json.loads(Path(path).read_text())
    digest=value.get('sha256') if isinstance(value,dict) else None
    if (not isinstance(digest,str) or len(digest)!=64 or not digest.startswith('397923af')
            or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError('weights metadata requires the full VGG16 SHA256, not its filename prefix')
    if value.get('filename','vgg16-397923af.pth')!='vgg16-397923af.pth':
        raise ValueError('unexpected VGG16 weights filename')
    return value


def build_bundle(output_path,*,runner_path=None,weights_metadata_path=None,runner_member=RUNNER,
                 extra_members=()):
    """Create a new pyz, optionally including repository-relative helper files."""
    repository=Path(__file__).resolve().parents[1]
    runner_member=_entrypoint(runner_member)
    if isinstance(extra_members,(str,bytes)) or not isinstance(extra_members,Sequence):
        raise ValueError('extra_members must be a sequence of repository-relative file names')
    extras=tuple(_relative(name) for name in extra_members)
    if len(extras)!=len(set(extras)):
        raise ValueError('duplicate extra member')
    output=Path(output_path)
    sidecar=output.with_suffix(output.suffix+'.manifest.json')
    if output.exists() or sidecar.exists():
        raise FileExistsError(output if output.exists() else sidecar)
    runner=Path(runner_path) if runner_path is not None else repository/runner_member
    files={}
    for path in sorted((repository/'external_baselines').rglob('*.py')):
        files[path.relative_to(repository).as_posix()]=path.read_bytes()
    for name in ('scene_classification/__init__.py','scene_classification/data.py',
                 'scene_classification/metrics.py','experiment_identity.py','grouping_identity.py',
                 'configs/dfagcn_independent_v1.json','configs/dfagcn_nwpu_smoke_v1.json'):
        files[name]=(repository/name).read_bytes()
    if runner_member in files:
        raise ValueError('entrypoint would overwrite another packaged source member')
    files[runner_member]=runner.read_bytes()
    third_party=repository/'results/evidence/dfagcn_pipeline_20261029/liblinear'
    for name in ('liblinear-v250-source.tar.gz','COPYRIGHT','provenance.json'):
        files['third_party/liblinear/'+name]=(third_party/name).read_bytes()
    if weights_metadata_path is not None:
        metadata=_weights_metadata(weights_metadata_path)
        files['bundled_weights_provenance.json']=json.dumps(metadata,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'
    reserved={'__main__.py',MANIFEST,RECEIPT,'bundled_weights_provenance.json'}
    for name in extras:
        if any(name==other or name.startswith(other+'/') or other.startswith(name+'/')
               for other in files.keys()|reserved):
            raise ValueError('extra member would overwrite a reserved or packaged member: '+name)
        source=_target(repository,name)
        if not source.is_file():
            raise ValueError('extra member must be an existing repository file: '+name)
        files[name]=source.read_bytes()
    for name in files:
        _relative(name)
    bootstrap=Path(__file__).read_bytes()
    description={'schema':SCHEMA,'runner':runner_member,
                 'files':{name:{'sha256':_sha_bytes(data),'bytes':len(data)}
                                          for name,data in sorted(files.items())},
                 'bootstrap_sha256':_sha_bytes(bootstrap),
                 'weight_file_included':False,'compiled_binaries_included':False}
    manifest=dict(description,bundle_id=_sha_bytes(_canonical(description)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
            archive.writestr('__main__.py',bootstrap)
            archive.writestr(MANIFEST,json.dumps(manifest,indent=2,sort_keys=True).encode()+b'\n')
            for name,data in sorted(files.items()):
                archive.writestr(name,data)
    record={'path':str(output.resolve()),'sha256':_sha_file(output),'bytes':output.stat().st_size,
            'bundle_id':manifest['bundle_id'],'manifest':manifest}
    _write_json(sidecar,record)
    return record


def _read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):
            raise ValueError('duplicate archive members')
        for name in names:
            _relative(name)
        manifest_bytes=archive.read(MANIFEST)
        manifest=json.loads(manifest_bytes)
        if not isinstance(manifest,dict) or manifest.get('schema')!=SCHEMA:
            raise ValueError('unsupported delivery manifest')
        expected_id=manifest.get('bundle_id')
        description={key:value for key,value in manifest.items() if key!='bundle_id'}
        if expected_id!=_sha_bytes(_canonical(description)):
            raise ValueError('delivery manifest changed')
        files=manifest.get('files')
        if not isinstance(files,dict) or set(names)!=set(files)|{'__main__.py',MANIFEST}:
            raise ValueError('unexpected or incomplete bundle members')
        runner_member=_entrypoint(manifest.get('runner',RUNNER))
        if runner_member not in files:
            raise ValueError('delivery entrypoint is not a packaged source member')
        if _sha_bytes(archive.read('__main__.py'))!=manifest['bootstrap_sha256']:
            raise ValueError('bootstrap bytes changed')
        payload={}
        for name,record in files.items():
            _relative(name)
            data=archive.read(name)
            if record!={'sha256':_sha_bytes(data),'bytes':len(data)}:
                raise ValueError('source bytes changed in archive: '+name)
            payload[name]=data
    return manifest,manifest_bytes,payload


def install_bundle(path):
    """Unpack once; refuse foreign, partial or changed workspaces on reuse."""
    path=Path(path).resolve()
    workspace=path.with_suffix('')
    manifest,manifest_bytes,payload=_read_bundle(path)
    if workspace.is_symlink():
        raise ValueError('workspace must not be a symbolic link')
    if workspace.exists():
        if not workspace.is_dir() or not (workspace/RECEIPT).is_file():
            raise ValueError('existing workspace is foreign or incomplete; preserve it and use a new bundle filename')
        receipt=json.loads((workspace/RECEIPT).read_text())
        if receipt!={'schema':SCHEMA,'bundle_id':manifest['bundle_id']}:
            raise ValueError('workspace receipt changed or belongs to another bundle')
        if (workspace/MANIFEST).read_bytes()!=manifest_bytes:
            raise ValueError('workspace delivery manifest changed')
        for name,record in manifest['files'].items():
            source=_target(workspace,name)
            if not source.is_file() or _sha_file(source)!=record['sha256']:
                raise ValueError('immutable source changed: '+name)
    else:
        workspace.mkdir(exist_ok=False)
        for name,data in payload.items():
            destination=_target(workspace,name)
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
        with (workspace/MANIFEST).open('xb') as stream:
            stream.write(manifest_bytes)
        _write_json(workspace/RECEIPT,{'schema':SCHEMA,'bundle_id':manifest['bundle_id']})
    return workspace,manifest


def _extract_tar(archive_path,destination):
    with tarfile.open(archive_path,'r:gz') as archive:
        members=archive.getmembers()
        seen=set()
        for member in members:
            name=_relative(member.name)
            if name in seen or not (member.isfile() or member.isdir()):
                raise ValueError('unsafe or duplicate LIBLINEAR source member: '+name)
            seen.add(name)
        destination.mkdir(exist_ok=False)
        for member in members:
            target=_target(destination,member.name)
            if member.isdir():
                target.mkdir(parents=True,exist_ok=True)
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as output, archive.extractfile(member) as source:
                    shutil.copyfileobj(source,output)


def build_linear(workspace):
    """Build the pinned official source on this platform; failed builds stay intact."""
    workspace=Path(workspace)
    archive=workspace/ARCHIVE
    provenance=json.loads((workspace/PROVENANCE).read_text())
    archive_sha=_sha_file(archive)
    if archive_sha!=provenance['source_archive_sha256']:
        raise ValueError('LIBLINEAR source archive changed')
    output=workspace/'liblinear'
    record_path=output/'build.json'
    if output.exists():
        if not record_path.is_file():
            raise ValueError('existing LIBLINEAR build is incomplete; preserve it and use a new bundle filename')
        record=json.loads(record_path.read_text())
        if not record.get('completed') or record.get('source_archive_sha256')!=archive_sha:
            raise ValueError('LIBLINEAR build record changed')
        if record.get('platform')!={'system':platform.system(),'machine':platform.machine()}:
            raise ValueError('LIBLINEAR build platform differs; preserve it and use a new bundle filename')
        for name in ('train','predict'):
            binary=output/'bin'/name
            if not binary.is_file() or _sha_file(binary)!=record['binaries'][name]['sha256']:
                raise ValueError('LIBLINEAR binary changed: '+name)
        return dict(record,reused=True)
    make=shutil.which('make')
    cxx=shutil.which('g++') or shutil.which('c++')
    cc=shutil.which('gcc') or shutil.which('cc')
    if not all((make,cxx,cc)):
        raise ValueError('make and C/C++ compilers are required; no packages are installed automatically')
    output.mkdir(exist_ok=False)
    source=output/'source'
    _extract_tar(archive,source)
    command=[make,'-C',str(source),'CXX='+cxx,'CC='+cc,'train','predict']
    started=time.monotonic()
    with (output/'build.log').open('x') as stream:
        run=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
    record={'schema':SCHEMA,'operation':'build-linear','source_archive_sha256':archive_sha,
            'source_tag':provenance['source_tag'],'source_git_commit':provenance['source_git_commit'],
            'license':provenance['license'],'command':command,'return_code':run.returncode,
            'platform':{'system':platform.system(),'machine':platform.machine()},
            'compiler_version':subprocess.run([cxx,'--version'],capture_output=True,text=True,check=True).stdout.strip(),
            'elapsed_seconds':time.monotonic()-started,'completed':run.returncode==0,'binaries':{}}
    if run.returncode==0:
        (output/'bin').mkdir()
        for name in ('train','predict'):
            binary=output/'bin'/name
            shutil.copy2(source/name,binary)
            record['binaries'][name]={'path':str(binary),'sha256':_sha_file(binary)}
    _write_json(record_path,record)
    if run.returncode!=0:
        raise RuntimeError('LIBLINEAR compilation failed; output preserved at '+str(output))
    return dict(record,reused=False)


def _bundle_entry(argv):
    original_entry=Path(sys.argv[0]).resolve()
    workspace,manifest=install_bundle(original_entry)
    if argv and argv[0]=='build-linear':
        if len(argv)!=1:
            raise ValueError('build-linear accepts no extra arguments')
        print(json.dumps(build_linear(workspace),indent=2,sort_keys=True))
        return 0
    if not argv or argv[0] not in ('inspect','status','run','--help','-h'):
        raise ValueError('use inspect, status, build-linear or run; nothing starts automatically')
    os.chdir(workspace)
    sys.path.insert(0,str(workspace))
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MKL_NUM_THREADS','1')
    os.environ['DFAGCN_DELIVERY_ENTRY']=str(original_entry)
    script=_target(workspace,_entrypoint(manifest.get('runner',RUNNER)))
    sys.argv=[str(script),*argv]
    runpy.run_path(str(script),run_name='__main__')
    return 0


def main(argv=None):
    arguments=sys.argv[1:] if argv is None else argv
    if zipfile.is_zipfile(sys.argv[0]):
        try:
            return _bundle_entry(arguments)
        except (ValueError,KeyError,OSError,RuntimeError,zipfile.BadZipFile) as exc:
            print('DFAGCN delivery failed: '+str(exc),file=sys.stderr)
            return 2
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--runner')
    parser.add_argument('--runner-member',default=RUNNER)
    parser.add_argument('--weights-metadata')
    parser.add_argument('--extra-member',action='append',default=[])
    args=parser.parse_args(arguments)
    result=build_bundle(args.output,runner_path=args.runner,weights_metadata_path=args.weights_metadata,
                        runner_member=args.runner_member,extra_members=args.extra_member)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
