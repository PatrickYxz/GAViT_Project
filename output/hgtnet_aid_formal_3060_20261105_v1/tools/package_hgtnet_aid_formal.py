"""Build a pinned source-only AID3060 formal; no install, download or train default."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hgtnet-aid-formal-delivery-v1'
MANIFEST = 'delivery_manifest.json'
RECEIPT = '.delivery_receipt.json'
RUNNER = 'external_baselines/hgtnet/aid_formal_entry.py'
SOURCE_FILES = (
    'external_baselines/__init__.py',
    'external_baselines/hgtnet/__init__.py',
    *('external_baselines/hgtnet/'+name+'.py' for name in (
        'aid_data','aid_entry','aid_formal_protocol','aid_formal_data','aid_formal_checkpoints','aid_formal_engine','aid_formal_entry','config','model','attention','graph',
        'pretrained','smoke','checkpoint','nwpu_engine','resources')),
    'configs/hgtnet_aid_independent_v0.json',
    'tools/inspect_hgtnet_aid_inputs.py',
    'tools/package_hgtnet_aid_formal.py','configs/hgtnet_aid_formal_v1.json',
)
WEIGHTS = {
    'filename':'swin_b-68c6b09e.pth',
    'sha256':'68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893',
    'bytes':351654207,
    'source_url':'https://download.pytorch.org/models/swin_b-68c6b09e.pth',
    'kind':'torchvision.Swin_B_Weights.IMAGENET1K_V1',
    'is_original_hgtnet_weights':False,
}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _json(value):
    return json.dumps(value,indent=2,sort_keys=True,allow_nan=False).encode()+b'\n'


def _relative(name):
    if (not isinstance(name,str) or not name or '\\' in name
            or any(ord(char)<32 for char in name)):
        raise ValueError('unsafe archive path: '+str(name))
    path=PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or path.as_posix()!=name or ':' in path.parts[0]):
        raise ValueError('unsafe archive path: '+name)
    return name


def _names(names):
    if len(names)!=len(set(names)):
        raise ValueError('duplicate archive members')
    seen=set(names)
    for name in names:
        _relative(name)
        if any(parent.as_posix() in seen for parent in PurePosixPath(name).parents
               if parent!=PurePosixPath('.')):
            raise ValueError('archive file/directory collision: '+name)


def _target(root,name):
    path=root/_relative(name)
    for current in (path,*path.parents):
        if current.is_symlink():
            raise ValueError('symbolic link is not an immutable source: '+str(current))
    return path


def build_bundle(output_path):
    """Create a new immutable scoped zipapp and sidecar; never overwrite."""
    output=Path(output_path).absolute()
    sidecar=output.with_suffix(output.suffix+'.manifest.json')
    for target in (output,sidecar):
        _target(target.parent,target.name)
        if target.exists(): raise FileExistsError(target)
    files={}
    for name in SOURCE_FILES:
        path=_target(ROOT,name)
        if not path.is_file(): raise ValueError('missing source file: '+name)
        files[name]=path.read_bytes()
    files['bundled_weights_provenance.json']=_json(WEIGHTS)
    bootstrap=Path(__file__).read_bytes()
    revision=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    if len(revision)!=40 or any(char not in '0123456789abcdef' for char in revision):
        raise ValueError('A full Git base commit is required for experiment provenance')
    description=dict(schema=SCHEMA,runner=RUNNER,target_gpu='3060',
                     build_provenance=dict(git_base_commit=revision,source_state='uncommitted_source_snapshot'),
                     files={name:dict(sha256=_sha(raw),bytes=len(raw)) for name,raw in sorted(files.items())},
                     bootstrap_sha256=_sha(bootstrap),weight_file_included=False,
                     compiled_binaries_included=False)
    manifest=dict(description,bundle_id=_sha(_canonical(description)))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream,zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,
                                                   compresslevel=9) as archive:
        for name,raw in [('__main__.py',bootstrap),(MANIFEST,_json(manifest)),*sorted(files.items())]:
            item=zipfile.ZipInfo(name,date_time=(2026,11,4,0,0,0))
            item.compress_type=zipfile.ZIP_DEFLATED
            item.external_attr=(stat.S_IFREG|0o644)<<16
            archive.writestr(item,raw)
    record=dict(path=str(output),sha256=_sha(output.read_bytes()),bytes=output.stat().st_size,
                bundle_id=manifest['bundle_id'],manifest=manifest)
    with sidecar.open('xb') as stream:stream.write(_json(record))
    return record


def _read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();_names(names)
        for item in archive.infolist():
            mode=item.external_attr>>16
            if item.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                raise ValueError('archive member must be a regular source file: '+item.filename)
        raw_manifest=archive.read(MANIFEST)
        manifest=json.loads(raw_manifest)
        if (not isinstance(manifest,dict) or manifest.get('schema')!=SCHEMA
                or manifest.get('target_gpu')!='3060' or manifest.get('runner')!=RUNNER):
            raise ValueError('unsupported AID3060 delivery identity')
        description={key:value for key,value in manifest.items() if key!='bundle_id'}
        if manifest.get('bundle_id')!=_sha(_canonical(description)):
            raise ValueError('delivery manifest changed')
        files=manifest.get('files')
        expected=set(SOURCE_FILES)|{'bundled_weights_provenance.json'}
        if (not isinstance(files,dict) or set(files)!=expected
                or set(names)!=expected|{'__main__.py',MANIFEST}):
            raise ValueError('unexpected or incomplete bundle members')
        if _sha(archive.read('__main__.py'))!=manifest.get('bootstrap_sha256'):
            raise ValueError('bootstrap bytes changed')
        payload={}
        for name,record in files.items():
            raw=archive.read(name)
            if record!={'sha256':_sha(raw),'bytes':len(raw)}:
                raise ValueError('source bytes changed in archive: '+name)
            payload[name]=raw
        if json.loads(payload['bundled_weights_provenance.json'])!=WEIGHTS:
            raise ValueError('candidate weights provenance changed')
    return manifest,raw_manifest,payload


def install_bundle(path):
    """Safely install into a new real directory, or verify the exact source set."""
    path=Path(path).absolute()
    _target(path.parent,path.name)
    workspace=path.with_suffix('')
    _target(workspace.parent,workspace.name)
    manifest,raw_manifest,payload=_read_bundle(path)
    if workspace.exists():
        receipt=_target(workspace,RECEIPT);metadata=_target(workspace,MANIFEST)
        if not workspace.is_dir() or not receipt.is_file() or not metadata.is_file():
            raise ValueError('existing workspace is foreign or incomplete; use a new bundle filename')
        if json.loads(receipt.read_text())!={'schema':SCHEMA,'bundle_id':manifest['bundle_id']}:
            raise ValueError('workspace receipt changed or belongs to another bundle')
        if metadata.read_bytes()!=raw_manifest:raise ValueError('workspace delivery manifest changed')
        for name,record in manifest['files'].items():
            source=_target(workspace,name)
            if not source.is_file() or _sha(source.read_bytes())!=record['sha256']:
                raise ValueError('immutable source changed: '+name)
        for source in workspace.rglob('*'):
            name=source.relative_to(workspace)
            if source.is_symlink():raise ValueError('workspace contains a symbolic link: '+str(name))
            if (source.is_file() and name.as_posix() not in manifest['files']
                    and (name.parts[0] in ('external_baselines','tools','configs')
                         or source.suffix in ('.py','.pyc','.pyd','.so'))):
                raise ValueError('unexpected immutable workspace source: '+str(name))
    else:
        workspace.mkdir(exist_ok=False)
        for name,raw in payload.items():
            target=_target(workspace,name);target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(raw)
        with (workspace/MANIFEST).open('xb') as stream:stream.write(raw_manifest)
        with (workspace/RECEIPT).open('xb') as stream:
            stream.write(_json({'schema':SCHEMA,'bundle_id':manifest['bundle_id']}))
    return workspace,manifest


def _bundle_entry(arguments):
    if not arguments or arguments[0] not in ('inspect','status','run','resume','--help','-h'):
        raise ValueError('use explicit inspect, status, run or resume; nothing starts automatically')
    entry=Path(sys.argv[0]).absolute()
    workspace,manifest=install_bundle(entry)
    os.chdir(workspace)
    sys.path.insert(0,str(workspace));sys.dont_write_bytecode=True
    os.environ['HGTNET_AID_FORMAL_DELIVERY_ENTRY']=str(entry)
    script=_target(workspace,manifest['runner'])
    sys.argv=[str(script),*arguments]
    runpy.run_path(str(script),run_name='__main__')
    return 0


def main(argv=None):
    arguments=sys.argv[1:] if argv is None else argv
    try:
        if zipfile.is_zipfile(sys.argv[0]):return _bundle_entry(arguments)
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--output',required=True)
        args=parser.parse_args(arguments)
        print(json.dumps(build_bundle(args.output),indent=2,sort_keys=True))
        return 0
    except (ValueError,KeyError,OSError,RuntimeError,zipfile.BadZipFile) as exc:
        print('HGTNet AID delivery failed: '+str(exc),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
