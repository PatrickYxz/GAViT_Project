"""Build and verify an offline, source-only HGTNet NWPU smoke zipapp.

The bundle contains no weights, dependency installer or compiled binaries.
Only an explicit inspect, status or run action dispatches the checked runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import stat
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'hgtnet-nwpu-delivery-v1'
MANIFEST = 'delivery_manifest.json'
RECEIPT = '.delivery_receipt.json'
RUNNER = 'tools/run_hgtnet_nwpu_smoke.py'
PROBE_RUNNER = 'external_baselines/hgtnet/batch_probe.py'
FORMAL_RUNNER = 'external_baselines/hgtnet/formal_entry.py'
FORMAL_RECIPE = 'configs/hgtnet_nwpu_formal_v1.json'
RUNNERS = (RUNNER, PROBE_RUNNER, FORMAL_RUNNER)
TARGET_GPUS = ('3080', '3090')
WEIGHTS = {
    'filename':'swin_b-68c6b09e.pth',
    'sha256':'68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893',
    'bytes':351654207,
    'source_url':'https://download.pytorch.org/models/swin_b-68c6b09e.pth',
    'kind':'torchvision.Swin_B_Weights.IMAGENET1K_V1',
    'is_original_hgtnet_weights':False,
}
FIXED = ('external_baselines/__init__.py', 'configs/hgtnet_independent_v0.json',
         'tools/run_dfagcn_nwpu_smoke.py', RUNNER)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _json_bytes(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'


def _relative(name):
    if (not isinstance(name, str) or not name or '\\' in name
            or any(ord(c) < 32 for c in name)):
        raise ValueError('unsafe archive path: ' + str(name))
    path = PurePosixPath(name)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or path.as_posix() != name or ':' in path.parts[0]):
        raise ValueError('unsafe archive path: ' + name)
    return name


def _names(names):
    if len(names) != len(set(names)):
        raise ValueError('duplicate archive members')
    seen = set(names)
    for name in names:
        _relative(name)
        if any(parent.as_posix() in seen for parent in PurePosixPath(name).parents
               if parent != PurePosixPath('.')):
            raise ValueError('archive file/directory collision: ' + name)


def _target(root, name):
    path = root / _relative(name)
    current = path
    while current != root:
        if current.is_symlink():
            raise ValueError('symbolic link is not an immutable source: ' + name)
        current = current.parent
    return path


def _payload_name(name):
    path = PurePosixPath(name)
    return (name in FIXED or name == FORMAL_RECIPE or name == 'bundled_weights_provenance.json'
            or (len(path.parts) == 3 and path.parts[:2] == ('external_baselines','hgtnet')
                and path.suffix == '.py'))


def build_bundle(output_path, *, target_gpu='3080', runner_member=RUNNER):
    """Create a new pinned source zipapp and its JSON sidecar; never overwrite."""
    if not isinstance(target_gpu, str) or target_gpu not in TARGET_GPUS:
        raise ValueError('target_gpu must be 3080 or 3090')
    if not isinstance(runner_member, str) or runner_member not in RUNNERS:
        raise ValueError('unsupported delivery runner')
    if runner_member == FORMAL_RUNNER and target_gpu != '3090':
        raise ValueError('formal delivery requires explicit 3090 profile')
    output = Path(output_path)
    sidecar = output.with_suffix(output.suffix + '.manifest.json')
    if output.exists() or output.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(output if output.exists() or output.is_symlink() else sidecar)
    names = list(FIXED) + sorted(path.relative_to(ROOT).as_posix()
                               for path in (ROOT/'external_baselines/hgtnet').glob('*.py'))
    if runner_member == FORMAL_RUNNER: names.append(FORMAL_RECIPE)
    _names(names)
    if runner_member not in names:
        raise ValueError('missing selected runner source: ' + runner_member)
    if 'external_baselines/hgtnet/__init__.py' not in names:
        raise ValueError('HGTNet source package is incomplete')
    files = {}
    for name in names:
        path = _target(ROOT, name)
        if not path.is_file():
            raise ValueError('missing source file: ' + name)
        files[name] = path.read_bytes()
    files['bundled_weights_provenance.json'] = _json_bytes(WEIGHTS)
    bootstrap = Path(__file__).read_bytes()
    description = {
        'schema':SCHEMA, 'runner':runner_member, 'target_gpu':target_gpu,
        'files':{name:{'sha256':_sha(raw), 'bytes':len(raw)} for name,raw in sorted(files.items())},
        'bootstrap_sha256':_sha(bootstrap),
        'weight_file_included':False, 'compiled_binaries_included':False,
    }
    manifest = dict(description, bundle_id=_sha(_canonical(description)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream, zipfile.ZipFile(
            stream, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in [('__main__.py', bootstrap), (MANIFEST, _json_bytes(manifest)),
                          *sorted(files.items())]:
            info = zipfile.ZipInfo(name, date_time=(2026,10,31,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, raw)
    record = {'path':str(output.resolve()), 'sha256':_sha(output.read_bytes()),
              'bytes':output.stat().st_size, 'bundle_id':manifest['bundle_id'], 'manifest':manifest}
    with sidecar.open('xb') as stream:
        stream.write(_json_bytes(record))
    return record


def _read_bundle(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        _names(names)
        for item in archive.infolist():
            mode = item.external_attr >> 16
            if item.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
                raise ValueError('archive member must be a regular source file: ' + item.filename)
        manifest_bytes = archive.read(MANIFEST)
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get('schema') != SCHEMA:
            raise ValueError('unsupported delivery manifest')
        if manifest.get('target_gpu', '3080') not in TARGET_GPUS:
            raise ValueError('unsupported delivery GPU target')
        description = {key:value for key,value in manifest.items() if key != 'bundle_id'}
        if manifest.get('bundle_id') != _sha(_canonical(description)):
            raise ValueError('delivery manifest changed')
        files = manifest.get('files')
        if (not isinstance(files, dict) or manifest.get('runner') not in RUNNERS
                or manifest['runner'] not in files
                or not set(FIXED).issubset(files)
                or 'external_baselines/hgtnet/__init__.py' not in files
                or 'bundled_weights_provenance.json' not in files
                or any(not _payload_name(name) for name in files)
                or set(names) != set(files) | {'__main__.py', MANIFEST}):
            raise ValueError('unexpected or incomplete bundle members')
        if ((FORMAL_RECIPE in files) != (manifest['runner'] == FORMAL_RUNNER)
                or (manifest['runner'] == FORMAL_RUNNER and manifest.get('target_gpu') != '3090')):
            raise ValueError('formal runner/recipe/profile mismatch')
        if _sha(archive.read('__main__.py')) != manifest.get('bootstrap_sha256'):
            raise ValueError('bootstrap bytes changed')
        payload = {}
        for name, record in files.items():
            raw = archive.read(name)
            if record != {'sha256':_sha(raw), 'bytes':len(raw)}:
                raise ValueError('source bytes changed in archive: ' + name)
            payload[name] = raw
        if json.loads(payload['bundled_weights_provenance.json']) != WEIGHTS:
            raise ValueError('candidate weights provenance changed')
    return manifest, manifest_bytes, payload


def install_bundle(path):
    """Install once, or verify every immutable source before reusing a workspace."""
    path = Path(path).resolve()
    workspace = path.with_suffix('')
    manifest, manifest_bytes, payload = _read_bundle(path)
    if workspace.is_symlink():
        raise ValueError('workspace must not be a symbolic link')
    if workspace.exists():
        receipt_path = _target(workspace, RECEIPT)
        manifest_path = _target(workspace, MANIFEST)
        if not workspace.is_dir() or not receipt_path.is_file() or not manifest_path.is_file():
            raise ValueError('existing workspace is foreign or incomplete; use a new bundle filename')
        if json.loads(receipt_path.read_text()) != {'schema':SCHEMA,'bundle_id':manifest['bundle_id']}:
            raise ValueError('workspace receipt changed or belongs to another bundle')
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError('workspace delivery manifest changed')
        for name, record in manifest['files'].items():
            source = _target(workspace, name)
            if not source.is_file() or _sha(source.read_bytes()) != record['sha256']:
                raise ValueError('immutable source changed: ' + name)
        for source in workspace.rglob('*'):
            name = source.relative_to(workspace)
            if source.is_symlink():
                raise ValueError('workspace contains a symbolic link: ' + name.as_posix())
            if (source.is_file() and name.as_posix() not in manifest['files']
                    and (name.parts[0] in ('external_baselines', 'tools', 'configs')
                         or source.suffix in ('.py', '.pyc', '.pyd', '.so'))):
                raise ValueError('unexpected immutable workspace source: ' + name.as_posix())
    else:
        workspace.mkdir(exist_ok=False)
        for name, raw in payload.items():
            destination = _target(workspace, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(raw)
        with (workspace/MANIFEST).open('xb') as stream:
            stream.write(manifest_bytes)
        with (workspace/RECEIPT).open('xb') as stream:
            stream.write(_json_bytes({'schema':SCHEMA,'bundle_id':manifest['bundle_id']}))
    return workspace, manifest


def _bundle_entry(argv):
    if not argv or argv[0] not in ('inspect', 'status', 'run', 'resume', '--help', '-h'):
        raise ValueError('use inspect, status or run; nothing starts automatically')
    original_entry = Path(sys.argv[0]).resolve()
    if argv[0] == 'resume' and _read_bundle(original_entry)[0]['runner'] != FORMAL_RUNNER:
        raise ValueError('resume is available only for the registered formal runner')
    workspace, manifest = install_bundle(original_entry)
    os.chdir(workspace)
    sys.path.insert(0, str(workspace))
    sys.dont_write_bytecode = True
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ['HGTNET_DELIVERY_ENTRY'] = str(original_entry)
    os.environ['HGTNET_TARGET_GPU'] = manifest.get('target_gpu', '3080')
    script = _target(workspace, manifest['runner'])
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name='__main__')
    return 0


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else argv
    try:
        if zipfile.is_zipfile(sys.argv[0]):
            return _bundle_entry(arguments)
        parser = argparse.ArgumentParser(description=__doc__)
        commands = parser.add_subparsers(dest='action', required=True)
        build = commands.add_parser('build', help='build a new source-only zipapp')
        build.add_argument('--output', required=True)
        build.add_argument('--target-gpu', choices=TARGET_GPUS, default='3080')
        build.add_argument('--runner-member', choices=RUNNERS, default=RUNNER)
        args = parser.parse_args(arguments)
        print(json.dumps(build_bundle(args.output, target_gpu=args.target_gpu,
            runner_member=args.runner_member), indent=2, sort_keys=True))
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        print('HGTNet delivery failed: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
