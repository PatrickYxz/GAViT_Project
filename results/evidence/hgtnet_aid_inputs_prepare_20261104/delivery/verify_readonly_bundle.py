"""Execute the actual zipapp under stdlib-import and filesystem-mutation guards."""
import builtins
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[4]
BUNDLE = ROOT / 'output/hgtnet_aid_inputs_20261104.pyz'
SOURCE = ROOT / 'tools/inspect_hgtnet_aid_inputs.py'
MISSING = Path('/private/tmp/gavit-hgtnet-aid-inputs-absent-20261104')
assert not MISSING.exists(), 'Guard fixture must remain absent'
source = SOURCE.read_bytes()
with zipfile.ZipFile(BUNDLE) as archive:
    assert archive.namelist() == ['__main__.py', 'source_identity.json']
    assert archive.testzip() is None
    assert archive.read('__main__.py') == source
    identity = json.loads(archive.read('source_identity.json'))
    assert identity['source_sha256'] == hashlib.sha256(source).hexdigest()
    assert identity['source_size_bytes'] == len(source)

original_import = builtins.__import__
seen_imports, opens, subprocesses = set(), [], []
stdlib = sys.stdlib_module_names

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    top = name.split('.')[0]
    if level:
        top = (globals or {}).get('__package__', '').split('.')[0]
    if top and top not in stdlib and top not in ('__main__',):
        raise AssertionError('non-stdlib import: ' + name)
    seen_imports.add(name)
    return original_import(name, globals, locals, fromlist, level)


def audit(event, arguments):
    if event == 'open':
        path, mode, flags = arguments
        if isinstance(mode, str) and any(character in mode for character in 'wax+'):
            raise AssertionError('file mutation through open mode')
        if isinstance(flags, int) and flags & (os.O_CREAT | os.O_WRONLY | os.O_RDWR | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('file mutation through open flags')
        opens.append(str(path))
    if event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.rename', 'os.link', 'os.symlink', 'os.truncate', 'os.chmod', 'os.chown', 'os.utime'}:
        raise AssertionError('filesystem mutation: ' + event)
    if event == 'subprocess.Popen':
        executable, command, *_ = arguments
        assert command == ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits']
        subprocesses.append(command)

sys.dont_write_bytecode = True
sys.argv = [str(BUNDLE), 'inspect', '--manifest', str(MISSING / 'manifest.json'),
            '--data-root', str(MISSING / 'data'), '--weights', str(MISSING / 'weights.pth')]
builtins.__import__ = guarded_import
sys.addaudithook(audit)
output = io.StringIO()
exit_code = None
with contextlib.redirect_stdout(output):
    try:
        runpy.run_path(str(BUNDLE), run_name='__main__')
    except SystemExit as exc:
        exit_code = exc.code
inventory = json.loads(output.getvalue())
assert exit_code == 0
assert inventory['status'] == 'INPUT_CHECK_INCOMPLETE'
assert inventory['manifest']['status'] == 'missing'
assert inventory['weights']['status'] == 'missing'
assert inventory['manifest']['test_image_bytes_read'] == 0
assert inventory['training_started'] is False
assert inventory['training_entry_available'] is False
assert len(subprocesses) == 1
assert not MISSING.exists()
print(json.dumps(dict(status='PASS', actual_final_zipapp_executed=True, exit_code=exit_code,
                     stdlib_import_guard=True, file_creation_and_mutation_guard=True,
                     opened_file_or_directory_count=len(opens), observed_imports=sorted(seen_imports),
                     subprocesses=subprocesses, inventory=inventory), sort_keys=True, indent=2))
