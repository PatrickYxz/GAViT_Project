import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import time
import zlib

ROOT = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
command = (ROOT / 'docs/paste_aid_efficiency.sh').read_text().strip()
assert len(command.splitlines()) == 1
for shell in ('bash', 'zsh'):
    subprocess.run([shell, '-n', '-c', command], check=True)
boot = shlex.split(command)
assert boot[:2] == ['python', '-c'] and len(boot) == 3
compile(boot[2], '<paste-bootstrap>', 'exec')
guide = (ROOT / 'docs/run_aid_efficiency.md').read_text()
assert command in guide
checks = {'bash_zsh_syntax': True, 'single_line_bootstrap_compiles': True}
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    fakebin = td / 'bin'
    fakebin.mkdir()
    python = Path('/private/tmp/gavit-aid-qa-20260919/bin/python')
    (fakebin / 'python').symlink_to(python)
    smi = fakebin / 'nvidia-smi'
    smi.write_text('#!/bin/sh\necho 999999999\n')
    smi.chmod(0o755)
    output = td / 'run'
    patched = command.replace('/home/featurize/work/aid_efficiency_20260922_01', str(output))
    # Keep all writes in the fixture. GPU guard fails before any real model load.
    pieces = shlex.split(patched)
    payload_start = pieces[2].index('base64.b64decode("') + len('base64.b64decode("')
    payload_end = pieces[2].index('"', payload_start)
    deployed_source = zlib.decompress(base64.b64decode(pieces[2][payload_start:payload_end]))
    assert deployed_source == (ROOT / 'tools/benchmark_aid.py').read_bytes()
    env = dict(os.environ, PATH=str(fakebin) + ':' + os.environ['PATH'])
    # Use a harmless fake persistent work path for lock creation via argv.
    patched = patched.replace('"--output",str(p)', '"--work",str(p),"--output",str(p)')
    started = subprocess.run(['zsh', '-c', patched], env=env, capture_output=True, text=True, check=True)
    deadline = time.monotonic() + 15
    while not (output / 'run.exit').exists() and time.monotonic() < deadline:
        time.sleep(.05)
    assert (output / 'benchmark_aid.py').read_bytes() == deployed_source
    assert (output / 'run.exit').read_text() == 'exit_code=1\n'
    failure = json.loads((output / 'failure.json').read_text())
    assert 'another compute process' in failure['error']
    assert not (output / 'summary.json').exists() and not (output / 'stages').exists()
    before = {str(p): p.read_bytes() for p in output.rglob('*') if p.is_file()}
    again = subprocess.run(['zsh', '-c', patched], env=env, capture_output=True, text=True)
    assert again.returncode != 0 and 'FileExistsError' in again.stderr
    assert all(Path(p).read_bytes() == value for p, value in before.items())
    checks.update(payload_identical_to_reviewed_source=True, detached_launch_writes_explicit_failure=True,
                  occupied_gpu_stops_before_workers=True, no_success_summary_on_failure=True,
                  repeated_launch_preserves_all_previous_files=True)
evidence = ROOT / 'results/evidence/aid_efficiency_local_20260922'
checks['probe_sha256'] = hashlib.sha256((ROOT / 'tools/benchmark_aid.py').read_bytes()).hexdigest()
(evidence / 'launcher.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
