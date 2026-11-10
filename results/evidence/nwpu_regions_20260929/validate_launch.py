"""Offline launcher check: decode exact source, mock child only, reject retries."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

here = Path(__file__).resolve().parent
root = here.parents[2]
command = (here / 'launch_command.txt').read_text().strip()
parts = shlex.split(command)
assert parts[:2] == ['python', '-c'] and len(parts) == 3
source = (root / 'tools/visualize_nwpu_regions.py').read_bytes()
compile(parts[2], '<launcher>', 'exec')
shells = {}
for shell in ('bash', 'zsh'):
    result = subprocess.run([shell, '-n'], input=command + '\n', text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    shells[shell] = {'exit': result.returncode, 'stderr': result.stderr}
with tempfile.TemporaryDirectory(prefix='nwpu-launch-') as tmp:
    code = parts[2].replace('/home/featurize/work', tmp)
    with patch('subprocess.Popen', return_value=SimpleNamespace(pid=1234)) as launch:
        exec(code, {})
        assert launch.call_count == 1
        args, kw = launch.call_args
        staged = Path(args[0][2])
        assert staged.read_bytes() == source
        assert args[0][3:] == ['--output', str(Path(tmp) / 'nwpu_regions42_20260929_01'), '--device', 'cuda']
        assert kw['start_new_session'] and kw['stdin'] == subprocess.DEVNULL
        assert kw['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        try:
            exec(code, {})
        except AssertionError as exc:
            assert 'already exists' in str(exc)
        else:
            raise AssertionError('Duplicate launch was accepted')
        assert launch.call_count == 1
with tempfile.TemporaryDirectory(prefix='nwpu-launch-corrupt-') as tmp:
    code = parts[2].replace('/home/featurize/work', tmp)
    staged = Path(tmp) / f'nwpu_region_tool_20260929_{hashlib.sha256(source).hexdigest()[:12]}.py'
    staged.write_bytes(b'different existing source')
    with patch('subprocess.Popen') as launch:
        try:
            exec(code, {})
        except AssertionError as exc:
            assert 'differs' in str(exc)
        else:
            raise AssertionError('Changed script was accepted')
        assert not launch.called and staged.read_bytes() == b'different existing source'
result = dict(exact_source_roundtrip=True, shell_syntax=shells, background_child_arguments=True,
              duplicate_launch_rejected=True, differing_staged_source_rejected=True,
              source_sha256=hashlib.sha256(source).hexdigest(), command_characters=len(command),
              scope='Local mocked launch; no remote server or CUDA run')
(here / 'launch_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
