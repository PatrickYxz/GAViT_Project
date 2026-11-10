import json
import pathlib
import subprocess
import sys
import time

root = pathlib.Path('/Users/patrick/Projects/GAViT_Project')
evidence = root / 'results/evidence/dfagcn_primitives_20261028'
command = sys.argv[2:]
label = sys.argv[1]
if not label.replace('_', '').isalnum():
    raise ValueError('simple evidence label required')
started = time.monotonic()
with (evidence / (label + '.log')).open('w') as stream:
    run = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
record = dict(command=command, exit_code=run.returncode, elapsed_seconds=time.monotonic() - started)
(evidence / (label + '.json')).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record))
print('\n'.join((evidence / (label + '.log')).read_text().splitlines()[-25:]))
sys.exit(run.returncode)
