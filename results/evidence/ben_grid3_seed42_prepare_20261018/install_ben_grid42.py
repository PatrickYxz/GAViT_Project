"""Install an isolated immutable experiment release, then optionally dispatch it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

NAMES={'ben_grid42_patch.py','run_ben_grid42.py','ben_grid42_tensor_checks.py'}


def install(payload,target,repo):
    target=Path(target)
    if set(payload['files'])!=NAMES:
        raise RuntimeError('Unexpected release file list')
    for name,text in payload['files'].items():
        compile(text,name,'exec')
    target.mkdir(parents=True,exist_ok=False)
    for name,text in payload['files'].items():
        (target/name).write_text(text)
    spec=importlib.util.spec_from_file_location('installed_grid42_exporter',target/'ben_grid42_patch.py')
    patch=importlib.util.module_from_spec(spec);spec.loader.exec_module(patch)
    actual=patch.export_snapshot(repo,target/'code')
    if actual!=payload['snapshot']:
        raise RuntimeError('Export differs from locally verified source manifest; preserve directory and inspect')
    release=dict(snapshot=actual,tools={name:hashlib.sha256((target/name).read_bytes()).hexdigest() for name in NAMES},
                 dataset='BigEarthNet-19',seed=42,formal_models=['fixed_K9'],historical_reference='fixed_K16_seed42_cd39be9')
    (target/'release.json').write_text(json.dumps(release,indent=2)+'\n')
    patch.verify_snapshot(target/'code',actual)
    print('BEN_GRID3_42_INSTALLED '+str(target),flush=True)
    return release


def dispatch(target):
    target=Path(target)
    # Exclusive log creation rejects a duplicate dispatch without overwriting logs.
    with (target/'run.log').open('xb') as out:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(target/'run_ben_grid42.py'),str(target)],
                              stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
    (target/'run.pid').write_text(str(proc.pid)+'\n')
    print(json.dumps(dict(status='BEN_GRID3_42_DISPATCHED',pid=proc.pid,output=str(target),
                         training_start_verified=False,sequence=['preflight','smoke','train30','best_test']),indent=2),flush=True)
