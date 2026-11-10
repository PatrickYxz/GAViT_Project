"""Read-only artifact gate for the already authorized soft43 queue; stdlib only."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def check_smoke(smoke, expected, expected_preflight, source_hashes):
    smoke = Path(smoke)
    if (smoke / 'run.exit').read_text().strip() != 'exit_code=0':
        raise RuntimeError('Smoke exit missing/nonzero')
    report = json.loads((smoke / 'summary.json').read_text())
    if report != expected:
        raise RuntimeError('Smoke summary differs from returned complete result')
    meta = json.loads((smoke / 'gavit.meta.json').read_text())
    if meta != report['metadata'] or digest(smoke / 'gavit.pth') != report['checkpoint_sha256']:
        raise RuntimeError('Smoke checkpoint/metadata differs from returned result')
    if json.loads((smoke / 'approved_preflight.json').read_text()) != expected_preflight:
        raise RuntimeError('Smoke approved preflight differs')
    info = json.loads((smoke / 'preflight.json').read_text())
    if info['status'] != 'ready' or info['counts'] != report['counts']:
        raise RuntimeError('Wrong engineering subset')
    if info['commit'] != report['commit'] or info['pretrained_sha256'] != 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36':
        raise RuntimeError('Wrong smoke source/pretraining')
    if set(info['subset_sha256']) != {'train', 'val'}:
        raise RuntimeError('Wrong subset identities')
    for split, sha in info['subset_sha256'].items():
        if digest(smoke / 'splits' / (split + '.csv')) != sha:
            raise RuntimeError('Subset CSV changed')
    for name, sha in source_hashes.items():
        if digest(smoke / name) != sha:
            raise RuntimeError('Smoke wrapper source changed: ' + name)
    for name in ('gavit.last.train_state.pth', 'gavit.assignment_probe/probe_manifest.json',
                 'gavit.assignment_probe/assignment_diagnostics.jsonl',
                 'gavit.assignment_probe/epoch_000.pt', 'gavit.assignment_probe/epoch_001.pt',
                 'gavit.assignment_probe/best.pt', 'gavit.assignment_probe/final.pt'):
        if not (smoke / name).is_file():
            raise RuntimeError('Missing smoke artifact: ' + name)
    return dict(status='BIGEARTH_SOFT43_SMOKE_ARTIFACTS_PASS',
                checkpoint_sha256=report['checkpoint_sha256'], counts=report['counts'],
                hardware=report['hardware'], reload_max_logit_diff=report['reload_max_logit_diff'])
