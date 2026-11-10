"""Detached stdlib fixtures: corrupt artifacts must prevent the paid-launch gate."""
import copy
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

E = Path(__file__).resolve().parent
R = E.parents[2]
spec = importlib.util.spec_from_file_location('artifact_gate', E / 'check_smoke.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
expected = json.loads((E / 'user_return.json').read_text())['summary']
preflight = json.loads((E.parent / 'seed43_bigearth_preflight_20261011/user_return.json').read_text())
source_hashes = json.loads((E / 'command_validation.json').read_text())['source_hashes']
source_paths = {
    'helper_source.py': R / 'tools/run_bigearth_soft43.py',
    'prepare_source.py': E.parent / 'seed43_bigearth_preflight_20261011/prepare_smoke_source.py',
    'train_reload_source.py': E.parent / 'seed43_bigearth_preflight_20261011/train_reload_smoke_source.py',
}
cases = []
for case in ('valid', 'bad_exit', 'bad_summary', 'bad_checkpoint', 'bad_metadata',
             'bad_preflight', 'bad_subset', 'changed_wrapper', 'missing_probe'):
    with tempfile.TemporaryDirectory(prefix='gavit-soft43-artifact-') as tmp:
        root = Path(tmp)
        result = copy.deepcopy(expected)
        # Synthetic weight bytes make these fixtures independent of real GPU checkpoint files.
        weight = b'detached fixture checkpoint'
        sha = hashlib.sha256(weight).hexdigest()
        result['checkpoint_sha256'] = result['metadata']['checkpoint_sha256'] = sha
        (root / 'gavit.pth').write_bytes(weight)
        (root / 'run.exit').write_text('exit_code=0\n')
        (root / 'summary.json').write_text(json.dumps(result))
        (root / 'gavit.meta.json').write_text(json.dumps(result['metadata']))
        (root / 'approved_preflight.json').write_text(json.dumps(preflight))
        (root / 'splits').mkdir()
        hashes = {}
        for split in ('train', 'val'):
            data = (split + ' synthetic split bytes').encode()
            (root / 'splits' / (split + '.csv')).write_bytes(data)
            hashes[split] = hashlib.sha256(data).hexdigest()
        (root / 'preflight.json').write_text(json.dumps(dict(status='ready',counts=result['counts'],
            commit=result['commit'],pretrained_sha256='f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36',subset_sha256=hashes)))
        for dst, source in source_paths.items():
            (root / dst).write_bytes(source.read_bytes())
        for name in ('gavit.last.train_state.pth', 'gavit.assignment_probe/probe_manifest.json',
                     'gavit.assignment_probe/assignment_diagnostics.jsonl',
                     'gavit.assignment_probe/epoch_000.pt', 'gavit.assignment_probe/epoch_001.pt',
                     'gavit.assignment_probe/best.pt', 'gavit.assignment_probe/final.pt'):
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b'detached artifact fixture')
        if case == 'bad_exit':
            (root / 'run.exit').write_text('exit_code=1\n')
        if case == 'bad_summary':
            broken = copy.deepcopy(result)
            broken['seed'] = 44
            (root / 'summary.json').write_text(json.dumps(broken))
        if case == 'bad_checkpoint':
            (root / 'gavit.pth').write_bytes(b'changed weights')
        if case == 'bad_metadata':
            (root / 'gavit.meta.json').write_text('{}')
        if case == 'bad_preflight':
            (root / 'approved_preflight.json').write_text('{}')
        if case == 'bad_subset':
            (root / 'splits/train.csv').write_bytes(b'changed split')
        if case == 'changed_wrapper':
            (root / 'helper_source.py').write_bytes(b'changed source')
        if case == 'missing_probe':
            (root / 'gavit.assignment_probe/best.pt').unlink()
        before = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        try:
            checked = gate.check_smoke(root, result, preflight, source_hashes)
            assert case == 'valid'
            assert checked['status'] == 'BIGEARTH_SOFT43_SMOKE_ARTIFACTS_PASS'
        except (RuntimeError, FileNotFoundError):
            assert case != 'valid'
        after = {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
        assert before == after, 'Read-only gate modified fixture artifacts'
        cases.append(case)
report = dict(status='artifact_gate_fixtures_pass',cases=cases,count=len(cases),
              scope='stdlib artifact corruption/reuse protection only; no model, CUDA, or real training launch')
(E / 'artifact_gate_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
