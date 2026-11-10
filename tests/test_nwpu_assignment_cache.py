"""Hand-derived geometry checks and real PT-cache CLI integration."""
import hashlib
import json
from pathlib import Path
import subprocess
import shlex
import sys

import pytest
import torch

from tools import analyze_nwpu_assignment_cache as audit


def fixture_tensors(soft=False):
    # Hand-authored 7x7 -> 4x4 partition, including unequal border cells.
    anchors = torch.tensor([0, 0, 1, 1, 2, 2, 3,
                            0, 0, 1, 1, 2, 2, 3,
                            4, 4, 5, 5, 6, 6, 7,
                            4, 4, 5, 5, 6, 6, 7,
                            8, 8, 9, 9, 10, 10, 11,
                            8, 8, 9, 9, 10, 10, 11,
                            12, 12, 13, 13, 14, 14, 15])
    mask = torch.tensor([[max(abs(int(a)//4-k//4), abs(int(a)%4-k%4)) <= (1 if soft else 0)
                          for k in range(16)] for a in anchors])
    s = mask.double() / mask.sum(1, keepdim=True)
    w = s / s.sum(0, keepdim=True)
    spatial = [(a, b) for a in range(16) for b in range(16)
               if abs(a//4-b//4) + abs(a%4-b%4) == 1]
    feature = [(a, b) for b in range(16) for a in
               [a for a in range(16) if abs(a//4-b//4) + abs(a%4-b%4) > 1][:2]]
    return dict(S=s, W=w, candidate_mask=mask, anchor_assignment=anchors,
                edge_index=torch.tensor(spatial + feature).T, logits=torch.zeros(45))


def test_disjoint_pooling_has_no_overlap_or_geometric_drift():
    t = fixture_tensors()
    m = audit.measure(t)
    assert m['scalars']['W_neighbor_overlap_mean'] == 0
    assert m['scalars']['S_centroid_drift_mean'] == 0
    assert m['scalars']['W_centroid_drift_mean'] == 0
    assert m['scalars']['W_effective_tokens_mean'] == 49 / 16
    assert m['regions']['W_effective_tokens'] == [4, 4, 4, 2, 4, 4, 4, 2, 4, 4, 4, 2, 2, 2, 2, 1]


def test_uniform_local_geometry_is_not_mislabelled_as_learned_collapse():
    t = fixture_tensors(True)
    m = audit.measure(t)['scalars']
    assert m['S_entropy_normalized_mean'] == pytest.approx(1)
    assert m['W_neighbor_overlap_mean'] == pytest.approx(0.6406926406926406)
    assert m['W_effective_tokens_mean'] == pytest.approx(18.777516340779265)
    assert m['W_centroid_drift_mean'] == pytest.approx(0.12012766872992803)
    assert m['W_center_rms_mean'] == pytest.approx(0.26416287889089013)


def test_attention_can_move_fixed_W_while_fixed_S_does_not_move():
    t = fixture_tensors()
    t['W'][[0, 1, 7, 8], 0] = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    m = audit.measure(t)
    assert m['scalars']['S_centroid_drift_mean'] == 0
    assert m['regions']['W_centroid_drift'][0] == pytest.approx(2**0.5 / 14)
    assert m['regions']['W_effective_tokens'][0] == 1


def test_identical_weights_have_overlap_one_and_effective_support_49():
    t = fixture_tensors(True)
    t['S'] = torch.full((49, 16), 1/16, dtype=torch.float64)
    t['W'] = torch.full((49, 16), 1/49, dtype=torch.float64)
    t['candidate_mask'] = torch.ones(49, 16, dtype=torch.bool)
    m = audit.measure(t)['scalars']
    assert m['W_neighbor_overlap_mean'] == pytest.approx(1)
    assert m['W_effective_tokens_mean'] == pytest.approx(49)


@pytest.mark.parametrize('corruption', ['row', 'column', 'mask', 'nan', 'anchor', 'graph', 'zero_mass'])
def test_invalid_tensor_cache_is_rejected(corruption):
    t = fixture_tensors(True)
    if corruption == 'row': t['S'][0] *= 2
    if corruption == 'column': t['W'][:, 0] *= 2
    if corruption == 'mask': t['candidate_mask'][0, 15] = True
    if corruption == 'nan': t['S'][0, 0] = float('nan')
    if corruption == 'anchor': t['anchor_assignment'][0] = 1
    if corruption == 'graph': t['edge_index'][:, 79] = t['edge_index'][:, 78]
    if corruption == 'zero_mass':
        t['S'][:, 0] = 0
        t['S'] /= t['S'].sum(1, keepdim=True)
    with pytest.raises(ValueError): audit.validate_tensors(t, 'soft')


def create_cache(root):
    root.mkdir()
    (root/'cases').mkdir()
    sources = {n: dict(checkpoint_sha256=audit.CHECKPOINTS[n], run_sha256=audit.RUN_HASHES[n],
                       cached_predictions_sha256='a'*64, best_epoch=29 if n == 'fixed' else 28)
               for n in ('fixed', 'soft')}
    records = [dict(path=f'class_{i//2:02d}/image_{i%2}.jpg', label=i//2, split='val') for i in range(90)]
    checks = dict(capture_max_logit_diff=0., reconstruction_max_logit_diff=0.,
                  restoration_max_logit_diff=0., cached_probability_max_error=0.,
                  S_row_error=0., W_column_error=0., spatial_edges=48, feature_edges=32,
                  gat_self_loops_not_drawn=True)
    def save_json(name, value): (root/name).write_text(json.dumps(value))
    save_json('identity.json', dict(manifest_sha256=audit.MANIFEST_SHA,
                                   checkpoint_sources=sources, selection=audit.SELECTION,
                                   sample_ids=[r['path'] for r in records]))
    save_json('selection.json', dict(records=records, checkpoint_sources=sources, selection=audit.SELECTION))
    for i, row in enumerate(records):
        preds = {n: row['label'] if i < (82 if n == 'fixed' else 85) else (row['label']+1)%45
                 for n in sources}
        for n in sources:
            tensors = fixture_tensors(n == 'soft')
            tensors['logits'][preds[n]] = 1
            torch.save(dict(path=row['path'], label=row['label'], source=sources[n], tensors=tensors,
                            checks=checks, cached_check_status='passed'), root/'cases'/f'{i:03d}_{n}.pt')
        save_json(f'cases/{i:03d}.json', dict(path=row['path'], label=row['label'], predictions=preds,
                                            checks={n: checks for n in sources}))
    save_json('summary.json', dict(status='complete', samples=90, classes=45, split='val',
                                  source_phase='calibrate', selection=audit.SELECTION,
                                  all_capture_checks_passed=True, spatial_edges=48, feature_edges=32,
                                  correct=dict(fixed=82, soft=85),
                                  paired=dict(both_correct=82, soft_corrects=3, soft_breaks=0, both_wrong=5),
                                  max_errors={k: checks[k] for k in audit.ERROR_KEYS}))
    (root/'run.exit').write_text('exit_code=0\n')
    return root


def tree_hash(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_cli_reads_real_pt_files_and_preserves_sources_and_existing_output(tmp_path):
    source = create_cache(tmp_path/'cache')
    out = tmp_path/'analysis'
    before = tree_hash(source)
    command = [sys.executable, audit.__file__, '--source', str(source), '--output', str(out)]
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert len(run.stdout.splitlines()) <= 16
    assert 'ASSIGNMENT CACHE COMPLETE' in run.stdout
    assert tree_hash(source) == before
    report = json.loads((out/'summary.json').read_text())
    assert report['paired']['soft_corrects'] == 3
    assert report['bucket_statistics']['soft_breaks']['samples'] == 0
    assert report['statistics']['fixed']['W_neighbor_overlap_mean']['mean'] == 0
    assert report['statistics']['soft']['W_neighbor_overlap_mean']['mean'] == pytest.approx(0.6406926407)
    assert report['statistics']['soft_minus_uniform_local']['W_neighbor_overlap_mean']['mean'] == pytest.approx(0)
    assert len(report['source_files_sha256']) == 274
    saved = tree_hash(out)
    rerun = subprocess.run(command, capture_output=True, text=True)
    assert rerun.returncode != 0
    assert tree_hash(out) == saved
    assert tree_hash(source) == before


@pytest.mark.parametrize('change', ['label', 'source', 'prediction', 'pending', 'missing'])
def test_partial_or_mismatched_evidence_fails_with_explicit_exit(tmp_path, change):
    source = create_cache(tmp_path/'cache')
    pt = source/'cases'/'000_soft.pt'
    p = torch.load(pt, weights_only=True)
    if change == 'label': p['label'] = 1
    if change == 'source': p['source']['checkpoint_sha256'] = 'f'*64
    if change == 'prediction': p['tensors']['logits'][1] = 2
    if change == 'pending': p['cached_check_status'] = 'pending'
    torch.save(p, pt)
    if change == 'missing': pt.unlink()
    out = tmp_path/'analysis'
    with pytest.raises((ValueError, FileNotFoundError)): audit.execute(source, out)
    assert (out/'run.exit').read_text().strip() == 'exit_code=1'
    assert json.loads((out/'summary.json').read_text())['status'] == 'failed'


def test_output_cannot_be_created_inside_source(tmp_path):
    source = create_cache(tmp_path/'cache')
    before = tree_hash(source)
    with pytest.raises(ValueError): audit.execute(source, source/'analysis')
    assert tree_hash(source) == before
    assert not (source/'analysis').exists()


def test_delivered_one_line_payload_runs_without_repo_or_script_file(tmp_path):
    source = create_cache(tmp_path/'cache')
    out = tmp_path/'analysis'
    line = (Path(__file__).parents[1]/'docs/launch_nwpu_assignment_cache_20261007.txt').read_text()
    assert len(line.splitlines()) == 1
    for shell in ('bash', 'zsh'):
        result = subprocess.run([shell, '-n'], input=line, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
    args = shlex.split(line)
    code = args[2].replace('/home/featurize/work/nwpu_regions42_20260929_01', str(source)).replace(
        '/home/featurize/work/nwpu_assignment_cache42_20261007_01', str(out))
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.splitlines()) <= 16
    report = json.loads((out/'summary.json').read_text())
    assert report['tool_sha256'] == hashlib.sha256(Path(audit.__file__).read_bytes()).hexdigest()
    assert report['statistics']['soft']['W_neighbor_overlap_mean']['mean'] == pytest.approx(0.6406926407)


def test_missing_source_is_an_explicit_failure_without_fake_complete(tmp_path):
    out = tmp_path/'analysis'
    with pytest.raises(FileNotFoundError): audit.execute(tmp_path/'missing', out)
    assert (out/'run.exit').read_text().strip() == 'exit_code=1'
    assert json.loads((out/'summary.json').read_text())['status'] == 'failed'
