import copy
from collections import Counter
import hashlib
import importlib
import io
import json
import random
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from test_soft_gavit import make_model


def api():
    return importlib.import_module('tools.visualize_nwpu_regions')


def record(path, image_bytes, split='val', label=0):
    im = Image.open(io.BytesIO(image_bytes))
    return dict(path=path, label=label, split=split,
                sha256=hashlib.sha256(image_bytes).hexdigest(),
                pixel_sha256=hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest())


def jpg():
    f = io.BytesIO()
    Image.new('RGB', (256, 256), (120, 80, 50)).save(f, format='JPEG')
    return f.getvalue()


def test_selects_two_per_class_from_val_independent_of_order_and_outcomes():
    rows = [dict(path=f'{c}/{i}.jpg', label=k, split=s)
            for k, c in enumerate(['a', 'b']) for s, i in [('test', 0), ('train', 1), ('val', 5), ('val', 3), ('val', 4)]]
    random.Random(8).shuffle(rows)
    manifest = {'class_names': ['a', 'b'], 'records': rows}
    chosen = api().select_cases(manifest)
    assert [r['path'] for r in chosen] == ['a/3.jpg', 'a/4.jpg', 'b/3.jpg', 'b/4.jpg']
    assert api().select_cases({**manifest, 'records': list(reversed(rows))}) == chosen
    with pytest.raises(ValueError, match='two'):
        api().select_cases({**manifest, 'records': [r for r in rows if r['path'] != 'a/3.jpg' and r['path'] != 'a/4.jpg']})
    with pytest.raises(ValueError, match='duplicate'):
        api().select_cases({**manifest, 'records': rows + [rows[0]]})


def test_zip_reads_only_original_member_without_extraction(tmp_path):
    raw = jpg()
    row = record('airport/a.jpg', raw)
    archive = tmp_path / 'data.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('NWPU-RESISC45/airport/a.jpg', raw)
    with api().ImageSource(tmp_path / 'absent', archive) as source:
        im, identity = source.read(row)
    assert im.size == (256, 256) and identity['source'] == 'zip'
    assert not (tmp_path / 'absent').exists()
    with api().ImageSource(tmp_path / 'absent', archive) as source:
        with pytest.raises(ValueError, match='hash'):
            source.read({**row, 'sha256': '0' * 64})
        with pytest.raises(ValueError, match='path'):
            source.read({**row, 'path': '../airport/a.jpg'})


def test_corrupt_existing_original_is_not_silently_replaced_by_zip(tmp_path):
    raw = jpg()
    root = tmp_path / 'data'
    (root / 'airport').mkdir(parents=True)
    (root / 'airport/a.jpg').write_bytes(b'corrupt')
    with zipfile.ZipFile(tmp_path / 'data.zip', 'w') as z:
        z.writestr('NWPU-RESISC45/airport/a.jpg', raw)
    with api().ImageSource(root, tmp_path / 'data.zip') as source:
        with pytest.raises(ValueError, match='hash'):
            source.read(record('airport/a.jpg', raw))


@pytest.mark.parametrize('grouping', ['attentive_spatial', 'soft_prototype'])
def test_capture_observes_real_edges_and_reconstructs_without_side_effects(grouping):
    model = make_model(grouping).eval()
    image = torch.randn(1, 3, 16, 16)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    graph_calls = []
    handle = model.graph_reasoning.register_forward_pre_hook(lambda m, args: graph_calls.append(args[1].clone()))
    tensors, report = api().capture(model, image)
    handle.remove()
    assert report['capture_max_logit_diff'] < 1e-5 and report['reconstruction_max_logit_diff'] < 1e-5
    assert torch.equal(tensors['edge_index'], graph_calls[1].cpu())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(not t.requires_grad and t.device.type == 'cpu' for t in tensors.values())
    assert tuple(tensors['S'].shape) == (49, 16)
    torch.testing.assert_close(tensors['S'].sum(-1), torch.ones(49))
    torch.testing.assert_close(tensors['W'].sum(0), torch.ones(16))
    assert report['spatial_edges'] == 48 and report['feature_edges'] == 32
    assert report['gat_self_loops_not_drawn'] is True
    if grouping == 'attentive_spatial':
        assert torch.equal(tensors['S'].argmax(-1), model.region_grouping.assignment)


def test_capture_failure_restores_hooks_and_observer():
    model = make_model().eval()
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    calls = 0
    original = model.forward
    def fail_second(x):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('injected capture failure')
        return original(x)
    with patch.object(model, 'forward', fail_second), pytest.raises(RuntimeError, match='injected'):
        api().capture(model, torch.randn(1, 3, 16, 16))
    assert model.region_grouping.assignment_observer is None
    assert before == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


def test_visualization_does_not_rebuild_or_relabel_edges(tmp_path):
    model = make_model().eval()
    tensors, report = api().capture(model, torch.randn(1, 3, 16, 16))
    bad = copy.deepcopy(tensors)
    bad['edge_index'][:, 48] = bad['edge_index'][:, 0]
    with pytest.raises(ValueError, match='edge'):
        api().validate_tensors(bad)
    bad = copy.deepcopy(tensors)
    bad['S'][0] *= .5
    with pytest.raises((ValueError, AssertionError)):
        api().validate_tensors(bad)
    svg = api().case_svg(Image.open(io.BytesIO(jpg())), tensors, tensors,
                         'airport/a.jpg', 'airport', 'airport', 'church')
    import xml.etree.ElementTree as ET
    root = ET.fromstring(svg)
    assert root.tag.endswith('svg')
    assert svg.count('class="edge spatial"') == 96
    assert svg.count('class="edge feature"') == 64
    assert 'argmax' in svg and 'self-loops' in svg and 'not GAT attention' in svg
    (tmp_path / 'figure.svg').write_text(svg)


def test_cached_probability_mismatch_is_a_gate():
    logits = torch.tensor([0., 1., 2.])
    probabilities = logits.softmax(0).tolist()
    row = {'label': '0', 'prediction': '2', **{f'p_{i}': str(p) for i, p in enumerate(probabilities)}}
    assert api().check_cached(logits, row, 0) < 1e-6
    with pytest.raises(ValueError, match='cached'):
        api().check_cached(logits + torch.tensor([1., 0., 0.]), row, 0)
    with pytest.raises(ValueError, match='label'):
        api().check_cached(logits, row, 1)


def test_existing_output_is_never_overwritten(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    (out / 'run.exit').write_text('old evidence')
    with pytest.raises(FileExistsError):
        api().execute(type('Args', (), {'output': out})())
    assert (out / 'run.exit').read_text() == 'old evidence'


def frozen_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    tool = api()
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'model.py').write_text('original model source')
    names = [f'class_{i:02d}' for i in range(45)]
    raw = jpg()
    rows = [record(f'{name}/{j:03d}.jpg', raw,
                   'train' if j < 112 else 'val' if j < 140 else 'test', label)
            for label, name in enumerate(names) for j in range(700)]
    manifest = dict(class_names=names, records=rows, dataset='NWPU-RESISC45')
    mp = tmp_path / 'manifest.json'
    mp.write_text(json.dumps(manifest))
    monkeypatch.setattr(tool, 'MANIFEST_SHA', tool.sha(mp))
    monkeypatch.setattr(tool.subprocess, 'check_output', lambda *a, **k: tool.COMMIT)
    calibration = tmp_path / 'calibration'
    hashes, ckpts = {}, {}
    for name, grouping in [('fixed', 'attentive_spatial'), ('soft', 'soft_prototype')]:
        out = calibration / name
        out.mkdir(parents=True)
        (out / 'best.pth').write_bytes(name.encode())
        ckpts[name] = tool.sha(out / 'best.pth')
        report = dict(status='complete', phase='calibrate', seed=42, model='gavit', dataset='NWPU-RESISC45',
                      architecture=dict(grouping=grouping), manifest_sha256=tool.MANIFEST_SHA,
                      checkpoint_sha256=ckpts[name], code=dict(files_sha256={'model.py': tool.sha(repo / 'model.py')}),
                      training={'lr': 0.0003}, pretrained_sha256='same', best_epoch=29)
        (out / 'run.json').write_text(json.dumps(report))
        hashes[name] = tool.sha(out / 'run.json')
        (out / 'run.exit').write_text('exit_code=0\n')
        (out / 'validation_predictions.csv').write_text('path,label,prediction\n' +
            ''.join(f'{r["path"]},{r["label"]},0\n' for r in rows if r['split'] == 'val'))
    monkeypatch.setattr(tool, 'RUN_HASHES', hashes)
    monkeypatch.setattr(tool, 'CHECKPOINTS', ckpts)
    return SimpleNamespace(repo=repo, manifest=mp, calibration=calibration)


def test_frozen_source_chain_and_changed_code_gate(tmp_path, monkeypatch):
    args = frozen_sources(tmp_path, monkeypatch)
    _, selected, _, _, _ = api().validate_sources(args)
    assert len(selected) == 90 and Counter(r['label'] for r in selected) == {i: 2 for i in range(45)}
    (args.repo / 'model.py').write_text('changed source')
    with pytest.raises(ValueError, match='source changed'):
        api().validate_sources(args)


def test_refit_and_wrong_calibration_weights_are_rejected(tmp_path, monkeypatch):
    args = frozen_sources(tmp_path, monkeypatch)
    report_path = args.calibration / 'fixed/run.json'
    report = json.loads(report_path.read_text())
    report['phase'] = 'refit'
    report_path.write_text(json.dumps(report))
    api().RUN_HASHES['fixed'] = api().sha(report_path)
    with pytest.raises(ValueError, match='calibration identity'):
        api().validate_sources(args)
    report['phase'] = 'calibrate'
    report_path.write_text(json.dumps(report))
    api().RUN_HASHES['fixed'] = api().sha(report_path)
    (args.calibration / 'soft/best.pth').write_bytes(b'wrong weight')
    with pytest.raises(ValueError, match='source hash'):
        api().validate_sources(args)


def test_display_pixels_match_model_resize():
    from scene_classification.modeling import image_transform
    import numpy as np
    raw = np.random.default_rng(4).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(raw)
    display = torch.tensor(np.array(api().resized_input(image))).permute(2, 0, 1).float() / 255
    actual = image_transform(False)(image)
    reconstructed = actual * torch.tensor([.229, .224, .225])[:, None, None] + torch.tensor([.485, .456, .406])[:, None, None]
    torch.testing.assert_close(display, reconstructed)


def test_end_to_end_gallery_all_90_cases_with_real_grouping_and_pyg(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scene_classification import modeling
    tool = api()
    names = [f'class_{i:02d}' for i in range(45)]
    raw = jpg()
    selected = [record(f'{name}/{i}.jpg', raw, label=label) for label, name in enumerate(names) for i in range(2)]
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for row in selected:
            z.writestr('NWPU-RESISC45/' + row['path'], raw)
    models, cached, reports = {}, {}, {}
    calibration = tmp_path / 'calibration'
    for name, grouping in [('fixed', 'attentive_spatial'), ('soft', 'soft_prototype')]:
        m = make_model(grouping).eval()
        m.classifier[-1] = torch.nn.Linear(768, 45)
        m.eval()
        models[grouping] = m
        (calibration / name).mkdir(parents=True)
        torch.save(m.state_dict(), calibration / name / 'best.pth')
        with torch.no_grad():
            probs = m(modeling.image_transform(False)(Image.open(io.BytesIO(raw))).unsqueeze(0))[0].softmax(0).tolist()
        cached[name] = {r['path']: dict(label=r['label'], prediction=probs.index(max(probs)),
                                        **{f'p_{i}': p for i, p in enumerate(probs)}) for r in selected}
        reports[name] = dict(architecture={'grouping': grouping}, code={})
    monkeypatch.setattr(modeling, 'build_model', lambda *a, architecture_config, **k: copy.deepcopy(models[architecture_config['grouping']]))
    monkeypatch.setattr(tool, 'validate_sources', lambda args: (dict(class_names=names), selected, reports, cached, dict(fixed={}, soft={})))
    args = SimpleNamespace(output=tmp_path / 'output', repo=tmp_path, calibration=calibration,
                           data_root=tmp_path / 'absent', archive=archive, device='cpu')
    summary = tool.execute(args)
    assert summary['status'] == 'complete' and summary['samples'] == 90
    assert (args.output / 'run.exit').read_text() == 'exit_code=0\n'
    assert (args.output / 'cases.html').read_text().count('<article>') == 90
    assert len(list((args.output / 'cases').glob('*.pt'))) == 180
    assert summary['input_sources'] == {'zip': 90}
    assert sum(summary['paired'].values()) == 90
    assert summary['max_errors']['cached_probability_max_error'] < 1e-5
    with zipfile.ZipFile(args.output / 'evidence.zip') as z:
        assert z.testzip() is None and len(z.namelist()) == 542
    # Keep a local synthetic fixture for visual QA; never label it as real NWPU.
    preview = Path('/private/tmp/nwpu_region_synthetic_preview.html')
    preview.write_text(tool.gallery([dict(path='SYNTHETIC LOCAL QA',
        svg=(args.output / 'cases/000.svg').read_text(), membership_svg=(args.output / 'cases/000_memberships.svg').read_text())]))


def test_failure_records_nonzero_and_retains_partial_evidence(tmp_path, monkeypatch):
    tool = api()
    def fail(args, out):
        (out / 'partial.pt').write_bytes(b'preserve')
        raise RuntimeError('injected failure')
    monkeypatch.setattr(tool, 'run', fail)
    args = type('Args', (), {'output': tmp_path / 'failed'})()
    with pytest.raises(RuntimeError, match='injected'):
        tool.execute(args)
    assert (args.output / 'run.exit').read_text() == 'exit_code=1\n'
    assert json.loads((args.output / 'summary.json').read_text())['status'] == 'failed'
    assert (args.output / 'partial.pt').read_bytes() == b'preserve'


def test_real_run_retains_original_pixels_and_capture_on_cache_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scene_classification import modeling
    tool = api()
    raw = jpg()
    row = record('airport/1.jpg', raw)
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('NWPU-RESISC45/' + row['path'], raw)
    model = make_model('attentive_spatial').eval()
    calibration = tmp_path / 'calibration'
    (calibration / 'fixed').mkdir(parents=True)
    torch.save(model.state_dict(), calibration / 'fixed/best.pth')
    reports = {'fixed': dict(architecture={}, code={})}
    cache = {'fixed': {row['path']: {'label': 0, 'prediction': 0, **{f'p_{i}': 0 for i in range(19)}}}}
    monkeypatch.setattr(tool, 'validate_sources', lambda args: (dict(class_names=['airport']), [row], reports, cache, {'fixed': {}}))
    monkeypatch.setattr(modeling, 'build_model', lambda *a, **k: copy.deepcopy(model))
    args = SimpleNamespace(repo=tmp_path, output=tmp_path / 'output', calibration=calibration,
                           data_root=tmp_path / 'missing', archive=archive, device='cpu')
    with pytest.raises(ValueError, match='cached'):
        tool.execute(args)
    assert (args.output / 'run.exit').read_text() == 'exit_code=1\n'
    assert (args.output / 'cases/000_original.png').is_file()
    assert json.loads((args.output / 'selection.json').read_text())['records'][0]['path'] == row['path']
    payload = torch.load(args.output / 'cases/000_fixed.pt', weights_only=True)
    assert payload['cached_check_status'] == 'failed'
    assert 'cached' in payload['error']
    assert payload['tensors']['S'].shape == (49, 16)
