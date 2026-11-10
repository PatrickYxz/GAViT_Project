import copy
import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from test_soft_gavit import make_model


def api():
    spec = importlib.util.find_spec('tools.diagnose_nwpu_feedback')
    assert spec is not None, 'NWPU feedback sensitivity diagnostic is not implemented'
    return importlib.import_module('tools.diagnose_nwpu_feedback')


def test_all_validation_images_are_selected_without_test_or_outcome_selection():
    names = [f'c{i:02d}' for i in range(45)]
    rows = [dict(path=f'{name}/{j:03d}.jpg', label=i, split='val')
            for i, name in enumerate(names) for j in range(28)]
    rows += [dict(path=f'{name}/test.jpg', label=i, split='test') for i, name in enumerate(names)]
    manifest = dict(class_names=names, records=list(reversed(rows)))
    selected = api().select_validation(manifest)
    assert len(selected) == 1260 and selected[0]['path'] == 'c00/000.jpg'
    assert all(row['split'] == 'val' for row in selected)
    with pytest.raises(ValueError):
        api().select_validation({**manifest, 'records': rows + [rows[0]]})
    with pytest.raises(ValueError):
        api().select_validation({**manifest, 'records': rows[1:]})


@pytest.mark.parametrize('grouping', ['attentive_spatial', 'soft_prototype'])
def test_cached_replay_matches_actual_scaled_forward_for_each_alpha(grouping):
    tool = api()
    torch.manual_seed(91)
    model = make_model(grouping).eval()
    inputs = torch.randn(2, 3, 16, 16)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    backbone_calls = []
    handle = model.backbone.register_forward_hook(lambda m,a,o: backbone_calls.append(1))
    capture, checks = tool.capture_batch(model, inputs)
    handle.remove()
    assert len(backbone_calls) == 1
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, before[k]) for k,v in model.state_dict().items())
    assert checks['reconstruction_max_logit_diff'] < 1e-5
    for i, alpha in enumerate(tool.ALPHAS):
        hook = model.feedback_proj.register_forward_hook(lambda m,a,o: o * alpha)
        try:
            with torch.inference_mode():
                actual = model(inputs)
        finally:
            hook.remove()
        torch.testing.assert_close(capture['scaled_logits'][i], actual, atol=1e-5, rtol=1e-5)
    with torch.inference_mode():
        expected_zero = model.classifier(model.backbone(inputs).mean(1))
    torch.testing.assert_close(capture['scaled_logits'][0], expected_zero, atol=1e-5, rtol=1e-5)
    assert all(t.device.type == 'cpu' and not t.requires_grad for t in capture.values())


def test_capture_failure_does_not_leave_hooks_or_change_training_state():
    tool = api()
    model = make_model().eval()
    # Fail after the diagnostic's first two capture hooks have run.
    def fail(*args):
        raise RuntimeError('deliberate failure')
    h = model.feedback_proj.register_forward_hook(fail)
    original = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    with pytest.raises(RuntimeError, match='deliberate'):
        tool.capture_batch(model, torch.randn(2, 3, 16, 16))
    assert original == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    h.remove()
    assert all(not m.training for m in model.modules())
    with pytest.raises(ValueError, match='eval'):
        tool.capture_batch(model.train(), torch.randn(1, 3, 16, 16))


def test_pairs_and_metrics_keep_gains_and_harms():
    tool = api()
    result = tool.paired_counts([0,1,2,0], [0,1,1,2], [1,1,2,2])
    assert result == dict(both_correct=1, corrects=1, breaks=1, both_wrong=1, net_correct=0)
    logits = torch.tensor([[2.,0.], [0.,2.], [2.,0.], [0.,2.]])
    report = tool.score_logits(logits, [0,1,1,1], ['a','b'])
    assert report['oa_percent'] == 75.
    assert report['macro_f1_percent'] == pytest.approx((200/3+80)/2)
    assert report['cross_entropy'] == pytest.approx(0.626928, abs=1e-6)
    logits[0,0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        tool.score_logits(logits, [0,1,1,1], ['a','b'])


def test_wrong_cached_prediction_blocks_completion():
    tool = api()
    logits = torch.tensor([[0.,1.], [1.,0.]])
    rows = [{'path':'a/a.jpg','label':0}, {'path':'b/b.jpg','label':1}]
    cached = {r['path']:dict(label=str(r['label']),prediction=str(int(z.argmax())),
                           **{f'p_{i}':str(p) for i,p in enumerate(z.softmax(0).tolist())})
              for r,z in zip(rows,logits)}
    from tools import visualize_nwpu_regions as base
    assert tool.check_cached_batch(base, logits, rows, cached) < 1e-6
    cached['a/a.jpg']['prediction']='0'
    with pytest.raises(ValueError, match='cached'):
        tool.check_cached_batch(base, logits, rows, cached)


@pytest.mark.parametrize('grouping', ['attentive_spatial', 'soft_prototype'])
def test_saved_cache_replays_without_images_backbone_or_graph(tmp_path, grouping):
    tool = api()
    model = make_model(grouping).eval()
    capture, _ = tool.capture_batch(model, torch.randn(3, 3, 16, 16))
    capture.update(labels=[0, 1, 2], class_names=[str(i) for i in range(19)],
                   paths=['a.jpg', 'b.jpg', 'c.jpg'], alphas=list(tool.ALPHAS),
                   identity={'scope': tool.SCOPE, 'sources': {'fixed': {'best_epoch': 29}}},
                   classifier_state=model.classifier.state_dict())
    path = tmp_path/'cache.pt'
    torch.save(capture, path)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    head = copy.deepcopy(model.classifier)
    del model
    # Rebuild the saved head state, with no original input or backbone access.
    with torch.no_grad():
        for p in head.parameters():
            p.zero_()
    head.load_state_dict(loaded['classifier_state'], strict=True)
    with torch.inference_mode():
        replay = torch.stack([head(loaded['mean_tokens'] + a*loaded['mean_feedback'])
                              for a in tool.ALPHAS])
    torch.testing.assert_close(replay, loaded['scaled_logits'], atol=1e-5, rtol=1e-5)
    rows = tool.analyze_cache(loaded)
    assert [r['alpha'] for r in rows] == list(tool.ALPHAS)
    assert all(r['samples'] == 3 for r in rows)
    assert rows[-1]['delta_oa_pp'] == rows[-1]['delta_macro_f1_pp'] == 0
    assert rows[-1]['paired_vs_original']['corrects'] == 0
    assert rows[-1]['paired_vs_original']['breaks'] == 0
    for row in rows:
        assert row['delta_oa_pp'] == pytest.approx(100*row['paired_vs_original']['net_correct']/3)
    json.dumps(rows, allow_nan=False)


def test_existing_output_preserved_and_failed_run_recorded(tmp_path, monkeypatch):
    tool = api()
    out=tmp_path/'old';out.mkdir();(out/'run.exit').write_text('old evidence')
    with pytest.raises(FileExistsError):
        tool.execute(SimpleNamespace(output=out))
    assert (out/'run.exit').read_text() == 'old evidence'
    def fail(*args):
        raise ValueError('identity mismatch')
    monkeypatch.setattr(tool, 'run', fail)
    new=tmp_path/'new'
    with pytest.raises(ValueError, match='identity mismatch'):
        tool.execute(SimpleNamespace(output=new))
    assert (new/'run.exit').read_text() == 'exit_code=1\n'
    assert json.loads((new/'summary.json').read_text())['status']=='failed'


def test_complete_pipeline_writes_replayable_outputs(tmp_path, monkeypatch):
    """Synthetic source fixture; genuine grouping, graph, capture and serialization."""
    tool = api()
    from scene_classification import modeling
    from tools import visualize_nwpu_regions as base
    names = [f'c{i:02d}' for i in range(45)]
    rows = [dict(path=f'{n}/{j:03d}.jpg', label=i, split='val')
            for i,n in enumerate(names) for j in range(28)]
    reports, cached = {}, {}
    x = torch.randn(3, 16, 16)
    def build(_name, _classes, *, architecture_config):
        m = make_model(architecture_config['grouping'])
        m.classifier[-1] = torch.nn.Linear(768, 45)
        return m
    for name, grouping in [('fixed', 'attentive_spatial'), ('soft', 'soft_prototype')]:
        folder = tmp_path/name
        folder.mkdir()
        arch = {'grouping': grouping}
        model = build('gavit', 45, architecture_config=arch).eval()
        torch.save(model.state_dict(), folder/'best.pth')
        with torch.inference_mode():
            z = model(x[None])[0]
        cached[name] = {r['path']: dict(label=str(r['label']), prediction=str(int(z.argmax())),
                       **{f'p_{i}':str(p) for i,p in enumerate(z.softmax(0).tolist())}) for r in rows}
        reports[name] = dict(architecture=arch, code={}, best_val_oa_percent=100/45)
    class Reader:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, row): return x, {'path':row['path'], 'source':'synthetic_fixture'}
    helper = SimpleNamespace(
        validate_sources=lambda a: (dict(class_names=names, records=rows), [], reports, cached, {}),
        ImageSource=Reader, MANIFEST_SHA='synthetic', check_cached=base.check_cached,
        sha=lambda p: hashlib.sha256(p.read_bytes() if hasattr(p, 'read_bytes') else open(p,'rb').read()).hexdigest())
    monkeypatch.setattr(tool, 'load_helper', lambda p:helper)
    monkeypatch.setattr(modeling, 'build_model', build)
    monkeypatch.setattr(modeling, 'image_transform', lambda training:lambda image:image)
    args = SimpleNamespace(helper=tmp_path/'helper', repo=tmp_path, calibration=tmp_path,
                           device='cpu', data_root=tmp_path, archive=tmp_path/'unused.zip',
                           output=tmp_path/'output')
    old_threads = torch.get_num_threads()
    old_path = list(__import__('sys').path)
    try:
        report = tool.execute(args)
    finally:
        torch.set_num_threads(old_threads)
        __import__('sys').path[:] = old_path
    assert report['status'] == 'complete' and report['samples'] == 1260
    assert (args.output/'run.exit').read_text() == 'exit_code=0\n'
    assert report['input_sources'] == {'synthetic_fixture':1260}
    assert len(json.loads((args.output/'inputs.json').read_text())) == 1260
    for name in ('fixed','soft'):
        assert len(report['curves'][name]) == 5
        assert len((args.output/(name+'_predictions.csv')).read_text().splitlines()) == 1261
        loaded = torch.load(args.output/(name+'_cache.pt'), map_location='cpu', weights_only=True)
        assert loaded['scaled_logits'].shape == (5,1260,45)
        assert loaded['paths'] == [r['path'] for r in rows]
        assert report['curves'][name][-1]['oa_percent'] == pytest.approx(100/45)
    assert 'Alpha zero is not a trained' in (args.output/'report.md').read_text()
