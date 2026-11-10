"""Pooling-only intervention through genuine grouping and graph modules."""
import copy
import importlib
import json
import ast
import base64
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import zlib

import pytest
import torch
from test_soft_gavit import make_model


def api():
    return importlib.import_module('tools.diagnose_nwpu_pooling')


def test_pool_intervention_matches_direct_pool_override_and_preserves_S_state_rng():
    torch.manual_seed(37)
    model = make_model().eval()
    x = torch.randn(2, 3, 16, 16)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result, checks = api().capture_pair(model, x)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(result['normal']['S'], result['membership_only']['S'])
    assert checks['restoration_max_logit_diff'] <= 1e-5
    assert not torch.equal(result['normal']['W'], result['membership_only']['W'])
    original = model.region_grouping._pool
    def independent_pool(tokens, scores, log_s):
        w = log_s.softmax(dim=1)
        return torch.bmm(w.transpose(1, 2), tokens), w
    try:
        model.region_grouping._pool = independent_pool
        with torch.no_grad(): expected = model(x)
    finally:
        model.region_grouping._pool = original
    torch.testing.assert_close(result['membership_only']['logits'], expected, atol=1e-5, rtol=1e-5)
    for mode in result:
        s = result[mode]
        assert s['edges'].shape == (2, 160)
        assert all(t.device.type == 'cpu' and not t.requires_grad for t in s.values())
        with torch.no_grad(): replay = model.classifier(s['pooled'])
        torch.testing.assert_close(replay, s['logits'], atol=1e-5, rtol=1e-5)


def test_hooks_and_observer_restore_after_intervened_graph_failure():
    model = make_model().eval()
    seen = []
    def observer(values): seen.append(values['S'].shape)
    model.region_grouping.assignment_observer = observer
    calls = []
    def fail_second(*args):
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('intervention graph failed')
    hook = model.graph_reasoning.register_forward_pre_hook(fail_second)
    counts = [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    try:
        with pytest.raises(RuntimeError, match='intervention graph failed'):
            api().capture_pair(model, torch.randn(1, 3, 16, 16))
        assert model.region_grouping.assignment_observer is observer
        assert counts == [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]
    finally:
        hook.remove()
    with torch.no_grad(): model(torch.randn(1, 3, 16, 16))
    assert seen


def test_no_intervention_when_attention_scores_constant():
    model = make_model().eval()
    for param in model.region_grouping.attn.parameters(): param.data.zero_()
    result, _ = api().capture_pair(model, torch.randn(2, 3, 16, 16))
    torch.testing.assert_close(result['normal']['W'], result['membership_only']['W'], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(result['normal']['logits'], result['membership_only']['logits'], atol=1e-5, rtol=1e-5)


def test_training_mode_and_nonsoft_grouping_rejected():
    with pytest.raises(ValueError): api().capture_pair(make_model().train(), torch.randn(1, 3, 16, 16))
    with pytest.raises(ValueError): api().capture_pair(make_model('attentive_spatial').eval(), torch.randn(1, 3, 16, 16))


def test_preserve_old_output_and_record_failed_new_output(tmp_path, monkeypatch):
    tool = api()
    old = tmp_path/'old'
    old.mkdir()
    (old/'run.exit').write_text('old evidence')
    with pytest.raises(FileExistsError): tool.execute(SimpleNamespace(output=old))
    assert (old/'run.exit').read_text() == 'old evidence'
    def fail(*args): raise ValueError('source mismatch')
    monkeypatch.setattr(tool, 'run', fail)
    new = tmp_path/'new'
    with pytest.raises(ValueError, match='source mismatch'): tool.execute(SimpleNamespace(output=new))
    assert (new/'run.exit').read_text() == 'exit_code=1\n'
    assert json.loads((new/'summary.json').read_text())['status'] == 'failed'


def test_frozen_helper_hash_rejected_before_import(tmp_path):
    path = tmp_path/'bad.py'
    path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='hash'): api().load_legacy(path)


def test_full_pipeline_synthetic_data_real_model_graph_serialization(tmp_path, monkeypatch):
    tool = api()
    from tools import diagnose_nwpu_feedback as legacy
    from tools import visualize_nwpu_regions as base
    from scene_classification import modeling
    torch.manual_seed(51)
    model = make_model().eval()
    model.classifier[-1] = torch.nn.Linear(768, 45)
    x = torch.randn(3, 16, 16)
    names = [f'c{i:02d}' for i in range(45)]
    rows = [dict(path=f'{n}/{j:03d}.jpg', label=i, split='val') for i,n in enumerate(names) for j in range(28)]
    with torch.no_grad(): z = model(x[None])[0]
    cached = {r['path']:dict(label=str(r['label']), prediction=str(int(z.argmax())),
                           **{f'p_{i}':str(p) for i,p in enumerate(z.softmax(0).tolist())}) for r in rows}
    ckpt = tmp_path/'calibration'/'soft'
    ckpt.mkdir(parents=True)
    torch.save(model.state_dict(), ckpt/'best.pth')
    report = dict(architecture={'grouping':'soft_prototype'}, code={}, best_val_oa_percent=100/45)
    class Reader:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,row): return x, dict(path=row['path'], source='synthetic_fixture')
    helper = SimpleNamespace(validate_sources=lambda args:(dict(class_names=names,records=rows),
                             [r for r in rows if r['path'].endswith(('000.jpg','001.jpg'))],
                             {'soft':report}, {'soft':cached}, {'soft':{}}), ImageSource=Reader,
                             check_cached=base.check_cached, sha=base.sha, MANIFEST_SHA='synthetic')
    monkeypatch.setattr(tool,'load_legacy',lambda path:legacy)
    monkeypatch.setattr(legacy,'load_helper',lambda path:helper)
    monkeypatch.setattr(modeling,'build_model',lambda *args,**kwargs:copy.deepcopy(model))
    monkeypatch.setattr(modeling,'image_transform',lambda training:lambda image:image)
    args = SimpleNamespace(output=tmp_path/'out', legacy=tmp_path/'legacy', helper=tmp_path/'helper',
                           repo=tmp_path, calibration=tmp_path/'calibration', manifest=tmp_path/'manifest',
                           data_root=tmp_path/'missing', archive=tmp_path/'archive', device='cpu')
    summary = tool.execute(args)
    assert summary['status'] == 'complete' and summary['samples'] == 1260
    assert summary['metrics']['normal']['correct'] == 28
    assert sum(summary['paired'][k] for k in ('both_correct','corrects','breaks','both_wrong')) == 1260
    assert (args.output/'run.exit').read_text() == 'exit_code=0\n'
    cache = torch.load(args.output/'prediction_cache.pt', weights_only=True)
    assert cache['normal_logits'].shape == cache['membership_only_logits'].shape == (1260,45)
    assert len(cache['paths']) == 1260 and cache['paths'][0] == 'c00/000.jpg'
    selected = torch.load(args.output/'selected_cases.pt', weights_only=True)
    assert len(selected['cases']) == 90
    head = copy.deepcopy(model.classifier).cpu().eval()
    head.load_state_dict(cache['classifier_state'], strict=True)
    with torch.inference_mode():
        for name in ('normal','membership_only'):
            torch.testing.assert_close(head(cache[name+'_pooled']),cache[name+'_logits'],atol=1e-5,rtol=1e-5)


def test_delivered_payload_is_current_script_and_parses_in_both_shells():
    line=(Path(__file__).parents[1]/'docs/launch_nwpu_pooling_20261008.txt').read_text()
    assert len(line.splitlines())==1
    code=shlex.split(line)[2]
    tree=ast.parse(code)
    encoded=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>500)
    assert zlib.decompress(base64.b64decode(encoded))==Path(api().__file__).read_bytes()
    for shell in ('bash','zsh'):
        result=subprocess.run([shell,'-n'],input=line,text=True,capture_output=True)
        assert result.returncode==0,result.stderr


@pytest.mark.parametrize('existing',[False,True])
def test_launcher_refuses_missing_inputs_or_existing_results_without_launch(tmp_path,existing):
    line=(Path(__file__).parents[1]/'docs/launch_nwpu_pooling_20261008.txt').read_text()
    code=shlex.split(line)[2].replace('/home/featurize/work/',str(tmp_path)+'/')
    out=tmp_path/'nwpu_pooling42_20261008_01'
    if existing:
        out.mkdir();(out/'run.exit').write_text('old evidence')
    result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True)
    assert result.returncode!=0
    assert ('Output/log exists' if existing else 'Missing persistent inputs') in result.stderr
    assert not list(tmp_path.glob('*.py')) and not list(tmp_path.glob('*.launch.log'))
    if existing: assert (out/'run.exit').read_text()=='old evidence'
    else: assert not out.exists()
