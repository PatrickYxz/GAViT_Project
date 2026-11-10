"""Real PyG replay checks; no images, pretrained backbone or training."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module('tools.diagnose_nwpu_neighbors')


def fixture():
    spec = importlib.util.spec_from_file_location('graph_for_test', ROOT/'models/graph_reasoning.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(17)
    graph = mod.GraphReasoning(in_dim=4, hidden_dim=2, num_heads=2, num_layers=2).eval()
    feedback = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)).eval()
    head = nn.Sequential(nn.LayerNorm(4), nn.Dropout(.1), nn.Linear(4, 3)).eval()
    for module in (graph, feedback, head):
        module.requires_grad_(False)
    regions = torch.randn(3, 4)
    s = torch.rand(5, 3).softmax(-1)
    tokens = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
    with torch.inference_mode():
        f = feedback(graph(regions, edges))
        # Independent token-level forward, not probe's mean-membership formula.
        pooled = (tokens + s @ f).mean(0)
        logits = head(pooled)
    case = dict(S=s, regions=regions, edges=edges, pooled=pooled, logits=logits)
    return graph, feedback, head, case, tokens.mean(0), (s @ f).mean(0)


def test_self_only_preserves_local_transform_and_independent_normal_replay():
    graph, feedback, head, case, mean, mean_f = fixture()
    state = {k: v.clone() for k, v in graph.state_dict().items()}
    result = api().replay_case(graph, feedback, head, case, mean, case['logits'], mean_f)
    with torch.inference_mode():
        # Separately process each node, with an explicit self edge, then full token readout.
        nodes = torch.cat([graph(x[None], torch.zeros(2, 1, dtype=torch.long)) for x in case['regions']])
        expected = head(mean + (case['S'] @ feedback(nodes)).mean(0))
    torch.testing.assert_close(result['self_only_logits'], expected)
    torch.testing.assert_close(result['normal_logits'], case['logits'])
    assert not torch.allclose(result['normal_logits'], result['self_only_logits'])
    assert result['internal_edges'] == {'normal': [7, 7], 'self_only': [3, 3]}
    assert all(torch.equal(state[k], v) for k, v in graph.state_dict().items())
    assert all(p.grad is None for m in (graph, feedback, head) for p in m.parameters())


@pytest.mark.parametrize('field', ['mean', 'pooled', 'logits', 'independent_logits', 'mean_feedback'])
def test_rejects_inconsistent_cache_instead_of_circular_reconstruction(field):
    graph, feedback, head, case, mean, mean_f = fixture()
    independent = case['logits'].clone()
    if field == 'mean': mean = mean + .3
    elif field == 'independent_logits': independent[0] += .3
    elif field == 'mean_feedback': mean_f[0] += .3
    else: case[field] = case[field] + .3
    with pytest.raises((ValueError, AssertionError)):
        api().replay_case(graph, feedback, head, case, mean, independent, mean_f)


def test_training_mode_rejected_before_intervention():
    graph, feedback, head, case, mean, mean_f = fixture()
    graph.train()
    with pytest.raises(ValueError, match='eval'):
        api().replay_case(graph, feedback, head, case, mean, case['logits'], mean_f)


def test_neighbor_deletion_does_not_change_other_node_local_output():
    graph, *_ = fixture()
    x = torch.randn(3, 4)
    changed = x.clone(); changed[1] *= -10
    empty = torch.empty((2, 0), dtype=torch.long)
    a, _ = api().audited_graph(graph, x, empty)
    b, _ = api().audited_graph(graph, changed, empty)
    torch.testing.assert_close(a[[0, 2]], b[[0, 2]])
    assert not torch.allclose(a[1], b[1])


def test_paired_direction_and_metrics_are_self_relative_to_original():
    a = torch.tensor([[4., 0.], [4., 0.], [0., 4.], [0., 4.]])
    b = torch.tensor([[4., 0.], [0., 4.], [0., 4.], [4., 0.]])
    labels = [0, 1, 1, 1]
    result = api().compare(a, b, labels)
    assert result['paired'] == dict(both_correct=2, self_only_corrects=1, self_only_breaks=1, both_wrong=0, net_correct=0)
    assert result['normal']['correct'] == result['self_only']['correct'] == 3
    assert result['normal']['oa_percent'] == 75


def test_failed_output_recorded_and_existing_directory_preserved(tmp_path):
    args = SimpleNamespace(output=tmp_path/'out', pooling=tmp_path/'missing', feedback=tmp_path/'none', calibration=tmp_path/'absent', repo=tmp_path/'repo', threads=2)
    with pytest.raises(FileNotFoundError): api().execute(args)
    assert (args.output/'run.exit').read_text().strip() == 'exit_code=1'
    assert json.loads((args.output/'summary.json').read_text())['status'] == 'failed'
    before = (args.output/'summary.json').read_bytes()
    with pytest.raises(FileExistsError): api().execute(args)
    assert (args.output/'summary.json').read_bytes() == before


@pytest.fixture
def source_files(tmp_path, monkeypatch):
    import hashlib
    tool = api()
    args = SimpleNamespace(pooling=tmp_path/'pool', feedback=tmp_path/'feedback', calibration=tmp_path/'calibration', repo=ROOT)
    for folder in (args.pooling, args.feedback, args.calibration):
        folder.mkdir(); (folder/'run.exit').write_text('exit_code=0\n')
    names = [f'class{i:02}' for i in range(45)]
    paths = [f'{name}/{j:03}.jpg' for name in names for j in range(28)]
    labels = [i for i in range(45) for _ in range(28)]
    logits = torch.zeros(1260, 45)
    for i, y in enumerate(labels): logits[i, y if i < 1181 else (y+1)%45] = 3
    source = dict(checkpoint_sha256=tool.CHECKPOINT_SHA, run_sha256=tool.RUN_SHA, best_epoch=28)
    code = {'git': {'commit': 'fixture', 'dirty': False}}
    common = dict(dataset='NWPU-RESISC45', seed=42, split='val', source_phase='calibrate', samples=1260, manifest_sha256=tool.MANIFEST_SHA)
    pi = dict(common, source=source, model_code=code, tool_sha256=tool.POOL_TOOL_SHA)
    # Real writer has a tuple embedded in PT, while JSON turns it into a list.
    fi = dict(common, sources={'soft': source}, source_code=code, tool_sha256=tool.FEEDBACK_TOOL_SHA, alphas=(0., .25, .5, .75, 1.))
    spatial = [(s, t) for s in range(16) for t in range(16) if abs(s//4-t//4)+abs(s%4-t%4) == 1]
    feature = [(s, t) for t in range(16) for s in [s for s in range(16) if abs(s//4-t//4)+abs(s%4-t%4)>1][:2]]
    edges = torch.tensor(spatial+feature).T.contiguous()
    cases = [dict(path=p, label=labels[i], tensors={'normal': dict(S=torch.full((49,16),1/16), W=torch.full((49,16),1/49), regions=torch.zeros(16,768), pooled=torch.zeros(768), logits=logits[i].clone(), edges=edges)}) for i,p in enumerate(paths) if i%28<2]
    cache = dict(paths=paths, labels=labels, class_names=names, normal_logits=logits, mean_tokens=torch.zeros(1260,768), mean_feedback=torch.zeros(1260,768), model='soft', identity=fi)
    (args.calibration/'best.pth').write_bytes(b'identity fixture only, not loaded as weights')
    report = dict(status='complete', code=code)
    (args.calibration/'run.json').write_text(json.dumps(report))
    ck_sha = hashlib.sha256((args.calibration/'best.pth').read_bytes()).hexdigest()
    run_sha = hashlib.sha256((args.calibration/'run.json').read_bytes()).hexdigest()
    source.update(checkpoint_sha256=ck_sha, run_sha256=run_sha)
    monkeypatch.setattr(tool,'CHECKPOINT_SHA',ck_sha); monkeypatch.setattr(tool,'RUN_SHA',run_sha)
    torch.save(dict(identity=pi,cases=cases), args.pooling/'selected_cases.pt')
    torch.save(cache,args.feedback/'soft_cache.pt')
    # Original cache size is fixed in the production tool; fixture is a different archive.
    monkeypatch.setattr(tool,'FEEDBACK_BYTES',(args.feedback/'soft_cache.pt').stat().st_size, raising=False)
    monkeypatch.setattr(tool,'FEEDBACK_SHA',tool.sha(args.feedback/'soft_cache.pt'))
    (args.pooling/'identity.json').write_text(json.dumps(pi)); (args.feedback/'identity.json').write_text(json.dumps(fi))
    artifact = dict(sha256=tool.sha(args.pooling/'selected_cases.pt'),bytes=(args.pooling/'selected_cases.pt').stat().st_size)
    (args.pooling/'summary.json').write_text(json.dumps(dict(status='complete',cache_artifacts={'selected_cases.pt':artifact})))
    (args.feedback/'summary.json').write_text(json.dumps(dict(status='complete')))
    return args


def test_real_tuple_identity_roundtrip_and_path_join(source_files):
    cases, cache, index, _, ident = api().load_inputs(source_files)
    assert len(cases) == 90 and ident['samples'] == 90
    assert all(cache['labels'][index[c['path']]] == c['label'] for c in cases)


@pytest.mark.parametrize('corruption', ['bytes', 'identity', 'missing'])
def test_corrupt_or_missing_cache_stops_before_replay(source_files, corruption):
    if corruption == 'bytes':
        with (source_files.pooling/'selected_cases.pt').open('ab') as f: f.write(b'damage')
    elif corruption == 'identity':
        p=source_files.pooling/'identity.json'; v=json.loads(p.read_text()); v['seed']=43; p.write_text(json.dumps(v))
    else: (source_files.feedback/'soft_cache.pt').unlink()
    with pytest.raises((ValueError,FileNotFoundError)): api().load_inputs(source_files)


@pytest.mark.parametrize('corruption', ['selection', 'label', 'edge', 'nonfinite'])
def test_wrong_case_join_and_graph_not_accepted(source_files, corruption):
    c=torch.load(source_files.pooling/'selected_cases.pt',weights_only=True)['cases']
    f=torch.load(source_files.feedback/'soft_cache.pt',weights_only=True)
    if corruption == 'selection': c[0]['path']=f['paths'][2]
    elif corruption == 'label': c[0]['label']=1
    elif corruption == 'edge': c[0]['tensors']['normal']['edges'][0,0]=0; c[0]['tensors']['normal']['edges'][1,0]=0
    else: c[0]['tensors']['normal']['regions'][0,0]=float('nan')
    with pytest.raises(ValueError): api().validate_cases(c,f)


def test_case_join_uses_paths_when_cache_and_case_orders_differ(source_files):
    c=torch.load(source_files.pooling/'selected_cases.pt',weights_only=True)['cases']
    f=torch.load(source_files.feedback/'soft_cache.pt',weights_only=True)
    for k in ('paths','labels'): f[k]=f[k][::-1]
    for k in ('normal_logits','mean_tokens','mean_feedback'): f[k]=f[k].flip(0)
    cases,index=api().validate_cases(c[::-1],f)
    assert all(f['paths'][index[row['path']]]==row['path'] and f['labels'][index[row['path']]]==row['label'] for row in cases)


def test_checkpoint_submodules_and_full_dimension_replay(tmp_path):
    graph_class = type(fixture()[0])
    graph = graph_class().eval()
    feedback=nn.Sequential(nn.Linear(1024,768),nn.LayerNorm(768)).eval()
    head=nn.Sequential(nn.LayerNorm(768),nn.Dropout(.1),nn.Linear(768,45)).eval()
    state={prefix+'.'+k:v for prefix,m in [('graph_reasoning',graph),('feedback_proj',feedback),('classifier',head)] for k,v in m.state_dict().items()}
    state['region_grouping.candidate_mask']=torch.ones(49,16,dtype=torch.bool)
    torch.save(state,tmp_path/'best.pth')
    a=dict(num_classes=45,num_regions=16,gat_hidden=256,gat_heads=4,gat_layers=2,dropout=.1,grouping='soft_prototype',edge_type='sparse_hybrid',integration='token_feedback')
    g,f,h,mask=api().load_modules(SimpleNamespace(repo=ROOT,calibration=tmp_path),dict(architecture=a),head.state_dict())
    assert not any(p.requires_grad for m in (g,f,h) for p in m.parameters())
    regions=torch.randn(16,768); S=torch.rand(49,16).softmax(-1); tokens=torch.randn(49,768)
    edges=torch.stack((torch.arange(16),torch.arange(16).roll(1)))
    with torch.inference_mode():
        mean_f=(S @ feedback(graph(regions,edges))).mean(0)
        pooled=(tokens + S @ feedback(graph(regions,edges))).mean(0)
        logits=head(pooled)
    result=api().replay_case(g,f,h,dict(S=S,regions=regions,edges=edges,pooled=pooled,logits=logits),tokens.mean(0),logits,mean_f)
    assert result['internal_edges']['self_only']==[16,16]
    bad={k:v.clone() for k,v in head.state_dict().items()}; bad['2.bias'][0]+=1
    with pytest.raises(ValueError,match='Cached head differs'):
        api().load_modules(SimpleNamespace(repo=ROOT,calibration=tmp_path),dict(architecture=a),bad)


def test_delivered_launcher_decodes_exact_tested_tool_and_shell_syntax(tmp_path):
    import ast, base64, shlex, subprocess, sys, zlib
    command=(ROOT/'docs/launch_nwpu_neighbors_20261010.txt').read_text()
    args=shlex.split(command); assert args[:2]==['python','-c'] and len(args)==3
    tree=ast.parse(args[2])
    blobs=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>1000]
    assert len(blobs)==1
    decoded=zlib.decompress(base64.b64decode(blobs[0]))
    assert decoded==(ROOT/'tools/diagnose_nwpu_neighbors.py').read_bytes()
    for shell in ('bash','zsh'):
        subprocess.run([shell,'-n'],input=command,text=True,check=True,capture_output=True)
    script=tmp_path/'probe.py'; script.write_bytes(decoded)
    help_run=subprocess.run([sys.executable,str(script),'--help'],text=True,capture_output=True)
    assert help_run.returncode==0 and '--output' in help_run.stdout


def test_delivered_launcher_missing_sources_and_old_outputs_stop_without_spawn(tmp_path):
    import ast, shlex, subprocess, sys
    command=shlex.split((ROOT/'docs/launch_nwpu_neighbors_20261010.txt').read_text())[2]
    code=command.replace('/home/featurize/work',str(tmp_path))
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True)
    assert result.returncode!=0 and 'Missing persistent sources' in result.stderr
    assert list(tmp_path.iterdir())==[]
    tree=ast.parse(code)
    required=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='required')
    for name in required:
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    out=tmp_path/'nwpu_neighbors42_20261010_01';out.mkdir();(out/'marker').write_text('preserve')
    result=subprocess.run([sys.executable,'-c',code],text=True,capture_output=True)
    assert result.returncode!=0 and 'Output/log already exists' in result.stderr
    assert (out/'marker').read_text()=='preserve'
    assert not list(tmp_path.glob('*.py')) and not list(tmp_path.glob('*.log'))
