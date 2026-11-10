"""Local-only QA. Model inference is never simulated or claimed by this file."""
import ast
import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
from unittest.mock import patch
import xml.etree.ElementTree as ET
import zlib

OUT=Path(__file__).resolve().parent
REPO=OUT.parents[2]
WT=REPO/'.worktrees/sparse-hybrid-4n-top2'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    m=load('visual_tool',WT/'tools/ben_grid_pair_visual.py')
    test=load('visual_test_support',WT/'tests/test_ben_grid_pair_visual.py')
    payload=json.loads((OUT/'release_payload.json').read_text())
    cfg=json.loads(payload['files']['config.json'])
    command=(OUT/'install_and_capture_command.txt').read_text().strip()
    argv=shlex.split(command)
    assert argv[:3]==['python','-B','-c']
    for shell in ('/bin/bash','/bin/zsh'):
        result=subprocess.run([shell,'-n','-c',command],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
    tree=ast.parse(argv[3])
    packed=next(ast.literal_eval(n.args[0]) for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    bootstrap=zlib.decompress(base64.b64decode(packed)).decode()
    assert hashlib.sha256(bootstrap.encode()).hexdigest() in argv[3]
    target='/home/featurize/work/ben_gridpair42_visual_20261020_01'
    assert bootstrap.count(target)==1
    with tempfile.TemporaryDirectory() as td:
        local_target=str(Path(td)/'installed')
        test_bootstrap=bootstrap.replace(target,local_target)
        # Only dispatch is stubbed; the real shipped decoder, installer and hash
        # checks execute. No torch/model behavior is mocked or run.
        with patch.object(subprocess,'run',return_value=subprocess.CompletedProcess([],0)) as dispatch:
            try:
                exec(compile(test_bootstrap,'<local-installer-fixture>','exec'),{})
            except SystemExit as e:
                assert e.code==0
            assert dispatch.call_count==1
            assert dispatch.call_args.args[0][-1]=='cpu'
        for name,sha in payload['sha256'].items():
            assert m.digest(Path(local_target)/name)==sha
        assert json.loads((Path(local_target)/'release.json').read_text())==payload['sha256']
        with patch.object(subprocess,'run') as dispatch:
            try:
                exec(compile(test_bootstrap,'<duplicate-installer-fixture>','exec'),{})
            except FileExistsError:
                pass
            else:
                raise AssertionError('Duplicate install was not rejected')
            dispatch.assert_not_called()
    fixture=OUT/'layout_fixture'; fixture.mkdir(exist_ok=True)
    archive=WT/'results/evidence/pair44_feedback_k_7ehbne/case_sheet_return_20260919'
    cases=[]; pngs=[]; checked_edges=0
    for index,item in enumerate(cfg['cases']):
        png=(archive/Path(item['png_path']).name).read_bytes()
        assert m.sha_bytes(png)==item['png_sha256']
        case=dict(row_index=item['row_index'],labels=item['labels'],models={})
        for k in (9,16):
            row=test.model_record(m,k)
            # Explicitly synthetic probabilities create both outcome directions
            # in the layout fixture; no archived or new model scores are used.
            row['probabilities']=[.3 if (i+index+(k==16))%2 else .7 for i in range(19)]
            case['models'][str(k)]=row
        svg=m.render_svg(case,png,fixture=True)
        (fixture/f'fixture_row_{item["row_index"]}.svg').write_text(svg)
        root=ET.fromstring(svg)
        drawn=[(int(n.get('data-k')),int(n.get('data-source')),int(n.get('data-target')))
               for n in root.iter() if n.get('data-source') is not None]
        expected=[(k,s,t) for k in (9,16) for s,t in test.edges(k)]
        assert sorted(drawn)==sorted(expected)
        checked_edges+=len(drawn)
        cases.append(case); pngs.append(png)
    evidence=dict(scope='LAYOUT_FIXTURE_ONLY',synthetic_edges=True,synthetic_probabilities=True,
                  model_inference_performed=False,original_pngs_preserved=True)
    page=m.render_html(cases,cfg['classes'],pngs,evidence,fixture=True)
    (fixture/'layout_fixture_not_results.html').write_text(page)
    embedded=page.split('<script type="application/json" id="capture-data">',1)[1].split('</script>',1)[0]
    assert json.loads(embedded)['cases']==cases
    assert 'LAYOUT FIXTURE' in page and page.count('<article>')==4
    assert checked_edges==488
    summary=dict(status='LOCAL_PREPARATION_CHECKS_PASS',command_bytes=len(command.encode()),
                 bash_and_zsh_syntax_pass=True,real_installer_fixture_pass=True,duplicate_install_rejected=True,
                 dispatch_stubbed=True,fixture_original_pngs=4,fixture_rendered_input_placements=12,
                 fixture_edges_verified=checked_edges,fixture_probabilities_are_synthetic=True,
                 fixture_path=str(fixture/'layout_fixture_not_results.html'),server_capture_executed=False,
                 tensor_inference_executed=False,source_and_payload_hashes=payload['sha256'])
    (OUT/'preparation_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
