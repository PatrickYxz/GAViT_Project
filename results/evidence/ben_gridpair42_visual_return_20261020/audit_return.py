"""Audit returned files as data; never execute returned Python, JS or pickle."""
import base64
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=Path('/Users/patrick/Downloads/ben_grid_pair_visual')
ARCHIVE=HERE/'server_return'


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    own=ROOT/'.worktrees/sparse-hybrid-4n-top2/tools/ben_grid_pair_visual.py'
    spec=importlib.util.spec_from_file_location('trusted_grid_capture_validation',own)
    tool=importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
    summary=read(SOURCE/'summary.json'); data=read(SOURCE/'captures.json')
    prepared=read(ROOT/'results/evidence/ben_gridpair42_visual_prepare_20261020/release_payload.json')
    assert (SOURCE/'run.exit').read_text().strip()=='exit_code=0'
    assert (SOURCE/'source_reports/run.exit').read_text().strip()=='exit_code=0'
    assert summary['status']=='complete' and summary['seed']==42 and summary['samples']==4
    assert summary['models']==[9,16] and summary['paired_capture_checks_passed']
    for relative,digest in summary['files_sha256'].items():
        p=Path(relative)
        assert not p.is_absolute() and '..' not in p.parts
        assert (SOURCE/p).resolve().is_relative_to(SOURCE.resolve())
        assert sha(SOURCE/p)==digest,relative
    assert read(SOURCE/'release.json')==prepared['sha256']
    for relative,digest in prepared['sha256'].items(): assert sha(SOURCE/relative)==digest
    cfg=read(SOURCE/'config.json'); training=read(SOURCE/'source_reports/summary.json')
    tool.validate_summary(training,cfg['checkpoints']['9']['sha256'],cfg['checkpoints']['16']['sha256'],cfg['snapshot']['snapshot_sha256'])
    assert training['stages']['test']['metrics']==cfg['reported_metrics']==read(SOURCE/'source_reports/test_metrics.json')
    assert read(SOURCE/'source_reports/train_run.json')==training['stages']['train']
    assert read(SOURCE/'source_reports/test_run.json')==training['stages']['test']
    assert read(SOURCE/'source_reports/loaded_checkpoints.json')==data['evidence']['checkpoints']
    assert data['classes']==cfg['classes'] and data['evidence']['source_snapshot']==cfg['snapshot']
    assert [c['row_index'] for c in data['cases']]==[97389,14680,28450,4439]
    page=(SOURCE/'comparison.html').read_text()
    embedded=page.split('<script type="application/json" id="capture-data">',1)[1].split('</script>',1)[0]
    assert json.loads(embedded)==data
    html_svgs=re.findall(r'<svg\b.*?</svg>',page,re.S)
    assert len(html_svgs)==4
    counts=Counter(); analysis=[]; edges_checked=0; pt_files=[]
    for case,item,html_svg in zip(data['cases'],cfg['cases'],html_svgs):
        assert case['labels']==item['labels'] and case['patch_path']==item['patch_path']
        png=SOURCE/'inputs'/Path(item['png_path']).name
        assert sha(png)==case['input_png_sha256']==item['png_sha256']
        pool={}
        for k in (9,16):
            record=case['models'][str(k)]
            assert record==read(SOURCE/'captures'/f'row_{case["row_index"]}_K{k}.json')
            assert record['num_regions']==k and record['checkpoint_sha256']==summary['checkpoint_sha256'][str(k)]==cfg['checkpoints'][str(k)]['sha256']
            assert tool.validate_topology(record['assignments'],record['edges'],k)==record['edge_kinds']
            tool.validate_pooling(record['assignments'],record['pooling_weights'],k)
            assert set(record['gate_max_abs_errors'])=={'capture','restoration','pooling','feedback'}
            assert all(v==0 for v in record['gate_max_abs_errors'].values())
            assert record['gat_internal_self_loops']==[True,True]
            if k==16: assert record['legacy_k16_max_logit_diff']==0
            tensor=SOURCE/record['tensor_path']
            assert sha(tensor)==record['tensor_sha256']
            with zipfile.ZipFile(tensor) as z:
                assert z.testzip() is None
                assert sum(n.endswith('data.pkl') for n in z.namelist())==1
            pt_files.append(str(tensor.relative_to(SOURCE)))
            edges_checked+=len(record['edges'])
            members=record['assignments']; w=record['pooling_weights']
            non_single=[r for r in range(k) if members.count(r)>1]
            maxima=[max(row[r] for row in w) for r in range(k)]
            pool[str(k)]=dict(region_max_mass=maxima,non_singleton_regions=len(non_single),
                              non_singleton_max_ge_0_9=sum(maxima[r]>=.9 for r in non_single),
                              region_winner_tokens=[max(range(49),key=lambda i:w[i][r]) for r in range(k)])
        expected=tool.compare_predictions(case['labels'],case['models']['9']['probabilities'],
                                          case['models']['16']['probabilities'],data['classes'])
        assert expected==case['comparison']
        counter=Counter(r['outcome'] for r in expected); counts.update(counter)
        for svg in ((SOURCE/'figures'/f'row_{case["row_index"]}.svg').read_text(),html_svg):
            tree=ET.fromstring(svg)
            placements=[n for n in tree.iter() if n.tag.endswith('image')]
            assert len(placements)==3
            assert all(base64.b64decode(n.get('href').split(',',1)[1])==png.read_bytes() for n in placements)
            actual=[(int(n.get('data-k')),int(n.get('data-source')),int(n.get('data-target')))
                    for n in tree.iter() if n.get('data-source') is not None]
            wanted=[(k,s,t) for k in (9,16) for s,t in case['models'][str(k)]['edges']]
            assert sorted(actual)==sorted(wanted)
            assert 'FIXTURE' not in svg
        analysis.append(dict(row_index=case['row_index'],labels=[data['classes'][i] for i,v in enumerate(case['labels']) if v],
                             all_class_rows=expected,counts=dict(counter),
                             threshold_flips=[r for r in expected if r['outcome'] in ('k16_only_correct','k9_only_correct')],
                             pooling_concentration=pool))
    assert edges_checked==488 and len(pt_files)==8
    assert dict(counts)==dict(both_correct=63,both_wrong=9,k16_only_correct=3,k9_only_correct=1)
    for path in SOURCE.rglob('*'):
        assert not path.is_symlink(),'Do not copy linked paths'
        if path.is_file():
            target=ARCHIVE/path.relative_to(SOURCE)
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): assert sha(target)==sha(path)
            else: shutil.copyfile(path,target)
    checks=dict(status='RETURNED_ARTIFACTS_AUDIT_PASS',source=str(SOURCE),archive=str(ARCHIVE),
                manifest_files_verified=len(summary['files_sha256']),capture_groups=8,actual_pre_gat_edges_verified=488,
                model_and_config_hashes_match_prepared_release=True,exit_codes_verified=[0,0],
                svg_and_html_original_input_placements_verified=24,svg_and_html_edges_verified=976,
                probabilities_and_outcomes_verified=76,threshold=.5,all_class_outcomes=dict(counts),
                raw_pt_hashes_and_zip_crc_verified=pt_files,raw_pt_deserialized=False,
                server_neural_forward_checks_verified_from_returned_reports=True,neural_forward_rerun_locally=False,
                figures_visually_inspected=False,
                capture_wall_seconds=summary['elapsed_seconds'],
                training=dict(best=training['stages']['train']['best'],final_epoch=30,
                              hardware=training['stages']['train']['hardware'],precision=training['stages']['train']['precision'],
                              stages_seconds={k:v['elapsed_seconds'] for k,v in training['stages'].items()},
                              parent_seconds=training['elapsed_seconds'],peak_cuda_allocated_bytes=training['stages']['train']['peak_cuda_allocated_bytes']),
                limits=['The four predetermined historical validation cases are not a population sample.',
                        'K9/K16 were trained in separate batches; this is not a pure causal edge ablation.',
                        'Local audit validates JSON, hashes, PT container CRC and export consistency; neural checks ran on the server.'])
    (HERE/'validation.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    (HERE/'case_analysis.json').write_text(json.dumps(dict(classes=data['classes'],cases=analysis,counts=dict(counts)),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:checks[k] for k in ('status','manifest_files_verified','actual_pre_gat_edges_verified','all_class_outcomes','capture_wall_seconds')},indent=2))
    print('pooling concentration',[(c['row_index'],{k:(r['non_singleton_max_ge_0_9'],r['non_singleton_regions']) for k,r in c['pooling_concentration'].items()}) for c in analysis])


if __name__=='__main__': main()
