"""Parse the user-pasted console return; this does not inspect server files."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def main():
    raw=(HERE/'user_return.txt').read_bytes()
    text=raw.decode()
    output='/home/featurize/work/ben_gridpair42_visual_20261020_01'
    assert 'BEN_GRID_PAIR_VISUAL_INSTALLED '+output in text
    assert 'ARTIFACT_GATE_PASS: completed30 / exact two weights / frozen source / 4 original PNGs / val labels' in text
    device_line=next(line for line in text.splitlines() if line.startswith('DEVICE '))
    hardware=json.loads(device_line.removeprefix('DEVICE ').split('; 4 inputs',1)[0])
    assert hardware['device']=='cpu' and hardware['gpu_identity'] is None
    assert hardware['packages']=={'torch':'2.2.2','torchvision':'0.17.2','timm':'1.0.27',
                                 'torch-geometric':'2.8.0','numpy':'1.26.4','Pillow':'10.3.0'}
    captures=[]
    for row,k,spatial,feature,gates in re.findall(
            r'^CAPTURE_PASS row=(\d+) K=(\d+); (\d+) spatial \+ (\d+) feature; gates=(\{[^\n]+\})$',text,re.M):
        row,k,spatial,feature=map(int,(row,k,spatial,feature))
        gates=ast.literal_eval(gates)
        assert (spatial,feature)=={9:(24,18),16:(48,32)}[k]
        assert set(gates)=={'capture','restoration','pooling','feedback'}
        assert all(type(v) in (int,float) and math.isfinite(v) and v==0 for v in gates.values())
        captures.append(dict(row_index=row,num_regions=k,spatial_edges=spatial,feature_edges=feature,
                             total_edges=spatial+feature,gate_max_abs_errors=gates))
    assert len(captures)==8
    assert {(r['row_index'],r['num_regions']) for r in captures}=={
        (row,k) for row in (97389,14680,28450,4439) for k in (9,16)}
    assert sum(row['total_edges'] for row in captures)==488
    assert 'BEN_GRID_PAIR_VISUAL_COMPLETE '+output in text
    assert 'RETURN_HTML '+output+'/comparison.html' in text
    assert 'RETURN_AUDIT_ZIP '+output+'/ben_grid_pair_visual.zip' in text
    value=dict(status='SERVER_CAPTURE_REPORTED_COMPLETE_ARTIFACTS_PENDING',
               evidence_scope='user-pasted console output; no independent server/file read',
               raw_sha256=hashlib.sha256(raw).hexdigest(),output=output,hardware_reported=hardware,
               artifact_gate_reported_pass=True,captures_reported=captures,
               capture_groups=8,gate_values_checked=32,all_reported_errors_zero=True,
               reported_total_pre_gat_edges=488,completion_marker_reported=True,
               html_path=output+'/comparison.html',zip_path=output+'/ben_grid_pair_visual.zip',
               explicit_exit_code_received=False,exit_code=None,
               returned_zip_received=False,raw_tensors_audited=False,
               paired_probabilities_received=False,figures_visually_inspected=False,
               next_action='Download and attach the existing ben_grid_pair_visual.zip; do not rerun capture.')
    (HERE/'validation.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    state_path=ROOT/'results/ben_grid3_seed42_status_20261018.json'
    state=json.loads(state_path.read_text())
    v=state['visualization']
    v.update(status=value['status'],server_capture_executed=True,server_capture_reported_complete=True,
             capture_evidence_basis=value['evidence_scope'],reported_groups=8,
             reported_artifact_gate_pass=True,reported_capture_errors_zero=True,
             output_html=value['html_path'],output_zip=value['zip_path'],
             run_return_evidence='results/evidence/ben_gridpair42_visual_run_20261020/validation.json',
             artifacts_received=False,paired_capture_audit_complete=False,
             paired_k9_k16_capture_complete=False,trained_model_figure_complete=False,
             completion_field_scope='server_capture_reported_complete is based on the pasted log; paired_k9_k16_capture_complete awaits returned-file audit',
             existing_tool_gap='Server log reports all eight real captures and ZIP export complete; returned artifacts and local figure analysis pending.')
    state['next_action']=value['next_action']
    state['last_updated']='2026-10-20'
    assert not state['full_artifact_audit_complete']
    assert state['formal_test_metrics']==json.loads((ROOT/'results/evidence/ben_grid3_seed42_metrics_20261020/metrics.json').read_text())
    state_path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:value[k] for k in ('status','capture_groups','gate_values_checked',
                                          'all_reported_errors_zero','returned_zip_received')},indent=2))


if __name__=='__main__':
    main()
