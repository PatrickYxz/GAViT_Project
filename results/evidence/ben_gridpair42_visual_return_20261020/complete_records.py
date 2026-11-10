"""Advance status only after returned data and final exports pass checks."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    audit=json.loads((HERE/'validation.json').read_text())
    export=json.loads((ROOT/'output/pdf/ben_gridpair42_20261020/export_qa.json').read_text())
    assert audit['status']=='RETURNED_ARTIFACTS_AUDIT_PASS' and export['status']=='EXPORT_CHECKS_PASS'
    summary=json.loads((HERE/'server_return/source_reports/summary.json').read_text())
    preflight=json.loads((HERE/'server_return/source_reports/preflight.json').read_text())
    assert preflight['status']=='BEN_GRID3_42_PREFLIGHT_PASS' and preflight['tensor_checks']['status']=='pass'
    train=summary['stages']['train']
    path=ROOT/'results/ben_grid3_seed42_status_20261018.json'
    state=json.loads(path.read_text())
    state.update(status='TRAINING_SUMMARY_TEST_METRICS_AUDITED_VISUALIZATION_COMPLETE',
                 final_epoch=30,best_epoch=train['best']['epoch'],training_summary_audit_complete=True,
                 preflight_pass_verified=True,smoke_pass_verified=True,
                 actual_training_hardware=train['hardware'],actual_training_precision=train['precision'],
                 training_resources={k:train[k] for k in ('elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','effective_training_images_per_second')},
                 parent_wall_seconds=summary['elapsed_seconds'],
                 training_summary_evidence='results/evidence/ben_gridpair42_visual_return_20261020/server_return/source_reports/summary.json',
                 full_artifact_audit_complete=False,
                 full_artifact_audit_scope='Returned training/capture summaries and outputs audited. Original full training history and trained checkpoint bytes remain on server; no local neural rerun.',
                 next_action='The trained K9/K16 visualization and analysis are complete. Continue the remaining Swin/fixed4x4/soft same-image comparison using existing checkpoints; no additional training is implied.')
    state['target_hardware']['uuid']=train['hardware']['gpu_identity'].split(',')[0]
    state['limitations'][2]='Training summary, epochs, UUID and resources are now archived. Earlier explicit startup flags reflect the missing original first-batch trace, not unstarted training; original full history/checkpoint bytes remain on server.'
    v=state['visualization']
    v.update(status='PAIRED_CAPTURE_AUDITED_FIGURES_COMPLETE',artifacts_received=True,
             artifacts_received_from=audit['source'],paired_capture_audit_complete=True,
             paired_k9_k16_capture_complete=True,trained_model_figure_complete=True,
             paired_class_outcomes=audit['all_class_outcomes'],actual_cpu_wall_seconds=audit['capture_wall_seconds'],
             raw_tensor_hash_crc_verified=True,raw_tensor_deserialized_locally=False,
             neural_forward_rerun_locally=False,return_artifact_audit='results/evidence/ben_gridpair42_visual_return_20261020/validation.json',
             pdf='output/pdf/ben_gridpair42_20261020/ben_gridpair42_cases.pdf',
             png='output/pdf/ben_gridpair42_20261020/graphs_predictions_300dpi.png',
             html='output/pdf/ben_gridpair42_20261020/comparison.html',
             analysis='docs/ben_gridpair42_visual_analysis_20261020.md',
             export_validation='output/pdf/ben_gridpair42_20261020/export_qa.json',
             completion_field_scope='Returned JSON, input/source hashes, topology, probabilities, PT hashes/CRC and exports verified; model forward checks ran on server.',
             existing_tool_gap='Resolved for the trained K9/K16 comparison; remaining three-model visual comparison is a separate next item.')
    path.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n')
    audit.update(figures_visually_inspected=True,figure_export_validation=v['export_validation'],
                 figure_pdf=v['pdf'],analysis=v['analysis'])
    (HERE/'validation.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    perfpath=ROOT/'results/ben_grid3_grid4_seed42_20261020.json'
    perf=json.loads(perfpath.read_text())
    perf['status']='TEST_METRICS_AND_TRAINING_SUMMARY_AUDITED'
    perf['evidence_scope']='Returned training/capture files audited; original full-test metrics unchanged. No local neural rerun.'
    perf['k9'].update(best_epoch=10,final_epoch=30,hardware_observed=train['hardware'],
                      best_validation=train['best'],training_summary_audit=state['training_summary_evidence'])
    perf['visualization_analysis']=v['analysis']
    perf['caveats'][-1]='Trained K9/K16 figures are now complete on four historical diagnostic validation inputs; these cases do not estimate population gains or isolate edge causality.'
    perfpath.write_text(json.dumps(perf,indent=2,ensure_ascii=False)+'\n')
    print(state['status']);print(v['status'])

if __name__=='__main__': main()
