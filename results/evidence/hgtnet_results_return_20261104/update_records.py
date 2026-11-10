"""Promote result provenance after raw predictions/history/metadata were audited."""
import csv,hashlib,io,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
analysis=json.loads((ROOT/'results/hgtnet_nwpu_seed42_analysis_20261104.json').read_text())
audit=json.loads((HERE/'validation.json').read_text())
pa=json.loads((HERE/'review_predictions/validation.json').read_text())
ha=json.loads((HERE/'review_history/validation.json').read_text())
assert audit['passed'] and analysis['audit']['passed']
# Review schemas expose a passed field; retain their original exact receipts.
assert pa.get('passed',pa.get('all_passed',False)) and ha.get('passed',ha.get('all_passed',False))
def backup(relative):
    p=ROOT/relative;q=HERE/'before_update'/relative
    if not q.exists():q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    return p
table=backup('results/external_baselines_nwpu_seed42_20261030.csv')
raw=table.read_bytes();lines=raw.splitlines(keepends=True);target=[]
for i,line in enumerate(lines):
    if line.startswith(b'HGTNet independent v1,'):target.append(i)
assert len(target)==1
i=target[0];fields=next(csv.reader(io.StringIO(lines[0].decode())))
row=dict(zip(fields,next(csv.reader(io.StringIO(lines[i].decode())))))
assert float(row['oa_percent'])==analysis['metrics']['oa_percent'] and float(row['macro_f1_percent'])==analysis['metrics']['macro_f1_percent']
row.update(correct=str(analysis['correct']),correct_basis='locally_recomputed_from_returned_test_predictions',
    evidence='results/evidence/hgtnet_results_return_20261104/files/test.json')
stream=io.StringIO(newline='');csv.DictWriter(stream,fieldnames=fields,lineterminator='\r\n').writerow(row)
lines[i]=stream.getvalue().encode();updated=b''.join(lines)
assert updated.splitlines(keepends=True)[:i]==raw.splitlines(keepends=True)[:i]
table.write_bytes(updated)
result_path=backup('results/hgtnet_nwpu_seed42_20261103.json');result=json.loads(result_path.read_text())
result.update(status='DOWNLOADED_RESULTS_METRICS_HISTORY_LOCALLY_RECOMPUTED',
    evidence_type='server_completed_formal_run_plus_downloaded_prediction_history_metadata_audit',
    metrics=analysis['metrics'],prediction_audit=dict(status='verified',date='2026-11-04',
        file='results/evidence/hgtnet_results_return_20261104/files/test.json',file_sha256=analysis['test_sha256'],
        samples=25200,correct=23718,errors=1482,metrics_recomputed=analysis['metrics'],
        per_class='results/hgtnet_nwpu_seed42_per_class_20261104.csv',
        independent_audit='results/evidence/hgtnet_results_return_20261104/review_predictions/validation.json'),
    history_audit=dict(date='2026-11-04',best_epoch=95,final_epoch=100,refit_epochs=95,
        best_validation_oa=analysis['history']['best_validation_oa'],final_validation_oa=analysis['history']['final_validation_oa'],
        best_minus_final_oa_pp=analysis['history']['best_minus_final_oa_pp'],
        file='results/evidence/hgtnet_results_return_20261104/files/history.json',
        independent_audit='results/evidence/hgtnet_results_return_20261104/review_history/validation.json'),
    offline_analysis='results/hgtnet_nwpu_seed42_analysis_20261104.json',
    metadata_archive_sha256=analysis['archive_sha256'],paired_vs_dfagcn=analysis['paired_vs_dfagcn'],
    limitations=analysis['limits'])
result['derived'].update(test_correct_from_predictions=23718,test_errors_from_predictions=1482)
result_path.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
before=HERE/'before_update/results/external_baselines_nwpu_seed42_20261030.csv'
receipt=dict(csv_old_sha256=hashlib.sha256(before.read_bytes()).hexdigest(),csv_new_sha256=hashlib.sha256(updated).hexdigest(),
    non_hgt_lines_unchanged=all(a==b for j,(a,b) in enumerate(zip(raw.splitlines(keepends=True),updated.splitlines(keepends=True))) if j!=i),
    old_and_new_metric_values_unchanged=True,new_correct_basis=row['correct_basis'],
    result_state=result['status'],original_archive_preserved=True)
(HERE/'record_update.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt,indent=2))
