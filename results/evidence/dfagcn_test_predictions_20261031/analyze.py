"""One-off reproducible offline analysis of the user-provided DFAGCN test record.

No model or dataset images are loaded. Truth labels follow the frozen pipeline's
class-directory rule; original manifest/image/model bytes are outside this audit.
"""
from pathlib import Path, PurePosixPath
from collections import Counter
import csv
import hashlib
import json
import math
import shutil


ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=Path('/Users/patrick/Downloads/test.json')
raw=SOURCE.read_bytes()
value=json.loads(raw)
record={k:v for k,v in value.items() if k!='record_sha256'}
seal=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert seal==value['record_sha256']
assert value['schema']=='dfagcn-independent-pipeline-v1'
assert value['phase']=='test' and value['mode']=='formal' and value['completed'] is True
reported=value['metrics'];classes=reported['per_class']
assert len(classes)==45 and [r['class_index'] for r in classes]==list(range(45))
names=[r['class_name'] for r in classes]
assert names==sorted(names) and len(set(names))==45
lookup={name:i for i,name in enumerate(names)}
paths,predictions=value['sample_paths'],value['predictions']
assert len(paths)==len(predictions)==25200 and len(set(paths))==25200 and paths==sorted(paths)
confusion=[[0]*45 for unused in names]
prediction_rows=[]
for relative,pred in zip(paths,predictions):
    path=PurePosixPath(relative)
    assert not path.is_absolute() and '..' not in path.parts and '\\' not in relative
    assert path.as_posix()==relative and len(path.parts)==2 and path.parts[0] in lookup
    assert type(pred) is int and 0<=pred<45
    truth=lookup[path.parts[0]]
    confusion[truth][pred]+=1
    prediction_rows.append({'sample_path':relative,'true_index':truth,'true_class':names[truth],
                            'predicted_index':pred,'predicted_class':names[pred],'correct':truth==pred})
assert all(sum(row)==560 for row in confusion)
assert confusion==reported['confusion_matrix']
assert reported['confusion_orientation']=='rows=true, columns=predicted' and reported['zero_division']==0
correct=sum(confusion[i][i] for i in range(45));assert correct==21806
total=25200;errors=total-correct
rows=[]
for i,name in enumerate(names):
    true_count=sum(confusion[i]);pred_count=sum(row[i] for row in confusion)
    tp=confusion[i][i];fp=pred_count-tp;fn=true_count-tp
    recall=100*tp/true_count
    precision=100*tp/pred_count if pred_count else 0.0
    f1=200*tp/(true_count+pred_count) if true_count+pred_count else 0.0
    original=classes[i]
    assert original['support']==true_count
    assert math.isclose(original['accuracy_percent'],recall,abs_tol=1e-10,rel_tol=0)
    assert math.isclose(original['f1_percent'],f1,abs_tol=1e-10,rel_tol=0)
    rows.append({'class_index':i,'class_name':name,'support':true_count,'predicted_count':pred_count,
                 'tp':tp,'fn':fn,'fp':fp,'accuracy_recall_percent':recall,'precision_percent':precision,
                 'f1_percent':f1,'share_of_all_errors_percent':100*fn/errors})
oa=100*correct/total;macro_f1=sum(row['f1_percent'] for row in rows)/45
independent_macro_f1=sum((2*row['tp']/(2*row['tp']+row['fp']+row['fn']))*100 for row in rows)/45
assert math.isclose(oa,reported['oa_percent'],abs_tol=1e-10,rel_tol=0)
assert math.isclose(macro_f1,reported['macro_f1_percent'],abs_tol=1e-10,rel_tol=0)
assert math.isclose(independent_macro_f1,reported['macro_f1_percent'],abs_tol=1e-10,rel_tol=0)
summary=json.loads((ROOT/'results/evidence/dfagcn_formal_3080_return_20261030/reported_summary.json').read_text())
assert value['manifest_sha256']==summary['manifest_sha256']
assert value['cache_content_sha256']==summary['test_extraction']['content_sha256']
assert value['cached_pipeline_elapsed_seconds']==summary['cached_test_seconds']
assert reported['oa_percent']==summary['test_metrics']['oa_percent']
assert reported['macro_f1_percent']==summary['test_metrics']['macro_f1_percent']
directed=[];pairs=[]
for i in range(45):
    for j in range(45):
        if i!=j and confusion[i][j]:
            directed.append({'true_class':names[i],'predicted_class':names[j],
                'count':confusion[i][j],'true_class_percent':100*confusion[i][j]/560,
                'all_errors_percent':100*confusion[i][j]/errors})
        if j>i and confusion[i][j]+confusion[j][i]:
            pairs.append({'class_a':names[i],'class_b':names[j],
                'a_to_b':confusion[i][j],'b_to_a':confusion[j][i],
                'total':confusion[i][j]+confusion[j][i],
                'all_errors_percent':100*(confusion[i][j]+confusion[j][i])/errors})
directed.sort(key=lambda r:(-r['count'],r['true_class'],r['predicted_class']))
pairs.sort(key=lambda r:(-r['total'],r['class_a'],r['class_b']))
weak=sorted(rows,key=lambda r:(r['accuracy_recall_percent'],r['class_name']))
strong=sorted(rows,key=lambda r:(-r['accuracy_recall_percent'],r['class_name']))

def write_csv(path,records):
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


shutil.copyfile(SOURCE,OUT/'test.json')
write_csv(ROOT/'results/dfagcn_nwpu_seed42_per_class_20261031.csv',rows)
write_csv(ROOT/'results/dfagcn_nwpu_seed42_predictions_20261031.csv',prediction_rows)
write_csv(ROOT/'results/dfagcn_nwpu_seed42_errors_20261031.csv',[r for r in prediction_rows if not r['correct']])
write_csv(ROOT/'results/dfagcn_nwpu_seed42_confusions_20261031.csv',directed)
write_csv(ROOT/'results/dfagcn_nwpu_seed42_confusion_pairs_20261031.csv',pairs)
with (ROOT/'results/dfagcn_nwpu_seed42_confusion_matrix_20261031.csv').open('x',newline='') as stream:
    writer=csv.writer(stream);writer.writerow(['true_class / predicted_class',*names])
    writer.writerows([[name,*row] for name,row in zip(names,confusion)])
analysis={'schema':'dfagcn-prediction-analysis-v1','analysis_date':'2026-10-31','experiment_date':'2026-10-30',
    'status':'DOWNLOADED_TEST_RECORD_METRICS_LOCALLY_RECOMPUTED','file_sha256':hashlib.sha256(raw).hexdigest(),
    'record_sha256':seal,'refit_record_sha256_reference_only':value['refit_record_sha256'],
    'manifest_sha256_reference_only':value['manifest_sha256'],'cache_content_sha256_reference_only':value['cache_content_sha256'],
    'class_names':names,'samples':total,'class_count':45,'support_per_class':560,'correct':correct,'errors':errors,
    'metrics':{'oa_percent':oa,'macro_f1_percent':macro_f1,'independent_formula_macro_f1_percent':independent_macro_f1,
               'reported_macro_f1_percent':reported['macro_f1_percent'],'absolute_tolerance':1e-10,
               'max_error_recomputed_vs_reported':max(abs(oa-reported['oa_percent']),abs(independent_macro_f1-reported['macro_f1_percent']))},
    'weakest_recall_classes':weak[:10],'strongest_recall_classes':strong[:10],
    'directed_confusions_top10':directed[:10],'mutual_confusion_pairs_top10':pairs[:10],
    'weakest5_false_negatives':sum(r['fn'] for r in weak[:5]),
    'weakest5_share_of_errors_percent':100*sum(r['fn'] for r in weak[:5])/errors,
    'top10_directed_error_count':sum(r['count'] for r in directed[:10]),
    'top10_directed_share_of_errors_percent':100*sum(r['count'] for r in directed[:10])/errors,
    'top5_pairs_error_count':sum(r['total'] for r in pairs[:5]),
    'top5_pairs_share_of_errors_percent':100*sum(r['total'] for r in pairs[:5])/errors,
    'confusion_matrix':confusion,'per_class':rows,
    'evidence_limits':['Truth reconstructed from class directory under the frozen pipeline naming rule; original manifest/image bytes not returned.',
        'Record checksum and summary references are consistent, not a direct refit/checkpoint/cache byte audit.',
        'No old-model full prediction files returned for paired correction/regression counts.',
        'Confusion structure is descriptive; no module-causal or original-paper performance inference.',
        'No training, GPU evaluation or source-model changes in this analysis.']}
target=ROOT/'results/dfagcn_nwpu_seed42_analysis_20261031.json'
with target.open('x') as stream:json.dump(analysis,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({k:analysis[k] for k in ('status','file_sha256','samples','correct','errors','metrics',
    'weakest5_share_of_errors_percent','top10_directed_share_of_errors_percent')},indent=2))
