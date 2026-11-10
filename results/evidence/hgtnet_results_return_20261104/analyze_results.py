"""Reproduce prediction/history analysis from returned JSON, with no ML imports."""
from collections import Counter
from fractions import Fraction
from pathlib import Path,PurePosixPath
import csv,hashlib,io,json,math,statistics,sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from external_baselines.hgtnet.formal_protocol import validate_binding,recipe_sha256
from external_baselines.hgtnet.resources import check_runtime
from tools.package_hgtnet_nwpu_smoke import _read_bundle
FILES=HERE/'files';OUT=ROOT/'results';checks={}
def read(name):return json.loads((FILES/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check(name,value):
    checks[name]=bool(value)
    if not value:raise ValueError('Audit failed: '+name)
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-10)
def write_new(path,raw):
    if isinstance(raw,str):raw=raw.encode()
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=raw:raise FileExistsError('Refuse different existing artifact: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(raw)
def write_json(path,value):write_new(path,json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def write_csv(path,rows,fields=None):
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]))
    writer.writeheader();writer.writerows(rows);write_new(path,stream.getvalue())
def metrics(cm,names):
    total=sum(map(sum,cm));correct=sum(cm[i][i] for i in range(len(names)));per=[]
    for i,name in enumerate(names):
        support=sum(cm[i]);predicted=sum(row[i] for row in cm);tp=cm[i][i]
        per.append(dict(class_index=i,class_name=name,support=support,predicted_count=predicted,tp=tp,
            fn=support-tp,fp=predicted-tp,recall_percent=100*tp/support if support else 0.,
            precision_percent=100*tp/predicted if predicted else 0.,f1_percent=200*tp/(support+predicted) if support+predicted else 0.))
    exact_f1=sum((Fraction(2*r['tp'],r['support']+r['predicted_count']) if r['support']+r['predicted_count'] else Fraction(0)) for r in per)/len(names)
    return dict(samples=total,correct=correct,errors=total-correct,oa_percent=100*correct/total,
        macro_f1_percent=float(100*exact_f1),per_class=per,confusion_matrix=cm)
def validate_metrics(packet,names,support):
    cm=packet['confusion_matrix']
    if len(cm)!=45 or any(len(row)!=45 or any(type(v) is not int or v<0 for v in row) for row in cm):raise ValueError('CM type/shape')
    m=metrics(cm,names)
    if not all(sum(row)==support for row in cm) or not close(m['oa_percent'],packet['accuracy_percent']) or not close(m['macro_f1_percent'],packet['macro_f1_percent']):raise ValueError('CM metric mismatch')
    if len(packet['per_class'])!=45:raise ValueError('Per-class count')
    for got,reported in zip(m['per_class'],packet['per_class']):
        if (got['class_name']!=reported['class'] or got['tp']!=reported['correct'] or got['support']!=reported['support']
                or not close(got['recall_percent'],reported['accuracy_percent']) or not close(got['f1_percent'],reported['f1_percent'])):
            raise ValueError('Per-class mismatch')
    return m

t=read('test.json');h=read('history.json');s=read('summary.json');b=read('binding.json')
init=read('initialization.json');reload=read('checkpoint_reload.json');runtime=read('attempts/0001/runtime.json')
check('test_sha_matches_completed_status',sha(FILES/'test.json')==s['test_sha256']=='869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae')
check('real_completed_metadata',s['completed'] is True and s['input_scope']==h['input_scope']==b['input_scope']==t['input_scope']=='real_nwpu')
check('binding_frozen_protocol',validate_binding(b)==b)
check('full_counts',s['counts']==b['counts']==dict(train=5040,val=1260,pool=6300,test=25200))
check('recipe_bound',s['recipe']==b['recipe'] and recipe_sha256(s['recipe'])==s['recipe_sha256']==b['recipe_sha256']==t['recipe_sha256'])
for key in ['manifest_sha256','config_sha256','weights_sha256']:check('identity:'+key,s[key]==b[key])
check('test_manifest',t['manifest_sha256']==b['manifest_sha256'])
check('runtime_original_returned',runtime==s['runtime'] and check_runtime(runtime,s['physical_gpu_uuid'],target_name='NVIDIA GeForce RTX 3090')==runtime)
check('init_original_equal',init['equal'] is True and init['calibration_state_sha256']==init['refit_state_sha256']==s['initial_state_sha256']==s['refit_initial_state_sha256'])
check('reload_original_match',reload==s['checkpoint_reload'] and reload['loaded'] is True and reload['reload_max_abs_error']==0)
check('checkpoint_references',t['refit_checkpoint_sha256']==reload['sha256']==s['checkpoints']['final_refit']['sha256'])
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz';manifest,_,payload=_read_bundle(bundle)
check('delivered_v1_sha',sha(bundle)=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1')
for name,digest in b['source_identity'].items():
    member='configs/hgtnet_nwpu_formal_v1.json' if name=='formal_recipe_bytes' else name
    check('returned_source:'+name,member in payload and hashlib.sha256(payload[member]).hexdigest()==digest)
names=b['class_names'];lookup={name:i for i,name in enumerate(names)}
check('45_sorted_class_names',names==sorted(names) and len(set(names))==45)
paths,predictions,labels=t['paths'],t['predictions'],t['labels']
check('25200_unique_ordered',len(paths)==len(predictions)==len(labels)==25200 and paths==sorted(paths) and len(set(paths))==25200)
check('exact_binding_test_order',paths==[r['path'] for r in b['records']['test']] and labels==[r['label'] for r in b['records']['test']])
check('validation_test_disjoint',not set(paths)&{r['path'] for r in b['records']['val']})
cm=[[0]*45 for _ in names];prediction_rows=[]
for path,label,pred in zip(paths,labels,predictions):
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or '\\' in path or len(p.parts)!=2 or p.as_posix()!=path:raise ValueError('Unsafe test path')
    if type(label) is not int or type(pred) is not int or not 0<=pred<45 or label!=lookup[p.parts[0]]:raise ValueError('Directory truth/prediction mismatch')
    cm[label][pred]+=1
    prediction_rows.append(dict(sample_path=path,true_index=label,true_class=names[label],predicted_index=pred,
        predicted_class=names[pred],correct=label==pred))
check('all25200_directory_truths_valid',True)
check('all2025_cm_cells',cm==t['metrics']['confusion_matrix'])
m=validate_metrics(t['metrics'],names,560)
check('test_metrics_summary_exact',t['metrics']==s['test_metrics'] and s['test']['metrics']==t['metrics'])
check('recomputed23718_1482',m['correct']==23718 and m['errors']==1482)
check('test_selected95',t['selected_epoch']==s['selected_epoch']==s['best_epoch']==s['refit_epochs']==95)
cal,refit=h['calibration'],h['refit']
check('history_summary_exact',cal==s['calibration_history'] and refit==s['refit_history'])
check('100cal_95refit',len(cal)==100 and len(refit)==95 and s['final_epoch']==100)
epoch_rows=[]
for phase,records,count,steps,offset in [('calibration',cal,5040,315,0),('refit',refit,6300,394,31500)]:
    check(phase+'_sequential_epochs',[r['epoch'] for r in records]==list(range(1,len(records)+1)))
    for row in records:
        epoch=row['epoch']
        check(f'{phase}:{epoch}:counts',row['epoch_optimizer_steps']==steps and row['optimizer_steps']==offset+epoch*steps)
        check(f'{phase}:{epoch}:times',0<row['train_seconds']<=row['epoch_precommit_seconds']<=row['epoch_wall_seconds'] and close(row['train_images_per_second'],count/row['train_seconds']))
        correct=row['train_accuracy_percent']*count/100
        check(f'{phase}:{epoch}:online_accuracy_integer',0<=correct<=count and math.isclose(correct,round(correct),abs_tol=1e-8))
        record=dict(phase=phase,epoch=epoch,optimizer_steps=row['optimizer_steps'],train_loss=row['train_loss'],
            train_accuracy_percent=row['train_accuracy_percent'],train_seconds=row['train_seconds'],
            train_images_per_second=row['train_images_per_second'],epoch_wall_seconds=row['epoch_wall_seconds'],
            epoch_precommit_seconds=row['epoch_precommit_seconds'],validation_seconds='',val_loss='',val_accuracy_percent='',val_macro_f1_percent='',val_correct='')
        if phase=='calibration':
            vm=validate_metrics(row['val_metrics'],names,28)
            check(f'calibration:{epoch}:val_correct',row['val_correct']==vm['correct'])
            record.update(validation_seconds=row['validation_seconds'],val_loss=row['val_loss'],val_accuracy_percent=vm['oa_percent'],val_macro_f1_percent=vm['macro_f1_percent'],val_correct=vm['correct'])
        epoch_rows.append(record)
best=max(cal,key=lambda r:r['val_correct']);final=cal[-1]
check('best95_earliest_max',best['epoch']==95 and best['val_metrics']==s['best_val_metrics'])
check('total68930_steps',refit[-1]['optimizer_steps']==s['optimizer_steps']==68930)
weak=sorted(m['per_class'],key=lambda r:(r['recall_percent'],r['class_name']));strong=sorted(m['per_class'],key=lambda r:(-r['recall_percent'],r['class_name']))
directed=[];pairs=[]
for i in range(45):
    for j in range(45):
        if i!=j and cm[i][j]:directed.append(dict(true_class=names[i],predicted_class=names[j],count=cm[i][j],true_class_percent=100*cm[i][j]/560,all_errors_percent=100*cm[i][j]/m['errors']))
        if j>i and cm[i][j]+cm[j][i]:pairs.append(dict(class_a=names[i],class_b=names[j],a_to_b=cm[i][j],b_to_a=cm[j][i],total=cm[i][j]+cm[j][i],all_errors_percent=100*(cm[i][j]+cm[j][i])/m['errors']))
directed.sort(key=lambda r:(-r['count'],r['true_class'],r['predicted_class']));pairs.sort(key=lambda r:(-r['total'],r['class_a'],r['class_b']))
dfpath=ROOT/'results/evidence/dfagcn_test_predictions_20261031/test.json';df=json.loads(dfpath.read_text())
check('prior_df_raw_sha',sha(dfpath)=='771dee2dcd1e676e4884c36654cc8fbcc4c067811362dbf9f3e997c903ccd4e9')
check('same_df_paths_manifest',df['sample_paths']==paths and df['manifest_sha256']==b['manifest_sha256'])
paired=Counter();paired_rows=[]
for path,label,hp,dp in zip(paths,labels,predictions,df['predictions']):
    category=('both_correct' if hp==label else 'regressed') if dp==label else ('corrected' if hp==label else 'both_wrong')
    paired[category]+=1
    paired_rows.append(dict(sample_path=path,true_index=label,true_class=names[label],df_predicted_index=dp,
        hgt_predicted_index=hp,df_predicted_class=names[dp],hgt_predicted_class=names[hp],outcome=category))
check('paired_all_directions',dict(paired)==dict(both_correct=21303,corrected=2415,regressed=503,both_wrong=979))
check('paired_net1912',paired['corrected']-paired['regressed']==23718-21806==1912)
phase_stats={}
for phase,rows in [('calibration',cal),('refit',refit)]:
    phase_stats[phase]=dict(epochs=len(rows),train_seconds=sum(r['train_seconds'] for r in rows),
        epoch_wall_seconds=sum(r['epoch_wall_seconds'] for r in rows),epoch_wall_mean_seconds=statistics.mean(r['epoch_wall_seconds'] for r in rows),
        validation_seconds=sum(r.get('validation_seconds',0) for r in rows))
phase_stats.update(test_evaluation_seconds=t['seconds'],test_eval_images_per_second=25200/t['seconds'],
    engine_outside_epoch_and_test_seconds=s['engine_wall_seconds']-sum(v['epoch_wall_seconds'] for v in phase_stats.values())-t['seconds'],
    wrapper_outside_engine_seconds=s['wall_seconds']-s['engine_wall_seconds'],
    scope='epoch wall includes loading/setup/checks/checkpoint/history; test eval includes input/forward/label collection, excludes final metric/file processing; not same-hardware inference benchmark')
result=dict(schema='hgtnet-offline-analysis-v1',analysis_date='2026-11-04',experiment_date='2026-11-03',
    status='DOWNLOADED_TEST_HISTORY_METADATA_LOCALLY_RECOMPUTED',archive_sha256=sha(HERE/'reported_results.tar.gz'),
    test_sha256=sha(FILES/'test.json'),summary_sha256=sha(FILES/'summary.json'),class_names=names,
    samples=25200,correct=23718,errors=1482,metrics={k:m[k] for k in ['oa_percent','macro_f1_percent']},
    reported_metrics=dict(oa_percent=t['metrics']['accuracy_percent'],macro_f1_percent=t['metrics']['macro_f1_percent']),
    per_class=m['per_class'],confusion_matrix=cm,weakest_recall_classes=weak[:10],strongest_recall_classes=strong[:10],
    directed_confusions_top10=directed[:10],mutual_confusion_pairs_top10=pairs[:10],
    weakest5_share_of_errors_percent=100*sum(r['fn'] for r in weak[:5])/1482,
    top5_pairs_share_of_errors_percent=100*sum(r['total'] for r in pairs[:5])/1482,
    history=dict(calibration_epochs=100,refit_epochs=95,best_epoch=95,max_tied_epochs=[r['epoch'] for r in cal if r['val_correct']==best['val_correct']],
        best_validation={k:best[k] for k in ['epoch','val_correct','val_loss','train_accuracy_percent','train_loss']},
        best_validation_oa=best['val_metrics']['accuracy_percent'],best_validation_macro_f1=best['val_metrics']['macro_f1_percent'],
        final_validation={k:final[k] for k in ['epoch','val_correct','val_loss','train_accuracy_percent','train_loss']},
        final_validation_oa=final['val_metrics']['accuracy_percent'],final_validation_macro_f1=final['val_metrics']['macro_f1_percent'],
        best_minus_final_oa_pp=best['val_metrics']['accuracy_percent']-final['val_metrics']['accuracy_percent'],
        final_refit=refit[-1]),phase_timing=phase_stats,paired_vs_dfagcn=dict(paired),
    audit=dict(checks=len(checks),passed=all(checks.values()),max_abs_metric_error=max(abs(m['oa_percent']-t['metrics']['accuracy_percent']),abs(m['macro_f1_percent']-t['metrics']['macro_f1_percent']))),
    limits=['Original model/checkpoint/optimizer snapshot/manifest/image bytes not returned; hash references and returned metadata do not replace a model-byte audit.',
        'Prediction labels/paths match disclosed binding and directory taxonomy; original full manifest membership/image hashes not independently revalidated.',
        'Validation per-epoch metrics recomputed from returned CMs; validation per-image predictions and hidden training randomness not independently replayed.',
        'No GAViT full predictions in this returned packet; HGT-fixed net4 examples is not paired correction/regression/significance/stability.',
        'Single seed independent whole-method comparison with differing recipes/backbones/hardware; no graph-module causality or original-paper numeric equivalence.',
        'No GPU/training or changes to model/source/recipe in this offline analysis.'])
prefix='hgtnet_nwpu_seed42_'
write_csv(OUT/(prefix+'per_class_20261104.csv'),m['per_class'])
write_csv(OUT/(prefix+'predictions_20261104.csv'),prediction_rows)
write_csv(OUT/(prefix+'errors_20261104.csv'),[r for r in prediction_rows if not r['correct']])
write_csv(OUT/(prefix+'confusions_20261104.csv'),directed)
write_csv(OUT/(prefix+'confusion_pairs_20261104.csv'),pairs)
write_csv(OUT/(prefix+'history_20261104.csv'),epoch_rows)
write_csv(OUT/(prefix+'paired_vs_dfagcn_20261104.csv'),paired_rows)
stream=io.StringIO(newline='');w=csv.writer(stream);w.writerow(['true_class / predicted_class',*names]);w.writerows([[n,*r] for n,r in zip(names,cm)])
write_new(OUT/(prefix+'confusion_matrix_20261104.csv'),stream.getvalue())
write_json(OUT/(prefix+'analysis_20261104.json'),result)
write_json(HERE/'validation.json',dict(checks=checks,count=len(checks),passed=all(checks.values()),
    test_sha256=sha(FILES/'test.json'),metrics_recomputed=result['metrics'],correct=23718,errors=1482))
print(json.dumps({k:result[k] for k in ['status','correct','errors','metrics','weakest_recall_classes','mutual_confusion_pairs_top10','history','phase_timing','paired_vs_dfagcn','audit']},indent=2))
