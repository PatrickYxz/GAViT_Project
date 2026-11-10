"""Independent stdlib history, selection, runtime and source-identity audit.

Read user files as data only. No model/image loading, extraction or execution.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
SOURCE=Path('/Users/patrick/Downloads/hgtnet_seed42_results_20261103.tar.tar')
EXPECTED={'test.json','history.json','summary.json','binding.json','initialization.json',
          'checkpoint_reload.json','attempts/0001/runtime.json'}
checks={}
def check(name,value):
    checks[name]=bool(value)
    if not value:raise AssertionError(name)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def close(a,b):return math.isclose(a,b,rel_tol=0,abs_tol=1e-10)

compressed=SOURCE.read_bytes()
check('archive_sha256',sha(compressed)=='4fc4d51cec3aac5900ac94cdde919818ba58cd0f76add625867623f54d052477')
with tarfile.open(SOURCE,'r:*') as archive:
    members=archive.getmembers()
    check('seven_unique_expected_regular_members',len(members)==7 and {m.name for m in members}==EXPECTED
          and all(m.isfile() and not m.issym() and not m.islnk() for m in members))
    check('archive_paths_and_sizes',sum(m.size for m in members)<20*2**20 and all(
          not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members))
    raw={m.name:archive.extractfile(m).read() for m in members}
data={name:json.loads(value) for name,value in raw.items()}
s=data['summary.json'];h=data['history.json'];b=data['binding.json'];t=data['test.json']
runtime=data['attempts/0001/runtime.json'];init=data['initialization.json'];reload=data['checkpoint_reload.json']
names=b['class_names'];cal=h['calibration'];refit=h['refit']
check('summary_complete_real_single_attempt',s['completed'] is True and s['formal_training_started'] is True
      and s['input_scope']=='real_nwpu' and s['attempt']==1 and s['resumed_from'] is None and s['pretraining_factory_calls']==3)
check('history_summary_exact',h['input_scope']=='real_nwpu' and cal==s['calibration_history'] and refit==s['refit_history'])
check('calibration100_refit95',len(cal)==100 and len(refit)==95 and [r['epoch'] for r in cal]==list(range(1,101))
      and [r['epoch'] for r in refit]==list(range(1,96)))
check('counts_45class_mapping',s['counts']==b['counts']==dict(train=5040,val=1260,pool=6300,test=25200)
      and len(names)==45 and names==sorted(names) and len(set(names))==45)
check('update_counts_each_epoch',all(r['epoch_optimizer_steps']==315 and r['optimizer_steps']==315*r['epoch'] for r in cal)
      and all(r['epoch_optimizer_steps']==394 and r['optimizer_steps']==31500+394*r['epoch'] for r in refit)
      and s['optimizer_steps']==68930)
all_epochs=cal+refit
check('all195_actual_wall_and_precommit',all('epoch_wall_seconds' in r and 'epoch_precommit_seconds' in r for r in all_epochs))
check('195_finite_train_and_timing',all(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0
      for k in ['train_loss','train_seconds','train_images_per_second','epoch_precommit_seconds','epoch_wall_seconds'])
      and 0<=r['train_accuracy_percent']<=100 for r in all_epochs))
check('195_throughput_arithmetic',all(close(r['train_images_per_second'],n/r['train_seconds'])
      for rows,n in [(cal,5040),(refit,6300)] for r in rows))
check('195_clock_nesting',all(r['epoch_wall_seconds']>=r['epoch_precommit_seconds']>=r['train_seconds'] for r in all_epochs)
      and all(r['epoch_precommit_seconds']>=r['train_seconds']+r['validation_seconds'] for r in cal))
check('195_online_train_accuracy_integer',all(close(r['train_accuracy_percent']*n/100,round(r['train_accuracy_percent']*n/100))
      for rows,n in [(cal,5040),(refit,6300)] for r in rows))

metric_details=[]
for row in cal:
    m=row['val_metrics'];cm=m['confusion_matrix'];pc=m['per_class']
    check('val_epoch%d_matrix_dimensions'%row['epoch'],len(cm)==45 and all(len(line)==45 for line in cm))
    check('val_epoch%d_support_and_values'%row['epoch'],all(type(v) is int and v>=0 for line in cm for v in line)
          and all(sum(line)==28 for line in cm))
    true=[sum(line) for line in cm];pred=[sum(cm[i][j] for i in range(45)) for j in range(45)]
    correct=sum(cm[i][i] for i in range(45));f1=[];recall=[]
    for i in range(45):
        precision=cm[i][i]/pred[i] if pred[i] else 0.;rec=cm[i][i]/true[i]
        f1.append(100*2*precision*rec/(precision+rec) if precision+rec else 0.)
        recall.append(100*rec)
    macro=statistics.mean(f1);oa=100*correct/1260
    check('val_epoch%d_perclass_metrics'%row['epoch'],len(pc)==45 and all(
          c['class']==names[i] and c['support']==true[i] and c['correct']==cm[i][i]
          and close(c['accuracy_percent'],recall[i]) and close(c['f1_percent'],f1[i]) for i,c in enumerate(pc)))
    check('val_epoch%d_global_metrics'%row['epoch'],row['val_correct']==correct and m['scope']=='formal_validation'
          and close(m['accuracy_percent'],oa) and close(m['macro_f1_percent'],macro)
          and math.isfinite(row['val_loss']) and row['val_loss']>0 and row['validation_seconds']>0)
    metric_details.append(dict(epoch=row['epoch'],correct=correct,oa=oa,macro_f1_recomputed=macro,
        max_abs_error=max(abs(m['accuracy_percent']-oa),abs(m['macro_f1_percent']-macro))))

maximum=max(r['val_correct'] for r in cal);winners=[r['epoch'] for r in cal if r['val_correct']==maximum]
best=cal[min(winners)-1];last=cal[-1]
check('earliest_validation_max95',winners==[95] and s['best_epoch']==s['selected_epoch']==s['refit_epochs']==min(winners)==95)
check('best_metrics_are_epoch95',s['best_val_metrics']==best['val_metrics'] and s['final_epoch']==100)
check('fresh_equal_initialization',init['equal'] is True and init['input_scope']=='real_nwpu'
      and init['calibration_state_sha256']==init['refit_state_sha256']==s['initial_state_sha256']==s['refit_initial_state_sha256']
      =='2e954f447be44e4a411fdcbf7673a62fe2876337ec555ab041cf96987fdaaf79')
check('reload_record_summary_equal',reload==s['checkpoint_reload'] and reload['loaded'] is True and reload['reload_max_abs_error']==0.0)
check('reload_refit_identity',reload['metadata']==dict(epoch=95,selected_epoch=95,phase='refit',initial_state_sha256=s['initial_state_sha256'])
      and reload['sha256']==s['checkpoints']['final_refit']['sha256']==t['refit_checkpoint_sha256'])
check('returned_test_raw_sha',sha(raw['test.json'])==s['test_sha256']=='869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae')
check('summary_test_packet_agrees',s['test']['rows']==25200 and s['test']['sha256']==s['test_sha256']
      and s['test']['metrics']==s['test_metrics']==t['metrics'] and s['test']['loss']==t['loss']
      and s['test']['seconds']==t['seconds'] and t['selected_epoch']==95)
for split,size,support in [('val',1260,28),('test',25200,560)]:
    rows=b['records'][split];paths=[r['path'] for r in rows]
    check(split+'_binding_complete_sorted_unique',len(rows)==size and len(set(paths))==size and paths==sorted(paths))
    check(split+'_binding_label_directory_support',all(type(r['label']) is int and 0<=r['label']<45
          and PurePosixPath(r['path']).parts[0]==names[r['label']] for r in rows)
          and Counter(r['label'] for r in rows)==dict.fromkeys(range(45),support))
check('val_test_disjoint',not(set(r['path'] for r in b['records']['val']) & set(r['path'] for r in b['records']['test'])))
check('test_order_labels_match_binding',t['paths']==[r['path'] for r in b['records']['test']]
      and t['labels']==[r['label'] for r in b['records']['test']])

bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
check('actual_delivered_bundle_sha',sha(bundle.read_bytes())=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1')
with zipfile.ZipFile(bundle) as archive:
    manifest=json.loads(archive.read('delivery_manifest.json'))
    for name,identity in manifest['files'].items():
        value=archive.read(name)
        check('payload_sha:'+name,sha(value)==identity['sha256'] and len(value)==identity['bytes'])
    expected_source={('formal_recipe_bytes' if k=='configs/hgtnet_nwpu_formal_v1.json' else k):v['sha256']
        for k,v in manifest['files'].items() if k!='bundled_weights_provenance.json'}
    check('all_returned_source_ids_match_shipped_payload',b['source_identity']==expected_source)
    cfg=json.loads(archive.read('configs/hgtnet_independent_v0.json'))
    recipe=json.loads(archive.read('configs/hgtnet_nwpu_formal_v1.json'))
    weights=json.loads(archive.read('bundled_weights_provenance.json'))
    check('config_canonical_sha',sha(canonical(cfg))==b['config_sha256']==s['config_sha256'])
    check('recipe_exact_canonical',s['recipe']==b['recipe']==recipe and sha(canonical(recipe))==s['recipe_sha256']==b['recipe_sha256']==t['recipe_sha256'])
    check('candidate_reference_sha',weights['sha256']==s['weights_sha256']==b['weights_sha256']=='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
check('manifest_reference_sha',s['manifest_sha256']==b['manifest_sha256']==t['manifest_sha256']=='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e')
check('actual_runtime_file_equals_summary',runtime==s['runtime'])
versions=dict(torch='2.2.2+cu121',torchvision='0.17.2+cu121',numpy='1.26.4',pillow='10.3.0',cuda='12.1',cudnn=8902)
check('actual_runtime_versions',runtime['python'].startswith('3.11.8 ') and all(runtime[k]==v for k,v in versions.items()))
flags=dict(amp=False,cpu_threads=8,cudnn_benchmark=False,cudnn_deterministic=True,cudnn_tf32=False,
           device='cuda:0',dtype='float32',matmul_tf32=False,seed=42,pid=2661)
check('actual_runtime_flags_types',all(type(runtime[k]) is type(v) and runtime[k]==v for k,v in flags.items()))
check('runtime_physical_uuid',runtime['physical_gpu_uuid']==runtime['cuda_visible_devices']==runtime['gpu']['uuid']==s['physical_gpu_uuid']
      =='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1' and runtime['gpu']['name']=='NVIDIA GeForce RTX 3090'
      and runtime['gpu']['memory_total_mib']==24576 and runtime['gpu']['driver_version']=='610.57.04')
d=runtime['cuda_device']
check('runtime_cuda_capacity_consistency',d['available'] is True and d['device_count']==1 and d['name']==runtime['gpu']['name']
      and d['total_memory_bytes']==25295323136 and d['cuda_total_mib']==d['total_memory_bytes']//2**20
      and d['nvml_total_mib']==24576 and d['interface_gap_bytes']==24576*2**20-d['total_memory_bytes']
      and d['interface_gap_mib']==d['interface_gap_bytes']/2**20==452.5)
previous=json.loads((ROOT/'results/evidence/hgtnet_formal_complete_return_20261103/status_reported.json').read_text())
for key,value in previous.items():
    if key not in s:continue
    if key in ['best_val_metrics','test_metrics']:
        check('prior_status:'+key,all(s[key][k]==v for k,v in value.items()))
    else:check('prior_status:'+key,s[key]==value)

phase=dict(calibration_train_seconds=sum(r['train_seconds'] for r in cal),calibration_validation_seconds=sum(r['validation_seconds'] for r in cal),
           calibration_epoch_wall_seconds=sum(r['epoch_wall_seconds'] for r in cal),refit_train_seconds=sum(r['train_seconds'] for r in refit),
           refit_epoch_wall_seconds=sum(r['epoch_wall_seconds'] for r in refit),test_seconds=t['seconds'])
phase['epoch_wall_plus_test_seconds']=phase['calibration_epoch_wall_seconds']+phase['refit_epoch_wall_seconds']+phase['test_seconds']
phase['engine_other_seconds']=s['engine_wall_seconds']-phase['epoch_wall_plus_test_seconds']
phase['wrapper_other_seconds']=s['wall_seconds']-s['engine_wall_seconds']
check('phase_times_fit_engine_and_wall',phase['engine_other_seconds']>=0 and phase['wrapper_other_seconds']>=0)
phase['calibration_training_weighted_images_per_second']=504000/phase['calibration_train_seconds']
phase['refit_training_weighted_images_per_second']=598500/phase['refit_train_seconds']
phase['calibration_epoch_wall_mean_seconds']=statistics.mean(r['epoch_wall_seconds'] for r in cal)
phase['refit_epoch_wall_mean_seconds']=statistics.mean(r['epoch_wall_seconds'] for r in refit)
result=dict(schema='independent-hgtnet-history-source-audit-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),
    archive_sha256=sha(compressed),member_sha256={k:sha(v) for k,v in raw.items()},
    history=dict(calibration_epochs=100,refit_epochs=95,optimizer_steps=68930,best_epochs=winners,
        best_epoch=95,best_val_oa_percent=best['val_metrics']['accuracy_percent'],best_val_macro_f1_percent=best['val_metrics']['macro_f1_percent'],
        best_val_loss=best['val_loss'],final_epoch=100,final_val_oa_percent=last['val_metrics']['accuracy_percent'],
        final_val_macro_f1_percent=last['val_metrics']['macro_f1_percent'],final_val_loss=last['val_loss'],
        best_minus_final_oa_pp=best['val_metrics']['accuracy_percent']-last['val_metrics']['accuracy_percent'],
        best_minus_final_macro_f1_pp=best['val_metrics']['macro_f1_percent']-last['val_metrics']['macro_f1_percent'],
        best_minus_final_correct=best['val_correct']-last['val_correct'],
        refit_final_online_train_accuracy_percent=refit[-1]['train_accuracy_percent'],
        validation_matrix_epochs_recomputed=100,all195_wall_present=True,
        max_val_metric_abs_error=max(r['max_abs_error'] for r in metric_details)),
    validation_epoch_metric_recomputations=metric_details,timing=phase,runtime=runtime,
    limits=['All seven returned JSON raw bytes were inspected, but no trained model tensor/checkpoint bytes, original NWPU images or complete original manifest were returned.',
        'Each validation confusion matrix/per-class/global metric is internally recomputed; individual per-epoch validation predictions are not in this export.',
        'Initial state equality and numerical checkpoint reload are verified consistent returned records, not a locally repeated model computation.',
        'Program wall and epoch sums are actual returned durations; rental fees and idle/preflight time are unavailable. GPU bitwise deterministic resume is not established.',
        'Best-versus-final validation change is descriptive; this audit does not identify its cause.'])
(OUT/'validation.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps(dict(checks=result['checks_count'],passed=result['passed'],history=result['history'],timing=phase),indent=2))
