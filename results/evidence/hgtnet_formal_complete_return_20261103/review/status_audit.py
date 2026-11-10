"""Independent reported-summary acceptance; no remote-model/prediction retrieval."""
import csv,hashlib,json,math,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
s=json.loads((HERE.parent/'status_reported.json').read_text());checks={}
expected=dict(schema='hgtnet-nwpu-formal-v1',state='complete',completed=True,exit_code=0,
    attempt=1,batch_size=16,best_epoch=95,selected_epoch=95,refit_epochs=95,final_epoch=100,
    formal_training_started=True,training_started=True,input_scope='real_nwpu',target_gpu='3090',
    optimizer_steps=68930,counts=dict(train=5040,val=1260,pool=6300,test=25200),
    physical_gpu_uuid='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
    config_sha256='af6ef410125e9d47c87acd0acec1a6db6e0119656793124ceafcb2379eedac25',
    recipe_sha256='ce72e1822d55632b6dd9b02cdb4dc32da7e3e77eedca815b83981ff559c1af85',
    manifest_sha256='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e',
    weights_sha256='68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893')
for key,value in expected.items():checks['summary_'+key]=type(s[key]) is type(value) and s[key]==value
checks['all74619858_parameters_trainable']=s['parameters']['total']==s['parameters']['trainable']==sum(s['parameters']['groups'].values())==74619858
checks['68930_budget']=100*math.ceil(5040/16)+95*math.ceil(6300/16)==s['optimizer_steps']
checks['fresh_initialization_equal']=s['initial_state_sha256']==s['refit_initial_state_sha256']==s['checkpoint_reload']['metadata']['initial_state_sha256']=='2e954f447be44e4a411fdcbf7673a62fe2876337ec555ab041cf96987fdaaf79'
checks['numeric_reload_zero']=s['checkpoint_reload']['loaded'] is True and s['checkpoint_reload']['reload_max_abs_error']==0.0 and s['checkpoint_reload']['reload_tolerance']==1e-6
checks['final_refit_matches_reload']=s['checkpoints']['final_refit']['sha256']==s['checkpoint_reload']['sha256']=='716835556fe989739fc39e2d278188739b4d4454a14b5bf4d8d2b9efe060db1c'
checks['reload_selected95']=s['checkpoint_reload']['metadata']['phase']=='refit' and s['checkpoint_reload']['metadata']['epoch']==s['checkpoint_reload']['metadata']['selected_epoch']==95
checks['checkpoint_reference_hash_formats']=all(bool(re.fullmatch('[0-9a-f]{64}',r['sha256'])) and type(r['bytes']) is int and r['bytes']>0 for r in s['checkpoints'].values())
checks['test_reference_hash_format']=bool(re.fullmatch('[0-9a-f]{64}',s['test_sha256']))
checks['test_reference_path']=s['test_path']==s['output']+'/test.json'
checks['reported_metrics_scopes']=s['test_metrics']['scope']=='formal_test' and s['best_val_metrics']['scope']=='formal_validation'
checks['reported_metrics_finite_in_range']=all(math.isfinite(s[m][k]) and 0<=s[m][k]<=100 for m in ['test_metrics','best_val_metrics'] for k in ['accuracy_percent','macro_f1_percent'])
test_correct=s['test_metrics']['accuracy_percent']*25200/100
val_correct=s['best_val_metrics']['accuracy_percent']*1260/100
checks['23718_correct_test_arithmetic']=math.isclose(test_correct,23718,abs_tol=1e-8) and 25200-23718==1482
checks['1173_correct_val_arithmetic']=math.isclose(val_correct,1173,abs_tol=1e-8)
checks['wall_scope_order']=0<s['engine_wall_seconds']<=s['wall_seconds'] and math.isfinite(s['wall_seconds'])
checks['positive_ordered_allocator_peaks']=0<s['cuda_peak_allocated_bytes']<=s['cuda_peak_reserved_bytes']<25295323136
checks['complete_progress_matches']=s['progress']['stage']=='complete' and s['progress']['completed'] is True and s['progress']['optimizer_steps']==68930 and s['progress']['attempt']==1 and s['progress']['pid']==2661
rows=list(csv.DictReader((ROOT/'results/external_baselines_nwpu_seed42_20261030.csv').open()))
models={r['model']:r for r in rows};names=['Swin-T','GAViT fixed4x4','GAViT local soft','DFAGCN independent v1']
checks['all_baselines_same_manifest_seed_count']=all(models[name]['manifest_sha256']==s['manifest_sha256'] and int(models[name]['seed'])==42 and int(models[name]['test_samples'])==25200 and int(models[name]['train'])==5040 and int(models[name]['val'])==1260 and int(models[name]['refit_pool'])==6300 for name in names)
deltas={name:dict(delta_oa_pp=s['test_metrics']['accuracy_percent']-float(models[name]['oa_percent']),
    delta_macro_f1_pp=s['test_metrics']['macro_f1_percent']-float(models[name]['macro_f1_percent']),
    delta_correct_arithmetic=23718-int(models[name]['correct'])) for name in names}
checks['four_correct_difference_vs_fixed']=deltas['GAViT fixed4x4']['delta_correct_arithmetic']==4 and math.isclose(deltas['GAViT fixed4x4']['delta_oa_pp'],4/25200*100,abs_tol=1e-10)
wall=s['wall_seconds'];hours=int(wall//3600);minutes=int(wall%3600//60);seconds=wall%60
derived=dict(test_correct_arithmetic=23718,test_errors_arithmetic=1482,val_correct_arithmetic=1173,
    wall_hours=wall/3600,wall_hms=[hours,minutes,seconds],
    allocated_gib=s['cuda_peak_allocated_bytes']/2**30,reserved_gib=s['cuda_peak_reserved_bytes']/2**30,
    setup_and_final_wrapper_seconds=wall-s['engine_wall_seconds'],deltas=deltas,
    four_samples_delta_pp=4/25200*100,baseline_rows_at_read=len(rows))
record=dict(schema='hgtnet-formal-status-independent-review-v1',checks=checks,checks_count=len(checks),passed=all(checks.values()),
    status_sha256=hashlib.sha256((HERE.parent/'status_reported.json').read_bytes()).hexdigest(),derived=derived,
    disposition='server-reported real formal completion accepted with matching identities; per-image independent recomputation pending',
    boundaries=['OA correct counts are arithmetic implied by the summary, not independently counted predictions','macroF1 not recomputed from per-image results','300MB model and original remote runtime/checkpoint/test files have not been directly retrieved','single seed and four-correct-image gap do not establish stable superiority, statistical significance or individual graph-module causality','different backbones/pretraining/recipes/budgets/hardware prevent an isolated efficiency or module claim'])
(HERE/'status_audit.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=record['passed'],failed=[k for k,v in checks.items() if not v],derived=derived),indent=2))
if not record['passed']:raise SystemExit(1)
