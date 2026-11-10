"""Accept reported formal result identities and update the disclosed seed42 table."""
import csv,hashlib,io,json,math,re,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];sys.path.insert(0,str(ROOT))
from tools.package_hgtnet_nwpu_smoke import _read_bundle
from external_baselines.hgtnet.formal_protocol import RECIPE,recipe_sha256,IDENTITIES
s=json.loads((HERE/'status_reported.json').read_text());checks={}
expected=dict(schema='hgtnet-nwpu-formal-v1',state='complete',completed=True,exit_code=0,attempt=1,
    batch_size=16,best_epoch=95,selected_epoch=95,refit_epochs=95,final_epoch=100,
    input_scope='real_nwpu',formal_training_started=True,training_started=True,target_gpu='3090',
    counts=dict(train=5040,val=1260,pool=6300,test=25200),
    physical_gpu_uuid='GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',**IDENTITIES,
    recipe_sha256=recipe_sha256(RECIPE))
for key,value in expected.items():checks[key]=type(s[key]) is type(value) and s[key]==value
checks['budget_steps68930']=s['optimizer_steps']==100*315+95*394==68930
checks['parameter_groups']=s['parameters']['groups']==dict(dual_attention=44077,graph=14066432,head=23085,patch_embed=6528,transformer=60479736)
checks['74619858_trainable_parameters']=sum(s['parameters']['groups'].values())==s['parameters']['total']==s['parameters']['trainable']==74619858
checks['equal_fresh_initialization']=s['initial_state_sha256']==s['refit_initial_state_sha256']=='2e954f447be44e4a411fdcbf7673a62fe2876337ec555ab041cf96987fdaaf79'
reload=s['checkpoint_reload'];checks['numeric_reload_zero']=reload['loaded'] is True and reload['reload_max_abs_error']==0.0 and reload['reload_tolerance']==1e-6
checks['reload_metadata']=reload['metadata']==dict(epoch=95,selected_epoch=95,phase='refit',initial_state_sha256=s['initial_state_sha256'])
checks['final_refit_reload_same_sha']=reload['sha256']==s['checkpoints']['final_refit']['sha256']=='716835556fe989739fc39e2d278188739b4d4454a14b5bf4d8d2b9efe060db1c'
checks['metrics_scopes']=s['test_metrics']['scope']=='formal_test' and s['best_val_metrics']['scope']=='formal_validation'
checks['metric_ranges']=all(type(m[k]) in (int,float) and math.isfinite(m[k]) and 0<=m[k]<=100 for m in [s['test_metrics'],s['best_val_metrics']] for k in ['accuracy_percent','macro_f1_percent'])
test_correct=s['test_metrics']['accuracy_percent']*25200/100;val_correct=s['best_val_metrics']['accuracy_percent']*1260/100
checks['test_correct23718_arithmetic']=math.isclose(test_correct,23718,abs_tol=1e-8)
checks['val_correct1173_arithmetic']=math.isclose(val_correct,1173,abs_tol=1e-8)
checks['current_attempt_wall']=s['wall_seconds']>s['engine_wall_seconds']>0 and all(math.isfinite(s[k]) for k in ['wall_seconds','engine_wall_seconds'])
checks['allocator_peak_order']=all(type(s[k]) is int for k in ['cuda_peak_allocated_bytes','cuda_peak_reserved_bytes']) and 0<s['cuda_peak_allocated_bytes']<=s['cuda_peak_reserved_bytes']<24576*2**20
checks['test_identity_reference']=re.fullmatch('[0-9a-f]{64}',s['test_sha256']) is not None and s['test_sha256']=='869986974e4801f2234fd2fa3bace383af2f092fd20359fbe1a252d4af7e37ae'
checks['test_path']=s['test_path']==s['output']+'/test.json'
for name,r in s['checkpoints'].items():checks['checkpoint_reference:'+name]=r['path']==s['output']+'/'+name+'.pt' and type(r['bytes']) is int and r['bytes']>0 and re.fullmatch('[0-9a-f]{64}',r['sha256']) is not None
checks['complete_progress']=s['progress']==dict(attempt=1,completed=True,formal_training_started=True,optimizer_steps=68930,pid=2661,stage='complete',training_started=True)
bundle=ROOT/'output/hgtnet_nwpu_formal_3090_20261103_v1.pyz'
checks['local_v1_sha']=hashlib.sha256(bundle.read_bytes()).hexdigest()=='b2bc9077cf37e75c83fbb72e7c4047d2fb9ede04db8fb4e4a56344500ff572e1'
manifest,_,payload=_read_bundle(bundle)
for name in ['formal_entry.py','formal_engine.py','formal_protocol.py']:
    p='external_baselines/hgtnet/'+name;checks['current_source:'+name]=payload[p]==(ROOT/p).read_bytes()
derived=dict(test_correct_arithmetic=23718,test_errors_arithmetic=1482,val_correct_arithmetic=1173,val_errors_arithmetic=87,
    allocated_gib=s['cuda_peak_allocated_bytes']/2**30,reserved_gib=s['cuda_peak_reserved_bytes']/2**30,
    wall_hours=s['wall_seconds']/3600,wall_hms='6h40m32.617s',engine_wall_hours=s['engine_wall_seconds']/3600,
    wrapper_wall_minus_engine_seconds=s['wall_seconds']-s['engine_wall_seconds'])
audit=dict(checks=checks,count=len(checks),passed=all(checks.values()),derived=derived,
    source='user-returned status, normalized locally; no direct server files retrieved',
    normalized_status_sha256=hashlib.sha256((HERE/'status_reported.json').read_bytes()).hexdigest())
(HERE/'validation.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
if not audit['passed']:raise SystemExit('result validation failed: '+str([k for k,v in checks.items() if not v]))
table=ROOT/'results/external_baselines_nwpu_seed42_20261030.csv';before=table.read_bytes()
rows=list(csv.DictReader(io.StringIO(before.decode())));fields=list(rows[0])
old=[r for r in rows if not r['model'].startswith('HGTNet')]
backup=HERE/'comparison_before_hgtnet.csv'
if not backup.exists():backup.write_bytes(before)
comparisons=[]
for row in old:
    comparisons.append(dict(model=row['model'],oa_percent=float(row['oa_percent']),macro_f1_percent=float(row['macro_f1_percent']),
        correct=int(row['correct']),hgt_minus_model_oa_pp=s['test_metrics']['accuracy_percent']-float(row['oa_percent']),
        hgt_minus_model_macro_f1_pp=s['test_metrics']['macro_f1_percent']-float(row['macro_f1_percent']),
        hgt_minus_model_net_correct_arithmetic=23718-int(row['correct'])))
swin=next(r for r in old if r['model']=='Swin-T');fixed=next(r for r in old if r['model']=='GAViT fixed4x4')
derived.update(hgt_params_over_fixed_ratio=74619858/int(fixed['neural_parameters']),fixed_fewer_parameters_percent=100*(1-int(fixed['neural_parameters'])/74619858))
new=dict(model='HGTNet independent v1',source_type='independent_reimplementation',dataset='NWPU-RESISC45',seed='42',
    manifest_sha256=s['manifest_sha256'],train='5040',val='1260',refit_pool='6300',test_samples='25200',
    oa_percent=str(s['test_metrics']['accuracy_percent']),macro_f1_percent=str(s['test_metrics']['macro_f1_percent']),correct='23718',
    correct_basis='arithmetic_from_server_reported_OA_predictions_pending',refit_epochs='95',gpu='RTX 3090 24GiB',
    backbone='partial torchvision Swin-B initialization',neural_parameters='74619858',parameter_scope='full_neural_model_all_trainable',
    delta_oa_vs_swin_pp=str(s['test_metrics']['accuracy_percent']-float(swin['oa_percent'])),
    delta_macro_f1_vs_swin_pp=str(s['test_metrics']['macro_f1_percent']-float(swin['macro_f1_percent'])),
    evidence='results/evidence/hgtnet_formal_complete_return_20261103/status_reported.json')
existing=[r for r in rows if r['model'].startswith('HGTNet')]
if existing:
    if existing!=[new]:raise ValueError('Existing HGTNet row differs; do not overwrite')
else:
    stream=io.StringIO(newline='');csv.DictWriter(stream,fieldnames=fields,lineterminator='\r\n').writerow(new)
    with table.open('ab') as f:f.write(stream.getvalue().encode())
    assert table.read_bytes().startswith(before),'previous rows changed'
record=dict(schema='hgtnet-formal-result-record-v1',status='SERVER_REPORTED_FORMAL_COMPLETE_IDENTITIES_MATCH_PREDICTIONS_PENDING',
    experiment_date='2026-11-03',source_type='independent_reimplementation',dataset='NWPU-RESISC45',seed=42,
    reported_status='results/evidence/hgtnet_formal_complete_return_20261103/status_reported.json',
    metrics=dict(oa_percent=s['test_metrics']['accuracy_percent'],macro_f1_percent=s['test_metrics']['macro_f1_percent']),
    best_epoch=95,final_epoch=100,refit_epochs=95,counts=s['counts'],status_reported=s,derived=derived,
    comparison_seed42=comparisons,parameters=s['parameters'],bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
    prediction_audit=dict(status='pending_returned_file',server_path=s['test_path'],expected_sha256=s['test_sha256']),
    limitations=['Metrics are server-reported; correct counts arithmetic from OA, macroF1 not independently recomputed from predictions.',
        'Checkpoint/model/test record/source/runtime/history original server bytes not retrieved; hashes are identity references.',
        'Single seed independent whole-method comparison; different backbones/pretraining/augmentation/optimizer/epochs/hardware.',
        'Four net correct examples vs fixed4x4 do not establish significance/stability or paired corrections/regressions.',
        'Current-attempt program wall excludes preflight and earlier rental idle time; no actual rental cost supplied.'])
(ROOT/'results/hgtnet_nwpu_seed42_20261103.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(checks=len(checks),passed=True,derived=derived,comparisons=comparisons),indent=2))
