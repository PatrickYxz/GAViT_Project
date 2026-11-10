from pathlib import Path
import hashlib,json,math,statistics
root=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2');e=root/'results/evidence/nwpu_spatial_calibration_20261010'
raw=(e/'closeout_server_return.txt').read_text();lines=raw.splitlines()
ident=json.loads(lines[0].removeprefix('IDENTITY '));metrics=json.loads(lines[1].removeprefix('METRICS '));paired=json.loads(lines[2].removeprefix('PAIRED '));columns=lines[3].removeprefix('CURVE_COLUMNS ').split(',');curve=[dict(zip(columns,row)) for row in json.loads(lines[4])]
previous=json.loads((e/'server_return.json').read_text())
assert ident['git']=={'commit':'67fe84d','dirty':False}
assert ident['checkpoint_sha256']==previous['checkpoint_sha256']
assert ident['manifest_sha256']=='ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
assert ident['pretrained_sha256']=='f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
assert ident['hardware']['device_name']=='NVIDIA GeForce RTX 2080 Ti' and ident['hardware']['torch_num_threads']==8
assert ident['graph']=={'actual_forward_checked':True,'edges_per_image':80,'feature_edges':32,'spatial_edges':48}
assert metrics['spatial']['OA']==previous['best_val_oa_percent']
assert all(abs(100*v['correct']/1260-v['OA'])<1e-10 and 0<=v['macro_F1']<=100 for v in metrics.values())
for k,c in paired.items():
    assert sum(c.values())==1260
    assert c['both_correct']+c['spatial_breaks']==metrics[k]['correct']
    assert c['both_correct']+c['spatial_corrects']==metrics['spatial']['correct']
    assert c['spatial_corrects']-c['spatial_breaks']==metrics['spatial']['correct']-metrics[k]['correct']
assert [x['epoch'] for x in curve]==list(range(1,31))
assert all(len(x)==6 and all(math.isfinite(v) for v in x.values()) and x['train_s']>0 and x['val_s']>0 and x['train_loss']>=0 for x in curve)
best=max(curve,key=lambda x:x['val_OA']);assert best['epoch']==previous['best_epoch']==24
assert abs(best['val_OA']-metrics['spatial']['OA'])<5e-7 and abs(best['macro_F1']-metrics['spatial']['macro_F1'])<5e-7
assert best['epoch']==max(curve,key=lambda x:x['macro_F1'])['epoch']
train_s=sum(x['train_s'] for x in curve);val_s=sum(x['val_s'] for x in curve)
summary={'date':'2026-10-10','status':'cached_metric_closeout_complete_candidate_closed',
'evidence_scope':'Server returned compact output after supplied readback assertions; local verification checks returned identities/arithmetic/rounded curve, not the original raw predictions or checkpoint',
'final_pass_marker_in_paste':'SPATIAL_CACHED_CLOSEOUT_PASS' in raw,
'marker_note':'Final marker omitted in paste; all substantive output is present, so no repeated command requested',
'identity':ident,'metrics':metrics,'paired':paired,
'deltas_vs_controls':{k:{'OA_pp':metrics['spatial']['OA']-v['OA'],'macro_F1_pp':metrics['spatial']['macro_F1']-v['macro_F1'],'net_correct':metrics['spatial']['correct']-v['correct']} for k,v in metrics.items() if k!='spatial'},
'curve_columns':columns,'curve':curve,'best':best,'final':curve[-1],
'resources_from_rounded_curve':{'train_total_s':train_s,'validation_total_s':val_s,'train_weighted_images_per_second':5040*30/train_s,'train_median_s':statistics.median(x['train_s'] for x in curve),'train_min_s':min(x['train_s'] for x in curve),'train_max_s':max(x['train_s'] for x in curve),'validation_min_s':min(x['val_s'] for x in curve),'validation_max_s':max(x['val_s'] for x in curve),'wall_minus_train_val_s':previous['elapsed_seconds']-train_s-val_s,'caveat':'Two-decimal timing values; remainder includes untimed phases and cannot locate the slowdown cause'},
'confirmed_decision':'Stop this candidate, no additional seeds/strength sweeps/refit/test/BigEarthNet; no further server readback needed for this closeout',
'limitations':['Single training seed on fixed validation split, not test/general stability','Full calibration S/W geometry was not returned and is not required to enforce the frozen stop rule','Per-class identities and prediction files remain on server; full raw backups not created locally','No unique causal explanation for accuracy drop or wall time variation'],
'source_sha256':hashlib.sha256(raw.encode()).hexdigest(),'local_checks_passed':True}
(e/'closeout_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ('status','deltas_vs_controls','best','final','resources_from_rounded_curve','local_checks_passed')},indent=2))
