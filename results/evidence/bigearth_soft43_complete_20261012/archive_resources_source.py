import csv
import hashlib
import io
import json
import math
from pathlib import Path

R=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
E=R/'results/evidence/bigearth_soft43_complete_20261012'
raw=Path('/Users/patrick/.codex/attachments/7337d954-107a-4454-9206-03c3d8eb3021/已粘贴的文本.txt').read_bytes()
(E/'reported_summary_output.txt').write_bytes(raw)
text=raw.decode()
summary,_=json.JSONDecoder().raw_decode(text[text.index('\n{')+1:])
def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
save(E/'resource_summary.json',summary)
before=json.loads((R/'results/bigearth_soft43_test_summary.json').read_text())
pf=json.loads((R/'results/evidence/seed43_bigearth_preflight_20261011/user_return.json').read_text())
train,test,h=summary['train'],summary['test'],summary['history']
assert summary['status']=='complete' and summary['seed']==43 and train['final_epoch']==30
assert all(x=='exit_code=0' for x in summary['stage_exits'].values()) and set(summary['stage_exits'])=={'gavit_train','gavit_test'}
assert train['counts']==dict(train=237871,val=122342) and train['parameters']==31466766
assert train['commit']==pf['soft_commit'] and train['pretrained_sha256']=='f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
assert train['splits']=={k:[v['count'],v['sha256']] for k,v in pf['splits'].items()}
assert train['hardware']==test['hardware']==pf['hardware'] and train['precision']==test['precision']==pf['precision']
assert train['checkpoint_sha256']==test['checkpoint_sha256']==before['test_metrics']['checkpoint_sha256']
assert h['epochs']==list(range(1,31)) and h['first']['epoch']==1 and h['last']['epoch']==30
assert h['last']==before['final_epoch']
assert train['best']['metric']=='val_mAP' and h['best']['epoch']==train['best']['epoch']==8
assert abs(h['best']['val_mAP_percent']-train['best']['value'])<=.000051
assert all(math.isfinite(v) for name in ['first','best','last'] for v in h[name].values())
assert h['best']['val_mAP_percent']>h['last']['val_mAP_percent']
for report in (train,test):
    assert all(math.isfinite(report[k]) and report[k]>0 for k in ['elapsed_seconds','peak_cuda_allocated_bytes','peak_cuda_reserved_bytes','main_process_peak_rss_kib'])
    assert report['peak_cuda_allocated_bytes']<=report['peak_cuda_reserved_bytes']
assert summary['queue_elapsed_seconds']>=train['elapsed_seconds']+test['elapsed_seconds']
assert summary['hourly_yuan']==1.67 and abs(summary['queue_time_estimated_yuan']-summary['queue_elapsed_seconds']*1.67/3600)<1e-12
assert abs(train['effective_train_images_per_second']-237871*30/train['elapsed_seconds'])<1e-9
g=train['graph_capture']
assert g['actual_forward_checked'] and g['images_checked']==8 and g['snapshots_checked']==3
assert (g['edges_per_image'],g['spatial_edges'],g['feature_edges'])==(80,48,32)
assert all(math.isfinite(v) and 0<=v<1e-5 for v in g['normalization_errors'].values())
assert train['probe_summary']['last_epoch']==30 and train['probe_summary']['split']=='val' and train['probe_summary']['samples']==8
resource={}
for name,report in [('train',train),('test',test)]:
    resource[name]=dict(elapsed_seconds=report['elapsed_seconds'],elapsed_hours=report['elapsed_seconds']/3600,
        allocated_peak_gib=report['peak_cuda_allocated_bytes']/2**30,reserved_peak_gib=report['peak_cuda_reserved_bytes']/2**30,
        main_process_rss_peak_gib=report['main_process_peak_rss_kib']/2**20)
nwpu=json.loads((R/'results/nwpu_seed43_test_summary.json').read_text())
cost=dict(hourly_yuan=1.67,bigearth_queue_hours=summary['queue_elapsed_seconds']/3600,
    bigearth_queue_estimated_yuan=summary['queue_time_estimated_yuan'],
    nwpu_queue_elapsed_seconds=nwpu['elapsed_seconds'],
    combined_queue_elapsed_seconds=summary['queue_elapsed_seconds']+nwpu['elapsed_seconds'],
    combined_estimated_yuan=(summary['queue_elapsed_seconds']+nwpu['elapsed_seconds'])*1.67/3600,
    scope='formal queue runtime only, not invoice; excludes extraction/smokes/idle time')
audit=dict(status='identity_history_resources_audit_pass',source='user compact summary return; not original server-file download',
    source_sha256=hashlib.sha256(raw).hexdigest(),
    checks=['complete/seed43/stage exits0/full30 epoch identities','train/test/previous exact metrics weight SHA chain',
        'code/pretrained/split SHA and counts','same approved GPU/software/precision in both stages',
        'best epoch8/full precision best consistent with rounded metadata','finite resources and 8-image three-snapshot graph/probe checks',
        'effective-throughput and runtime-cost arithmetic'],resources=resource,cost=cost,
    queue_minus_measured_stages_seconds=summary['queue_elapsed_seconds']-train['elapsed_seconds']-test['elapsed_seconds'],
    limits=['local return includes epoch identities and first/best/last rows, not full 30-row curve',
        'best selected from full server history by returned command assertions; no local independent best recomputation',
        'checkpoint/predictions/probe tensors not downloaded','timings/memory include IO and artifact checks; RSS excludes loader workers',
        'hardware/recipe/history differences remain in model comparisons; no fair model-only speed claim'])
save(E/'resource_validation.json',audit)
save(E/'seed43_formal_cost.json',cost)
save(E/'test_summary_before_resources.json',before)
current=before.copy()
current.update(best_epoch=8,best_val_mAP_percent=h['best']['val_mAP_percent'],best_val_mAP_metadata_rounded=train['best']['value'],
    total_elapsed_seconds=summary['queue_elapsed_seconds'],train_stage_resources=train,test_stage_resources=test,
    resource_return_sha256=audit['source_sha256'],resources=resource,history_summary=h,cost=cost,
    resource_archival_status='complete')
current['limits']=[x for x in before['limits'] if 'reports still pending' not in x]+audit['limits']
save(R/'results/bigearth_soft43_test_summary.json',current)
# Preserve historical CSV bytes; replace only the last new soft43 row.
p=R/'results/bigearth_comparison.csv';csv_before=p.read_bytes();save_before=E/'comparison_table_before_resource_update.csv';save_before.write_bytes(csv_before)
rows=list(csv.DictReader(io.StringIO(csv_before.decode())))
row=rows[-1];assert row['Model']=='GAViT soft' and row['Seed']=='43'
row.update({'Best_Val_mAP(%)':str(h['best']['val_mAP_percent']),'Best_Epoch':'8',
    'Peak_GPU_Memory(GiB)':str(resource['train']['allocated_peak_gib']),
    'Train_Wall_Time(h)':str(resource['train']['elapsed_hours']),
    'Test_Time':str(test['elapsed_seconds'])+' s (whole measured stage)',
    'Notes':'Formal soft43 full30 and119825 test complete; queue/train/test exits0. Best epoch8/full-history val mAP78.99004691745016; metadata rounded78.9900. Precise metrics/19AP and checkpoint hash chain agree. Current4090 GPU UUID unchanged; code/pretraining/splits/software/precision verified. Queue22941.731309s; train22679.135110s/test197.353206s; train allocated3.662695GiB/reserved3.943359GiB, test allocated0.595415GiB. Queue runtime cost at1.67yuan/h10.642414yuan, not bill incl recovery/idle. Peak/timing include artifact checks; RSS excludes loader workers; effective314.656images/s includes validation/IO, not pure model speed. 8val initial/best/final probes normalized/48+32 edges; full30 epoch IDs but only first/best/last rows returned locally. Old43 rounded controls/fixed saved21/code/hardware confounds retained. Evidence results/evidence/bigearth_soft43_complete_20261012/reported_summary_output.txt'})
buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]),lineterminator='\n');writer.writerow(row)
lines=csv_before.splitlines(keepends=True);assert b'Formal soft43 full30' in lines[-1]
p.write_bytes(b''.join(lines[:-1])+buffer.getvalue().encode())
assert p.read_bytes().startswith(b''.join(lines[:-1]))
# Keep the two-seed metrics and statistics unchanged, only resolve their obsolete pending note.
p=R/'results/bigearth_three_model_two_seed_summary.json';x=json.loads(p.read_text());x['limits']=[t for t in x['limits'] if 'reports still pending' not in t]
x['soft43_resources']='bigearth_soft43_test_summary.json';save(p,x)
(E/'archive_resources_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(resources=resource,cost=cost,best=h['best'],queue_minus_stage_seconds=audit['queue_minus_measured_stages_seconds']),indent=2))
