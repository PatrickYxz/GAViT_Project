"""Archive returned BEN soft42 evidence and refresh descriptive reporting data."""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP, localcontext
import copy
import csv
import hashlib
import importlib.util
import json
import math
import re

ROOT = Path('/Users/patrick/Projects/GAViT_Project')
WT = ROOT / '.worktrees/sparse-hybrid-4n-top2'
E = Path(__file__).resolve().parent
SOURCE = Path('/Users/patrick/.codex/attachments/e3d92fce-64b6-4fb9-ae0a-e565a01ba532/已粘贴的文本.txt')
DATE = '2026-10-17'


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dec(value):
    return Decimal(str(value))


def fmt(value, signed=False):
    d = dec(value).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return format(d, '+.2f' if signed else '.2f')


def stats(values):
    with localcontext() as ctx:
        ctx.prec = 40
        a = [dec(x) for x in values]
        mean = sum(a) / len(a)
        sd = (sum((x - mean) ** 2 for x in a) / (len(a) - 1)).sqrt() if len(a) > 1 else None
        return dict(n=len(a), mean=float(mean), sample_sd_ddof1=float(sd) if sd is not None else None,
                    mean_decimal=str(mean), sample_sd_decimal=str(sd) if sd is not None else None)


def three_model_summary(rows, selected=False):
    result = dict(dataset='BEN', seeds=[r['seed'] for r in rows])
    if selected:
        result.update(n=len(rows), requested_seeds=[42,43,46,47], missing_soft_seeds=[])
    for model in ('swin','fixed','soft'):
        result[model] = stats([r[model] for r in rows])
    for ref in ('swin','fixed'):
        result['paired_soft_minus_'+ref+'_pp'] = stats([dec(r['soft'])-dec(r[ref]) for r in rows])
    result['scope'] = 'Same-seed descriptive comparison; historical code/hardware differences and reported46/47 evidence limits retained'
    return result


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, WT/'tools'/(name+'.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


raw = SOURCE.read_bytes()
text = raw.decode()
start = text.index('\n{') + 1
summary, length = json.JSONDecoder().raw_decode(text[start:])
log_tail = text[start+length:]
exits = re.findall(r'^exit_code=(\d+)\s*$', text[:start], flags=re.M)
checks = []


def check(ok, label):
    if not ok:
        raise AssertionError(label)
    checks.append(label)


worker = load_tool('run_bigearth_soft42')
base = load_tool('run_bigearth_seed45')
ready = read(ROOT/'results/evidence/bigearth_soft42_launch_20261016/preflight.json')
check(exits == ['0','0','0'], 'Queue/train/test exit markers all0 in requested order')
check(summary['status']=='complete' and summary['dataset']=='BigEarthNet-19' and summary['seed']==42, 'Complete formal BEN42 identity')
check(summary['controls']==ready['controls'], 'Old42 control identities unchanged')
check(set(summary['stages'])=={'gavit_train','gavit_test'}, 'Both formal stages present')
train, test = summary['stages']['gavit_train'], summary['stages']['gavit_test']
for stage, report in summary['stages'].items():
    worker.check_report(report, stage, ready, base)
    check(True, stage+' passes frozen report validator')
    check(report['launcher_sha256']==sha(WT/'tools/run_bigearth_soft42.py'), stage+' uses the repaired worker SHA')
worker.check_metadata(train['metadata'])
check(train['final_epoch']==30 and train['counts']=={'train':237871,'val':122342}, 'Full30 and original train/validation counts')
check(train['best']==train['metadata']['best'] and train['best']['epoch']==10, 'Best validation checkpoint epoch10')
weight = train['checkpoint_sha256']
m = test['metrics']
check(weight==train['metadata']['checkpoint_sha256']==test['checkpoint_sha256']==m['checkpoint_sha256'], 'Train/metadata/test/metrics checkpoint SHA chain agrees')
check(train['checkpoint']==test['checkpoint']==train['metadata']['checkpoint_path'], 'Test evaluates trained best checkpoint path')
check(train['hardware']==test['hardware']==ready['hardware'] and train['precision']==test['precision']==ready['precision'], 'Same4090 UUID/software/precision across formal stages and preflight')
check(train['pretrained_sha256']==base.PRETRAINED_SHA, 'Frozen ImageNet weight identity')
probe = train['graph_capture']
check(probe['actual_forward_checked'] and probe['images_checked']==8 and probe['snapshots_checked']==3, 'Reported initial/best/final eight-image graph probes checked')
check((probe['edges_per_image'],probe['spatial_edges'],probe['feature_edges'])==(80,48,32), 'Actual48+32 graph capture')
check(all(0<=v<1e-5 for v in probe['normalization_errors'].values()), 'S/W normalization within tolerance')
check(train['probe_summary']['last_epoch']==30 and train['probe_summary']['split']=='val', 'Final30 validation probe receipt')
epoch_line = next(line for line in log_tail.splitlines() if line.startswith('EPOCH RESOURCE '))
last = json.loads(epoch_line[len('EPOCH RESOURCE '):])
check(last['epoch']==30 and last['next_epoch_lr']==0, 'Log confirms final30 and completed cosine schedule')
check(all(tag in log_tail for tag in ('STAGE COMPLETE gavit_train','STAGE COMPLETE gavit_test','BIGEARTH SOFT42 COMPLETE /home/featurize/work/bigearth_soft42_20261013_01')), 'Both stages and queue completion markers')
check('Test samples: 119,825' in log_tail and '3745/3745' in log_tail, 'Full test count and last test batch in log')
printed = re.findall(r'^\s+(\d+\.\d)%\s+(.+)$', log_tail, flags=re.M)
check(len(printed)==19 and set(k for _,k in printed)==set(m['per_class_ap_percent']), '19 printed classes match exact metric keys')
check(all(abs(float(v)-m['per_class_ap_percent'][k])<=.050000001 for v,k in printed), 'All19 class log values agree within printing precision')
check(abs(sum(m['per_class_ap_percent'].values())/19-m['mAP_percent'])<1e-8, 'mAP equals mean of all19 exact AP values')
check(summary['elapsed_seconds']>=train['elapsed_seconds']+test['elapsed_seconds']>0, 'Queue and stage timing reconcile')
check(abs(train['effective_train_images_per_second']-237871*30/train['elapsed_seconds'])<1e-8, 'Reported effective throughput arithmetic')

(E/'before').mkdir(exist_ok=False)
targets = [ROOT/'results/supervisor_progress_data_20261012.json', ROOT/'results/ben_seed42_seed43_vs_swin_20261013.json',
           ROOT/'results/bigearth_soft42_status_20261013.json', ROOT/'docs/ben_seed42_seed43_vs_swin_20261013.md',
           ROOT/'docs/supervisor_progress_report_20261012.md', ROOT/'docs/supervisor_email_draft_20261012.md',
           ROOT/'docs/run_ben_soft42_20261013.md', WT/'results/bigearth_comparison.csv']
for p in targets:
    (E/'before'/p.name).write_bytes(p.read_bytes())
(E/'server_return.txt').write_bytes(raw)
save(E/'summary.json', summary)
save(E/'final_epoch_from_log.json', last)
(E/'log_tail.txt').write_text(log_tail)

progress_path = ROOT/'results/supervisor_progress_data_20261012.json'
progress = read(progress_path)
for collection in ('rows','selected_four_seed_rows'):
    row = next(r for r in progress[collection] if r['dataset']=='BEN' and r['seed']==42)
    check(row['soft'] is None, collection+' originally has missing soft42')
    row.update(soft=m['mAP_percent'], soft_macro_f1=m['macro_f1_percent'], soft_micro_f1=m['micro_f1_percent'],
               soft_evidence='results/evidence/bigearth_soft42_complete_20261017/summary.json',
               soft_status='formal_complete_exit0_full30_test119825', soft_gpu_identity=train['hardware']['gpu_identity'])
selected = [r for r in progress['selected_four_seed_rows'] if r['dataset']=='BEN']
check([r['seed'] for r in selected]==[42,43,46,47], 'Requested four-seed cohort preserved')
new_selected = three_model_summary(selected, selected=True)
new_all = three_model_summary([r for r in progress['rows'] if r['dataset']=='BEN' and r.get('soft') is not None])
for key,new in (('selected_soft_same_seed_summaries',new_selected),('soft_three_model_same_seed_summaries',new_all)):
    progress[key] = [new if x['dataset']=='BEN' else x for x in progress[key]]
progress['reporting_selection']['missing_soft_results']['BEN'] = []
progress['last_updated'] = DATE
progress['ben_soft42_complete_since'] = DATE
progress['sources'].append(dict(path=str(E/'summary.json'),sha256=sha(E/'summary.json')))
save(progress_path, progress)

per_path = ROOT/'results/ben_seed42_seed43_vs_swin_20261013.json'
per = read(per_path)
reference = per['per_seed']['42']['fixed']['all19_rows']
check(set(r['class'] for r in reference)==set(m['per_class_ap_percent']), 'Swin42 per-class labels match formal soft42')
rows=[]
for ref in reference:
    delta = dec(m['per_class_ap_percent'][ref['class']])-dec(ref['swin_AP_percent'])
    lower, upper = delta-Decimal('.05'), delta+Decimal('.05')
    rows.append(dict(dataset='BigEarthNet-19',seed=42,**{'class':ref['class']},class_zh=ref['class_zh'],
        reference_model='Swin-T',candidate_model='soft',metric='test_AP_percent',swin_AP_percent=ref['swin_AP_percent'],
        candidate_AP_percent=m['per_class_ap_percent'][ref['class']],delta_AP_pp_approx=float(delta),delta_decimal=str(delta),
        rounding_uncertainty_pp=.05,delta_interval_pp=[float(lower),float(upper)],
        direction_at_reported_precision='increase' if lower>0 else 'decrease' if upper<0 else 'unresolved_at_rounding',
        statistical_significance_tested=False))
large = sorted([r for r in rows if abs(dec(r['delta_decimal']))>=1],key=lambda x:dec(x['delta_decimal']),reverse=True)
directions={name:sum(r['direction_at_reported_precision']==name for r in rows) for name in ('increase','decrease','unresolved_at_rounding')}
soft42=dict(reference='Swin-T',seed=42,status='formal_test_complete_user_return_reconciled',test_mAP_percent=m['mAP_percent'],
            per_class_AP=m['per_class_ap_percent'],n_class_rows=19,all19_rows=rows,large_numeric_changes=large,
            large_numeric_changes_display_rule='absolute AP change>=1pp; display filter only, not statistical significance or model selection',
            direction_counts=directions,source=str(E/'summary.json'))
per['per_seed']['42']['soft']=soft42
per['last_updated']=DATE
per['source_files'].append(dict(path=str(E/'summary.json'),sha256=sha(E/'summary.json')))
per['limits']=[s for s in per['limits'] if 'soft42 has no formal' not in s]
per['limits'].append('Formal soft42 uses ed480e6/current4090 and30epochs; historical Swin42 has no sidecar and fixed42 uses cd39be9/historical hardware')
save(per_path,per)

def table(data, candidate):
    lines=['| 类别 | Swin AP % | '+candidate+' AP % | 对Swin约 pp |','|---|---:|---:|---:|']
    return '\n'.join(lines+['| '+r['class_zh']+' | '+fmt(r['swin_AP_percent'])+' | '+fmt(r['candidate_AP_percent'])+' | '+fmt(r['delta_decimal'],True)+' |' for r in data])

gains=[r for r in large if dec(r['delta_decimal'])>0]
losses=sorted([r for r in large if dec(r['delta_decimal'])<0],key=lambda x:dec(x['delta_decimal']))
class_block='''## seed42：软分配GAViT对Swin

正式test mAP：Swin70.90%、软分配'''+fmt(m['mAP_percent'])+'%，约'+fmt(dec(m['mAP_percent'])-dec('70.9'),True)+'''pp。soft42完整30轮，按验证mAP选择第10轮best，完整测试119825张；不使用smoke分数。

'''+table(gains+losses,'软分配GAViT')+'''

主要改善是天然草地/稀疏植被、多年生作物、荒原/石南灌丛/硬叶植被和阔叶林；主要退化是海岸湿地、沙滩/沙丘/沙地、工业或商业区。全部19类见附录D。这里的“主要”按数值幅度描述，不代表统计显著。

'''
p=ROOT/'docs/ben_seed42_seed43_vs_swin_20261013.md';doc=p.read_text();a=doc.index('## seed42：软分配GAViT对Swin');b=doc.index('## seed43：固定GAViT对Swin',a);doc=doc[:a]+class_block+doc[b:]
doc=doc.replace('2026-10-13更新。','2026-10-17更新。',1).replace('soft43为完整精度','soft42/43为完整精度')
insert='\n软分配对Swin也有明显的方向变化：海岸湿地在42约−19.64pp、43约+9.81pp；沙滩在42约−4.43pp、43约+12.47pp；天然草地在42约+6.24pp、43约−1.32pp。不能以43的一次改善概括软分配在这些类别的稳定收益。\n'
doc=doc.replace('## 证据与精度',insert+'\n## 证据与精度',1)
i=doc.index('来源：');doc=doc[:i]+'## 附录D：seed42软分配对Swin全部19类\n\n'+table(rows,'软分配GAViT')+'\n\n'+doc[i:]
doc=doc.replace('全部数值与缺项状态','全部数值与证据范围')+'\n新增soft42正式来源：[完成回传与核对](../results/evidence/bigearth_soft42_complete_20261017/validation.json)。\n'
p.write_text(doc)

single=[]
for name,control in (('Swin-T',summary['controls']['swin']),('固定GAViT',summary['controls']['gavit'])):
    v=[float(x.rstrip('%')) for x in control['metrics_printed']]
    single.append(dict(model=name,mAP_percent=v[0],macro_f1_percent=v[1],micro_f1_percent=v[2]))
single.append(dict(model='软分配GAViT',mAP_percent=m['mAP_percent'],macro_f1_percent=m['macro_f1_percent'],micro_f1_percent=m['micro_f1_percent']))
cost = summary['elapsed_seconds']/3600*1.67
report_data=dict(date=DATE,dataset='BigEarthNet-19',seed42_models=single,selected_seeds=[42,43,46,47],
    seed_rows=selected,summary=new_selected,full_common_history_summary=new_all,
    soft42_best=train['best'],soft42_final_val_mAP_percent=last['val_mAP_percent'],soft42_checkpoint_sha256=weight,
    per_class_source=str(per_path),resources=dict(queue_seconds=summary['elapsed_seconds'],train_seconds=train['elapsed_seconds'],
    test_seconds=test['elapsed_seconds'],peak_train_allocated_bytes=train['peak_cuda_allocated_bytes'],
    estimated_runtime_cost_yuan_at_1_67_per_hour=cost,actual_bill_yuan=None),
    evidence_limits=['User-returned complete logs/metadata/metrics; weights and prediction arrays not downloaded locally',
       'Seed46/47 aggregate scores user confirmed; original artifacts/F1/per-class/UUID still pending',
       'Legacy42 controls and43 execution differences remain; specified four-seed subset is descriptive, chosen after prior results'])
save(ROOT/'results/ben_four_seed_three_model_20261017.json', report_data)

lines=['# BEN软分配seed42完成与指定四seed比较','',DATE+'。正式回传总/训练/test三项exit0，30轮完整，验证best第10轮，119825张完整test。原始摘要及权重SHA链一致；这是回传核对，没有在本地重新训练或下载全部权重/预测。','',
       '| seed42模型 | mAP % | macro-F1 % | micro-F1 % |','|---|---:|---:|---:|']
lines += ['| '+x['model']+' | '+fmt(x['mAP_percent'])+' | '+fmt(x['macro_f1_percent'])+' | '+fmt(x['micro_f1_percent'])+' |' for x in single]
lines += ['', '## 指定四seed的主指标','', '| seed | Swin mAP % | 固定 mAP % | soft mAP % | soft−Swin pp | soft−固定 pp |','|---|---:|---:|---:|---:|---:|']
lines += ['| '+str(x['seed'])+' | '+fmt(x['swin'])+' | '+fmt(x['fixed'])+' | '+fmt(x['soft'])+' | '+fmt(dec(x['soft'])-dec(x['swin']),True)+' | '+fmt(dec(x['soft'])-dec(x['fixed']),True)+' |' for x in selected]
lines += ['', '| 模型 | n | mAP均值±样本SD % |','|---|---:|---:|']
for key,label in (('swin','Swin-T'),('fixed','固定GAViT'),('soft','软分配GAViT')):
    z=new_selected[key];lines.append('| '+label+' | 4 | '+fmt(z['mean'])+'±'+fmt(z['sample_sd_ddof1'])+' |')
lines += ['', '在指定42/43/46/47中，soft对Swin为2升2降（平均'+fmt(new_selected['paired_soft_minus_swin_pp']['mean'],True)+'pp）；对固定版仅43为正，平均'+fmt(new_selected['paired_soft_minus_fixed_pp']['mean'],True)+'pp。固定版在该子集平均更高、分数样本SD更小。保留42的负面结果，不选择性删除或另找正向seed。',
          '', '## seed42软分配对Swin的类别变化','',table(gains+losses,'软分配GAViT'),'',
          '全部19类与固定42/43、soft43分开列在[逐seed类别表](ben_seed42_seed43_vs_swin_20261013.md)。Swin类AP原值仅一位小数，差值约有±0.05pp打印不确定性；没有类别级显著性检验。海岸湿地、沙滩、草地在soft42/43中的变化方向相反，不能宣称稳定改善。',
          '', '## 资源与解释范围','',
          '队列实测'+fmt(summary['elapsed_seconds']/3600)+'小时（约5小时46分48秒），训练'+fmt(train['elapsed_seconds']/3600)+'小时，test'+fmt(test['elapsed_seconds'])+'秒。训练allocated峰值'+fmt(train['peak_cuda_allocated_bytes']/2**30)+'GiB；这是整阶段测量。按此前1.67元/小时折算队列运行约'+fmt(cost)+'元，实际账单/准备/空闲费用未提供。有效吞吐包含验证和IO，不能当作模型前向速度对比。',
          '', '历史代码、显卡/UUID、fixed43保存21轮与新soft满30轮等限制保留。42旧Swin无sidecar，不能新增其权重与日志强绑定证明。46/47原始工件及F1/逐类仍待补齐；四seed完整F1均值尚不能计算。指定四seed为已有结果上的汇报子集，历史44/45保留。',
          '', '材料：[完整精度统计](../results/ben_four_seed_three_model_20261017.json)、[四seed主指标CSV](../results/ben_four_seed_three_model_20261017.csv)、[正式原始回传](../results/evidence/bigearth_soft42_complete_20261017/server_return.txt)、[核对与限制](../results/evidence/bigearth_soft42_complete_20261017/validation.json)。']
(ROOT/'docs/ben_four_seed_three_model_20261017.md').write_text('\n'.join(lines)+'\n')

headers=next(csv.reader((WT/'results/bigearth_comparison.csv').open()))
values={k:'' for k in headers}
values.update({'Model':'GAViT soft','Variant':'sparse-hybrid K16 soft_prototype 4n top2 token_feedback local_prototype_cosine_v1',
 'Dataset':'BigEarthNet-19','Seed':42,'Best_Val_mAP(%)':train['best']['value'],'Best_Epoch':10,
 'Final_Val_mAP(%)':last['val_mAP_percent'],'Test_mAP(%)':m['mAP_percent'],
 'Test_Macro_F1(%)':m['macro_f1_percent'],'Test_Micro_F1(%)':m['micro_f1_percent'],'Params(M)':train['parameters']/1e6,
 'Epochs':30,'Batch_Size':32,'LR':.0003,'Weight_Decay':.0001,'Optimizer':'AdamW','Scheduler':'CosineAnnealingLR',
 'Threshold':.5,'Git_Commit':'ed480e6','GPU':'NVIDIA GeForce RTX 4090','Peak_GPU_Memory(GiB)':train['peak_cuda_allocated_bytes']/2**30,
 'Train_Wall_Time(h)':train['elapsed_seconds']/3600,'Test_Time':str(test['elapsed_seconds'])+' s (whole measured stage)',
 'Notes':'Formal soft42 full30 and119825 test complete; queue/train/test exit0; best10/val78.7500; exact19AP and checkpoint SHA chain agree: '+weight+'. Final30 val75.70723466069043. GPU UUID44825cf0-bd98-bfdc-efbc-0100238a6789; repaired workerSHA '+train['launcher_sha256']+'. Original scheduler failures preserved; no smoke retraining. Queue20808.021594s, runtime at historical1.67yuan/h about9.65yuan; bill/setup/idle unknown. Throughput includes validation/IO and probes, not pure model speed. Legacy42 control provenance and code/hardware differences retained. Evidence: ../../results/evidence/bigearth_soft42_complete_20261017/summary.json'})
check(not any(row['Model']=='GAViT soft' and row['Seed']=='42' for row in csv.DictReader((WT/'results/bigearth_comparison.csv').open())), 'Official table has no existing soft42 duplicate')
csv_headers=['Seed','Swin_mAP_percent','Fixed_mAP_percent','Soft_mAP_percent','Fixed_minus_Swin_pp','Soft_minus_Swin_pp','Soft_minus_Fixed_pp','Soft_evidence']
csv_rows=[]
for x in selected:
    csv_rows.append([x['seed'],x['swin'],x['fixed'],x['soft'],float(dec(x['fixed'])-dec(x['swin'])),float(dec(x['soft'])-dec(x['swin'])),float(dec(x['soft'])-dec(x['fixed'])),
                     'formal_archived' if x['seed'] in (42,43) else 'user_reported_original_artifacts_pending'])
save(E/'csv_job.json', dict(official_path=str(WT/'results/bigearth_comparison.csv'),original_official_sha256=sha(WT/'results/bigearth_comparison.csv'),
    official_headers=headers,official_new_row=[values[k] for k in headers],
    new_path=str(ROOT/'results/ben_four_seed_three_model_20261017.csv'),new_headers=csv_headers,new_rows=csv_rows,
    display_rows=[[str(x[0])]+[fmt(z) for z in x[1:7]]+[x[7]] for x in csv_rows]))

save(E/'validation.json',dict(status='formal_return_reconciled',received_date=DATE,checks=checks,source_attachment=str(SOURCE),
    source_sha256=sha(SOURCE),exit_codes=dict(queue=0,train=0,test=0),checkpoint_sha256=weight,
    source_worker_sha256=train['launcher_sha256'],report_metrics=m,
    best_val_precision='Metadata value rounded to4 decimals; exact epoch10 history value not returned locally',
    final_epoch_log=last,history_full_locally_archived=False,remote_worker_checks_full_history=True,
    evidence_scope='Returned complete stage receipts/logs, no direct GPU rerun or local checkpoint/prediction-array download',
    baseline_limit='Legacy Swin42 lacks sidecar; archived rounded log and current returned digest only',
    four_seed_evidence_limit='46/47 mAP user confirmed; full artifacts/auxiliary metrics/per-class/UUID still missing',
    runtime_cost=dict(hours=summary['elapsed_seconds']/3600,reference_price_yuan_per_hour=1.67,estimated_queue_cost_yuan=cost,actual_bill_yuan=None)))
print(json.dumps(dict(checks=len(checks),metrics=m,selected_summary=new_selected,direction_counts=directions),ensure_ascii=False,indent=2))
