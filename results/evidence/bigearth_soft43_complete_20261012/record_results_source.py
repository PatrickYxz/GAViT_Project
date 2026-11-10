import ast
import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

R=Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
E=R/'results/evidence/bigearth_soft43_complete_20261012'
E.mkdir(exist_ok=False)
raw=Path('/Users/patrick/.codex/attachments/c680e337-4961-4c32-b42d-ce1413e75ae0/已粘贴的文本.txt').read_bytes()
(E/'reported_output.txt').write_bytes(raw)
text=raw.decode()
metrics,_=json.JSONDecoder().raw_decode(text[text.rindex('\n{')+1:])
def save(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
save(E/'metrics.json',metrics)
exits=[int(x) for x in re.findall(r'^exit_code=(\d+)$',text,re.M)]
assert exits==[0,0,0]
assert 'STAGE COMPLETE gavit_train' in text and 'STAGE COMPLETE gavit_test' in text
assert 'BIGEARTH SOFT43 COMPLETE /home/featurize/work/bigearth_soft43_20261011_01' in text
assert metrics['samples']==119825 and metrics['f1_threshold']==.5
assert len(metrics['checkpoint_sha256'])==64
classes=metrics['per_class_ap_percent'];assert len(classes)==19
assert all(math.isfinite(x) and 0<=x<=100 for x in classes.values())
assert abs(statistics.mean(classes.values())-metrics['mAP_percent'])<1e-10
assert all(math.isfinite(metrics[k]) and 0<=metrics[k]<=100 for k in ('mAP_percent','macro_f1_percent','micro_f1_percent'))
final=json.loads(re.search(r'^EPOCH RESOURCE (.+)$',text,re.M)[1])
assert final['epoch']==30 and final['next_epoch_lr']==0 and all(math.isfinite(v) for v in final.values())
soft_final=json.loads(re.search(r'^SOFT EPOCH (.+)$',text,re.M)[1])
assert soft_final['epoch']==30 and soft_final['val_mAP_percent']==final['val_mAP_percent']
device_line=next(line for line in text.splitlines() if line.startswith('DEVICE '))
hardware_text,precision_text=device_line[len('DEVICE '):].split(' PRECISION ')
hardware,precision=ast.literal_eval(hardware_text),ast.literal_eval(precision_text)
preflight=json.loads((R/'results/evidence/seed43_bigearth_preflight_20261011/user_return.json').read_text())
assert hardware==preflight['hardware'] and precision==preflight['precision']
best_printed=float(re.search(r'Best Validation mAP: ([0-9.]+)%',text)[1])
assert '3745/3745' in text and 'Test samples: 119,825' in text
controls=list(csv.DictReader((R/'results/bigearth_comparison.csv').open()))
ctrl={}
for model,name in [('swin','Swin-T Baseline'),('fixed','GAViT v2')]:
    found=[r for r in controls if r['Seed']=='43' and r['Model']==name]
    assert len(found)==1
    row=found[0];ctrl[model]={k:float(row[c]) for k,c in [('mAP_percent','Test_mAP(%)'),('macro_f1_percent','Test_Macro_F1(%)'),('micro_f1_percent','Test_Micro_F1(%)')]}
    ctrl[model]['precision']='printed_one_decimal'
rows=[]
for model in ('swin','fixed','soft'):
    m=metrics if model=='soft' else ctrl[model]
    rows.append(dict(seed=43,model=model,**{k:m[k] for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')},
        precision='full_returned' if model=='soft' else 'printed_one_decimal',final_epoch=21 if model=='fixed' else 30,
        gpu='RTX4090 train / RTX3060 test' if model=='fixed' else 'RTX4090',code='ed480e6' if model=='soft' else 'bb9f1cf'))
diffs={model:{k:metrics[k]-ctrl[model][k] for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')} for model in ('swin','fixed')}
old_classes=list(csv.DictReader((R/'results/bigearth_seed43_per_class_comparison.csv').open()))
assert [r['Class'] for r in old_classes]==list(classes)
class_rows=[]
for row in old_classes:
    c=row['Class'];s,f,a=float(row['Swin_AP(%)']),float(row['GAViT_AP(%)']),classes[c]
    rec=dict(Class=c,Swin_AP_percent_printed=s,Fixed_AP_percent_printed=f,Soft_AP_percent=a)
    for name,base in [('swin',s),('fixed',f)]:
        d=a-base
        rec['soft_minus_'+name+'_pp_approx']=d
        rec['soft_minus_'+name+'_pp_lower']=d-.05
        rec['soft_minus_'+name+'_pp_upper']=d+.05
        rec['direction_vs_'+name]='up' if d>.05 else 'down' if d<-.05 else 'unresolved_at_reference_rounding'
    class_rows.append(rec)
with (R/'results/bigearth_seed43_three_model_per_class.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)
limits=['seed43 controls are rounded one-decimal metrics; deltas/means/SD involving them are approximate',
    'fixed43 saved21 epochs/best8 vs Swin43 full30/best6 vs soft43 full30; code/historical hardware differ',
    'server checkpoint/predictions/probe tensors not downloaded; saved result is user-returned evidence',
    'best validation printed79.0 only; exact best/epoch and total/stage resource reports still pending',
    'n=2 descriptive evidence; no significance/stability/causal claim']
report=dict(status='formal_training_and_test_complete_from_user_evidence',dataset='BigEarthNet-19',seed=43,
    source_sha256=hashlib.sha256(raw).hexdigest(),exits=dict(zip(['run.exit','gavit_train/run.exit','gavit_test/run.exit'],exits)),
    tests=rows,test_metrics=metrics,soft_minus_controls_pp_approx=diffs,final_epoch=final,
    best_val_mAP_printed=best_printed,best_epoch=None,hardware=hardware,precision=precision,
    per_class_counts={name:{d:sum(r['direction_vs_'+name]==d for r in class_rows) for d in ['up','down','unresolved_at_reference_rounding']} for name in ('swin','fixed')},
    limits=limits,total_elapsed_seconds=None,train_stage_resources=None,test_stage_resources=None)
save(E/'comparison.json',report)
save(E/'epoch30.json',dict(resource=final,soft_epoch=soft_final))
save(E/'validation.json',dict(status='reported_log_and_metrics_audit_pass',checks=['three exits0 and queue/train/test completion markers',
    'epoch30/finite final values/zero nextLR','119825samples/3745batches/threshold0.5','19class AP average equals precise mAP',
    'test environment equals passed preflight','old43 controls/class names match current archived tables'],source_sha256=report['source_sha256'],limits=limits))
save(R/'results/bigearth_soft43_test_summary.json',report)
old45=json.loads((R/'results/evidence/bigearth_seed45_20260922_01/summary.json').read_text())
soft45=json.loads((R/'results/evidence/bigearth_soft45_complete_20261001/metrics.json').read_text())
for model,stage in [('swin','swin_test'),('fixed','gavit_test'),('soft',None)]:
    m=soft45 if model=='soft' else old45['stages'][stage]['metrics']
    rows.append(dict(seed=45,model=model,**{k:m[k] for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')},
        precision='full_returned',final_epoch=30,gpu='RTX2080Ti',code='ed480e6' if model=='soft' else 'deb2b45'))
stats={m:{k:dict(mean=statistics.mean(r[k] for r in rows if r['model']==m),sample_sd=statistics.stdev(r[k] for r in rows if r['model']==m),
    precision='approximate_if_rounded_seed43_control') for k in ('mAP_percent','macro_f1_percent','micro_f1_percent')} for m in ('swin','fixed','soft')}
pairs={}
for base in ('swin','fixed'):
    pairs[base]={}
    for k in ('mAP_percent','macro_f1_percent','micro_f1_percent'):
        values={}
        for seed in (43,45):
            x={r['model']:r for r in rows if r['seed']==seed}
            values[seed]=x['soft'][k]-x[base][k]
        pairs[base][k]=dict(by_seed=values,mean=statistics.mean(values.values()),sample_sd=statistics.stdev(values.values()),precision='approximate_seed43_reference')
save(R/'results/bigearth_three_model_two_seed_summary.json',dict(dataset='BigEarthNet-19',seeds=[43,45],n=2,rows=rows,
    statistics=stats,soft_minus_controls_pp=pairs,limits=limits,sd_definition='sample SD ddof=1; paired differences first, then summary'))
with (R/'results/bigearth_three_model_two_seed_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
# Append only this new formal result, preserving all historical row bytes.
csv_path=R/'results/bigearth_comparison.csv'
assert not any(r['Model']=='GAViT soft' and r['Seed']=='43' for r in controls)
columns=list(controls[0]);new={k:'' for k in columns}
new.update({'Model':'GAViT soft','Variant':'sparse-hybrid K16 soft_prototype 4n top2 token_feedback local_prototype_cosine_v1',
    'Dataset':'BigEarthNet-19','Seed':'43','Best_Val_mAP(%)':str(best_printed),'Final_Val_mAP(%)':str(final['val_mAP_percent']),
    'Test_mAP(%)':str(metrics['mAP_percent']),'Test_Macro_F1(%)':str(metrics['macro_f1_percent']),'Test_Micro_F1(%)':str(metrics['micro_f1_percent']),
    'Params(M)':'31.466766','Epochs':'30','Batch_Size':'32','LR':'0.0003','Weight_Decay':'0.0001','Optimizer':'AdamW',
    'Scheduler':'CosineAnnealingLR','Threshold':'0.5','Git_Commit':'ed480e6','GPU':'NVIDIA GeForce RTX 4090',
    'Test_Time':'02:53 (printed test loop only; whole stage pending)',
    'Notes':'Formal soft43 full30 and119825 test complete; queue/train/test exits0. Precise metrics and19AP returned. Checkpoint '+metrics['checkpoint_sha256']+'. Best val79.0 is printed1decimal; exact best/epoch/resources/cost pending. Frozen code validator supports parameter/protocol identities; full run summary not yet returned. Same current4090 UUID as passed smoke; old43 controls rounded; fixed43 saved21 epochs and old code/hardware confounds retained. Evidence results/evidence/bigearth_soft43_complete_20261012/reported_output.txt'})
with csv_path.open('a',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns,lineterminator="\n");w.writerow(new)
names=dict(swin='Swin-T',fixed='固定4×4 GAViT',soft='soft_prototype GAViT')
table='| 模型 | test mAP % | macro-F1 % | micro-F1 % | 保存轮数 |\n|---|---:|---:|---:|---:|\n'
for row in rows[:3]:
    fmt='.6f' if row['model']=='soft' else '.1f'
    table+=f"| {names[row['model']]} | {row['mAP_percent']:{fmt}} | {row['macro_f1_percent']:{fmt}} | {row['micro_f1_percent']:{fmt}} | {row['final_epoch']} |\n"
diff_table='| soft比较 | ΔmAP约 pp | Δmacro-F1约 pp | Δmicro-F1约 pp |\n|---|---:|---:|---:|\n'
for base,d in diffs.items():
    diff_table+=f"| soft − {names[base]} | {d['mAP_percent']:+.4f} | {d['macro_f1_percent']:+.4f} | {d['micro_f1_percent']:+.4f} |\n"
scene='| 类别 | 固定43打印AP % | soft43 AP % | 差约 pp |\n|---|---:|---:|---:|\n'
for c in sorted(class_rows,key=lambda x:x['soft_minus_fixed_pp_approx'],reverse=True)[:3]+sorted(class_rows,key=lambda x:x['soft_minus_fixed_pp_approx'])[:3]:
    scene+=f"| {c['Class']} | {c['Fixed_AP_percent_printed']:.1f} | {c['Soft_AP_percent']:.4f} | {c['soft_minus_fixed_pp_approx']:+.4f} |\n"
(R/'results/bigearth_soft43_ablation.md').write_text('# BigEarthNet seed43 三模型正式对比（2026-10-12）\n\n'
    'soft43已完成30轮与119825张test，总/训练/test退出码均0，两个STAGE COMPLETE及总完成标记返回。支持完整训练和评估已结束，不需重跑。\n\n'+table+'\n'+diff_table+
    '\n旧Swin/fixed43只返回一位小数，因此差值为近似；对应参考舍入误差约±0.05pp。此seed中soft三项指标均高于两个旧对照，最小micro-F1提升也大于舍入误差。旧fixed43仅保存21轮/best8，Swin43完整30/best6；旧bb9f1cf、历史4090训练/3060固定test，新soft ed480e6/current4090，不能宣称三模型完全同条件或仅分组改动的因果提升。完整30轮的seed45另保留，不重训旧43。\n\n'+scene+
    '\n相对fixed43，舍入范围可确定8类升、8类降，另外3类方向不确定（农业与自然植被、针叶林、海洋水域）。soft沙滩/沙丘/沙地与海岸湿地改善较大，草地/稀疏植被、工业商业区和复杂耕作退步；不是所有场景同时提升。soft精确AP与旧打印AP作差时按±0.05pp范围判断，不能把接近0的差写成确定涨跌；这些是预测结果描述，不是机制/显著性证明。[全部19类CSV](bigearth_seed43_three_model_per_class.csv)。\n\n'
    '最终第30轮loss0.0600898171、val mAP75.69968198%、macro-F1 71.32167879%、nextLR0；best val仅打印79.0%，精确值和best epoch待摘要。测试加载gavit_train/gavit.pth（按val mAP选best），checkpoint SHA '+metrics['checkpoint_sha256']+'。冻结调度器在保存完整history/训练状态/8val初始-best-final探针和权重链验收后打印完成；本地没有下载完整曲线/权重/预测张量。\n\n'
    '实际测试设备/软件/精度与批准4090预检相同：GPU-cede40c0-f93d-7c8e-1006-cf6183e53592，FP32/noAMP/matmul TF32false/cuDNNtrue。最后一轮检查点间754.068秒与test循环02:53不是全程资源；总/训练/test完整耗时、显存、有效吞吐、best epoch待一次只读紧凑摘要，不据最后一轮乘30或按短测估算实际费用。\n\n'
    '证据：[原始字节](evidence/bigearth_soft43_complete_20261012/reported_output.txt)、[核验](evidence/bigearth_soft43_complete_20261012/validation.json)、[JSON](bigearth_soft43_test_summary.json)。[43/45双seed三模型描述表](bigearth_three_model_two_seed_summary.md)：soft对fixed两次mAP/macro-F1提升，对Swin一升一降；NWPU42/43的soft仍均低于Swin，不宣称稳定跨数据集收益。三seed与公平效率/外部方法实测仍有缺口。\n')
two='| seed | Swin mAP% | fixed mAP% | soft mAP% | soft−Swin约 pp | soft−fixed约 pp |\n|---|---:|---:|---:|---:|---:|\n'
for seed in (43,45):
    x={r['model']:r for r in rows if r['seed']==seed}
    two+=f"| {seed} | {x['swin']['mAP_percent']:.4f} | {x['fixed']['mAP_percent']:.4f} | {x['soft']['mAP_percent']:.4f} | {x['soft']['mAP_percent']-x['swin']['mAP_percent']:+.4f} | {x['soft']['mAP_percent']-x['fixed']['mAP_percent']:+.4f} |\n"
mean='| 模型 | mAP均值±样本SD % | macro-F1均值±样本SD % | micro-F1均值±样本SD % |\n|---|---:|---:|---:|\n'
for m in ('swin','fixed','soft'):
    x=stats[m];mean+=f"| {names[m]} | {x['mAP_percent']['mean']:.4f} ± {x['mAP_percent']['sample_sd']:.4f} | {x['macro_f1_percent']['mean']:.4f} ± {x['macro_f1_percent']['sample_sd']:.4f} | {x['micro_f1_percent']['mean']:.4f} ± {x['micro_f1_percent']['sample_sd']:.4f} |\n"
(R/'results/bigearth_three_model_two_seed_summary.md').write_text('# BigEarthNet seed43/45 三模型描述汇总（2026-10-12）\n\n'+two+'\n'+mean+
    '\nseed43旧Swin/fixed仅一位小数，表中相关差、均值与SD为打印中心值计算的近似；多写小数不代表精度提高。soft两次均有精确回传指标。SD为ddof=1，配对差先按同seed计算再求汇总，n=2不作显著性或稳定性结论。\n\n'
    f"soft−fixed mAP平均约{pairs['fixed']['mAP_percent']['mean']:+.4f}pp，两次均升；macro-F1两次均升，micro-F1一升一降、均差约{pairs['fixed']['micro_f1_percent']['mean']:+.4f}pp。soft−Swin mAP平均约{pairs['swin']['mAP_percent']['mean']:+.4f}pp，43升45降。当前支持BigEarth上soft相对固定的两次主要指标改善观察，尚未建立稳定优于Swin或跨数据集一致优势。\n\n"
    '43对照fixed仅保存21轮且历史代码/训练-test硬件不同；45三模型完整30轮，旧对照deb2b45/soft ed480e6、2080Ti型号相同但物理UUID不同。43soft4090、45soft2080Ti，不能把变化仅归因seed或GPU。相同119825 test不累计成更多独立测试样本，不能拿Swin/fixed四seed均值和soft两seed均值冒充共同seed配对。\n\n'
    '共同类别观察：seed43与45，soft相对fixed均改善海岸湿地、沙滩/沙丘/沙地，均降低草地/稀疏植被；只作为有限类别描述，AP与罕见类支持量/机制仍需现有预测分析。NWPU soft两个seed均低于Swin；OA不与mAP跨任务平均。最终三seed要求未满足，不按结果筛选seed或自行新起训练。\n\n'
    '来源：[seed43](bigearth_soft43_ablation.md)、[seed45](bigearth_soft45_ablation.md)；[精确/打印精度分列CSV](bigearth_three_model_two_seed_comparison.csv)、[结构化统计](bigearth_three_model_two_seed_summary.json)。\n')
(E/'record_results_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(dict(deltas_approx=diffs,class_counts=report['per_class_counts'],pair_stats=pairs),indent=2))
