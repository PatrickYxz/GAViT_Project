import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

R = Path('/Users/patrick/Projects/GAViT_Project/.worktrees/sparse-hybrid-4n-top2')
E = R / 'results/evidence/seed43_nwpu_final_20261011'
E.mkdir(exist_ok=False)
raw = Path('/Users/patrick/.codex/attachments/628c4a15-88eb-491c-8d0e-a1f6debb247c/已粘贴的文本.txt').read_bytes()
(E / 'server_return.txt').write_bytes(raw)
txt = raw.decode()
new, end = json.JSONDecoder().raw_decode(txt[txt.index('\n{')+1:])
old_path = R / 'results/evidence/nwpu_refit_test42_20260929/server_return.json'
old = json.loads(old_path.read_text())
def save(p, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
save(E / 'reported_summary.json', new)
assert new['exit'] == 'exit_code=0' and new['status'] == 'complete' and new['seed'] == 43
assert old['queue_exit'] == 'exit_code=0' and old['queue_summary']['status'] == 'complete'
pf = new['preflight']
assert pf['commit'] == 'ed480e60fea03ed786fdba70c144de5e815c34f5'
assert pf['manifest_sha256'] == 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
assert pf['pretrained_sha256'] == 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
assert pf['counts'] == dict(train=5040, val=1260, test=25200)
assert pf['seed'] == 43 and pf['gpu_profile']['uuid'] == 'GPU-cede40c0-f93d-7c8e-1006-cf6183e53592'
assert pf['gpu_profile']['name'] == 'NVIDIA GeForce RTX 4090'
models = ('swin', 'fixed', 'soft')
assert set(new['tests']) == set(models) and len(new['training']) == 6
assert pf['stages'] == [m + '_' + s for s in ('calibrate','refit','test') for m in models]
rows = []
params = dict(swin=27555495, fixed=31386920, soft=31486760)
for seed, tests in ((42, old['queue_summary']['tests']), (43, new['tests'])):
    for m in models:
        t = tests[m]
        assert t['samples'] == 25200 and t['manifest_sha256'] == pf['manifest_sha256']
        assert t['training_phase'] == 'refit' and t['fit_scope'] == 'train_plus_val'
        assert all(math.isfinite(t[k]) and 0 <= t[k] <= 100 for k in ('oa_percent','macro_f1_percent'))
        correct = round(t['oa_percent'] * 252)
        assert abs(correct/252 - t['oa_percent']) < 1e-10
        epoch = old['refit_run_subsets'][m]['final_epoch'] if seed == 42 else new['selected_epochs'][m]
        if seed == 43:
            a, b = new['training'][m+'_calibrate'], new['training'][m+'_refit']
            assert a['final_epoch'] == 30 and a['best_epoch'] == b['final_epoch'] == epoch
            assert a['counts'] == dict(train=5040, val=1260) and b['counts'] == dict(train=6300, val=0)
            assert b['best_epoch'] is None and b['checkpoint_sha256'] == t['checkpoint_sha256']
            assert a['checkpoint_sha256'] != b['checkpoint_sha256']
            for stage in (a,b):
                assert stage['total_parameters'] == params[m]
                assert 0 <= stage['reload_max_logit_diff'] <= 1e-5
                assert all(math.isfinite(stage[k]) and stage[k] > 0 for k in ('elapsed_seconds','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes'))
                assert stage['cuda_peak_allocated_bytes'] <= stage['cuda_peak_reserved_bytes']
        rows.append(dict(seed=seed, model=m, oa_percent=t['oa_percent'], macro_f1_percent=t['macro_f1_percent'],
            delta_oa_vs_swin_pp=t['oa_percent']-tests['swin']['oa_percent'],
            delta_macro_f1_vs_swin_pp=t['macro_f1_percent']-tests['swin']['macro_f1_percent'],
            correct=correct, samples=25200, refit_epochs=epoch, parameters=params[m],
            gpu='RTX 2080 Ti' if seed==42 else 'RTX 4090', checkpoint_sha256=t['checkpoint_sha256']))
means = {}
for m in models:
    x = [r for r in rows if r['model']==m]
    means[m] = {k:dict(mean=statistics.mean(r[k] for r in x), sample_sd=statistics.stdev(r[k] for r in x))
        for k in ('oa_percent','macro_f1_percent','delta_oa_vs_swin_pp','delta_macro_f1_vs_swin_pp')}
limits = ['n=2 descriptive statistics only; no significance or stability claim',
          'seed42 RTX2080Ti and seed43 RTX4090; hardware changes together with seed',
          'compact user-returned reports; original checkpoint/predictions/confusion arrays not downloaded',
          'preflight training_started=false is the historical preflight snapshot, not current completion status']
audit = dict(status='reported_summary_audit_pass', source='user terminal return, not direct server access',
    source_sha256=hashlib.sha256(raw).hexdigest(), seed42_source_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
    checks=['exit0/status/seed/code/data/pretrained/GPU identity','calibrate/refit counts and selected epochs',
        'refit checkpoint SHA equals test SHA','finite metrics/resources/reload <=1e-5','OA matches integer correct counts'],
    limits=limits, gpu=pf['gpu'], total_elapsed_seconds=new['elapsed_seconds'],
    hourly_yuan=1.67, run_time_estimated_yuan=new['elapsed_seconds']/3600*1.67)
save(E/'validation.json', audit)
save(R/'results/nwpu_seed43_test_summary.json', dict(dataset='NWPU-RESISC45',seed=43,status='complete',
    source='evidence/seed43_nwpu_final_20261011/reported_summary.json', source_sha256=audit['source_sha256'],
    rows=[r for r in rows if r['seed']==43], selected_epochs=new['selected_epochs'],
    resources=new['training'],elapsed_seconds=new['elapsed_seconds'],run_time_estimated_yuan=audit['run_time_estimated_yuan'],
    preflight=pf,limits=limits))
save(R/'results/nwpu_two_seed_summary.json', dict(dataset='NWPU-RESISC45', seeds=[42,43], n=2, rows=rows,
    statistics=means,sd_definition='sample SD ddof=1; paired differences computed per seed first',limits=limits))
with (R/'results/nwpu_two_seed_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
names = dict(swin='Swin-T',fixed='固定4×4 GAViT',soft='soft_prototype GAViT')
table = '| 模型 | test OA % | macro-F1 % | ΔOA vs Swin pp | 正确数/25200 | refit轮数 |\n|---|---:|---:|---:|---:|---:|\n'
for r in rows[3:]:
    table += f"| {names[r['model']]} | {r['oa_percent']:.6f} | {r['macro_f1_percent']:.6f} | {r['delta_oa_vs_swin_pp']:+.6f} | {r['correct']} | {r['refit_epochs']} |\n"
resource_table='| 模型 | 参数量 | calibrate秒 | refit秒 | refit allocated / reserved GiB |\n|---|---:|---:|---:|---:|\n'
for m in models:
    a,b=new['training'][m+'_calibrate'],new['training'][m+'_refit']
    resource_table+=f"| {names[m]} | {params[m]} | {a['elapsed_seconds']:.3f} | {b['elapsed_seconds']:.3f} | {b['cuda_peak_allocated_bytes']/2**30:.4f} / {b['cuda_peak_reserved_bytes']/2**30:.4f} |\n"
(R/'results/nwpu_seed43_test_comparison.md').write_text('# NWPU seed43 三模型正式测试（2026-10-11）\n\n'
    '用户回传总exit0、complete、六训练阶段与三模型25200张test的紧凑摘要。固定版与soft版均低于Swin；保留全部负面结果，不重试选分。\n\n'+table+
    '\n固定相对Swin净少对23张，soft净少对58张；soft相对fixed净少对35张。这是正确数净差，尚未读取43的逐样本预测，不能拆成纠正/新增错误或指定受益类别。\n\n'
    '协议：ed480e60fea03ed786fdba70c144de5e815c34f5；冻结manifest SHA ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e；ImageNet SHA f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36。5040/1260各校准30轮，以最高val OA首个同分选23/26/29；重新初始化6300池各自refit；各自refit末轮测试。batch32/workers2，CE/AdamW3e-4/wd1e-4/cosine周期30，FP32/no AMP/双TF32-off。完整预先命令见[执行说明](../docs/run_seed43_three_models_20261011.md)。紧凑摘要确认选轮身份；固定/soft的校准OA与校准macro-F1未回传，不填写。\n\n'
    f"GPU：{pf['gpu']}。输出`/home/featurize/work/nwpu_three43_20261011_01`；各模型`*_refit/last.pth`与`*_test`报告的SHA对应，完整SHA见[JSON](nwpu_seed43_test_summary.json)。\n\n"+resource_table+
    f"\n总队列{new['elapsed_seconds']:.3f}秒=1小时0分17.33秒，按1.67元/h折算{audit['run_time_estimated_yuan']:.4f}元（约1.68元）；这是本次运行时间折算，不是包含恢复/短测/空闲的账单。训练阶段资源包括检查点重建等；不同refit轮数与IO耗时不用于推导公平推理加速比。\n\n"
    '证据：[原文](evidence/seed43_nwpu_final_20261011/server_return.txt)、[解析](evidence/seed43_nwpu_final_20261011/reported_summary.json)、[核验](evidence/seed43_nwpu_final_20261011/validation.json)。调度器之前已检查原报告/权重/完整history/混淆和预测文件；本轮本地仅审计回传字段，没有下载原权重、完整配置、逐类准确率或混淆矩阵。\n\n'
    '与seed42合并见[双seed比较](nwpu_two_seed_summary.md)。seed42为2080Ti、43为4090，变化不能仅归因随机性或GPU；两seed仍不满足最终三seed要求。BigEarth soft43已通过预检，尚未训练。\n')
double='| seed | GPU | Swin OA% | 固定 OA% | soft OA% | 固定−Swin pp | soft−Swin pp |\n|---|---|---:|---:|---:|---:|---:|\n'
for seed in (42,43):
    x={r['model']:r for r in rows if r['seed']==seed}
    double+=f"| {seed} | {x['swin']['gpu']} | {x['swin']['oa_percent']:.6f} | {x['fixed']['oa_percent']:.6f} | {x['soft']['oa_percent']:.6f} | {x['fixed']['delta_oa_vs_swin_pp']:+.6f} | {x['soft']['delta_oa_vs_swin_pp']:+.6f} |\n"
mean_table='| 模型 | OA均值±样本SD % | macro-F1均值±样本SD % | ΔOA均值±样本SD pp |\n|---|---:|---:|---:|\n'
for m in models:
    s=means[m]
    mean_table+=f"| {names[m]} | {s['oa_percent']['mean']:.6f} ± {s['oa_percent']['sample_sd']:.6f} | {s['macro_f1_percent']['mean']:.6f} ± {s['macro_f1_percent']['sample_sd']:.6f} | {s['delta_oa_vs_swin_pp']['mean']:+.6f} ± {s['delta_oa_vs_swin_pp']['sample_sd']:.6f} |\n"
(R/'results/nwpu_two_seed_summary.md').write_text('# NWPU 固定协议 seed42/43 描述统计（2026-10-11）\n\n'+double+'\n'+mean_table+
    '\n固定版1次升、1次降，平均仅+0.039683pp；soft两次均低于Swin，平均−0.146825pp。当前不能宣称稳定优势，也不能以n=2证明等效或方法整体无效。样本SD使用ddof=1，先按同seed求配对差再汇总。两个seed共用同一25200张test，不把预测数量相加成独立样本。\n\n'
    '两seed冻结数据、预训练、源码和选择规则相同，但42使用RTX2080Ti、43使用RTX4090，实际refit预算由各自校准决定。硬件与seed一起变化，本表是历史配对结果的描述汇总，不是独立随机性或显卡因果实验。最终三seed仍缺一次；不根据这次负面结果自行追加训练。\n\n'
    '资源：固定版比Swin多3,831,425参数，soft比Swin多3,931,265参数；soft比fixed多99,840参数。当前未建立值得额外复杂度的稳定净收益。公平推理速度/显存与43逐类及混淆分析仍需读取现有产物，不需重训。\n\n'
    '数据：[精确CSV](nwpu_two_seed_comparison.csv)、[统计JSON](nwpu_two_seed_summary.json)；来源：[seed42](nwpu_seed42_test_comparison.md)、[seed43](nwpu_seed43_test_comparison.md)。约96%的历史NWPU协议不混入本表；OA不与BigEarth mAP跨数据集平均。\n')
print(json.dumps(dict(audit=audit,statistics=means),indent=2))
