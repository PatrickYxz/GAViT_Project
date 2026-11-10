"""Archive user-returned summary and verify arithmetic, not remote tensor bytes."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path('/Users/patrick/.codex/attachments/9e2d8073-34f8-4aad-89f3-2630c4a5099d/已粘贴的文本.txt')
raw = SOURCE.read_bytes()
archived = HERE/'terminal_return.txt'
if archived.exists():
    assert archived.read_bytes() == raw, 'Archived source differs; preserve it'
else:
    archived.write_bytes(raw)
text = raw.decode('utf-8')
assert '\nexit_code=0\n' in text
start = text.index('{')
summary, end = json.JSONDecoder().raw_decode(text[start:])
assert text[start+end:].strip() == '(base) ➜ ~'
assert summary['status'] == 'complete'
assert (summary['dataset'],summary['split'],summary['source_phase'],summary['seed'],summary['samples']) == (
    'NWPU-RESISC45','val','calibrate',42,1260)
assert summary['alphas'] == [0.,.25,.5,.75,1.]
assert set(summary['curves']) == {'fixed','soft'}
assert summary['input_sources'] == {'zip':1260}
prior = json.loads((ROOT/'results/evidence/nwpu_calibration42_20260928/server_return.json').read_text())['runs_returned_subset']
tool = ROOT/'tools/diagnose_nwpu_feedback.py'
tool_sha = hashlib.sha256(tool.read_bytes()).hexdigest()
assert tool_sha == '4cd5175261d271638a00e4a4c733f329ecde1c07b646b901c960f0181237eea3'
checks = {}
for name, rows in summary['curves'].items():
    assert [r['alpha'] for r in rows] == summary['alphas']
    baseline = rows[-1]
    assert math.isclose(baseline['oa_percent'],prior[name]['best_val_oa_percent'],abs_tol=1e-10)
    base_ce = baseline['cross_entropy'] - baseline['delta_cross_entropy']
    for r in rows:
        p = r['paired_vs_original']
        assert all(type(p[k]) is int and 0<=p[k]<=1260 for k in ('both_correct','corrects','breaks','both_wrong'))
        assert sum(p[k] for k in ('both_correct','corrects','breaks','both_wrong')) == 1260
        assert p['both_correct']+p['breaks'] == baseline['correct']
        assert p['both_correct']+p['corrects'] == r['correct']
        assert p['net_correct'] == p['corrects']-p['breaks'] == r['correct']-baseline['correct']
        assert math.isclose(r['oa_percent'],100*r['correct']/1260,abs_tol=1e-10)
        assert math.isclose(r['delta_oa_pp'],100*p['net_correct']/1260,abs_tol=1e-10)
        assert math.isclose(r['delta_macro_f1_pp'],r['macro_f1_percent']-baseline['macro_f1_percent'],abs_tol=1e-10)
        assert math.isclose(r['cross_entropy']-r['delta_cross_entropy'],base_ce,abs_tol=1e-10)
        assert 0<=r['macro_f1_percent']<=100 and r['cross_entropy']>=0
        assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
    assert baseline['paired_vs_original']['corrects'] == baseline['paired_vs_original']['breaks'] == 0
    errors = summary['max_errors'][name]
    assert all(math.isfinite(v) and v>=0 for v in errors.values())
    assert errors['cached_probability_max_error'] <= 1e-5
    artifact = summary['cache_artifacts'][name]
    assert len(artifact['cache_sha256']) == 64 and int(artifact['cache_sha256'],16)>=0
    assert artifact['cache_bytes']>0
    checks[name] = dict(
        original_oa_matches_prior=True,all_pair_counts_and_deltas_consistent=True,
        all_lower_alpha_oa_worse=all(r['oa_percent']<baseline['oa_percent'] for r in rows[:-1]),
        all_lower_alpha_macro_f1_worse=all(r['macro_f1_percent']<baseline['macro_f1_percent'] for r in rows[:-1]),
        all_lower_alpha_cross_entropy_worse=all(r['cross_entropy']>baseline['cross_entropy'] for r in rows[:-1]),
        zero_feedback_net_correct=rows[0]['paired_vs_original']['net_correct'],
        cpu_replay_max_logit_diff=errors['cpu_replay_max_logit_diff'])
assert 0<sum(r['elapsed_seconds'] for r in summary['model_timing'].values())<=summary['elapsed_seconds']
record = dict(status='passed',source_sha256=hashlib.sha256(raw).hexdigest(),
              scope='Local arithmetic and prior-OA cross-check of user-returned summary; no remote identity/log/cache download or independent prediction replay.',
              delivered_tool_sha256=tool_sha,run_exit='exit_code=0',checks=checks,
              elapsed_seconds=summary['elapsed_seconds'],
              cpu_replay_note='Server code checks elementwise atol=1e-5 plus rtol=1e-5 and exact argmax. Soft max absolute error slightly exceeds atol alone; max absolute error cannot independently recheck that mixed criterion.')
(HERE/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
(HERE/'validation.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')

lines=['# NWPU seed42反馈幅度诊断：减弱反馈未改善验证结果','',
       '2026-10-03回传收尾。退出码0、summary complete；1260张校准val全部直接来自原ZIP，总耗时46.331209秒。原强度α=1在本次预设五档中同时取得最高OA/macro-F1、最低交叉熵，降低强度的四档全部变差。当前证据不支持“反馈整体太强，简单调小就能改善”这一有限假设。','',
       '来源：[终端原文](evidence/nwpu_feedback42_20261003/terminal_return.txt)、[精确摘要](evidence/nwpu_feedback42_20261003/summary.json)、[本地算术核验](evidence/nwpu_feedback42_20261003/validation.json)。未下载PT缓存、逐类结果、identity.json或完整日志，不声称已本地复算F1/损失、检查逐类趋势或重新执行模型。','',
       '## 协议与结果','',
       '冻结NWPU seed42校准best：fixed epoch29、soft epoch28，5040张训练/1260张验证。源码ed480e6；清单与权重由交付脚本校验后才进入捕获。两模型各推理一次全val，共用缓存重放α=0/0.25/0.5/0.75/1。没有训练、重新选轮或test评估；不是6300图refit权重。','',
       '| 模型 | α | OA % | macro-F1 % | CE | 净正确数变化 | 纠正 / 新增错误 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name,rows in summary['curves'].items():
    for r in rows:
        p=r['paired_vs_original']
        lines.append(f"| {name} | {r['alpha']:g} | {r['oa_percent']:.6f} | {r['macro_f1_percent']:.6f} | {r['cross_entropy']:.6f} | {p['net_correct']:+d} | {p['corrects']} / {p['breaks']} |")
lines += ['', '变化均以同模型α=1为参照；纠正与新增错误表示降低反馈造成的预测变化。OA并非随α严格单调，但原强度在全部预设点中最好；CE随α增大下降。', '',
          '完全关闭反馈时，fixed纠正8个原错误、另增加17个错误；soft纠正12个、另增加21个；两者均净少对9张，OA下降0.714286个百分点。这是在联合训练模型中移除分支的影响，不是额外图训练相对独立Swin的贡献。α=0与Swin或另一模型恰好相同的正确数也不表示预测或结构相同。', '',
          '## 完整性与数值边界','',
          'α=1的fixed/soft OA为93.015873%/93.730159%，与已归档校准best完全一致；两模型α=1纠正/新增错误均0。全部10行四格配对计数、正确数/OA及差值、F1差值和CE差值算术一致。macro-F1与CE本体为服务器报告，缺逐样本缓存时不作本地独立重算声明。','',
          '捕获重建最大logit差两者均2.8610e-6；原缓存概率误差分别2.9802e-7/3.8743e-7。CPU缓存重放最大绝对差为8.5831e-6/1.0490e-5。脚本使用逐元素atol=1e-5与rtol=1e-5组合，并要求argmax完全一致；soft绝对最大值略超atol本身不等于违反组合标准。没有张量，不能仅从最大误差独立复验相对容差；complete对应服务器已通过其检查。fixed α=1 CE差2.98e-8为重建与原前向的浮点差，不影响分类结论。','',
          '服务器输出根目录：`/home/featurize/work/nwpu_feedback42_20261002_01`。缓存fixed/soft分别9,306,352/9,306,340字节；SHA256保留在精确摘要中，本地未读取远程缓存字节重哈希。模型阶段耗时21.568224/14.124649秒，峰值CUDA allocated 638999040/639792640字节；时间包含读盘、重建、捕获、写缓存等，不是公平推理速度对比。摘要未包含本次GPU型号，不以此前2080Ti记录冒充当前实测硬件。','',
          '## 研究结论与下一步','',
          '本次支持：已训练模型在这些验证样本上依赖原幅度反馈，直接整体削弱的收益不抵损害。它不证明图结构比无图训练更好，也不证明软分配比固定分组更好；两个模型分别联合训练，α=0的表征/分类器仍受原图分支训练影响。它更不能直接解释BigEarth正式test或证明反馈是唯一根因。','',
          '这条幅度假设到此收尾：保留原配置和全部正负结果，不继续扫描α>1、细化网格、重跑test或直接新增门控训练。已有正式test主表保持不变。下一步先把本结果纳入失败分析；若继续架构研究，应为区域表示/图关系质量另立可验证问题，先复用已有缓存和图册，不能从本结果直接指定一个保证提升的新模块。导师要求的多seed、外部方法对比和新结构公平效率仍未完成。','',
          '论文可用表述：在冻结的NWPU校准权重上，我们预设五种图反馈幅度进行全验证集扰动。对固定分组和软分组模型，所有低于原幅度的设置都降低了OA和macro-F1，并提高交叉熵；完全移除反馈均使正确数减少9/1260。因此，结果未支持用简单整体缩小反馈来修复当前差距。该诊断仅描述已训练网络的条件性依赖，不能替代独立训练的模块消融或多seed泛化比较。', '']
(ROOT/'results/nwpu_feedback_sensitivity_seed42.md').write_text('\n'.join(lines))
print(json.dumps(record,indent=2))
