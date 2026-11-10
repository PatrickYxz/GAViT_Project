"""Check the chapter's reported tables against archived summaries, without GPU work."""
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DRAFT = ROOT / 'docs/thesis_experiments_discussion_20261006.md'
text = DRAFT.read_text()
source_paths = []
checks = []
table_rows = []


def read(relative):
    path = ROOT / relative
    source_paths.append(path)
    return json.loads(path.read_text())


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9), (a, b)


def row(*cells):
    line = '| ' + ' | '.join(map(str, cells)) + ' |'
    assert text.splitlines().count(line) == 1, line
    table_rows.append(line)


aid = read('results/aid_multiseed_summary.json')
ben = read('results/bigearth_four_seed_summary.json')
soft = read('results/soft_assignment_cross_dataset_summary.json')
eff = read('results/aid_efficiency_summary.json')
ffn = read('results/nwpu_ffn_calibration42_summary.json')
feedback = read('results/evidence/nwpu_feedback42_20261003/summary.json')
gallery = read('results/evidence/nwpu_regions_review_20260929/display_statistics.json')

assert aid['training_seeds_completed'] == [42, 43, 44]
assert ben['seeds'] == [42, 43, 44, 45]
for r in aid['per_seed']:
    s, g = r['swin'], r['gavit']
    close(g['oa_percent'] - s['oa_percent'], r['delta_oa_pp'])
    c = r['paired_discordance_counts']
    assert sum(c[k] for k in ('both_correct', 'gavit_corrects', 'gavit_breaks', 'both_wrong')) == 5000
    assert c['both_correct'] + c['gavit_corrects'] == g['correct']
    assert c['both_correct'] + c['gavit_breaks'] == s['correct']
    row(r['seed'], f"{s['oa_percent']:.2f}", f"{g['oa_percent']:.2f}",
        f"{r['delta_oa_pp']:+.2f}", f"{s['macro_f1_percent']:.4f}", f"{g['macro_f1_percent']:.4f}")
for key, label, delta_key in [('oa_percent', 'OA %', 'delta_oa_pp'), ('macro_f1_percent', 'macro-F1 %', 'delta_macro_f1_pp')]:
    cells = []
    values = {}
    for model in ('swin', 'gavit'):
        values[model] = [r[model][key] for r in aid['per_seed']]
        mean, sd = statistics.mean(values[model]), statistics.stdev(values[model])
        close(mean, aid['statistics'][model][key]['mean'])
        close(sd, aid['statistics'][model][key]['sample_sd_ddof1'])
        cells.append(f'{mean:.4f} ± {sd:.4f}')
    delta = [g - s for g, s in zip(values['gavit'], values['swin'])]
    close(statistics.mean(delta), aid['paired_delta_statistics'][delta_key]['mean_pp'])
    close(statistics.stdev(delta), aid['paired_delta_statistics'][delta_key]['sample_sd_pp_ddof1'])
    row(label, *cells, f'{statistics.mean(delta):+.4f} ± {statistics.stdev(delta):.4f}')
checks.append('AID: all three seeds, paired counts, means, sample SD and formatted tables')

for i, seed in enumerate(ben['seeds']):
    s = ben['models']['swin']['mAP_percent']['values'][i]
    g = ben['models']['gavit']['mAP_percent']['values'][i]
    digits = 4 if seed == 45 else 1
    row(seed, f'{s:.{digits}f}', f'{g:.{digits}f}', f'{g-s:+.{digits}f}')
for key, label in [('mAP_percent', 'mAP %'), ('macro_f1_percent', 'macro-F1 %'), ('micro_f1_percent', 'micro-F1 %')]:
    cells, values = [], {}
    for model in ('swin', 'gavit'):
        r = ben['models'][model][key]
        values[model] = r['values']
        mean, sd = statistics.mean(r['values']), statistics.stdev(r['values'])
        close(mean, r['mean']); close(sd, r['sample_sd'])
        cells.append(f'{mean:.2f} ± {sd:.2f}')
    delta = [g - s for g, s in zip(values['gavit'], values['swin'])]
    r = ben['paired_gavit_minus_swin'][key]
    close(statistics.mean(delta), r['mean_pp']); close(statistics.stdev(delta), r['sample_sd_pp'])
    row(label, *cells, f'{statistics.mean(delta):+.2f} ± {statistics.stdev(delta):.2f}')
checks.append('BigEarth: four historical seeds, rounded-input caveat retained; means and sample SD recomputed')

names = {'swin': 'Swin-T', 'fixed': '固定4×4 GAViT', 'soft': '软分组GAViT'}
for r in soft['rows']:
    cells = [names[r['variant']], f"{r['primary_percent']:.4f}", f"{r['macro_f1_percent']:.4f}"]
    if r['dataset'] == 'NWPU-RESISC45':
        assert r['seed'] == 42 and r['samples'] == 25200
        cells += [f"{r['parameters']/1e6:.4f}", r['final_fit_epochs']]
    else:
        assert r['seed'] == 45 and r['samples'] == 119825
        cells += [f"{r['micro_f1_percent']:.4f}", f"{r['parameters']/1e6:.4f}"]
    row(*cells)
for dataset, deltas in soft['deltas_pp'].items():
    rows = {r['variant']: r for r in soft['rows'] if r['dataset'] == dataset}
    for comparison, values in deltas.items():
        control = comparison.removeprefix('soft_vs_')
        for metric, delta in values.items():
            close(rows['soft'][metric] - rows[control][metric], delta)
    assert rows['soft']['parameters'] - rows['fixed']['parameters'] == 99840
close(sum(soft['bigearth_class_ap_delta_pp'].values()) / 19,
      soft['deltas_pp']['BigEarthNet-19']['soft_vs_fixed']['primary_percent'])
close(soft['coastal_mAP_contribution_pp'] + soft['other18_mAP_contribution_pp'],
      soft['deltas_pp']['BigEarthNet-19']['soft_vs_fixed']['primary_percent'])
checks.append('Soft: six test rows, dataset/seed separation, all stored deltas and 19-class AP decomposition')

for key in ('swin', 'swin_ffn', 'soft', 'soft_ffn'):
    r = ffn['models'][key]
    label = {'swin': 'Swin-T', 'swin_ffn': 'Swin-T + FFN', 'soft': '软分组GAViT', 'soft_ffn': '软分组GAViT + FFN'}[key]
    close(r['correct'] / 1260 * 100, r['oa_percent'])
    row(label, f"{r['oa_percent']:.4f}", f"{r['macro_f1_percent']:.4f}", r['best_epoch'], f"{r['correct']:,}")
for name, c in ffn['paired_validation'].items():
    a, b = name.split('__versus__')
    assert sum(c[k] for k in ('both_correct', 'corrects', 'new_errors', 'both_wrong')) == 1260
    assert c['both_correct'] + c['corrects'] == ffn['models'][a]['correct']
    assert c['both_correct'] + c['new_errors'] == ffn['models'][b]['correct']
    assert c['corrects'] - c['new_errors'] == c['net'] == sum(c['class_net'].values())
for control in ('swin', 'soft'):
    assert ffn['models'][control + '_ffn']['total_parameters'] - ffn['models'][control]['total_parameters'] == 297408
assert ffn['phase'] == 'calibrate' and ffn['seed'] == 42
assert all(ffn['models']['soft_ffn']['oa_percent'] < ffn['models'][k]['oa_percent'] for k in ('swin', 'soft', 'swin_ffn'))
checks.append('FFN: four validation rows, OA/counts, paired margins/class sums, added parameters and stopping comparisons')

er = {(r['model'], r['batch_size']): r for r in eff['rows']}
s, g = er['swin', 32], er['gavit', 32]
param_change = (g['parameters'] / s['parameters'] - 1) * 100
row('参数量 M', f"{s['parameters']/1e6:.4f}", f"{g['parameters']/1e6:.4f}", f'{param_change:+.2f}%')
for label, batch, field, nested in [('batch1平均延迟 ms', 1, 'batch_ms_mean', True), ('batch32吞吐 图/秒', 32, 'images_per_second', True), ('batch32峰值allocated MiB', 32, 'peak_allocated_mib', False)]:
    r1, r2 = er['swin', batch], er['gavit', batch]
    v1, v2 = (r1['timing'][field], r2['timing'][field]) if nested else (r1[field], r2[field])
    row(label, f'{v1:.3f}', f'{v2:.3f}', f'{(v2/v1-1)*100:+.2f}%')
checks.append('Efficiency: all four displayed rows regenerated; relative changes computed from recorded values')

assert feedback['samples'] == 1260 and feedback['alphas'] == [0, .25, .5, .75, 1]
for model, rows in feedback['curves'].items():
    original = rows[-1]
    assert all(r['oa_percent'] < original['oa_percent'] and r['macro_f1_percent'] < original['macro_f1_percent'] and r['cross_entropy'] > original['cross_entropy'] for r in rows[:-1])
    assert rows[0]['paired_vs_original']['net_correct'] == -9
assert f"{gallery['argmax_changed_fraction']*100:.2f}%" in text
assert f"{gallery['feature_edges_common_fraction']*100:.2f}%" in text
checks.append('Feedback direction at all prespecified points and gallery display percentages checked against archived reports')

links = []
for doc in (DRAFT, ROOT / 'docs/external_methods_refresh_20261006.md'):
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', doc.read_text()):
        if target.startswith(('https://', 'http://', '#')):
            continue
        p = (doc.parent / target.split('#')[0]).resolve()
        if p == Path(__file__).parent / 'validation.json':
            continue
        assert p.is_file(), (doc.name, target)
        links.append(str(p.relative_to(ROOT)))
checks.append('All local links and both existing figure files in the two new documents exist')

result = {'status': 'PASS', 'date': '2026-10-06',
          'scope': 'Archived-summary arithmetic, formatted numerical table rows and local-link validation; no raw-prediction rerun, GPU forward, full backup or independent scientific peer review',
          'table_rows_checked': len(table_rows), 'checks': checks,
          'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
          'draft_sha256': hashlib.sha256(DRAFT.read_bytes()).hexdigest(),
          'local_link_targets_checked': sorted(set(links)),
          'independent_agent_review': 'Not completed; both delegated read-only audit attempts returned usage-limit errors; main agent performed the checks above'}
(Path(__file__).parent / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'table_rows_checked': len(table_rows), 'checks': len(checks), 'local_link_targets': len(set(links))}))
