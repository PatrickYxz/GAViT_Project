"""Validate the user-returned summary; keep source precision and provenance."""
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path('/Users/patrick/.codex/attachments/13d352ff-0186-4655-bd31-85cfbe6d1ba3/已粘贴的文本.txt')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def csv_text(fields, rows, header=True):
    out = io.StringIO(newline='')
    w = csv.DictWriter(out, fieldnames=fields, lineterminator='\n')
    if header:
        w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def main():
    data = SOURCE.read_bytes()
    (HERE / 'reported_summary_output.txt').write_bytes(data)
    text = data.decode('utf-8')
    assert re.search(r'^(?:QUEUE_EXIT|UEUE_EXIT): exit_code=0$', text, re.M)
    exits = ast.literal_eval(re.search(r'^STAGE_EXITS: (.+)$', text, re.M).group(1))
    expected_stages = ('swin_smoke', 'gavit_smoke', 'swin_train', 'swin_test', 'gavit_train', 'gavit_test')
    assert set(exits) == set(expected_stages) and set(exits.values()) == {'exit_code=0'}
    start = text.index('\n{') + 1
    result, end = json.JSONDecoder().raw_decode(text[start:])
    assert text[start + end:].strip() in ('', '(base) ➜ ~')
    assert result['status'] == 'complete' and result['dataset'] == 'BigEarthNet-19' and result['seed'] == 45
    stages = result['stages']
    assert set(stages) == set(expected_stages)
    identity = result['common_identity']
    assert identity['commit'] == 'deb2b457e86b6fa2feccb5a305e099b6d9ec6dfe'
    assert identity['launcher_sha256'] == sha(ROOT / 'tools/run_bigearth_seed45.py')
    assert identity['pretrained_sha256'] == 'f7e9953f51dd5e339023b3ec97dea7bc3c6ca61259c2a80031da08c3006a9d36'
    # Frozen counts and hashes from the previously audited launcher, not raw data.
    code = ast.parse((ROOT / 'tools/run_bigearth_seed45.py').read_text())
    frozen = next(ast.literal_eval(n.value) for n in code.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'FROZEN' for t in n.targets))
    assert identity['splits'] == {k: list(v) for k, v in frozen.items()}
    classes_ast = ast.parse((ROOT / 'models/bigearth_dataset.py').read_text())
    classes = next(ast.literal_eval(n.value) for n in classes_ast.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CLASSES_19' for t in n.targets))
    recipe = {'seed': 45, 'epochs': 30, 'batch_size': 32, 'lr': 0.0003, 'weight_decay': 0.0001, 'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingLR', 'loss': 'BCEWithLogitsLoss', 'f1_threshold': 0.5}
    for stage, r in stages.items():
        assert r['stage'] == stage and r['model'] == stage.split('_')[0]
        assert r['status'] == 'complete' and r['seed'] == 45
        assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0
        assert r['peak_cuda_reserved_bytes'] >= r['peak_cuda_allocated_bytes'] > 0
        if stage.endswith('_smoke'):
            assert r['final_epoch'] == 1 and r['reload_max_logit_diff'] < 1e-5
    for model, params in [('swin', 27535501), ('gavit', 31366926)]:
        train, test = stages[model + '_train'], stages[model + '_test']['metrics']
        meta = train['metadata']
        assert train['final_epoch'] == 30 and train['parameters'] == params
        assert meta['model'] == model and meta['dataset'] == 'BigEarthNet-19'
        assert meta['training'] == recipe and meta['execution']['run_stage'] == 'formal'
        assert meta['architecture']['num_classes'] == 19 and meta['architecture']['dropout'] == 0.1
        assert meta['git'] == {'commit': 'deb2b45', 'dirty': False}
        assert meta['best'] == train['best'] and 1 <= train['best']['epoch'] <= 30
        assert meta['checkpoint_path'] == train['checkpoint']
        assert re.fullmatch('[a-f0-9]{64}', train['checkpoint_sha256'])
        assert train['checkpoint_sha256'] == meta['checkpoint_sha256'] == test['checkpoint_sha256']
        assert train['checkpoint_sha256'] != stages[model + '_smoke']['checkpoint_sha256']
        assert test['samples'] == 119825 and test['f1_threshold'] == 0.5
        assert list(test['per_class_ap_percent']) == classes
        assert all(math.isfinite(v) and 0 <= v <= 100 for v in test['per_class_ap_percent'].values())
        assert math.isclose(statistics.mean(test['per_class_ap_percent'].values()), test['mAP_percent'], abs_tol=1e-10)
    graph = stages['gavit_train']['metadata']['graph_topology']
    assert graph['name'] == 'sparse_hybrid_4n_top2' and graph['feature_k'] == 2
    assert (graph['spatial_directed_edges'], graph['feature_directed_edges'], graph['total_directed_edges']) == (48, 32, 80)
    assert stages['gavit_train']['metadata']['architecture'] == {'num_classes': 19, 'num_regions': 16, 'knn_k': 5, 'gat_hidden': 256, 'gat_heads': 4, 'gat_layers': 2, 'dropout': 0.1, 'grouping': 'attentive_spatial', 'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'}
    hours = sum(r['elapsed_seconds'] for r in stages.values()) / 3600
    assert math.isclose(hours, result['stage_measured_hours_excluding_preflight_and_idle'], abs_tol=1e-10)
    old = json.loads((HERE / 'reported_metrics.json').read_text())
    g = stages['gavit_test']['metrics']
    assert all(abs(g['per_class_ap_percent'][k] - v) <= .050001 for k, v in old['reported_per_class_ap_percent'].items())
    write_json(HERE / 'summary.json', result)
    write_json(HERE / 'summary_provenance.json', {'source_attachment': str(SOURCE), 'source_sha256': sha(SOURCE), 'is_original_server_summary_file': False, 'normalization': 'JSON object extracted unchanged; retained original terminal prefix UEUE_EXIT, exit dictionary, and trailing shell prompt in reported_summary_output.txt.', 'boundary': 'Reported identity and checkpoint-hash chain verified. Original checkpoint bytes, full history, prediction arrays and per-stage identity fields omitted by the readout command were not locally received.'})

    a, b = stages['swin_test']['metrics'], stages['gavit_test']['metrics']
    metric_keys = ('mAP_percent', 'macro_f1_percent', 'micro_f1_percent')
    differences = {k: b[k] - a[k] for k in metric_keys}
    class_rows = [{'Class': c, 'Swin_AP(%)': a['per_class_ap_percent'][c], 'GAViT_AP(%)': b['per_class_ap_percent'][c], 'Delta_AP(pp)': b['per_class_ap_percent'][c] - a['per_class_ap_percent'][c], 'Contribution_to_macro_mAP(pp)': (b['per_class_ap_percent'][c] - a['per_class_ap_percent'][c]) / 19} for c in classes]
    assert math.isclose(sum(r['Contribution_to_macro_mAP(pp)'] for r in class_rows), differences['mAP_percent'], abs_tol=1e-10)
    (ROOT / 'results/bigearth_seed45_per_class_comparison.csv').write_text(csv_text(list(class_rows[0]), class_rows))

    ledger = ROOT / 'results/bigearth_comparison.csv'
    before = ledger.read_bytes()
    rows = list(csv.DictReader(io.StringIO(before.decode('utf-8'))))
    fields = list(rows[0])
    add = []
    for model in ('swin', 'gavit'):
        train, test_stage = stages[model + '_train'], stages[model + '_test']
        test = test_stage['metrics']
        row = dict.fromkeys(fields, '')
        row.update({'Model': 'Swin-T Baseline' if model == 'swin' else 'GAViT v2', 'Variant': 'global_pool' if model == 'swin' else 'sparse-hybrid K16 attentive_spatial 4n top2 token_feedback', 'Dataset': 'BigEarthNet-19', 'Seed': '45', 'Best_Val_mAP(%)': train['best']['value'], 'Best_Epoch': train['best']['epoch'], 'Final_Val_mAP(%)': '' if model == 'swin' else old['last_epoch_exact']['val_mAP_percent'], 'Test_mAP(%)': test['mAP_percent'], 'Test_Macro_F1(%)': test['macro_f1_percent'], 'Test_Micro_F1(%)': test['micro_f1_percent'], 'Params(M)': train['parameters'] / 1e6, 'Epochs': 30, 'Batch_Size': 32, 'LR': 0.0003, 'Weight_Decay': 0.0001, 'Optimizer': 'AdamW', 'Scheduler': 'CosineAnnealingLR', 'Threshold': .5, 'Git_Commit': 'deb2b45', 'GPU': identity['hardware']['gpu'], 'Peak_GPU_Memory(GiB)': train['peak_cuda_allocated_bytes'] / 2**30, 'Train_Wall_Time(h)': train['elapsed_seconds'] / 3600, 'Test_Time': f"{test_stage['elapsed_seconds']:.6f} s (entire measured stage)", 'Notes': 'Seed45 formal 30/30; queue and all six stage exits reported 0. Exact returned metrics; best metadata/train/test hash chain agrees, checkpoint bytes not received locally. Same-run shared GPU/software/precision supported by common identity and verified launcher hash. CUDA peak includes saved-state verification; RSS excludes workers. Stage timing excludes preflight/import/idle. Swin final val/history pending; GAViT final val from preceding same-run epoch30 excerpt. Source: results/evidence/bigearth_seed45_20260922_01/summary.json'})
        existing = [r for r in rows if r['Seed'] == '45' and r['Model'] == row['Model']]
        if existing:
            assert len(existing) == 1 and all(str(v) == existing[0][k] for k, v in row.items())
        else:
            add.append(row)
    if add:
        assert len(add) == 2
        ledger.write_bytes(before + csv_text(fields, add, header=False).encode())
        assert ledger.read_bytes().startswith(before)
    rows = list(csv.DictReader(ledger.open(newline='')))
    selected = {}
    for model in ('swin', 'gavit'):
        selected[model] = []
        for seed in ('42', '43', '44', '45'):
            group = [r for r in rows if r['Seed'] == seed and (r['Model'] == 'Swin-T Baseline' if model == 'swin' else 'sparse-hybrid' in r['Variant'])]
            assert len(group) == 1
            selected[model].append(group[0])
    mapping = {'mAP_percent': 'Test_mAP(%)', 'macro_f1_percent': 'Test_Macro_F1(%)', 'micro_f1_percent': 'Test_Micro_F1(%)'}
    models, paired = {}, {}
    for model in ('swin', 'gavit'):
        models[model] = {}
        for key, column in mapping.items():
            values = [float(r[column]) for r in selected[model]]
            models[model][key] = {'values': values, 'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
    for key in mapping:
        values = [g - s for s, g in zip(models['swin'][key]['values'], models['gavit'][key]['values'])]
        paired[key] = {'values': values, 'mean_pp': statistics.mean(values), 'sample_sd_pp': statistics.stdev(values)}
        assert math.isclose(paired[key]['mean_pp'], models['gavit'][key]['mean'] - models['swin'][key]['mean'], abs_tol=1e-10)
    caveat = 'Descriptive fixed-split summary of all predefined seeds42/43/44/45; ddof=1. Seeds42-44 retain original one-decimal reports; seed45 uses precise returned metrics, so aggregate precision is limited by historical rounding. Hardware/execution differ: GAViT43/44 historically stopped21/26; Swin44 observed through27 with final exit unconfirmed. Both45 completed30/30 with reported exit0. Not pure seed variance, equivalence or significance evidence. No scores excluded or replaced.'
    summary = {'dataset': 'BigEarthNet-19', 'seeds': [42, 43, 44, 45], 'n': 4, 'models': models, 'paired_gavit_minus_swin': paired, 'caveat': caveat}
    write_json(ROOT / 'results/bigearth_four_seed_summary.json', summary)
    summary_path = ROOT / 'results/bigearth_multiseed_summary.csv'
    old_summary = summary_path.read_bytes()
    snapshot = HERE / 'three_seed_summary_before_update.csv'
    if not snapshot.exists():
        assert '42;43;44,3,' in old_summary.decode()
        snapshot.write_bytes(old_summary)
    old_rows = list(csv.DictReader(io.StringIO(old_summary.decode())))
    out = []
    for model, title in [('swin', 'Swin-T Baseline'), ('gavit', 'Sparse-hybrid GAViT')]:
        out.append({'Model': title, 'Seeds': '42;43;44;45', 'N': 4, 'Test_mAP_mean(%)': models[model]['mAP_percent']['mean'], 'Test_mAP_sample_SD(pp)': models[model]['mAP_percent']['sample_sd'], 'Test_Macro_F1_mean(%)': models[model]['macro_f1_percent']['mean'], 'Test_Macro_F1_sample_SD(pp)': models[model]['macro_f1_percent']['sample_sd'], 'Test_Micro_F1_mean(%)': models[model]['micro_f1_percent']['mean'], 'Test_Micro_F1_sample_SD(pp)': models[model]['micro_f1_percent']['sample_sd'], 'Notes': caveat})
    summary_path.write_text(csv_text(list(old_rows[0]), out))
    coast = next(r for r in class_rows if r['Class'] == 'Coastal wetlands')
    analysis = {'status': 'reported_summary_validated', 'exit_codes': exits, 'queue_exit': 0, 'seed45_gavit_minus_swin_pp': differences, 'per_class': class_rows, 'classes_improved': sum(r['Delta_AP(pp)'] > 0 for r in class_rows), 'classes_declined': sum(r['Delta_AP(pp)'] < 0 for r in class_rows), 'coastal_wetlands_share_of_net_mAP_gap': coast['Contribution_to_macro_mAP(pp)'] / differences['mAP_percent'], 'stage_measured_hours': hours, 'reference_rate_yuan_per_hour_assumed': .69, 'stage_cost_at_reference_rate_yuan_not_bill': hours * .69, 'not_received': ['checkpoint bytes', 'complete history.json', 'predictions.npz', 'original six run.json files'], 'checks': ['six exit0 and complete stages', 'two formal30 epochs', 'seed45 smoke reload tolerance', 'known launcher byte hash', 'frozen split/pretrained identities', 'checkpoint report hash chain', '19 AP means and classes', 'previous rounded GAViT excerpt consistency', 'stage timing sum', 'paired and model means agree', 'historical ledger bytes preserved']}
    write_json(HERE / 'summary_validation.json', analysis)
    print(json.dumps({k: analysis[k] for k in ['seed45_gavit_minus_swin_pp', 'stage_measured_hours', 'coastal_wetlands_share_of_net_mAP_gap']}, indent=2))


if __name__ == '__main__':
    main()
