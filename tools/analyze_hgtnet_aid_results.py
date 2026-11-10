"""Reconstruct AID diagnostics from immutable JSON records, without ML execution.

Only Matplotlib/Numpy are imported, and only inside the plotting boundary.
No source images, tensors, checkpoints, GPU libraries or network are used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/evidence/hgtnet_aid_formal_complete_20261106/raw'
PREFIX = 'hgtnet_aid_seed42'
DATE = '20261106'
MANIFEST_SHA = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def metrics_from_confusion(matrix, class_names):
    n = len(class_names)
    if n < 2 or len(set(class_names)) != n or len(matrix) != n:
        raise ValueError('invalid class map or confusion matrix')
    if any(len(row) != n or any(type(x) is not int or x < 0 for x in row) for row in matrix):
        raise ValueError('confusion matrix must contain nonnegative integer counts')
    total = sum(map(sum, matrix))
    if total == 0:
        raise ValueError('empty predictions')
    rows = []
    for i, name in enumerate(class_names):
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        tp = matrix[i][i]
        rows.append({'class_index': i, 'class_name': name, 'support': support,
                     'predicted_count': predicted, 'correct': tp, 'errors': support - tp,
                     'false_positives': predicted - tp,
                     'accuracy_percent': 100 * tp / support if support else 0.0,
                     'precision_percent': 100 * tp / predicted if predicted else 0.0,
                     'f1_percent': 200 * tp / (support + predicted) if support + predicted else 0.0})
    correct = sum(matrix[i][i] for i in range(n))
    return {'accuracy_percent': 100 * correct / total,
            'macro_f1_percent': sum(r['f1_percent'] for r in rows) / n,
            'samples': total, 'correct': correct, 'errors': total - correct,
            'confusion_matrix': matrix, 'per_class': rows}


def reconstruct_metrics(labels, predictions, class_names):
    if len(labels) != len(predictions) or not labels:
        raise ValueError('empty or mismatched prediction arrays')
    n = len(class_names)
    if any(type(x) is not int or not 0 <= x < n for x in [*labels, *predictions]):
        raise ValueError('class indices must be integers within the class map')
    matrix = [[0] * n for _ in range(n)]
    for truth, prediction in zip(labels, predictions):
        matrix[truth][prediction] += 1
    return metrics_from_confusion(matrix, class_names)


def validate_predictions(packet, manifest, split):
    rows = sorted((r for r in manifest['records'] if r['split'] == split), key=lambda r: r['path'])
    if packet['paths'] != [r['path'] for r in rows]:
        raise ValueError('prediction paths differ from original ordered split')
    if packet['labels'] != [r['label'] for r in rows]:
        raise ValueError('prediction labels differ from original ordered split')
    reconstruct_metrics(packet['labels'], packet['predictions'], manifest['class_names'])
    return rows


def earliest_best_epoch(history):
    if not history or [r['epoch'] for r in history] != list(range(1, len(history) + 1)):
        raise ValueError('history must retain every epoch in chronological order')
    if any(type(r['val_correct']) is not int or r['val_correct'] < 0 for r in history):
        raise ValueError('validation selection needs integer correct counts')
    return max(history, key=lambda r: r['val_correct'])['epoch']


def assert_metrics(actual, recorded):
    if actual['confusion_matrix'] != recorded['confusion_matrix']:
        raise ValueError('reported confusion matrix differs from reconstructed predictions')
    for key in ('accuracy_percent', 'macro_f1_percent'):
        if not math.isclose(actual[key], recorded[key], rel_tol=0, abs_tol=1e-10):
            raise ValueError('reported aggregate metric differs: ' + key)
    if len(actual['per_class']) != len(recorded['per_class']):
        raise ValueError('reported class count differs')
    for expected, row in zip(actual['per_class'], recorded['per_class']):
        if row.get('class', row.get('class_name')) != expected['class_name']:
            raise ValueError('reported class order differs')
        for key in ('support', 'correct', 'accuracy_percent', 'f1_percent'):
            if not math.isclose(expected[key], row[key], rel_tol=0, abs_tol=1e-10):
                raise ValueError('reported per-class metric differs: ' + key)


def write_csv(path, rows, fields=None):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def analyse(raw):
    """Verify original bytes/order and derive complete numerical diagnostics."""
    raw = Path(raw)
    hashes_before = {p.relative_to(raw).as_posix(): sha(p) for p in sorted(raw.rglob('*')) if p.is_file()}
    download = read_json(raw / 'download_manifest.json')
    for item in download['files']:
        p = raw / item['name']
        if sha(p) != item['sha256'] or p.stat().st_size != item['bytes']:
            raise ValueError('downloaded artifact identity mismatch: ' + item['name'])
    manifest_path = raw / 'aid_protocol_20260920_03.json'
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError('original AID manifest bytes changed')
    manifest = read_json(manifest_path)
    names = manifest['class_names']
    if manifest['dataset'] != 'AID' or len(names) != 30 or manifest['counts'] != {'train': 4000, 'val': 1000, 'test': 5000}:
        raise ValueError('original AID taxonomy/counts changed')
    binding = read_json(raw / 'binding.json')
    summary = read_json(raw / 'summary.json')
    history = read_json(raw / 'history.json')
    if binding['class_names'] != names or binding['manifest_sha256'] != MANIFEST_SHA:
        raise ValueError('binding class/manifest identity differs')
    for split in ('val', 'test'):
        expected = [{'path': r['path'], 'label': r['label']} for r in
                    sorted((r for r in manifest['records'] if r['split'] == split), key=lambda r: r['path'])]
        if binding['records'][split] != expected:
            raise ValueError('binding record order differs from original manifest')
    packets, metrics = {}, {}
    for key, filename, split in [('test', 'test.json', 'test'), ('best_validation', 'best_val_predictions.json', 'val'),
                                 ('final_validation', 'last_val_predictions.json', 'val')]:
        packet = read_json(raw / filename)
        validate_predictions(packet, manifest, split)
        if (packet['manifest_sha256'] != MANIFEST_SHA or packet['recipe_sha256'] != binding['recipe_sha256']
                or packet['input_scope'] != 'real_aid'):
            raise ValueError('evaluation identity differs')
        actual = reconstruct_metrics(packet['labels'], packet['predictions'], names)
        assert_metrics(actual, packet['metrics'])
        packets[key], metrics[key] = packet, actual
    cal, refit = history['calibration'], history['refit']
    selected = earliest_best_epoch(cal)
    if (len(cal) != 100 or len(refit) != 96 or selected != 96
            or [r['epoch'] for r in refit] != list(range(1, 97))
            or selected != summary['selected_epoch'] or selected != packets['best_validation']['epoch']
            or packets['final_validation']['epoch'] != 100
            or summary['calibration_history'] != cal or summary['refit_history'] != refit):
        raise ValueError('history or earliest validation winner differs')
    for phase, rows, steps, base in [('calibration', cal, 250, 0), ('refit', refit, 313, 25000)]:
        for row in rows:
            if row['epoch_optimizer_steps'] != steps or row['optimizer_steps'] != base + row['epoch'] * steps:
                raise ValueError('accepted optimizer trajectory differs')
            if phase == 'calibration':
                reconstructed = metrics_from_confusion(row['val_metrics']['confusion_matrix'], names)
                assert_metrics(reconstructed, row['val_metrics'])
                if reconstructed['correct'] != row['val_correct'] or reconstructed['samples'] != 1000:
                    raise ValueError('validation correct count differs from its matrix')
    assert_metrics(metrics['best_validation'], cal[selected - 1]['val_metrics'])
    assert_metrics(metrics['final_validation'], cal[-1]['val_metrics'])
    assert_metrics(metrics['test'], summary['test_metrics'])
    if refit[-1]['optimizer_steps'] != summary['optimizer_steps'] or summary['optimizer_steps'] != 55048:
        raise ValueError('final accepted optimizer step differs')
    root_audit = read_json(raw.parent / 'root_audit.json')
    independent = read_json(raw.parent / 'independent_audit.json')
    if root_audit['state'] != 'PASS' or not all(x['passed'] for x in independent['checks']):
        raise ValueError('existing completion audit did not pass')
    assert_metrics(metrics['test'], dict(root_audit['test_metrics'],
                   confusion_matrix=root_audit['confusion_matrix'], per_class=root_audit['per_class']))
    im = independent['metrics']
    if (metrics['test']['confusion_matrix'] != im['test_confusion_matrix']
            or not math.isclose(metrics['test']['macro_f1_percent'], im['test_macro_f1_percent'], abs_tol=1e-10)
            or metrics['test']['accuracy_percent'] != im['test_accuracy_percent']):
        raise ValueError('independent completion audit differs')
    matrix = metrics['test']['confusion_matrix']
    predictions = [{'index': i, 'path': path, 'true_label': label, 'true_class': names[label],
                    'predicted_label': prediction, 'predicted_class': names[prediction],
                    'correct': label == prediction}
                   for i, (path, label, prediction) in enumerate(zip(packets['test']['paths'], packets['test']['labels'],
                                                                  packets['test']['predictions']))]
    directional = [{'true_label': i, 'true_class': names[i], 'predicted_label': j, 'predicted_class': names[j],
                    'count': matrix[i][j], 'percent_of_true_class': 100 * matrix[i][j] / sum(matrix[i])}
                   for i in range(len(names)) for j in range(len(names)) if i != j]
    directional.sort(key=lambda r: (-r['count'], r['true_label'], r['predicted_label']))
    pairs = [{'class_a': names[i], 'class_b': names[j], 'a_to_b': matrix[i][j], 'b_to_a': matrix[j][i],
              'total': matrix[i][j] + matrix[j][i]}
             for i in range(len(names)) for j in range(i + 1, len(names))]
    pairs.sort(key=lambda r: (-r['total'], r['class_a'], r['class_b']))
    history_rows = []
    for phase, rows in [('calibration', cal), ('refit', refit)]:
        for row in rows:
            value = {'phase': phase, **{k: v for k, v in row.items() if k != 'val_metrics'}}
            value['val_accuracy_percent'] = row.get('val_metrics', {}).get('accuracy_percent')
            value['val_macro_f1_percent'] = row.get('val_metrics', {}).get('macro_f1_percent')
            history_rows.append(value)
    phase_resources = {}
    for phase, rows in [('calibration', cal), ('refit', refit)]:
        phase_resources[phase] = {
            'epochs': len(rows), 'train_seconds_sum': sum(r['train_seconds'] for r in rows),
            'epoch_wall_seconds_sum_available': sum(r.get('epoch_wall_seconds', 0) for r in rows),
            'epoch_wall_missing_epochs': [r['epoch'] for r in rows if 'epoch_wall_seconds' not in r],
            'online_train_images_per_second_median': statistics.median(r['train_images_per_second'] for r in rows)}
    result = {
        'schema': 'hgtnet-aid-seed42-offline-analysis-v1', 'dataset': 'AID', 'seed': 42,
        'scope': 'single-seed independent nonofficial complete-method diagnostics; offline record analysis',
        'source_raw_directory': str(raw.relative_to(ROOT)), 'source_files_sha256': hashes_before,
        'downloaded_files_verified': len(download['files']), 'manifest_sha256': MANIFEST_SHA,
        'recipe_sha256': binding['recipe_sha256'], 'counts': binding['counts'], 'class_names': names,
        'metrics': metrics, 'selected_epoch': selected, 'final_calibration_epoch': 100, 'refit_epochs': 96,
        'accepted_optimizer_steps': summary['optimizer_steps'],
        'final_calibration': {k: v for k, v in cal[-1].items() if k != 'val_metrics'},
        'final_refit': refit[-1], 'parameters': summary['parameters'], 'recipe': summary['recipe'],
        'directional_confusions_all_off_diagonal': directional, 'bidirectional_confusion_pairs': pairs,
        'nonzero_directional_confusion_count': sum(r['count'] > 0 for r in directional),
        'resource': {
            'resumed_attempt': summary['attempt'], 'gpu': summary['runtime']['gpu'],
            'attempt_wall_seconds': summary['wall_seconds'], 'attempt_engine_seconds': summary['engine_wall_seconds'],
            'attempt_peak_allocated_bytes': summary['cuda_peak_allocated_bytes'],
            'attempt_peak_reserved_bytes': summary['cuda_peak_reserved_bytes'],
            'wall_scope': summary['wall_scope'], 'peak_scope': summary['memory_scope'],
            'whole_experiment_wall_seconds': None, 'whole_experiment_peak_bytes': None, 'actual_cost': None,
            'accepted_history': phase_resources,
            'accepted_history_scope': 'Completed accepted epochs only; refit75 wall field absent. Excludes discarded partial updates/setup and is not total experiment wall or billing.',
            'test_evaluation_seconds': packets['test']['seconds'],
            'test_evaluation_timing_scope': 'Recorded final test evaluation only; not a hardware-normalized inference benchmark.',
            'discarded_partial_optimizer_steps': summary['discarded_partial_optimizer_steps']},
        'audit_agreement': {'root_audit': 'PASS', 'independent_audit_checks_passed': len(independent['checks']),
                            'metrics_tolerance': 1e-10, 'all_classes_all_confusion_cells': True},
        'checkpoint_verification': {'local_model_loading_performed': False,
            'scope': 'Local returned receipts and prior audits only; checkpoint binary hashes/reload were verified on server.',
            'final_refit_sha256': packets['test']['refit_checkpoint_sha256']},
        'limitations': [
            'One seed42 independent implementation; no multiseed significance or unified-backbone/recipe superiority claim.',
            'Class errors/confusions are descriptive associations and do not establish a module-specific mechanism.',
            'Online train metrics aggregate changing weights; validation is epoch-end evaluation; refit has no validation.',
            'All test paths/labels checked against the original manifest; no image or model loading, test replay or new training.',
            'Current-attempt resources cover resumed attempt2 only. Whole-experiment wall/peak/cost are unavailable.',
            'No paired correction/regression counts versus Swin/GAViT are inferred from aggregate totals.']}
    if hashes_before != {p.relative_to(raw).as_posix(): sha(p) for p in sorted(raw.rglob('*')) if p.is_file()}:
        raise ValueError('raw artifacts changed during offline analysis')
    return result, predictions, history_rows


def export_numeric(result, predictions, history_rows, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f'{PREFIX}'
    paths = {}
    tables = {'per_class': result['metrics']['test']['per_class'], 'predictions': predictions,
              'errors': [r for r in predictions if not r['correct']],
              'confusions': result['directional_confusions_all_off_diagonal'],
              'confusion_pairs': result['bidirectional_confusion_pairs'], 'history': history_rows}
    for name, rows in tables.items():
        path = output / f'{stem}_{name}_{DATE}.csv'
        fields = list(dict.fromkeys(k for row in rows for k in row))
        write_csv(path, rows, fields)
        paths[name] = str(path.relative_to(ROOT))
    path = output / f'{stem}_confusion_matrix_{DATE}.csv'
    names = result['class_names']
    write_csv(path, [dict(true_class=name, **dict(zip(names, row))) for name, row in
                     zip(names, result['metrics']['test']['confusion_matrix'])], ['true_class', *names])
    paths['confusion_matrix'] = str(path.relative_to(ROOT))
    return paths


def plot_figures(result, history_rows, figure_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'hgtnet-aid-seed42-20261106'})
    artifacts = []
    def save(fig, name):
        for suffix in ('png', 'svg'):
            path = figure_dir / f'{PREFIX}_{name}.{suffix}'
            metadata = {'Date': None} if suffix == 'svg' else {'Software': 'Offline AID analysis'}
            fig.savefig(path, dpi=200, bbox_inches='tight', metadata=metadata)
            artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size})
        plt.close(fig)

    # Every epoch is plotted without smoothing; refit never uses test/validation.
    cal = [r for r in history_rows if r['phase'] == 'calibration']
    refit = [r for r in history_rows if r['phase'] == 'refit']
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 7.6), layout='constrained')
    x = [r['epoch'] for r in cal]
    axes[0, 0].plot(x, [r['train_loss'] for r in cal], label='Online train', color='#2f6690')
    axes[0, 0].plot(x, [r['val_loss'] for r in cal], label='Validation', color='#d47728')
    axes[0, 0].set(title='(a) Calibration loss: all 100 epochs', ylabel='Cross-entropy loss', yscale='log')
    axes[0, 1].plot(x, [r['train_accuracy_percent'] for r in cal], label='Online train accuracy', color='#2f6690')
    axes[0, 1].plot(x, [r['val_accuracy_percent'] for r in cal], label='Validation accuracy', color='#d47728')
    axes[0, 1].plot(x, [r['val_macro_f1_percent'] for r in cal], label='Validation macro-F1', color='#5a7f41')
    axes[0, 1].set(title='(b) Calibration scores: best 96 / final 100', ylabel='Score (%)', ylim=(55, 101))
    for ax in axes[0]:
        ax.axvline(96, color='#8b3d7c', linestyle='--', lw=1, label='Selected epoch 96')
        ax.axvline(100, color='#444444', linestyle=':', lw=1, label='Final epoch 100')
        ax.set_xlim(1, 102)
    axes[0, 1].scatter([96, 100], [97.2, 97.1], c=['#8b3d7c', '#444444'], s=26, zorder=5)
    axes[0, 1].annotate('96: 97.2%\n100: 97.1%', xy=(96, 97.2), xytext=(63, 76),
                         arrowprops={'arrowstyle': '-', 'color': '#777777'}, fontsize=9)
    rx = [r['epoch'] for r in refit]
    axes[1, 0].plot(rx, [r['train_loss'] for r in refit], color='#2f6690', label='Fresh pool refit')
    axes[1, 0].set(title='(c) Fresh refit loss: all 96 epochs', ylabel='Cross-entropy loss', yscale='log')
    axes[1, 1].plot(rx, [r['train_accuracy_percent'] for r in refit], color='#2f6690', label='Online pool-train accuracy')
    axes[1, 1].set(title='(d) Fresh refit accuracy (training only)', ylabel='Accuracy (%)', ylim=(60, 101))
    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('Independent HGTNet / AID / seed 42 — unsmoothed accepted history', fontsize=13)
    save(fig, 'training_history')

    rows = sorted(result['metrics']['test']['per_class'], key=lambda r: (r['accuracy_percent'], r['class_index']))
    fig, ax = plt.subplots(figsize=(10.8, 10.2), layout='constrained')
    y = np.arange(len(rows))
    ax.hlines(y, [r['accuracy_percent'] for r in rows], [r['f1_percent'] for r in rows], color='#c7c7c7', lw=1.4)
    ax.scatter([r['accuracy_percent'] for r in rows], y, color='#2f6690', s=32, label='Recall / per-class accuracy')
    ax.scatter([r['f1_percent'] for r in rows], y, color='#d47728', marker='s', s=26, label='F1')
    ax.set_yticks(y, [f"{r['class_name']}  (n={r['support']})" for r in rows])
    ax.invert_yaxis()
    ax.set(xlim=(82, 101), xlabel='Test score (%)', title='All 30 AID classes — sorted by recall (ascending)')
    ax.grid(axis='x', alpha=.22)
    ax.legend(loc='lower left')
    save(fig, 'all_class_performance')

    names = result['class_names']
    matrix = np.array(result['metrics']['test']['confusion_matrix'])
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(12.8, 11.0), layout='constrained')
    im = ax.imshow(normalized, vmin=0, vmax=100, cmap='Blues', interpolation='nearest')
    ticks = range(len(names))
    ax.set_xticks(ticks, names, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(ticks, names, fontsize=8)
    ax.set(xlabel='Predicted class', ylabel='True class', title='Complete 30 × 30 AID test confusion matrix (row-normalized)')
    for i in ticks:
        for j in ticks:
            if matrix[i, j] > 0:
                ax.text(j, i, f'{normalized[i,j]:.0f}' if i == j else f'{normalized[i,j]:.1f}',
                        ha='center', va='center', fontsize=6.5,
                        color='white' if normalized[i, j] > 50 else '#222222')
    fig.colorbar(im, ax=ax, shrink=.75, label='Share of true class (%)', pad=.02)
    save(fig, 'confusion_matrix')

    pairs = result['bidirectional_confusion_pairs'][:10]
    fig, ax = plt.subplots(figsize=(11.7, 5.7), layout='constrained')
    y = np.arange(len(pairs))
    ax.barh(y, [r['a_to_b'] for r in pairs], color='#2f6690', label='First class → second')
    ax.barh(y, [r['b_to_a'] for r in pairs], left=[r['a_to_b'] for r in pairs],
            color='#d47728', label='Second class → first')
    ax.set_yticks(y, [f"{r['class_a']} / {r['class_b']}" for r in pairs], fontsize=9)
    ax.invert_yaxis()
    for i, row in enumerate(pairs):
        ax.text(row['total'] + .15, i, f"{row['a_to_b']} + {row['b_to_a']} = {row['total']}", va='center', fontsize=9)
    ax.set(xlim=(0, max(r['total'] for r in pairs) + 4), xlabel='Misclassified test images',
           title='Top 10 bidirectional confusion pairs — both directions retained')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=.18)
    save(fig, 'top10_bidirectional_confusions')
    return {'matplotlib': matplotlib.__version__, 'numpy': np.__version__, 'artifacts': artifacts,
            'confusion_normalization': '100 * count / true-class support; true rows, predicted columns',
            'history_scope': '100 calibration + 96 fresh refit epochs, no smoothing or omissions'}


def report_markdown(result, paths):
    test = result['metrics']['test']
    best = result['metrics']['best_validation']
    final = result['metrics']['final_validation']
    lowest = sorted(test['per_class'], key=lambda r: (r['accuracy_percent'], r['class_index']))[:6]
    pairs = result['bidirectional_confusion_pairs'][:10]
    resource = result['resource']
    lines = ['# HGTNet AID seed42 离线诊断（2026-11-06）', '',
             f"从原manifest和5000条有序预测独立重算：Accuracy **{test['accuracy_percent']:.2f}%**，macro-F1 **{test['macro_f1_percent']:.12f}%**；{test['correct']}正确、{test['errors']}错误。全部30类和30×30混淆矩阵与现有root/独立审计一致。", '',
             '这是单seed独立非官方实现的完整方法比较。类别与混淆分析属于描述性诊断，不能据此推断图模块因果贡献、统计显著性或统一配方优势。', '',
             '## 数据与选择', '',
             '- 原AID manifest SHA256：`' + result['manifest_sha256'] + '`。固定train4000、val1000、fresh pool5000、test5000；30类不等量；原Industrial两张重复图均在train。',
             '- 全部测试路径和标签必须逐项等于原manifest有序test split；遇到错序或标签变化直接拒绝，不自动对齐。46项下载文件SHA/大小再次核对；所有原始文件保持不变。',
             f"- 校准100轮，按验证正确数严格增加选择最早最佳轮96：验证Accuracy {best['accuracy_percent']:.2f}%，macro-F1 {best['macro_f1_percent']:.12f}%。",
             f"- 最后校准轮100：验证Accuracy {final['accuracy_percent']:.2f}%，macro-F1 {final['macro_f1_percent']:.12f}%，loss {result['final_calibration']['val_loss']:.12f}；训练Accuracy {result['final_calibration']['train_accuracy_percent']:.2f}%，loss {result['final_calibration']['train_loss']:.12f}。",
             f"- 等初态fresh pool refit96轮；末轮训练Accuracy {result['final_refit']['train_accuracy_percent']:.2f}%，loss {result['final_refit']['train_loss']:.12f}。55048次接受更新；训练100%不等于test指标。", '',
             '## 全类别与混淆', '',
             '| Class | Support | Errors | Recall (%) | F1 (%) |', '|---|---:|---:|---:|---:|']
    lines += [f"| {r['class_name']} | {r['support']} | {r['errors']} | {r['accuracy_percent']:.6f} | {r['f1_percent']:.6f} |" for r in test['per_class']]
    lines += ['', '低召回类别依次为：' + '、'.join(f"{r['class_name']} {r['accuracy_percent']:.2f}% ({r['errors']}/{r['support']}错)" for r in lowest) + '。这是该已训练模型在固定test split上的分布，不代表全数据集或模块机制。', '',
              f"全部870个非对角方向（含零）保存在CSV，其中{result['nonzero_directional_confusion_count']}个方向非零、合计154张错图；435个无序类别对保留双向计数。下面仅按合计错数列出前10对，不能替代完整矩阵。", '',
              '| Class A | Class B | A → B | B → A | Total |', '|---|---|---:|---:|---:|']
    lines += [f"| {r['class_a']} | {r['class_b']} | {r['a_to_b']} | {r['b_to_a']} | {r['total']} |" for r in pairs]
    lines += ['', '## 完整训练历史与资源范围', '',
              f"模型参数{result['parameters']['total']:,}，均可训练。当前完成summary的attempt2墙钟{resource['attempt_wall_seconds']:.6f}秒（{resource['attempt_wall_seconds']/60:.2f}分钟），峰值allocated {resource['attempt_peak_allocated_bytes']/2**30:.6f} GiB、reserved {resource['attempt_peak_reserved_bytes']/2**30:.6f} GiB；仅覆盖恢复后的当前调用。全实验墙钟、全程峰值与费用均未获得。",
              '恢复按refit75已接受提交继续，旧实例180次未提交更新被丢弃并重算，未双计。历史CSV保留全部100+96轮；refit75没有epoch_wall_seconds，空值保留。可获得的轮时求和只覆盖接受轨迹中已记录的完整轮，排除废弃更新、setup等，不能当总耗时或账单。',
              f"最终test评估记录{resource['test_evaluation_seconds']:.6f}秒；其测量范围不是跨硬件推理速度基准。模型重载误差0属于服务器原记录，本分析没有本地下载/加载大模型。", '',
              '在线train指标聚合轮内不断变化的参数，validation是轮末评估；fresh refit没有validation。全部轮次原样绘制，无平滑或挑选。best96和final100同时标注。', '',
              '## 工件', '']
    lines += [f"- [{name}]({Path(path).name})" for name, path in paths.items()]
    lines += ['', '四张图均提供PNG和SVG。CM为按true-class support归一化，true为行、predicted为列，保留全部类别与对角线。', '']
    for name in ('training_history', 'all_class_performance', 'confusion_matrix', 'top10_bidirectional_confusions'):
        base = f'figures/{PREFIX}_{DATE}/{PREFIX}_{name}'
        lines += [f'![{name}]({base}.png)', f'[SVG]({base}.svg)', '']
    lines += ['## 复现边界', '', '脚本只读原JSON、下载清单和现有两个审计，使用标准库重算全部指标；仅绘图时导入Matplotlib/Numpy。没有网络、GPU模型执行、新训练、图片解码或test前向。没有从aggregate总数推断与Swin/GAViT配对的纠正/新增错误数。', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=RAW)
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--figures', type=Path, default=ROOT / f'results/figures/{PREFIX}_{DATE}')
    args = parser.parse_args(argv)
    result, predictions, history_rows = analyse(args.raw)
    paths = export_numeric(result, predictions, history_rows, args.output)
    result['figures'] = plot_figures(result, history_rows, args.figures)
    result['outputs'] = paths
    result['analysis_runtime'] = {'python': platform.python_version(), 'executable': sys.executable,
                                  'script_sha256': sha(__file__)}
    json_path = args.output / f'{PREFIX}_analysis_{DATE}.json'
    md_path = args.output / f'{PREFIX}_analysis_{DATE}.md'
    write_json(json_path, result)
    md_path.write_text(report_markdown(result, paths), encoding='utf-8')
    print(json.dumps({'state': 'PASS', 'accuracy_percent': result['metrics']['test']['accuracy_percent'],
                      'macro_f1_percent': result['metrics']['test']['macro_f1_percent'],
                      'test_rows': len(predictions), 'error_rows': result['metrics']['test']['errors'],
                      'history_rows': len(history_rows), 'selected_epoch': result['selected_epoch'],
                      'json': str(json_path), 'report': str(md_path), 'figures': len(result['figures']['artifacts'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
