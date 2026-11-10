"""Render the verified AID records at their final 14.5 cm thesis width.

Run with the existing local plotting dependencies (Python 3.14):
PYTHONPATH=/private/tmp/gavit-aid-plotdeps-20261106 \
MPLCONFIGDIR=/private/tmp/gavit-aid-mplconfig-20261106 \
python3 thesis/ntu_gavit/provenance/render_external_aid_figures_20261108.py

Only the two new thesis PNGs are written. Original data, analysis and figures
are read-only. All numeric work uses the existing verified offline analyser.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
WIDTH_INCHES = 14.5 / 2.54
DPI = 300
OUTPUT = ROOT / 'thesis/ntu_gavit/images/external_aid_20261108'
RAW = ROOT / 'results/evidence/hgtnet_aid_formal_complete_20261106/raw'
OLD_FIGURES = ROOT / 'results/figures/hgtnet_aid_seed42_20261106'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes():
    paths = [p for p in RAW.rglob('*') if p.is_file()]
    paths += [p for p in OLD_FIGURES.iterdir() if p.is_file()]
    paths += list((ROOT / 'results').glob('hgtnet_aid_seed42_*.csv'))
    paths += [ROOT / 'tools/analyze_hgtnet_aid_results.py',
              ROOT / 'results/hgtnet_aid_seed42_analysis_20261106.json',
              ROOT / 'results/hgtnet_aid_seed42_analysis_20261106.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def save_png(fig, name):
    path = OUTPUT / (name + '.png')
    # No tight bounding box: preserving the physical width preserves point sizes.
    fig.savefig(path, dpi=DPI, facecolor='white', metadata={'Software': 'Verified AID thesis rendering'})
    plt.close(fig)
    with Image.open(path) as image:
        size, dpi = image.size, image.info.get('dpi')
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
            'pixel_size': size, 'dpi': dpi, 'intended_width_cm': 14.5}


def history_figure(rows):
    calibration = [r for r in rows if r['phase'] == 'calibration']
    refit = [r for r in rows if r['phase'] == 'refit']
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_INCHES, 6.35))
    fig.subplots_adjust(left=.105, right=.985, bottom=.085, top=.86, wspace=.36, hspace=.46)
    fig.suptitle('Independent HGTNet / AID / seed 42\nComplete accepted history, without smoothing',
                 fontsize=10, y=.98, linespacing=1.35)
    x = [r['epoch'] for r in calibration]
    axes[0, 0].plot(x, [r['train_loss'] for r in calibration], color='#2f6690', label='Online train', lw=1.1)
    axes[0, 0].plot(x, [r['val_loss'] for r in calibration], color='#d47728', label='Validation', lw=1.1)
    axes[0, 0].set(title='(a) Calibration loss\nAll 100 epochs', ylabel='Cross-entropy loss', yscale='log')
    axes[0, 0].legend(loc='upper right', fontsize=8, handlelength=1.4)
    axes[0, 1].plot(x, [r['train_accuracy_percent'] for r in calibration], color='#2f6690',
                    label='Online train acc.', lw=1.1)
    axes[0, 1].plot(x, [r['val_accuracy_percent'] for r in calibration], color='#d47728',
                    label='Validation acc.', lw=1.1)
    axes[0, 1].plot(x, [r['val_macro_f1_percent'] for r in calibration], color='#5a7f41',
                    label='Validation F1', lw=1.1)
    axes[0, 1].set(title='(b) Calibration scores\nBest 96; final 100', ylabel='Score (%)', ylim=(55, 102))
    axes[0, 1].legend(loc='lower left', fontsize=8, handlelength=1.4)
    for ax in axes[0]:
        ax.axvline(96, color='#8b3d7c', linestyle='--', lw=.9)
        ax.axvline(100, color='#444444', linestyle=':', lw=.9)
        ax.set_xlim(1, 103)
        ax.set_xticks([1, 25, 50, 75, 100])
    axes[0, 1].scatter([96, 100], [97.2, 97.1], c=['#8b3d7c', '#444444'], s=15, zorder=6)
    axes[0, 1].annotate('96: 97.2%\n100: 97.1%', xy=(96, 97.2), xytext=(38, 81),
                         arrowprops={'arrowstyle': '-', 'color': '#777777', 'lw': .7}, fontsize=8,
                         linespacing=1.3)
    rx = [r['epoch'] for r in refit]
    axes[1, 0].plot(rx, [r['train_loss'] for r in refit], color='#2f6690', lw=1.1)
    axes[1, 0].set(title='(c) Fresh pool-refit loss\nAll 96 epochs', ylabel='Cross-entropy loss', yscale='log')
    axes[1, 1].plot(rx, [r['train_accuracy_percent'] for r in refit], color='#2f6690', lw=1.1)
    axes[1, 1].set(title='(d) Fresh pool-refit acc.\nTraining only; all 96 epochs',
                    ylabel='Online train accuracy (%)', ylim=(60, 102))
    for ax in axes[1]:
        ax.set_xlim(1, 98)
        ax.set_xticks([1, 24, 48, 72, 96])
    for ax in axes.flat:
        ax.set_xlabel('Epoch', labelpad=3)
        ax.grid(alpha=.18, lw=.6)
        ax.tick_params(labelsize=8, pad=2)
    return save_png(fig, 'hgtnet_aid_seed42_training_history')


def confusion_figure(result, narrow_font):
    names = result['class_names']
    matrix = np.array(result['metrics']['test']['confusion_matrix'])
    normalized = matrix / matrix.sum(axis=1, keepdims=True) * 100
    # Reserve space for every full class name at 8 pt and use the final thesis width.
    fig = plt.figure(figsize=(WIDTH_INCHES, 6.85))
    ax = fig.add_axes([.19, .275, .66, .66 * WIDTH_INCHES / 6.85])
    image = ax.imshow(normalized, vmin=0, vmax=100, cmap='Blues', interpolation='nearest')
    indices = range(len(names))
    ax.set_xticks(indices, names, rotation=90, ha='center', fontsize=8, fontfamily='Arial Narrow')
    ax.set_yticks(indices, names, fontsize=8, fontfamily='Arial Narrow')
    ax.tick_params(length=2, pad=2)
    ax.set_xlabel('Predicted class', fontsize=8, labelpad=5)
    ax.set_ylabel('True class', fontsize=8, labelpad=5)
    ax.set_title('AID test confusion matrix\nAll 30 classes; row-normalized (%)', fontsize=9, pad=9)
    for i in indices:
        for j in indices:
            if matrix[i, j] > 0:
                value = normalized[i, j]
                # Keep every nonzero direction visibly nonzero after integer display.
                label = '<1' if value < 1 else str(int(round(value)))
                ax.text(j, i, label, ha='center', va='center', fontproperties=narrow_font,
                        color='white' if value > 50 else '#222222')
    cax = fig.add_axes([.875, .36, .018, .37])
    colorbar = fig.colorbar(image, cax=cax)
    colorbar.set_ticks([0, 20, 40, 60, 80, 100])
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    colorbar.set_label('Share of true class (%)', fontsize=8, labelpad=3)
    return save_png(fig, 'hgtnet_aid_seed42_confusion_matrix')


def main():
    before = protected_hashes()
    spec = importlib.util.spec_from_file_location('verified_aid_analysis', ROOT / 'tools/analyze_hgtnet_aid_results.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    result, _, history_rows = analysis.analyse(RAW)
    stored = json.loads((ROOT / 'results/hgtnet_aid_seed42_analysis_20261106.json').read_text())
    assert stored['metrics'] == result['metrics']
    assert len(result['class_names']) == 30 and result['nonzero_directional_confusion_count'] == 83
    assert len(history_rows) == 196 and result['selected_epoch'] == 96 and result['final_calibration_epoch'] == 100
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'legend.fontsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    narrow_path = font_manager.findfont(FontProperties(family='Arial Narrow'), fallback_to_default=False)
    narrow_font = FontProperties(fname=narrow_path, size=7)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts = [history_figure(history_rows), confusion_figure(result, narrow_font)]
    assert protected_hashes() == before, 'protected source or old derived artifact changed'
    print(json.dumps({'state': 'PASS', 'python': sys.version.split()[0], 'matplotlib': matplotlib.__version__,
                      'script_sha256': sha(__file__), 'source_analysis_sha256': sha(ROOT / 'tools/analyze_hgtnet_aid_results.py'),
                      'source_manifest_sha256': result['manifest_sha256'], 'classes': 30,
                      'nonzero_off_diagonal_directions': 83, 'calibration_epochs': 100, 'refit_epochs': 96,
                      'selected_epoch': 96, 'final_calibration_epoch': 100,
                      'axis_class_legend_font_pt': 8, 'cm_numeric_font_pt': 7,
                      'cm_numeric_font': narrow_path,
                      'cm_numeric_display': 'Percentages rounded to integer; nonzero values below 1% shown as <1. Color uses unrounded percentages.',
                      'protected_original_files_unchanged': len(before), 'artifacts': artifacts}, indent=2))


if __name__ == '__main__':
    main()
