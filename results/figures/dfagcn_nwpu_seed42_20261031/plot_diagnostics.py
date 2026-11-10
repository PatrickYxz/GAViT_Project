#!/usr/bin/env python3
"""Plot already-computed prediction diagnostics; no fitting or evaluation runs.

Reproduce with:
MPLCONFIGDIR=/private/tmp/dfagcn-mpl-20261031 \
PYTHONPATH=/private/tmp/dfagcn-plotdeps-20261031 \
/private/tmp/gavit-dfagcn-cpu-20261027/bin/python \
results/figures/dfagcn_nwpu_seed42_20261031/plot_diagnostics.py
"""
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dfagcn-mpl-20261031")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "results/evidence/dfagcn_test_predictions_20261031/figures"
SOURCE = ROOT / "results/dfagcn_nwpu_seed42_analysis_20261031.json"
d = json.loads(SOURCE.read_text())
cm = np.asarray(d["confusion_matrix"], dtype=np.int64)
names = d["class_names"]
assert cm.shape == (45, 45)
assert np.all(cm.sum(axis=1) == 560)
assert cm.sum() == 25200 and cm.trace() == 21806
assert cm.sum() - cm.trace() == 3394
norm = cm / cm.sum(axis=1, keepdims=True) * 100
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})
label = lambda s: s.replace("_", " ")
saved = []
def save(fig, stem):
    for ext in ("png", "pdf", "svg"):
        path = OUT / f"{stem}.{ext}"
        # Preserve requested physical dimensions for the two-panel figure.
        fig.savefig(path, dpi=220)
        saved.append(path)
    plt.close(fig)

weak = d["weakest_recall_classes"][:10]
top = d["directed_confusions_top10"][:10]
for r in weak:
    i = names.index(r["class_name"])
    assert abs(norm[i, i] - r["accuracy_recall_percent"]) < 1e-10
for r in top:
    assert int(cm[names.index(r["true_class"]), names.index(r["predicted_class"])]) == r["count"]

fig = plt.figure(figsize=(13.5, 6.6))
ax1 = fig.add_axes([.155, .155, .285, .66])
ax2 = fig.add_axes([.775, .155, .195, .66])
fig.suptitle("DFAGCN independent implementation | NWPU-RESISC45 | seed 42", y=.965, fontsize=15, weight="bold")
fig.text(.5, .907, "Test set: 25,200 images / 45 classes / 560 per class; 3,394 errors", ha="center", fontsize=11, color="#444444")
y = np.arange(10)
vals = [r["accuracy_recall_percent"] for r in weak]
ax1.barh(y, vals, color="#377eb8", height=.66)
ax1.set_yticks(y, [label(r["class_name"]) for r in weak])
ax1.invert_yaxis()
ax1.set_xlim(0, 100)
ax1.set_xticks([0, 20, 40, 60, 80, 100])
ax1.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax1.set_xlabel("Per-class accuracy / recall (%)")
ax1.set_title("(a) Ten lowest class recalls", loc="left", pad=14)
ax1.set_axisbelow(True)
ax1.grid(axis="x", color="#e5e5e5", linewidth=.7)
for i, v in enumerate(vals):
    ax1.text(v+1.2, i, f"{v:.2f}%", va="center", fontsize=9.5)
counts = [r["count"] for r in top]
ax2.barh(y, counts, color="#d57836", height=.66)
ax2.set_yticks(y, [label(r["true_class"])+" → "+label(r["predicted_class"]) for r in top], fontsize=9.5)
ax2.invert_yaxis()
ax2.set_xlim(0, 95)
ax2.set_xticks([0, 20, 40, 60, 80])
ax2.set_xlabel("Misclassified images (count)")
ax2.set_title("(b) Ten largest directed confusions", loc="right", pad=14)
ax2.set_axisbelow(True)
ax2.grid(axis="x", color="#e5e5e5", linewidth=.7)
for i, v in enumerate(counts):
    ax2.text(v+1.5, i, str(v), va="center", fontsize=10)
fig.text(.5, .035, "Arrows indicate true class → predicted class. Descriptive diagnostics for one trained model and one seed.", ha="center", fontsize=9.5, color="#444444")
save(fig, "dfagcn_nwpu_seed42_error_summary")

fig, ax = plt.subplots(figsize=(17.5, 16.8))
fig.subplots_adjust(left=.16, bottom=.245, right=.885, top=.895)
edges = np.arange(46) - .5
im = ax.pcolormesh(edges, edges, norm, cmap="Blues", vmin=0, vmax=100,
                   shading="flat", edgecolors="#d9e3ec", linewidth=.22,
                   rasterized=False)
ax.set_aspect("equal")
ax.invert_yaxis()
ticks = np.arange(45)
ax.set_xticks(ticks, [label(n) for n in names], rotation=65, ha="right", rotation_mode="anchor", fontsize=10)
ax.set_yticks(ticks, [label(n) for n in names], fontsize=10)
ax.set_xlabel("Predicted class", fontsize=13, labelpad=14)
ax.set_ylabel("True class", fontsize=13, labelpad=14)
ax.set_xticks(np.arange(-.5, 45, 1), minor=True)
ax.set_yticks(np.arange(-.5, 45, 1), minor=True)
ax.grid(which="minor", color="#d9e3ec", linewidth=.22)
ax.tick_params(which="minor", bottom=False, left=False)
for i in range(45):
    ax.text(i, i, f"{norm[i,i]:.1f}", ha="center", va="center", fontsize=6.4, color="white" if norm[i,i] > 55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=.032, pad=.025, ticks=np.arange(0, 101, 20))
cb.solids.set_rasterized(False)
cb.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
cb.set_label("Share of true-class images (%)", fontsize=12, labelpad=12)
fig.suptitle("DFAGCN independent implementation | NWPU-RESISC45 | seed 42", y=.963, fontsize=18, weight="bold")
fig.text(.5, .931, "Row-normalized confusion matrix: each true-class row sums to 100% (560 images)", ha="center", fontsize=13)
fig.text(.5, .908, "25,200 test images; accuracy 86.5317%; 3,394 errors. Diagonal annotations are percentages.", ha="center", fontsize=11.5, color="#444444")
fig.text(.5, .045, "All 45 classes shown in dataset label order. Colors represent row percentages, not raw counts.", ha="center", fontsize=11, color="#444444")
save(fig, "dfagcn_nwpu_seed42_confusion_matrix_row_normalized")

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {
    "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
    "script": str(Path(__file__).resolve().relative_to(ROOT)), "script_sha256": sha(Path(__file__).resolve()),
    "python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
    "checks": {"matrix_shape": list(cm.shape), "samples": int(cm.sum()), "correct": int(cm.trace()), "errors": int(cm.sum()-cm.trace()), "support_min": int(cm.sum(axis=1).min()), "support_max": int(cm.sum(axis=1).max()), "row_sum_percent_max_abs_error": float(np.max(np.abs(norm.sum(axis=1)-100))), "top10_annotations_checked_against_matrix": True},
    "artifacts": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in saved],
    "interpretation_limits": "Single model, single seed, descriptive prediction analysis only; no causal attribution to model modules.",
}
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "plot_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps(manifest, indent=2))
