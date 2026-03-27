"""
compare_models.py — Compare Swin-T Baseline vs GAViT on the test set.

Produces:
  1. Per-class accuracy comparison bar chart (sorted by GAViT - Baseline delta)
  2. Confusion matrix for each model
  3. Confusion matrix *difference* (GAViT minus Baseline): highlights which
     confusions GAViT fixes (blue) or introduces (red)
  4. CSV with per-class accuracy for both models
  5. Summary of samples that Baseline got wrong but GAViT got right (and vice versa)

Usage:
    python compare_models.py
    python compare_models.py --swin_ckpt checkpoints/best_swin.pth --gavit_ckpt checkpoints/best_gavit_K9_spatial.pth
"""

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
try:
    from sklearn.metrics import confusion_matrix
except ImportError:
    def confusion_matrix(y_true, y_pred, labels=None):
        """Fallback confusion matrix without sklearn."""
        n = len(labels)
        cm = np.zeros((n, n), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1
        return cm
from tqdm import tqdm

from models.gavit import GAViT
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--swin_ckpt",   type=str, default="baselines/swin_baseline/checkpoints/best_swin.pth")
parser.add_argument("--gavit_ckpt",  type=str, default="checkpoints/best_gavit_K9_spatial.pth")
parser.add_argument("--grouping",    type=str, default="spatial")
parser.add_argument("--num_regions", type=int, default=9)
parser.add_argument("--knn_k",      type=int, default=5)
parser.add_argument("--gat_layers",  type=int, default=2)
parser.add_argument("--gat_heads",   type=int, default=4)
parser.add_argument("--gat_hidden",  type=int, default=256)
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--tag",         type=str, default="", help="suffix for output files, e.g. 'fusion'")
args = parser.parse_args()

TAG = f"_{args.tag}" if args.tag else ""

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 45
set_seed(args.seed)

# =============================================================================
# DATA
# =============================================================================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
idx_to_class = {v: k for k, v in test_set.class_to_idx.items()}
class_names = [idx_to_class[i] for i in range(NUM_CLASSES)]

# =============================================================================
# LOAD MODELS
# =============================================================================
print("Loading Swin-T Baseline...")
swin_model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=NUM_CLASSES)
swin_model.load_state_dict(torch.load(args.swin_ckpt, map_location=DEVICE))
swin_model.to(DEVICE).eval()

print("Loading GAViT...")
gavit_model = GAViT(
    num_classes=NUM_CLASSES,
    num_regions=args.num_regions,
    knn_k=args.knn_k,
    gat_hidden=args.gat_hidden,
    gat_heads=args.gat_heads,
    gat_layers=args.gat_layers,
    dropout=0.0,
    grouping=args.grouping,
    pretrained=False,
    freeze_backbone=False,
).to(DEVICE)
gavit_model.load_state_dict(torch.load(args.gavit_ckpt, map_location=DEVICE))
gavit_model.eval()

print(f"Both models loaded. Device: {DEVICE}\n")

# =============================================================================
# INFERENCE
# =============================================================================
all_labels = []
swin_preds = []
gavit_preds = []

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Evaluating"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        swin_out = swin_model(imgs).argmax(dim=1)
        gavit_out = gavit_model(imgs).argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        swin_preds.extend(swin_out.cpu().numpy())
        gavit_preds.extend(gavit_out.cpu().numpy())

all_labels = np.array(all_labels)
swin_preds = np.array(swin_preds)
gavit_preds = np.array(gavit_preds)

# =============================================================================
# OVERALL ACCURACY
# =============================================================================
swin_acc = 100.0 * (swin_preds == all_labels).mean()
gavit_acc = 100.0 * (gavit_preds == all_labels).mean()
print(f"Swin-T Baseline Test Acc: {swin_acc:.1f}%")
print(f"GAViT Test Acc:           {gavit_acc:.1f}%")
print(f"Delta:                    {gavit_acc - swin_acc:+.1f}%\n")

# =============================================================================
# PER-CLASS ACCURACY
# =============================================================================
swin_per_class = []
gavit_per_class = []

for c in range(NUM_CLASSES):
    mask = (all_labels == c)
    swin_per_class.append(100.0 * (swin_preds[mask] == c).mean())
    gavit_per_class.append(100.0 * (gavit_preds[mask] == c).mean())

swin_per_class = np.array(swin_per_class)
gavit_per_class = np.array(gavit_per_class)
delta = gavit_per_class - swin_per_class

# Save CSV
csv_path = f"results/per_class_accuracy{TAG}.csv"
with open(csv_path, "w") as f:
    f.write("class,swin_acc,gavit_acc,delta\n")
    for i in range(NUM_CLASSES):
        f.write(f"{class_names[i]},{swin_per_class[i]:.1f},{gavit_per_class[i]:.1f},{delta[i]:+.1f}\n")
print(f"Saved: {csv_path}")

# =============================================================================
# PLOT 1: Per-class accuracy delta (sorted)
# =============================================================================
sort_idx = np.argsort(delta)
sorted_names = [class_names[i] for i in sort_idx]
sorted_delta = delta[sort_idx]

fig, ax = plt.subplots(figsize=(10, 12))
colors = ["#e74c3c" if d < 0 else "#27ae60" for d in sorted_delta]
y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_delta, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=7)
ax.set_xlabel("GAViT - Baseline Accuracy (%)", fontsize=11)
ax.set_title(f"Per-Class Accuracy Change: GAViT vs Swin-T Baseline\n"
             f"(Overall: Swin {swin_acc:.1f}% → GAViT {gavit_acc:.1f}%, delta {gavit_acc-swin_acc:+.1f}%)",
             fontsize=11, fontweight="bold")
ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
ax.invert_yaxis()

plt.tight_layout()
path1 = os.path.join(SAVE_DIR, f"per_class_accuracy_delta{TAG}.png")
plt.savefig(path1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path1}")

# =============================================================================
# PLOT 2: Confusion matrices (Baseline, GAViT, Difference)
# =============================================================================
cm_swin = confusion_matrix(all_labels, swin_preds, labels=range(NUM_CLASSES))
cm_gavit = confusion_matrix(all_labels, gavit_preds, labels=range(NUM_CLASSES))

# Normalize by row (true label) to get per-class rates
cm_swin_norm = cm_swin.astype(float) / cm_swin.sum(axis=1, keepdims=True) * 100
cm_gavit_norm = cm_gavit.astype(float) / cm_gavit.sum(axis=1, keepdims=True) * 100
cm_diff = cm_gavit_norm - cm_swin_norm

fig, axes = plt.subplots(1, 3, figsize=(30, 9))

# (a) Swin-T confusion matrix
im0 = axes[0].imshow(cm_swin_norm, cmap="Blues", vmin=0, vmax=100)
axes[0].set_title(f"(a) Swin-T Baseline (Test Acc {swin_acc:.1f}%)", fontsize=11)
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("True")
axes[0].set_xticks(range(NUM_CLASSES))
axes[0].set_yticks(range(NUM_CLASSES))
axes[0].set_xticklabels(class_names, rotation=90, fontsize=5)
axes[0].set_yticklabels(class_names, fontsize=5)
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# (b) GAViT confusion matrix
im1 = axes[1].imshow(cm_gavit_norm, cmap="Blues", vmin=0, vmax=100)
axes[1].set_title(f"(b) GAViT (Test Acc {gavit_acc:.1f}%)", fontsize=11)
axes[1].set_xlabel("Predicted")
axes[1].set_xticks(range(NUM_CLASSES))
axes[1].set_yticks(range(NUM_CLASSES))
axes[1].set_xticklabels(class_names, rotation=90, fontsize=5)
axes[1].set_yticklabels(class_names, fontsize=5)
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# (c) Difference: GAViT - Baseline
#     Diagonal: blue = GAViT improved (higher correct rate)
#     Off-diagonal: blue = GAViT reduced this confusion; red = GAViT increased it
vmax_diff = max(abs(cm_diff.min()), abs(cm_diff.max()), 1)
norm_diff = TwoSlopeNorm(vmin=-vmax_diff, vcenter=0, vmax=vmax_diff)
im2 = axes[2].imshow(cm_diff, cmap="RdBu", norm=norm_diff)
axes[2].set_title("(c) Difference: GAViT − Baseline\n(Blue=improved, Red=worsened)", fontsize=11)
axes[2].set_xlabel("Predicted")
axes[2].set_xticks(range(NUM_CLASSES))
axes[2].set_yticks(range(NUM_CLASSES))
axes[2].set_xticklabels(class_names, rotation=90, fontsize=5)
axes[2].set_yticklabels(class_names, fontsize=5)
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
path2 = os.path.join(SAVE_DIR, f"confusion_matrix_comparison{TAG}.png")
plt.savefig(path2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path2}")

# =============================================================================
# PLOT 3: Zoomed difference matrix — only classes with |delta| > 0
# =============================================================================
changed_idx = np.where(np.abs(delta) > 0.1)[0]
if len(changed_idx) > 0:
    cm_diff_sub = cm_diff[np.ix_(changed_idx, changed_idx)]
    sub_names = [class_names[i] for i in changed_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(sub_names) * 0.5), max(5, len(sub_names) * 0.45)))
    vmax_sub = max(abs(cm_diff_sub.min()), abs(cm_diff_sub.max()), 1)
    norm_sub = TwoSlopeNorm(vmin=-vmax_sub, vcenter=0, vmax=vmax_sub)
    im = ax.imshow(cm_diff_sub, cmap="RdBu", norm=norm_sub)
    ax.set_xticks(range(len(sub_names)))
    ax.set_yticks(range(len(sub_names)))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sub_names, fontsize=8)
    ax.set_title("Confusion Difference (zoomed: classes with accuracy change)\n"
                 "Blue = GAViT improved, Red = GAViT worsened", fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # Annotate cells with values
    for i in range(len(sub_names)):
        for j in range(len(sub_names)):
            val = cm_diff_sub[i, j]
            if abs(val) > 0.5:
                ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                        fontsize=6, color="black" if abs(val) < vmax_sub * 0.5 else "white")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path3 = os.path.join(SAVE_DIR, f"confusion_matrix_diff_zoomed{TAG}.png")
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path3}")

# =============================================================================
# SAMPLE-LEVEL ANALYSIS
# =============================================================================
# Samples Baseline got wrong, GAViT got right
fixed_mask = (swin_preds != all_labels) & (gavit_preds == all_labels)
# Samples Baseline got right, GAViT got wrong
broken_mask = (swin_preds == all_labels) & (gavit_preds != all_labels)

n_fixed = fixed_mask.sum()
n_broken = broken_mask.sum()
print(f"\nSample-level analysis:")
print(f"  Baseline wrong → GAViT right (fixed):  {n_fixed}")
print(f"  Baseline right → GAViT wrong (broken): {n_broken}")
print(f"  Net improvement: {n_fixed - n_broken} samples")

# Which classes benefited most from fixes?
if n_fixed > 0:
    fixed_classes = all_labels[fixed_mask]
    fix_counts = defaultdict(int)
    for c in fixed_classes:
        fix_counts[class_names[c]] += 1
    print(f"\nTop classes where GAViT fixed Baseline errors:")
    for cls, cnt in sorted(fix_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls:30s}  {cnt} samples fixed")

# Which confusions were fixed?
if n_fixed > 0:
    print(f"\nTop confusion pairs fixed by GAViT (true→baseline_pred, now correct):")
    confusion_fixes = defaultdict(int)
    fixed_indices = np.where(fixed_mask)[0]
    for idx in fixed_indices:
        true_cls = class_names[all_labels[idx]]
        wrong_cls = class_names[swin_preds[idx]]
        confusion_fixes[(true_cls, wrong_cls)] += 1
    for (true_c, wrong_c), cnt in sorted(confusion_fixes.items(), key=lambda x: -x[1])[:15]:
        print(f"  {true_c:25s} → {wrong_c:25s}  ({cnt} fixed)")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Swin-T Baseline Test Acc: {swin_acc:.1f}%")
print(f"GAViT Test Acc:           {gavit_acc:.1f}%")
print(f"Delta:                    {gavit_acc - swin_acc:+.1f}%")
print(f"Relative error reduction: {(1 - (100 - gavit_acc) / (100 - swin_acc)) * 100:.1f}%")

print(f"\nTop 5 classes where GAViT improved most:")
top_improved = np.argsort(delta)[::-1][:5]
for i in top_improved:
    print(f"  {class_names[i]:25s}  {swin_per_class[i]:.1f}% → {gavit_per_class[i]:.1f}%  ({delta[i]:+.1f}%)")

print(f"\nTop 5 classes where GAViT declined:")
top_declined = np.argsort(delta)[:5]
for i in top_declined:
    if delta[i] < 0:
        print(f"  {class_names[i]:25s}  {swin_per_class[i]:.1f}% → {gavit_per_class[i]:.1f}%  ({delta[i]:+.1f}%)")

print("\nDone.")
