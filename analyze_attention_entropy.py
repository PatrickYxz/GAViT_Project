"""
analyze_attention_entropy.py — Quantify GAT attention entropy across scene types.

Hypothesis:
  - Complex scenes (airport, stadium, harbor) → lower entropy (focused attention)
  - Uniform scenes (forest, desert, sea_ice) → higher entropy (diffuse attention)
  - This demonstrates that GAViT *dynamically adapts* its relational reasoning
    based on scene complexity — evidence for the value of graph-augmented models.

Outputs:
  1. Per-class mean entropy bar chart:  results/figures/attention_entropy_barplot.png
  2. Complex vs uniform box plot:       results/figures/attention_entropy_boxplot.png
  3. CSV with per-image entropy:        results/attention_entropy.csv

Usage:
    python analyze_attention_entropy.py
    python analyze_attention_entropy.py --max_per_class 50
"""

import os
import argparse
import math
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from models.gavit import GAViT
from models.graph_construction import build_knn_graph
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",    type=str, default="checkpoints/best_gavit_K9_spatial.pth")
parser.add_argument("--grouping",      type=str, default="spatial", choices=["spatial", "kmeans"])
parser.add_argument("--num_regions",   type=int, default=9)
parser.add_argument("--knn_k",         type=int, default=5)
parser.add_argument("--gat_layers",    type=int, default=2)
parser.add_argument("--gat_heads",     type=int, default=4)
parser.add_argument("--gat_hidden",    type=int, default=256)
parser.add_argument("--max_per_class", type=int, default=30,
                    help="Max images per class to analyze (for speed).")
parser.add_argument("--seed",          type=int, default=42)
args = parser.parse_args()

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

# Scene type categorization
COMPLEX_CLASSES = [
    "airport", "stadium", "harbor", "railway_station",
    "bridge", "overpass", "freeway", "intersection",
    "industrial_area", "commercial_area",
]
UNIFORM_CLASSES = [
    "forest", "desert", "sea_ice", "snowberg",
    "meadow", "wetland", "beach", "cloud",
    "lake", "mountain",
]

# =============================================================================
# LOAD MODEL
# =============================================================================
print(f"Loading checkpoint: {args.checkpoint}")
model = GAViT(
    num_classes=45,
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

state = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Model loaded. K={args.num_regions}, grouping={args.grouping}, device={DEVICE}\n")

# =============================================================================
# DATA
# =============================================================================
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=val_tf)
class_to_idx = test_set.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Build per-class index lists
class_indices = defaultdict(list)
for idx, (_, label) in enumerate(test_set.samples):
    class_indices[idx_to_class[label]].append(idx)

# =============================================================================
# ATTENTION ENTROPY COMPUTATION
# =============================================================================
def compute_node_entropy(alpha, edge_index, num_nodes):
    """
    Compute per-node attention entropy.

    For each target node j, collect attention weights alpha_{ij} from all
    source nodes i that point to j. These form a distribution (softmax
    already applied by GATConv). Entropy = -sum(p * log(p)).

    Args:
        alpha:      (E, num_heads) attention weights
        edge_index: (2, E)
        num_nodes:  K

    Returns:
        entropy_per_node: (K,) — mean over heads
    """
    E, H = alpha.shape
    src, tgt = edge_index[0], edge_index[1]

    entropies = torch.zeros(num_nodes, H)
    for j in range(num_nodes):
        mask = (tgt == j)
        if mask.sum() == 0:
            continue
        # alpha values for edges pointing to node j: (num_neighbors, H)
        a_j = alpha[mask]  # already softmax-normalized by GATConv per target
        # Entropy: -sum(p * log(p))
        log_a = torch.log(a_j + 1e-10)
        ent = -(a_j * log_a).sum(dim=0)  # (H,)
        entropies[j] = ent

    # Mean over heads, then mean over nodes → scalar per image
    return entropies.mean(dim=1)  # (K,)


def extract_attention(model, img_tensor, num_regions, knn_k):
    """Run forward pass and return last-layer attention weights."""
    x = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        tokens = model.backbone(x)
        region_feats, _ = model.region_grouping(tokens)
        edge_index, edge_weight, batch = build_knn_graph(region_feats, k=knn_k)

        h = region_feats.reshape(num_regions, -1)
        h = model.graph_reasoning.input_proj(h)

        ei_last = None
        alpha_last = None
        for gat, norm in zip(model.graph_reasoning.gat_layers,
                              model.graph_reasoning.norms):
            residual = h
            h_out, (ei_ret, alpha) = gat(h, edge_index, return_attention_weights=True)
            ei_last = ei_ret      # (2, E') — includes self-loops added by GATConv
            alpha_last = alpha    # (E', num_heads)
            h = norm(h_out + residual)
            h = model.graph_reasoning.act(h)

    return ei_last.cpu(), alpha_last.cpu(), num_regions


# =============================================================================
# MAIN LOOP: compute entropy for all test images
# =============================================================================
import random

print("Computing attention entropy across test set...")
class_entropies = defaultdict(list)  # class_name -> list of mean-entropy values

total = 0
for cls_name, indices in sorted(class_indices.items()):
    # Sample up to max_per_class images per class
    sampled = random.sample(indices, min(args.max_per_class, len(indices)))

    for idx in sampled:
        img_tensor, label = test_set[idx]
        edge_index, alpha, K = extract_attention(model, img_tensor, args.num_regions, args.knn_k)

        node_ent = compute_node_entropy(alpha, edge_index, K)
        mean_ent = node_ent.mean().item()
        class_entropies[cls_name].append(mean_ent)
        total += 1

    print(f"  {cls_name}: {len(sampled)} images, mean entropy = {np.mean(class_entropies[cls_name]):.4f}")

print(f"\nTotal images analyzed: {total}")

# =============================================================================
# SAVE CSV
# =============================================================================
csv_path = "results/attention_entropy.csv"
with open(csv_path, "w") as f:
    f.write("class,image_idx,mean_entropy,scene_type\n")
    for cls_name, ent_list in sorted(class_entropies.items()):
        if cls_name in COMPLEX_CLASSES:
            stype = "complex"
        elif cls_name in UNIFORM_CLASSES:
            stype = "uniform"
        else:
            stype = "other"
        for i, ent in enumerate(ent_list):
            f.write(f"{cls_name},{i},{ent:.6f},{stype}\n")
print(f"Saved: {csv_path}")

# =============================================================================
# PLOT 1: Per-class mean entropy bar chart (all 45 classes)
# =============================================================================
class_names = sorted(class_entropies.keys())
mean_per_class = [np.mean(class_entropies[c]) for c in class_names]
std_per_class = [np.std(class_entropies[c]) for c in class_names]

# Sort by mean entropy
sort_idx = np.argsort(mean_per_class)
sorted_names = [class_names[i] for i in sort_idx]
sorted_means = [mean_per_class[i] for i in sort_idx]
sorted_stds = [std_per_class[i] for i in sort_idx]

# Color bars by scene type
bar_colors = []
for name in sorted_names:
    if name in COMPLEX_CLASSES:
        bar_colors.append("#e74c3c")  # red
    elif name in UNIFORM_CLASSES:
        bar_colors.append("#3498db")  # blue
    else:
        bar_colors.append("#95a5a6")  # grey

fig, ax = plt.subplots(figsize=(14, 8))
y_pos = np.arange(len(sorted_names))
ax.barh(y_pos, sorted_means, xerr=sorted_stds, color=bar_colors,
        edgecolor="white", linewidth=0.5, capsize=2)
ax.set_yticks(y_pos)
ax.set_yticklabels(sorted_names, fontsize=7)
ax.set_xlabel("Mean Attention Entropy (last GAT layer)", fontsize=11)
ax.set_title("GAT Attention Entropy by Scene Class\n"
             "(Red = complex scenes, Blue = uniform scenes, Grey = other)",
             fontsize=12, fontweight="bold")
ax.invert_yaxis()

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="Complex scenes"),
    Patch(facecolor="#3498db", label="Uniform scenes"),
    Patch(facecolor="#95a5a6", label="Other"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

plt.tight_layout()
barplot_path = os.path.join(SAVE_DIR, "attention_entropy_barplot.png")
plt.savefig(barplot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {barplot_path}")

# =============================================================================
# PLOT 2: Complex vs Uniform box plot
# =============================================================================
complex_ents = []
uniform_ents = []
for cls_name, ent_list in class_entropies.items():
    if cls_name in COMPLEX_CLASSES:
        complex_ents.extend(ent_list)
    elif cls_name in UNIFORM_CLASSES:
        uniform_ents.extend(ent_list)

fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot(
    [complex_ents, uniform_ents],
    labels=["Complex Scenes", "Uniform Scenes"],
    patch_artist=True,
    widths=0.5,
)
bp["boxes"][0].set_facecolor("#e74c3c")
bp["boxes"][0].set_alpha(0.6)
bp["boxes"][1].set_facecolor("#3498db")
bp["boxes"][1].set_alpha(0.6)

ax.set_ylabel("Attention Entropy", fontsize=11)
ax.set_title("GAT Attention Entropy: Complex vs Uniform Scenes", fontsize=12, fontweight="bold")

# Add mean markers
for i, data in enumerate([complex_ents, uniform_ents]):
    m = np.mean(data)
    ax.scatter(i + 1, m, marker="D", color="black", s=40, zorder=3, label=f"Mean={m:.4f}" if i == 0 else f"Mean={m:.4f}")

ax.legend(fontsize=9)

# Statistical test (Mann-Whitney U, no scipy dependency)
def mann_whitney_u(x, y):
    """Compute Mann-Whitney U statistic and approximate p-value (normal approx)."""
    nx, ny = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.empty_like(combined)
    order = combined.argsort()
    ranks[order] = np.arange(1, len(combined) + 1)
    # Handle ties: average ranks for tied values
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i + 1, j + 1))
            ranks[order[i:j]] = avg_rank
        i = j
    u1 = np.sum(ranks[:nx]) - nx * (nx + 1) / 2
    u2 = nx * ny - u1
    u_stat = min(u1, u2)
    # Normal approximation for p-value
    mu = nx * ny / 2
    sigma = np.sqrt(nx * ny * (nx + ny + 1) / 12)
    z = abs(u_stat - mu) / sigma
    # Two-sided p-value (approximate using z-score)
    p_val = 2 * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi) if z < 8 else 0.0
    return u_stat, p_val

stat, p_value = mann_whitney_u(np.array(complex_ents), np.array(uniform_ents))
ax.text(0.5, 0.02,
        f"Mann-Whitney U test: U={stat:.0f}, p={p_value:.2e}\n"
        f"Complex: {np.mean(complex_ents):.4f} ± {np.std(complex_ents):.4f}  "
        f"(n={len(complex_ents)})\n"
        f"Uniform: {np.mean(uniform_ents):.4f} ± {np.std(uniform_ents):.4f}  "
        f"(n={len(uniform_ents)})",
        transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
boxplot_path = os.path.join(SAVE_DIR, "attention_entropy_boxplot.png")
plt.savefig(boxplot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {boxplot_path}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("ATTENTION ENTROPY ANALYSIS SUMMARY")
print("=" * 60)
print(f"Complex scenes ({len(COMPLEX_CLASSES)} classes, {len(complex_ents)} images):")
print(f"  Mean entropy: {np.mean(complex_ents):.4f} ± {np.std(complex_ents):.4f}")
print(f"Uniform scenes ({len(UNIFORM_CLASSES)} classes, {len(uniform_ents)} images):")
print(f"  Mean entropy: {np.mean(uniform_ents):.4f} ± {np.std(uniform_ents):.4f}")
print(f"Mann-Whitney U test: U={stat:.0f}, p={p_value:.2e}")
if p_value < 0.05:
    print("→ Statistically significant difference (p < 0.05)")
    if np.mean(complex_ents) < np.mean(uniform_ents):
        print("→ Complex scenes have LOWER entropy (more focused attention) ✓")
    else:
        print("→ Complex scenes have HIGHER entropy (more diffuse attention)")
else:
    print("→ No statistically significant difference (p >= 0.05)")

# Top 5 lowest and highest entropy classes
print(f"\nTop 5 lowest entropy (most focused):")
for i in sort_idx[:5]:
    print(f"  {class_names[i]:25s}  {mean_per_class[i]:.4f}")
print(f"Top 5 highest entropy (most diffuse):")
for i in sort_idx[-5:]:
    print(f"  {class_names[i]:25s}  {mean_per_class[i]:.4f}")

print("\nDone.")