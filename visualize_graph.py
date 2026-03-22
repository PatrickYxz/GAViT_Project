"""
visualize_graph.py — Visualize GAViT's region graph and GAT attention weights.

For each sampled image, produces a 2×2 figure:
  (a) Original image
  (b) Region assignment map (7×7 patches colored by region ID, overlaid on image)
  (c) kNN graph: nodes at region centroids, edge width ∝ cosine similarity
  (d) GAT attention: same graph, edge width/color ∝ last-layer mean attention

Purpose: demonstrate that GAViT learns *meaningful* relational structure,
         not just higher accuracy (addresses "relational modeling" value).

Usage:
    python visualize_graph.py
    python visualize_graph.py --classes airport stadium forest --num_per_class 3
    python visualize_graph.py --num_images 12 --grouping spatial
"""

import os
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from torchvision import datasets, transforms
from PIL import Image

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
parser.add_argument("--classes",       type=str, nargs="*", default=None,
                    help="Class names to visualize. Default: auto-select diverse classes.")
parser.add_argument("--num_per_class", type=int, default=2)
parser.add_argument("--num_images",    type=int, default=None,
                    help="Total images to sample randomly (ignores --classes if set).")
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

# Default classes: diverse scene types for interesting graphs
DEFAULT_CLASSES = [
    "airport",          # complex: runway + terminal + apron
    "stadium",          # complex: pitch + stands
    "harbor",           # complex: water + dock + ships
    "dense_residential",# complex: grid of buildings
    "forest",           # uniform: should produce weak edges
    "desert",           # uniform: baseline comparison
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
    dropout=0.0,        # disable dropout at inference
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
# Inverse-normalize for display
INV_MEAN = torch.tensor([0.485, 0.456, 0.406])
INV_STD  = torch.tensor([0.229, 0.224, 0.225])

def unnormalize(t):
    """(3,H,W) normalized tensor → (H,W,3) numpy uint8"""
    t = t.cpu() * INV_STD[:, None, None] + INV_MEAN[:, None, None]
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t, 0, 1)

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=val_tf)
class_to_idx = test_set.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Build per-class index lists
class_indices = {}
for idx, (_, label) in enumerate(test_set.samples):
    c = idx_to_class[label]
    class_indices.setdefault(c, []).append(idx)

# Select images to visualize
if args.num_images is not None:
    all_idx = random.sample(range(len(test_set)), min(args.num_images, len(test_set)))
    selected = [(i, idx_to_class[test_set.targets[i]]) for i in all_idx]
else:
    class_list = args.classes if args.classes else DEFAULT_CLASSES
    # Filter to classes that exist in this dataset
    class_list = [c for c in class_list if c in class_indices]
    if not class_list:
        print(f"Warning: none of the specified classes found. Using random sample.")
        all_idx = random.sample(range(len(test_set)), 8)
        selected = [(i, idx_to_class[test_set.targets[i]]) for i in all_idx]
    else:
        selected = []
        for c in class_list:
            idxs = random.sample(class_indices[c], min(args.num_per_class, len(class_indices[c])))
            selected.extend([(i, c) for i in idxs])

print(f"Visualizing {len(selected)} images: {[c for _, c in selected]}\n")

# =============================================================================
# FORWARD PASS WITH ATTENTION EXTRACTION
# =============================================================================
def forward_with_attention(model, img_tensor):
    """
    Run GAViT forward on a single image and extract all intermediate results.

    Returns dict with:
        img_np:          (224, 224, 3) float [0,1]  — for display
        assignments:     (49,)    — region ID per patch
        region_features: (K, 768) — before GAT
        edge_index:      (2, E)   — kNN edges
        edge_weight:     (E,)     — cosine similarity
        attn_per_layer:  list of (E,)  — mean attention across heads, per GAT layer
        pred_class:      str
        true_class:      str (filled outside)
    """
    img_np = unnormalize(img_tensor)
    x = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 1. Backbone
        tokens = model.backbone(x)                              # (1, 49, 768)

        # 2. Region grouping
        region_feats, assignments = model.region_grouping(tokens)  # (1,K,768), (1,49)

        # 3. Graph construction
        edge_index, edge_weight, batch = build_knn_graph(region_feats, k=model.knn_k)

        # 4. Graph reasoning — manual forward to capture attention
        h = region_feats.reshape(args.num_regions, -1)           # (K, 768)
        h = model.graph_reasoning.input_proj(h)                  # (K, out_dim)

        attn_per_layer = []
        for gat, norm in zip(model.graph_reasoning.gat_layers,
                              model.graph_reasoning.norms):
            residual = h
            h_out, (_, alpha) = gat(h, edge_index, return_attention_weights=True)
            # alpha: (E, num_heads)
            mean_attn = alpha.mean(dim=1).cpu()                  # (E,)
            attn_per_layer.append(mean_attn)
            h = norm(h_out + residual)
            h = model.graph_reasoning.act(h)
            # skip dropout at inference

        # 5. Classify
        from torch_geometric.nn import global_mean_pool
        h_pooled = global_mean_pool(h, batch)
        logits = model.classifier(h_pooled)
        pred = logits.argmax(dim=1).item()

    return {
        "img_np":          img_np,
        "assignments":     assignments[0].cpu(),             # (49,)
        "region_features": region_feats[0].cpu(),            # (K, 768)
        "edge_index":      edge_index.cpu(),                 # (2, E)  — no batch offset for B=1
        "edge_weight":     edge_weight.cpu(),                # (E,)
        "attn_per_layer":  attn_per_layer,                   # list[(E,)]
        "pred_idx":        pred,
    }

# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================
# Region colormap (K=9 → 9 distinct colors)
REGION_CMAP = plt.cm.get_cmap("tab10", args.num_regions)

def make_region_overlay(img_np, assignments, alpha=0.45):
    """
    Overlay region colors onto original image.
    assignments: (49,) with values in [0, K)
    Returns: (224, 224, 3) float [0,1]
    """
    G = 7
    assignment_grid = assignments.reshape(G, G).numpy()           # (7,7)

    # Create color map for each patch
    color_grid = np.array([REGION_CMAP(int(r))[:3] for r in assignment_grid.flat])
    color_grid = color_grid.reshape(G, G, 3)

    # Upsample to 224×224 (each patch → 32×32 pixels)
    patch_h, patch_w = 224 // G, 224 // G
    color_full = np.kron(color_grid, np.ones((patch_h, patch_w, 1)))
    # Handle rounding: crop or pad to exactly 224×224
    color_full = color_full[:224, :224, :]

    overlay = img_np * (1 - alpha) + color_full * alpha
    return np.clip(overlay, 0, 1)

def get_node_positions(K, grouping, img_size=224):
    """
    Compute node (x, y) positions in image pixel coordinates.
    For spatial grouping: nodes sit at region centroids on a regular grid.
    """
    import math
    k_side = int(math.sqrt(K))
    positions = {}
    for k in range(K):
        r, c = divmod(k, k_side)
        x = (c + 0.5) / k_side * img_size   # horizontal (col)
        y = (r + 0.5) / k_side * img_size   # vertical (row)
        positions[k] = (x, y)
    return positions

def draw_graph(ax, img_np, node_positions, edge_index, edge_vals,
               title, node_size=180, cmap_name="YlOrRd"):
    """Draw graph on top of image background."""
    ax.imshow(img_np)
    ax.set_title(title, fontsize=10, pad=4)
    ax.axis("off")

    K = len(node_positions)
    # Normalize edge values to [0,1]
    ev = edge_vals.numpy()
    ev_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-8)
    edge_cmap = plt.cm.get_cmap(cmap_name)

    # Draw edges
    src_nodes = edge_index[0].numpy()
    tgt_nodes = edge_index[1].numpy()
    for e_idx, (s, t) in enumerate(zip(src_nodes, tgt_nodes)):
        xs, ys = node_positions[s]
        xt, yt = node_positions[t]
        v = ev_norm[e_idx]
        color = edge_cmap(v)
        lw = 0.5 + v * 3.5
        ax.plot([xs, xt], [ys, yt], color=color, linewidth=lw,
                alpha=0.7, zorder=2)

    # Draw nodes
    node_cmap = plt.cm.get_cmap("tab10", K)
    for k, (x, y) in node_positions.items():
        color = node_cmap(k)
        ax.scatter(x, y, s=node_size, c=[color], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(x, y, str(k), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    # Colorbar-style legend
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=ev.min(), vmax=ev.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

# =============================================================================
# MAIN VISUALIZATION LOOP
# =============================================================================
node_pos = get_node_positions(args.num_regions, args.grouping)

for sample_idx, (dataset_idx, true_class) in enumerate(selected):
    img_tensor, label = test_set[dataset_idx]
    result = forward_with_attention(model, img_tensor)
    pred_class = idx_to_class[result["pred_idx"]]

    # --- Figure: 2×2 layout ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(
        f"Class: {true_class}  |  Pred: {pred_class}  "
        f"({'✓' if pred_class == true_class else '✗'})",
        fontsize=12, fontweight="bold"
    )

    # (a) Original image
    axes[0, 0].imshow(result["img_np"])
    axes[0, 0].set_title("(a) Original Image", fontsize=10, pad=4)
    axes[0, 0].axis("off")

    # (b) Region assignment overlay
    overlay = make_region_overlay(result["img_np"], result["assignments"])
    axes[0, 1].imshow(overlay)
    axes[0, 1].set_title(f"(b) Region Assignment (K={args.num_regions})", fontsize=10, pad=4)
    axes[0, 1].axis("off")
    # Add region ID labels at node positions
    for k, (x, y) in node_pos.items():
        axes[0, 1].text(x, y, f"R{k}", ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=REGION_CMAP(k), alpha=0.8))

    # (c) kNN graph (cosine similarity)
    draw_graph(
        axes[1, 0], result["img_np"], node_pos,
        result["edge_index"], result["edge_weight"],
        title=f"(c) kNN Graph — Edge: cosine similarity (k={args.knn_k})",
        cmap_name="Blues",
    )

    # (d) GAT attention graph (last layer)
    last_attn = result["attn_per_layer"][-1]  # (E,)
    draw_graph(
        axes[1, 1], result["img_np"], node_pos,
        result["edge_index"], last_attn,
        title=f"(d) GAT Attention — Last Layer (mean over {args.gat_heads} heads)",
        cmap_name="Reds",
    )

    plt.tight_layout()
    fname = f"graph_vis_{true_class}_{sample_idx:02d}.png"
    save_path = os.path.join(SAVE_DIR, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  [pred={pred_class}, gt={true_class}]")

# =============================================================================
# SUMMARY FIGURE: all images in a grid (original | attention side-by-side)
# =============================================================================
n = len(selected)
ncols = 4   # pairs: original, attention, original, attention
nrows = (n + 1) // 2

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
if nrows == 1:
    axes = axes[np.newaxis, :]

for i, (dataset_idx, true_class) in enumerate(selected):
    row = i // 2
    col_base = (i % 2) * 2

    img_tensor, label = test_set[dataset_idx]
    result = forward_with_attention(model, img_tensor)
    pred_class = idx_to_class[result["pred_idx"]]
    correct = "✓" if pred_class == true_class else "✗"

    # Original
    axes[row, col_base].imshow(result["img_np"])
    axes[row, col_base].set_title(f"{true_class}\n{correct}", fontsize=8)
    axes[row, col_base].axis("off")

    # GAT attention graph
    ax_g = axes[row, col_base + 1]
    last_attn = result["attn_per_layer"][-1]
    draw_graph(ax_g, result["img_np"], node_pos,
               result["edge_index"], last_attn,
               title="GAT Attention", cmap_name="Reds", node_size=100)

# Hide unused axes
for i in range(n, nrows * 2):
    row = i // 2
    col_base = (i % 2) * 2
    for dc in range(2):
        if col_base + dc < ncols:
            axes[row, col_base + dc].axis("off")

fig.suptitle(
    f"GAViT Region Graph Visualization  |  K={args.num_regions} {args.grouping} grouping",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
summary_path = os.path.join(SAVE_DIR, f"graph_vis_summary_{args.grouping}_K{args.num_regions}.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSummary figure saved: {summary_path}")
print("Done.")
