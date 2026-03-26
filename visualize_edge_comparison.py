"""
visualize_edge_comparison.py — Compare GAT attention patterns across edge types.

For each sampled image, produces a 1×3 figure:
  (a) Original image
  (b) GAT attention with spatial adjacency edges
  (c) GAT attention with cosine kNN edges

Purpose: demonstrate that different edge construction strategies lead to
         different relational reasoning patterns, even when final accuracy
         is similar. This addresses "demonstrating the value of relational
         modeling" beyond accuracy numbers.

Usage:
    python visualize_edge_comparison.py
    python visualize_edge_comparison.py --classes airport stadium forest
"""

import os
import argparse
import random
import math

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torchvision import datasets, transforms

from models.gavit import GAViT
from models.graph_construction import build_knn_graph, build_spatial_graph
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_spatial", type=str,
                    default="checkpoints/best_gavit_K9_spatial_spatial.pth")
parser.add_argument("--ckpt_knn",     type=str,
                    default="checkpoints/best_gavit_K9_spatial_knn.pth")
parser.add_argument("--grouping",     type=str, default="spatial")
parser.add_argument("--num_regions",  type=int, default=9)
parser.add_argument("--knn_k",        type=int, default=5)
parser.add_argument("--gat_layers",   type=int, default=2)
parser.add_argument("--gat_heads",    type=int, default=4)
parser.add_argument("--gat_hidden",   type=int, default=256)
parser.add_argument("--classes",      type=str, nargs="*", default=None)
parser.add_argument("--num_per_class", type=int, default=2)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
SAVE_DIR = "results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

DEFAULT_CLASSES = [
    "airport", "stadium", "harbor",
    "dense_residential", "forest", "desert",
]

# =============================================================================
# LOAD TWO MODELS
# =============================================================================
def load_model(checkpoint, edge_type):
    model = GAViT(
        num_classes=45,
        num_regions=args.num_regions,
        knn_k=args.knn_k,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=0.0,
        grouping=args.grouping,
        edge_type=edge_type,
        pretrained=False,
        freeze_backbone=False,
    ).to(DEVICE)
    state = torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

print(f"Loading spatial-edge model: {args.ckpt_spatial}")
model_spatial = load_model(args.ckpt_spatial, "spatial")
print(f"Loading kNN-edge model: {args.ckpt_knn}")
model_knn = load_model(args.ckpt_knn, "knn")
print()

# =============================================================================
# DATA
# =============================================================================
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
INV_MEAN = torch.tensor([0.485, 0.456, 0.406])
INV_STD  = torch.tensor([0.229, 0.224, 0.225])

def unnormalize(t):
    t = t.cpu() * INV_STD[:, None, None] + INV_MEAN[:, None, None]
    t = t.permute(1, 2, 0).numpy()
    return np.clip(t, 0, 1)

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=val_tf)
class_to_idx = test_set.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

class_indices = {}
for idx, (_, label) in enumerate(test_set.samples):
    c = idx_to_class[label]
    class_indices.setdefault(c, []).append(idx)

class_list = args.classes if args.classes else DEFAULT_CLASSES
class_list = [c for c in class_list if c in class_indices]
selected = []
for c in class_list:
    idxs = random.sample(class_indices[c], min(args.num_per_class, len(class_indices[c])))
    selected.extend([(i, c) for i in idxs])

print(f"Visualizing {len(selected)} images\n")

# =============================================================================
# FORWARD WITH ATTENTION
# =============================================================================
def forward_with_attention(model, img_tensor, edge_type):
    x = img_tensor.unsqueeze(0).to(DEVICE)
    K = args.num_regions

    with torch.no_grad():
        tokens = model.backbone(x)
        region_feats, assignments = model.region_grouping(tokens)

        # Build graph based on edge type
        if edge_type == "spatial":
            edge_index, edge_weight, batch = build_spatial_graph(K, 1, x.device)
        else:
            edge_index, edge_weight, batch = build_knn_graph(region_feats, k=model.knn_k)

        # Manual GAT forward to capture attention
        h = region_feats.reshape(K, -1)
        h = model.graph_reasoning.input_proj(h)

        attn_per_layer = []
        for gat, norm in zip(model.graph_reasoning.gat_layers,
                             model.graph_reasoning.norms):
            residual = h
            h_out, (ei_attn, alpha) = gat(h, edge_index, return_attention_weights=True)
            mean_attn = alpha.mean(dim=1).cpu()
            attn_per_layer.append(mean_attn)
            h = norm(h_out + residual)
            h = model.graph_reasoning.act(h)

        from torch_geometric.nn import global_mean_pool
        h_pooled = global_mean_pool(h, batch)
        logits = model.classifier(h_pooled)
        pred = logits.argmax(dim=1).item()

    return {
        "edge_index":     edge_index.cpu(),
        "edge_weight":    edge_weight.cpu(),
        "attn_per_layer": attn_per_layer,
        "pred_idx":       pred,
    }

# =============================================================================
# DRAWING
# =============================================================================
def get_node_positions(K, img_size=224):
    side = int(math.isqrt(K))
    positions = {}
    for k in range(K):
        r, c = divmod(k, side)
        x = (c + 0.5) / side * img_size
        y = (r + 0.5) / side * img_size
        positions[k] = (x, y)
    return positions

def draw_graph(ax, img_np, node_positions, edge_index, edge_vals,
               title, node_size=180, cmap_name="Reds"):
    ax.imshow(img_np)
    ax.set_title(title, fontsize=9, pad=4)
    ax.axis("off")

    K = len(node_positions)
    ev = edge_vals.numpy()
    ev_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-8)
    edge_cmap = plt.colormaps.get_cmap(cmap_name)

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

    node_cmap = plt.colormaps.get_cmap("tab10")
    for k, (x, y) in node_positions.items():
        color = node_cmap(k)
        ax.scatter(x, y, s=node_size, c=[color], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(x, y, str(k), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=4)

    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=ev.min(), vmax=ev.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

# =============================================================================
# MAIN: PER-IMAGE COMPARISON
# =============================================================================
node_pos = get_node_positions(args.num_regions)

for sample_idx, (dataset_idx, true_class) in enumerate(selected):
    img_tensor, label = test_set[dataset_idx]
    img_np = unnormalize(img_tensor)

    res_sp  = forward_with_attention(model_spatial, img_tensor, "spatial")
    res_knn = forward_with_attention(model_knn,     img_tensor, "knn")

    pred_sp  = idx_to_class[res_sp["pred_idx"]]
    pred_knn = idx_to_class[res_knn["pred_idx"]]

    # --- 1×3 figure ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Class: {true_class}", fontsize=13, fontweight="bold")

    # (a) Original
    axes[0].imshow(img_np)
    axes[0].set_title("(a) Original Image", fontsize=10, pad=4)
    axes[0].axis("off")

    # (b) Spatial edge attention
    last_attn_sp = res_sp["attn_per_layer"][-1]
    draw_graph(axes[1], img_np, node_pos,
               res_sp["edge_index"], last_attn_sp,
               title=f"(b) Spatial Adj — GAT Attention\npred: {pred_sp}",
               cmap_name="Reds")

    # (c) kNN edge attention
    last_attn_knn = res_knn["attn_per_layer"][-1]
    draw_graph(axes[2], img_np, node_pos,
               res_knn["edge_index"], last_attn_knn,
               title=f"(c) Cosine kNN — GAT Attention\npred: {pred_knn}",
               cmap_name="Blues")

    plt.tight_layout()
    fname = f"edge_compare_{true_class}_{sample_idx:02d}.png"
    save_path = os.path.join(SAVE_DIR, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# =============================================================================
# SUMMARY FIGURE
# =============================================================================
n = len(selected)
fig, axes = plt.subplots(n, 3, figsize=(15, n * 4.5))
if n == 1:
    axes = axes[np.newaxis, :]

for i, (dataset_idx, true_class) in enumerate(selected):
    img_tensor, label = test_set[dataset_idx]
    img_np = unnormalize(img_tensor)

    res_sp  = forward_with_attention(model_spatial, img_tensor, "spatial")
    res_knn = forward_with_attention(model_knn,     img_tensor, "knn")

    # Original
    axes[i, 0].imshow(img_np)
    axes[i, 0].set_title(f"{true_class}", fontsize=9, fontweight="bold")
    axes[i, 0].axis("off")

    # Spatial attention
    draw_graph(axes[i, 1], img_np, node_pos,
               res_sp["edge_index"], res_sp["attn_per_layer"][-1],
               title="Spatial Adj", cmap_name="Reds", node_size=100)

    # kNN attention
    draw_graph(axes[i, 2], img_np, node_pos,
               res_knn["edge_index"], res_knn["attn_per_layer"][-1],
               title="Cosine kNN", cmap_name="Blues", node_size=100)

# Column headers
fig.text(0.18, 0.99, "Original", ha="center", fontsize=12, fontweight="bold")
fig.text(0.50, 0.99, "Spatial Adjacency Edges", ha="center", fontsize=12, fontweight="bold")
fig.text(0.82, 0.99, "Cosine kNN Edges", ha="center", fontsize=12, fontweight="bold")

fig.suptitle(
    "Edge Construction Ablation: GAT Attention Comparison",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
summary_path = os.path.join(SAVE_DIR, "edge_comparison_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSummary: {summary_path}")
print("Done.")
