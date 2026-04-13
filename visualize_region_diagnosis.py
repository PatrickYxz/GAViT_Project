"""
visualize_region_diagnosis.py — Diagnose whether regions have semantic meaning.

Per Prof Wang's guidance: visualize region partitions and graph connections for
airport, bridge, church, etc. Check whether regions correspond to meaningful
parts and whether edges link semantically related areas.

For each image, produces a 2x2 figure:
  (a) Original image with 4x4 grid overlay
  (b) Region assignment — each patch colored by region ID
  (c) Intra-region attention weights — brighter = higher weight within its region
  (d) Graph structure — nodes at region centroids, edges = kNN with GAT attention

Usage:
    python visualize_region_diagnosis.py
    python visualize_region_diagnosis.py --checkpoint checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth
    python visualize_region_diagnosis.py --classes airport bridge church stadium forest
"""

import os
import math
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from torchvision import datasets, transforms

from models.gavit import GAViT
from models.graph_construction import build_knn_graph
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str,
                    default="checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth")
parser.add_argument("--num_regions",  type=int, default=16)
parser.add_argument("--grouping",     type=str, default="attentive_spatial")
parser.add_argument("--edge_type",    type=str, default="knn")
parser.add_argument("--integration",  type=str, default="token_feedback")
parser.add_argument("--knn_k",        type=int, default=5)
parser.add_argument("--gat_hidden",   type=int, default=256)
parser.add_argument("--gat_heads",    type=int, default=4)
parser.add_argument("--gat_layers",   type=int, default=2)
parser.add_argument("--classes",      type=str, nargs="*",
                    default=["airport", "bridge", "church", "stadium", "harbor", "forest"])
parser.add_argument("--num_per_class", type=int, default=2)
parser.add_argument("--seed",         type=int, default=42)
args = parser.parse_args()

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    "datasets/NWPU-RESISC45_split"
)
SAVE_DIR = "results/figures/region_diagnosis"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

GRID_SIZE = 7   # Swin-T outputs 7x7 tokens
K_SIDE = int(math.sqrt(args.num_regions))

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
    edge_type=args.edge_type,
    integration=args.integration,
    pretrained=False,
    freeze_backbone=False,
).to(DEVICE)

state = torch.load(args.checkpoint, map_location=DEVICE)
model.load_state_dict(state)
model.eval()
print(f"Model loaded. K={args.num_regions}, grouping={args.grouping}, "
      f"integration={args.integration}, device={DEVICE}\n")

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
    return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=val_tf)
class_to_idx = test_set.class_to_idx
idx_to_class = {v: k for k, v in class_to_idx.items()}

class_indices = {}
for idx, (_, label) in enumerate(test_set.samples):
    c = idx_to_class[label]
    class_indices.setdefault(c, []).append(idx)

class_list = [c for c in args.classes if c in class_indices]
selected = []
for c in class_list:
    idxs = random.sample(class_indices[c], min(args.num_per_class, len(class_indices[c])))
    selected.extend([(i, c) for i in idxs])

print(f"Visualizing {len(selected)} images: {[c for _, c in selected]}\n")

# =============================================================================
# FORWARD PASS — extract regions, attention weights, graph, GAT attention
# =============================================================================
def forward_diagnosis(model, img_tensor):
    img_np = unnormalize(img_tensor)
    x = img_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        # 1. Backbone
        tokens = model.backbone(x)  # (1, 49, 768)

        # 2. Region grouping — also extract intra-region attention weights
        region_grouping = model.region_grouping
        B, N, D = tokens.shape
        K = region_grouping.K

        # Compute attention scores (same as AttentiveSpatialGrouping.forward)
        scores = region_grouping.attn(tokens)              # (1, 49, 1)
        mask = region_grouping.membership.unsqueeze(0)     # (1, 49, K)
        masked_scores = scores.expand(-1, -1, K)           # (1, 49, K)
        masked_scores = masked_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(masked_scores, dim=1) # (1, 49, K)
        attn_weights = attn_weights.masked_fill(mask == 0, 0.0)

        # Per-token attention weight = its weight within its assigned region
        assignments = region_grouping.assignment  # (49,)
        token_attn = attn_weights[0, torch.arange(N), assignments].cpu()  # (49,)

        # Region features
        region_feats = torch.einsum("bnk,bnd->bkd", attn_weights, tokens)  # (1, K, D)

        # 3. Graph construction
        edge_index, edge_weight, batch = build_knn_graph(region_feats, k=model.knn_k)

        # 4. GAT forward — extract attention
        h = region_feats.reshape(K, -1)
        h = model.graph_reasoning.input_proj(h)

        attn_per_layer = []
        for gat, norm in zip(model.graph_reasoning.gat_layers,
                              model.graph_reasoning.norms):
            residual = h
            h_out, (_, alpha) = gat(h, edge_index, return_attention_weights=True)
            attn_per_layer.append(alpha.mean(dim=1).cpu())
            h = norm(h_out + residual)
            h = model.graph_reasoning.act(h)

        # 5. Prediction
        if model.integration == "token_feedback":
            x_refined_2d = h.unsqueeze(0)  # (1, K, out_dim)
            region_feedback = model.feedback_proj(x_refined_2d)
            assign_batch = assignments.unsqueeze(0)  # (1, N)
            token_fb = region_feedback.gather(
                1, assign_batch.unsqueeze(-1).expand(-1, -1, region_feedback.size(-1))
            )
            updated_tokens = tokens + token_fb
            pooled = updated_tokens.mean(dim=1)
            logits = model.classifier(pooled)
        else:
            from torch_geometric.nn import global_mean_pool
            h_pooled = global_mean_pool(h, batch)
            logits = model.classifier(h_pooled)

        pred = logits.argmax(dim=1).item()

    return {
        "img_np":         img_np,
        "assignments":    assignments.cpu(),        # (49,)
        "token_attn":     token_attn,               # (49,) attention weight per token
        "edge_index":     edge_index.cpu(),
        "edge_weight":    edge_weight.cpu(),
        "gat_attn":       attn_per_layer[-1],       # last layer GAT attention (E,)
        "pred_idx":       pred,
    }

# =============================================================================
# VISUALIZATION
# =============================================================================
REGION_CMAP = plt.cm.get_cmap("tab20", args.num_regions)
PATCH_SIZE = 224 // GRID_SIZE  # 32 pixels per patch

def get_patch_rect(token_idx):
    """Get (x, y) top-left corner of a token's patch in pixel coords."""
    r, c = divmod(token_idx, GRID_SIZE)
    return c * PATCH_SIZE, r * PATCH_SIZE

def get_region_centroid(region_id):
    """Get pixel (x, y) center of a region."""
    # Find all tokens in this region
    assignment = model.region_grouping.assignment.cpu()
    token_ids = (assignment == region_id).nonzero(as_tuple=True)[0]
    if len(token_ids) == 0:
        return (0, 0)
    # Average pixel center
    ys, xs = [], []
    for t in token_ids:
        r, c = divmod(t.item(), GRID_SIZE)
        xs.append((c + 0.5) * PATCH_SIZE)
        ys.append((r + 0.5) * PATCH_SIZE)
    return np.mean(xs), np.mean(ys)


def plot_panel_a(ax, img_np):
    """(a) Original image with grid lines."""
    ax.imshow(img_np)
    # Draw grid lines
    for i in range(1, GRID_SIZE):
        pos = i * PATCH_SIZE
        ax.axhline(y=pos, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(x=pos, color="white", linewidth=0.5, alpha=0.5)
    # Draw region boundaries (thicker)
    for i in range(1, K_SIDE):
        # Approximate region boundary positions
        row_boundary = round(i / K_SIDE * GRID_SIZE) * PATCH_SIZE
        col_boundary = round(i / K_SIDE * GRID_SIZE) * PATCH_SIZE
        ax.axhline(y=row_boundary, color="yellow", linewidth=2, alpha=0.8)
        ax.axvline(x=col_boundary, color="yellow", linewidth=2, alpha=0.8)
    ax.set_title("(a) Image + 4x4 grid", fontsize=10, pad=4)
    ax.axis("off")


def plot_panel_b(ax, img_np, assignments):
    """(b) Region assignment — patches colored by region ID."""
    G = GRID_SIZE
    grid = assignments.reshape(G, G).numpy()

    # Create color overlay
    color_map = np.zeros((G, G, 4))
    for i in range(G):
        for j in range(G):
            color_map[i, j] = REGION_CMAP(int(grid[i, j]))

    # Upsample
    color_full = np.kron(color_map[:, :, :3], np.ones((PATCH_SIZE, PATCH_SIZE, 1)))
    color_full = color_full[:224, :224, :]

    overlay = img_np * 0.5 + color_full * 0.5
    ax.imshow(np.clip(overlay, 0, 1))

    # Label each region
    for k in range(args.num_regions):
        cx, cy = get_region_centroid(k)
        ax.text(cx, cy, f"R{k}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.15", fc=REGION_CMAP(k), alpha=0.85))

    # Count tokens per region
    counts = [(assignments == k).sum().item() for k in range(args.num_regions)]
    ax.set_title(f"(b) Regions (tokens/region: {min(counts)}-{max(counts)})", fontsize=10, pad=4)
    ax.axis("off")


def plot_panel_c(ax, img_np, assignments, token_attn):
    """(c) Intra-region attention — brightness = token importance within region."""
    G = GRID_SIZE

    # Normalize attention per region for display
    attn_display = np.zeros((G, G))
    for k in range(args.num_regions):
        mask = (assignments == k)
        if mask.sum() > 0:
            region_attns = token_attn[mask]
            # Normalize to [0.2, 1.0] for visibility
            if region_attns.max() > region_attns.min():
                norm_attns = (region_attns - region_attns.min()) / (region_attns.max() - region_attns.min())
                norm_attns = 0.2 + 0.8 * norm_attns
            else:
                norm_attns = torch.ones_like(region_attns)
            idx = 0
            for t in range(G * G):
                if mask[t]:
                    r, c = divmod(t, G)
                    attn_display[r, c] = norm_attns[idx].item()
                    idx += 1

    # Upsample attention map
    attn_full = np.kron(attn_display, np.ones((PATCH_SIZE, PATCH_SIZE)))
    attn_full = attn_full[:224, :224]

    # Apply as brightness modulation
    modulated = img_np * attn_full[:, :, None]
    ax.imshow(np.clip(modulated, 0, 1))

    # Draw region boundaries
    for i in range(1, K_SIDE):
        row_boundary = round(i / K_SIDE * GRID_SIZE) * PATCH_SIZE
        col_boundary = round(i / K_SIDE * GRID_SIZE) * PATCH_SIZE
        ax.axhline(y=row_boundary, color="yellow", linewidth=1.5, alpha=0.7)
        ax.axvline(x=col_boundary, color="yellow", linewidth=1.5, alpha=0.7)

    # Show raw attention values on each patch
    for t in range(G * G):
        r, c = divmod(t, G)
        px = (c + 0.5) * PATCH_SIZE
        py = (r + 0.5) * PATCH_SIZE
        val = token_attn[t].item()
        ax.text(px, py, f"{val:.2f}", ha="center", va="center",
                fontsize=5, color="white", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.4))

    ax.set_title("(c) Intra-region attention (brighter = more important)", fontsize=10, pad=4)
    ax.axis("off")


def plot_panel_d(ax, img_np, edge_index, gat_attn):
    """(d) Graph structure with GAT attention on edges."""
    ax.imshow(img_np, alpha=0.6)

    K = args.num_regions
    # Node positions = region centroids
    node_pos = {k: get_region_centroid(k) for k in range(K)}

    # Draw edges with GAT attention
    ev = gat_attn.numpy()
    ev_norm = (ev - ev.min()) / (ev.max() - ev.min() + 1e-8)
    edge_cmap = plt.cm.get_cmap("Reds")

    src_nodes = edge_index[0].numpy()
    tgt_nodes = edge_index[1].numpy()
    for e_idx, (s, t) in enumerate(zip(src_nodes, tgt_nodes)):
        xs, ys = node_pos[s]
        xt, yt = node_pos[t]
        v = ev_norm[e_idx]
        ax.annotate("", xy=(xt, yt), xytext=(xs, ys),
                     arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.1",
                                     color=edge_cmap(v),
                                     linewidth=0.5 + v * 3, alpha=0.7))

    # Draw nodes
    for k, (x, y) in node_pos.items():
        ax.scatter(x, y, s=200, c=[REGION_CMAP(k)], zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.text(x, y, f"R{k}", ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=Normalize(vmin=ev.min(), vmax=ev.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="GAT attn")

    ax.set_title(f"(d) Graph: kNN-{args.knn_k} + GAT attention", fontsize=10, pad=4)
    ax.axis("off")


# =============================================================================
# MAIN LOOP
# =============================================================================
for sample_idx, (dataset_idx, true_class) in enumerate(selected):
    img_tensor, label = test_set[dataset_idx]
    result = forward_diagnosis(model, img_tensor)
    pred_class = idx_to_class[result["pred_idx"]]
    correct = "CORRECT" if pred_class == true_class else f"WRONG (pred: {pred_class})"

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(
        f"Region Diagnosis: {true_class} — {correct}\n"
        f"K={args.num_regions}, grouping={args.grouping}, integration={args.integration}",
        fontsize=13, fontweight="bold"
    )

    plot_panel_a(axes[0, 0], result["img_np"])
    plot_panel_b(axes[0, 1], result["img_np"], result["assignments"])
    plot_panel_c(axes[1, 0], result["img_np"], result["assignments"], result["token_attn"])
    plot_panel_d(axes[1, 1], result["img_np"], result["edge_index"], result["gat_attn"])

    plt.tight_layout()
    fname = f"region_diag_{true_class}_{sample_idx:02d}.png"
    save_path = os.path.join(SAVE_DIR, fname)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}  [{correct}]")

# =============================================================================
# PRINT REGION ASSIGNMENT TABLE (for reference)
# =============================================================================
print(f"\n{'='*60}")
print(f"Region assignment table (7x7 tokens -> {args.num_regions} regions)")
print(f"{'='*60}")
assignment = model.region_grouping.assignment.cpu().reshape(GRID_SIZE, GRID_SIZE).numpy()
for i in range(GRID_SIZE):
    row = " ".join(f"R{int(assignment[i, j]):2d}" for j in range(GRID_SIZE))
    print(f"  row {i}: {row}")
counts = [(model.region_grouping.assignment.cpu() == k).sum().item()
          for k in range(args.num_regions)]
print(f"\nTokens per region: {counts}")
print(f"Min={min(counts)}, Max={max(counts)}")

print(f"\nAll figures saved to: {SAVE_DIR}/")
print("Done.")
