# CLAUDE.md — GAViT Project Context

## Project Overview

This is an NTU Master's thesis project supervised by Professor Wang Lipo. The goal is to build **GAViT (Graph-Augmented Vision Transformer)** — a hybrid architecture that augments a Swin Transformer with a dynamic graph neural network module to improve remote sensing image scene classification.

The core idea: Swin Transformer processes image patches independently via attention, but does not explicitly model semantic relationships between image regions. GAViT adds a graph reasoning layer on top of the transformer to capture inter-region relationships.

## Architecture Pipeline

```
Input Image (224×224 RGB)
    │
    ▼
┌─────────────────────────────┐
│  Swin Transformer Backbone  │  (Swin-T, pretrained, from timm)
│  4×4 patch embed → 4 stages │
│  56×56 → 28×28 → 14×14 → 7×7
└─────────────┬───────────────┘
              │ Output: 49 tokens, each 768-dim
              ▼
┌─────────────────────────────┐
│  Region Grouping Module     │  (k-means or attention-based clustering)
│  49 tokens → K region nodes │  K is a hyperparameter (e.g., 9)
│  Clustering in feature space│
└─────────────┬───────────────┘
              │ Output: K region-level feature vectors
              ▼
┌─────────────────────────────┐
│  Dynamic Graph Construction │
│  - Nodes = region features  │
│  - Edges = feature similarity, spatial adjacency, or kNN │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Graph Neural Network Layers│  (GAT or message-passing, 1–2 layers)
│  Refine region features via │
│  neighbor aggregation       │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Graph Pooling → Classifier │  (mean/max pool over nodes → FC → 45 classes)
└─────────────────────────────┘
```

## Dataset

**Primary:** NWPU-RESISC45
- 45 scene categories, 700 images each, 256×256 pixels
- Split: 70% train / 15% val / 15% test (random seed 42)
- Split script: `split_nwpu.py`
- Data path pattern: `datasets/NWPU-RESISC45_split/{train,val,test}/{class_name}/`

**Secondary (optional, for generalization testing):** AID (Aerial Image Dataset), 30 classes, 10,000 images.

## Current Codebase

| File | Purpose |
|------|---------|
| `split_nwpu.py` | Splits raw NWPU-RESISC45 into train/val/test |
| `train_swin_baseline.py` | Trains Swin-T baseline (pretrained, 30 epochs, AdamW, cosine LR) |
| `test_swin.py` | Evaluates saved checkpoint on test set |

**Baseline results:** ~96% validation and test accuracy with Swin-T pretrained on ImageNet.

## Key Technical Details

- **Framework:** PyTorch + timm (for Swin-T) + PyTorch Geometric (for GNN layers)
- **Backbone:** `swin_tiny_patch4_window7_224` from timm, pretrained=True
- **Swin-T output:** 49 tokens (7×7 spatial grid), each 768-dimensional
- **Training config:** batch_size=32, lr=3e-4, AdamW, CosineAnnealingLR, 30 epochs
- **GPU:** NTU School of EEE GPU server (CUDA)
- **Image preprocessing:** Resize 224×224, normalize with ImageNet mean/std

## What Needs to Be Built

### 1. Region Grouping Module
- Input: 49 tokens of shape `(B, 49, 768)` from Swin-T last stage
- Method: k-means clustering in feature space (start simple)
- Output: K region features of shape `(B, K, 768)`, computed by averaging tokens within each cluster
- K is a hyperparameter to tune (try 4, 9, 16)
- Keep spatial position info for edge construction (each token has a known (row, col) in the 7×7 grid)

### 2. Dynamic Graph Construction
- Nodes: K region feature vectors
- Edges: experiment with multiple strategies and compare:
  - (a) Feature similarity: cosine similarity > threshold
  - (b) Spatial adjacency: regions that contain spatially neighboring tokens
  - (c) kNN in feature space
- Edge weights can be continuous (similarity scores) or binary
- Output: adjacency matrix or edge_index in PyG format

### 3. Graph Reasoning Layers
- Use PyTorch Geometric (PyG) layers: `GATConv`, `GCNConv`, or `TransformerConv`
- Start with 1–2 layers of GAT
- Input: node features `(K, 768)`, edge_index
- Output: updated node features `(K, 768)`

### 4. Classification Head
- Pool graph node features → single vector (mean pool or attention pool)
- FC layer → 45 classes
- Loss: CrossEntropyLoss (same as baseline)

### 5. Full GAViT Model
- Combine all modules into one `nn.Module`
- The Swin backbone can be frozen or fine-tuned (experiment with both)
- Training script should log: train loss, val accuracy, per-epoch metrics
- Save best model checkpoint based on val accuracy

## Experiments to Run

1. **Baseline:** Swin-T only (already done, ~96%)
2. **Swin-T + Region Grouping only** (pool grouped regions, no GNN)
3. **Swin-T + Region Grouping + GNN** (the full GAViT)
4. **Ablation on K** (number of region groups)
5. **Ablation on edge construction** (similarity vs spatial vs kNN)
6. **Ablation on GNN layers** (1 vs 2 layers, GAT vs GCN)
7. **(Optional)** Test on AID dataset

## Code Style & Conventions

- Python 3.8+, PyTorch style
- Use `timm` for backbone, `torch_geometric` for GNN layers
- Config variables at top of each script (DATA_ROOT, NUM_CLASSES, BATCH_SIZE, etc.)
- Use `tqdm` for progress bars
- Device handling: `"cuda" if torch.cuda.is_available() else "cpu"`
- Checkpoint saving: `checkpoints/` directory
- Chinese comments are acceptable but English is preferred for new code
- Print metrics clearly per epoch, save best model by val accuracy

## Important Constraints from Supervisor

- All results must be reproducible (set random seeds)
- Report averages and standard deviations when results vary across runs
- Accuracy format: one decimal place (e.g., 96.3%, not 96.3729%)
- Ablation studies are required to justify each component
- Do not fake results — all results subject to verification
- Compare with published methods in tables
- Statistical significance tests (Wilcoxon/Friedman) if needed

## Directory Structure (Target)

```
GAViT_Project/
├── datasets/
│   ├── NWPU-RESISC45/          # Raw dataset
│   └── NWPU-RESISC45_split/    # Split into train/val/test
├── checkpoints/                 # Saved model weights
├── split_nwpu.py
├── train_swin_baseline.py
├── test_swin.py
├── models/
│   ├── __init__.py
│   ├── swin_backbone.py        # Swin-T feature extractor (no classification head)
│   ├── region_grouping.py      # Token clustering module
│   ├── graph_construction.py   # Build graph from regions
│   ├── graph_reasoning.py      # GNN layers (GAT/GCN)
│   └── gavit.py                # Full GAViT model combining all modules
├── train_gavit.py              # Training script for GAViT
├── test_gavit.py               # Evaluation script for GAViT
├── utils.py                    # Shared utilities (metrics, logging, seeds)
└── CLAUDE.md                   # This file
```