"""
test_bigearth.py — Evaluate a BigEarthNet checkpoint on the test set.

The GAViT architecture is reconstructed from the checkpoint's sidecar
metadata (<ckpt>.meta.json) when available; explicit CLI overrides that
conflict with the metadata are rejected. For legacy checkpoints without a
sidecar, architecture falls back to the CLI flags below.

Usage:
    # Swin-T baseline
    python test_bigearth.py --model swin \
        --ckpt checkpoints/best_bigearth_swin_seed42.pth

    # GAViT v2
    python test_bigearth.py --model gavit \
        --ckpt checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42.pth
"""

import os
import csv
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, classification_report

from experiment_identity import validate_checkpoint_identity
from models.bigearth_dataset import BigEarthNetDataset, CLASSES_19, NUM_CLASSES
from models.gavit import GAViT
from models.swin_features import pool_swin_features
from utils import load_checkpoint_metadata, resolve_arch_config, set_seed

# =============================================================================
# Architecture flags default to None -> taken from checkpoint metadata
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model",       type=str, required=True, choices=["swin", "gavit"])
parser.add_argument("--ckpt",        type=str, required=True)
parser.add_argument("--data_dir",    type=str,
                    default=os.environ.get("BIGEARTH_ROOT",
                        "datasets/BigEarthNet-RGB_split"))
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--num_regions", type=int, default=None)
parser.add_argument("--knn_k",       type=int, default=None)
parser.add_argument("--gat_hidden",  type=int, default=None)
parser.add_argument("--gat_heads",   type=int, default=None)
parser.add_argument("--gat_layers",  type=int, default=None)
parser.add_argument("--grouping",    type=str, default=None,
                    choices=["kmeans", "spatial", "attentive_spatial"])
parser.add_argument("--edge_type",   type=str, default=None,
                    choices=["knn", "spatial", "hybrid", "sparse_hybrid"])
parser.add_argument("--integration", type=str, default=None,
                    choices=["token_feedback", "fusion"])
parser.add_argument("--dropout",     type=float, default=None)
parser.add_argument("--seed",        type=int, default=42)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(args.seed)

# =============================================================================
# Data
# =============================================================================
def load_split(csv_path):
    patch_dirs, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_dirs.append(row["patch_path"])
            label_vec = np.array([float(row[f"label_{i}"]) for i in range(NUM_CLASSES)],
                                  dtype=np.float32)
            labels.append(label_vec)
    return patch_dirs, np.stack(labels)

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_patches, test_labels = load_split(os.path.join(args.data_dir, "test.csv"))
test_set    = BigEarthNetDataset(test_patches, test_labels, transform=test_tf)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=False)  # 2026-09-09 hotfix: same pinned-pool host OOM as the val loader; test set is 119825 images
print(f"Test samples: {len(test_set):,}")

# =============================================================================
# Model (architecture reconstructed from checkpoint metadata when available)
# =============================================================================
metadata = load_checkpoint_metadata(args.ckpt)
if metadata is None:
    print(f"WARNING: no sidecar metadata found for {args.ckpt}; "
          "falling back to CLI architecture flags.")
else:
    validate_checkpoint_identity(
        metadata,
        expected_model=args.model,
        expected_dataset="BigEarthNet-19",
        expected_num_classes=NUM_CLASSES,
    )

if args.model == "swin":
    swin_dropout = args.dropout
    if metadata is not None:
        swin_dropout = metadata.get("architecture", {}).get("dropout", 0.1)
        if args.dropout is not None and args.dropout != swin_dropout:
            raise ValueError(
                f"--dropout {args.dropout} conflicts with checkpoint metadata "
                f"(dropout={swin_dropout})"
            )
    if swin_dropout is None:
        swin_dropout = 0.1

    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=False, num_classes=0
    )
    swin_dim = backbone.num_features

    class SwinBaseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone   = backbone
            self.classifier = nn.Sequential(
                nn.LayerNorm(swin_dim),
                nn.Dropout(swin_dropout),
                nn.Linear(swin_dim, NUM_CLASSES),
            )
        def forward(self, x):
            feat = self.backbone.forward_features(x)
            feat = pool_swin_features(feat)
            return self.classifier(feat)

    model = SwinBaseline().to(DEVICE)
    print("Architecture source: " + ("metadata" if metadata else "cli"))
else:
    ARCH_DEFAULTS = {
        "num_classes": NUM_CLASSES,
        "num_regions": 16,
        "knn_k":       5,
        "gat_hidden":  256,
        "gat_heads":   4,
        "gat_layers":  2,
        "dropout":     0.1,
        "grouping":    "attentive_spatial",
        "edge_type":   "knn",
        "integration": "token_feedback",
    }
    arch, arch_source = resolve_arch_config(
        {k: getattr(args, k) for k in ARCH_DEFAULTS if k != "num_classes"},
        metadata,
        ARCH_DEFAULTS,
    )
    print(f"Architecture source: {arch_source}")
    model = GAViT(
        num_classes=arch["num_classes"],
        num_regions=arch["num_regions"],
        knn_k=arch["knn_k"],
        gat_hidden=arch["gat_hidden"],
        gat_heads=arch["gat_heads"],
        gat_layers=arch["gat_layers"],
        dropout=arch["dropout"],
        grouping=arch["grouping"],
        edge_type=arch["edge_type"],
        integration=arch["integration"],
        pretrained=False,
        freeze_backbone=False,
    ).to(DEVICE)

model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded: {args.ckpt}")

# =============================================================================
# Evaluate
# =============================================================================
all_logits, all_labels = [], []
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        logits = model(imgs.to(DEVICE))
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())

logits_np = np.concatenate(all_logits)
labels_np = np.concatenate(all_labels)
probs_np  = 1 / (1 + np.exp(-logits_np))
preds_np  = (probs_np >= 0.5).astype(int)

# mAP
ap_per_class = []
for c in range(NUM_CLASSES):
    if labels_np[:, c].sum() > 0:
        ap_per_class.append(average_precision_score(labels_np[:, c], probs_np[:, c]))
mAP     = float(np.mean(ap_per_class)) * 100.0

# F1
macro_f1 = f1_score(labels_np, preds_np, average="macro",  zero_division=0) * 100.0
micro_f1 = f1_score(labels_np, preds_np, average="micro",  zero_division=0) * 100.0

print(f"\n{'='*50}")
print(f"Test mAP   (macro): {mAP:.1f}%")
print(f"Test F1    (macro): {macro_f1:.1f}%")
print(f"Test F1    (micro): {micro_f1:.1f}%")
print(f"{'='*50}")

# Per-class AP
print("\nPer-class Average Precision:")
for c, cls_name in enumerate(CLASSES_19):
    if labels_np[:, c].sum() > 0:
        ap = average_precision_score(labels_np[:, c], probs_np[:, c]) * 100.0
        print(f"  {ap:5.1f}%  {cls_name}")
    else:
        print(f"  {'N/A':>5}   {cls_name}")
