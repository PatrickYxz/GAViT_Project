"""
test_bigearth.py — Evaluate a BigEarthNet checkpoint on the test set.

Usage:
    # Swin-T baseline
    python test_bigearth.py --model swin \
        --ckpt checkpoints/best_bigearth_swin.pth

    # GAViT v2
    python test_bigearth.py --model gavit \
        --ckpt checkpoints/best_bigearth_gavit_K16_attentive_spatial_knn_token_feedback.pth \
        --num_regions 16 --grouping attentive_spatial --integration token_feedback
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

from models.bigearth_dataset import BigEarthNetDataset, CLASSES_19, NUM_CLASSES
from models.gavit import GAViT
from utils import set_seed

# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model",       type=str, required=True, choices=["swin", "gavit"])
parser.add_argument("--ckpt",        type=str, required=True)
parser.add_argument("--data_dir",    type=str,
                    default=os.environ.get("BIGEARTH_ROOT",
                        "datasets/BigEarthNet-RGB_split"))
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--num_regions", type=int, default=16)
parser.add_argument("--knn_k",       type=int, default=5)
parser.add_argument("--gat_hidden",  type=int, default=256)
parser.add_argument("--gat_heads",   type=int, default=4)
parser.add_argument("--gat_layers",  type=int, default=2)
parser.add_argument("--grouping",    type=str, default="attentive_spatial")
parser.add_argument("--edge_type",   type=str, default="knn")
parser.add_argument("--integration", type=str, default="token_feedback")
parser.add_argument("--dropout",     type=float, default=0.1)
args = parser.parse_args()

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED)

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
                         num_workers=4, pin_memory=True)
print(f"Test samples: {len(test_set):,}")

# =============================================================================
# Model
# =============================================================================
if args.model == "swin":
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
                nn.Dropout(args.dropout),
                nn.Linear(swin_dim, NUM_CLASSES),
            )
        def forward(self, x):
            feat = self.backbone.forward_features(x)
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            return self.classifier(feat)

    model = SwinBaseline().to(DEVICE)
else:
    model = GAViT(
        num_classes=NUM_CLASSES,
        num_regions=args.num_regions,
        knn_k=args.knn_k,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        grouping=args.grouping,
        edge_type=args.edge_type,
        integration=args.integration,
        pretrained=False,
        freeze_backbone=False,
    ).to(DEVICE)

model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
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