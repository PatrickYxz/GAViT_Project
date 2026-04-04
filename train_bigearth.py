"""
train_bigearth.py — Multi-label training on BigEarthNet-RGB.

Supports two models:
  --model swin    → Swin-T baseline (global pool → FC → 19 logits)
  --model gavit   → GAViT v2 (token_feedback, attentive_spatial)

Loss:   BCEWithLogitsLoss (multi-label)
Metric: mAP (mean Average Precision, macro), Macro-F1 @ threshold 0.5

Usage:
    python train_bigearth.py --model swin
    python train_bigearth.py --model gavit --num_regions 16

Results saved to: checkpoints/best_bigearth_{model}.pth
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
from sklearn.metrics import average_precision_score, f1_score

from models.bigearth_dataset import BigEarthNetDataset, CLASSES_19, NUM_CLASSES
from models.gavit import GAViT
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model",        type=str, default="gavit",
                    choices=["swin", "gavit"])
parser.add_argument("--data_dir",     type=str,
                    default=os.environ.get("BIGEARTH_ROOT",
                        "datasets/BigEarthNet-RGB_split"))
parser.add_argument("--epochs",       type=int, default=30)
parser.add_argument("--batch_size",   type=int, default=32)
parser.add_argument("--lr",           type=float, default=3e-4)
parser.add_argument("--num_regions",  type=int, default=16)
parser.add_argument("--knn_k",        type=int, default=5)
parser.add_argument("--gat_hidden",   type=int, default=256)
parser.add_argument("--gat_heads",    type=int, default=4)
parser.add_argument("--gat_layers",   type=int, default=2)
parser.add_argument("--grouping",     type=str, default="attentive_spatial")
parser.add_argument("--edge_type",    type=str, default="knn")
parser.add_argument("--integration",  type=str, default="token_feedback")
parser.add_argument("--dropout",      type=float, default=0.1)
args = parser.parse_args()

SEED    = 42
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(SEED)

CKPT_NAME = f"best_bigearth_{args.model}"
if args.model == "gavit":
    CKPT_NAME += f"_K{args.num_regions}_{args.grouping}_{args.edge_type}_{args.integration}"
CKPT_PATH = os.path.join("checkpoints", f"{CKPT_NAME}.pth")
os.makedirs("checkpoints", exist_ok=True)

# =============================================================================
# Data
# =============================================================================
def load_split(csv_path: str):
    patch_dirs, labels = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            patch_dirs.append(row["patch_path"])
            label_vec = np.array([float(row[f"label_{i}"]) for i in range(NUM_CLASSES)],
                                  dtype=np.float32)
            labels.append(label_vec)
    return patch_dirs, np.stack(labels)


train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_patches, train_labels = load_split(os.path.join(args.data_dir, "train.csv"))
val_patches,   val_labels   = load_split(os.path.join(args.data_dir, "val.csv"))

train_set    = BigEarthNetDataset(train_patches, train_labels, transform=train_tf)
val_set      = BigEarthNetDataset(val_patches,   val_labels,   transform=val_tf)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

print(f"Train: {len(train_set):,}  |  Val: {len(val_set):,}")

# =============================================================================
# Model
# =============================================================================
if args.model == "swin":
    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
    )
    swin_dim = backbone.num_features  # 768

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
            feat = self.backbone.forward_features(x)   # (B, H*W, C) or (B, C)
            if feat.dim() == 3:
                feat = feat.mean(dim=1)
            return self.classifier(feat)

    model = SwinBaseline().to(DEVICE)

else:  # gavit
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
        pretrained=True,
        freeze_backbone=False,
    ).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model: {args.model.upper()} | {total_params:,} params | Device: {DEVICE}")

# =============================================================================
# Loss / Optimizer / Scheduler
# =============================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# =============================================================================
# Eval helpers
# =============================================================================
def evaluate(loader):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            logits = model(imgs)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits_np = np.concatenate(all_logits)   # (N, 19)
    labels_np = np.concatenate(all_labels)   # (N, 19)
    probs_np  = 1 / (1 + np.exp(-logits_np)) # sigmoid

    # mAP: macro (per-class AP averaged); skip classes with no positive samples
    ap_per_class = []
    for c in range(NUM_CLASSES):
        if labels_np[:, c].sum() > 0:
            ap_per_class.append(
                average_precision_score(labels_np[:, c], probs_np[:, c])
            )
    mAP = float(np.mean(ap_per_class)) * 100.0

    # Macro F1 at threshold 0.5
    preds_np = (probs_np >= 0.5).astype(int)
    f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0) * 100.0

    return mAP, f1

# =============================================================================
# Training loop
# =============================================================================
best_mAP = 0.0
print(f"\n{'='*60}")
print(f"Training {args.model.upper()} on BigEarthNet-19 | {args.epochs} epochs")
print(f"{'='*60}\n")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}] Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    scheduler.step()
    avg_loss = total_loss / len(train_set)

    val_mAP, val_f1 = evaluate(val_loader)
    print(f"Epoch [{epoch:2d}/{args.epochs}]  "
          f"Loss: {avg_loss:.4f}  |  Val mAP: {val_mAP:.1f}%  |  Val F1: {val_f1:.1f}%")

    if val_mAP > best_mAP:
        best_mAP = val_mAP
        torch.save(model.state_dict(), CKPT_PATH)
        print(f"  Best model saved -> {CKPT_PATH}")

print(f"\nBest Validation mAP: {best_mAP:.1f}%")