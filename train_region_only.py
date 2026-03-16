"""
train_region_only.py — Ablation: Swin-T + Region Grouping (no GNN).

Experiment purpose:
    Verify whether region grouping alone (without graph reasoning) improves
    over the plain Swin-T baseline. This is a required ablation step before
    claiming GAViT's GNN module is beneficial.

Experiment plan:
    Run this script with GROUPING = "spatial" first (deterministic, fast),
    then optionally with "kmeans" (feature-space, may vary across runs → run 3x).

Usage:
    python train_region_only.py
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.swin_region_only import SwinRegionOnly
from utils import set_seed

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
SAVE_DIR  = "checkpoints"

NUM_CLASSES     = 45
BATCH_SIZE      = 32
EPOCHS          = 30
LR              = 3e-4
SEED            = 42

NUM_REGIONS     = 9         # K: try 1 (=baseline-like), 4, 9, 16
GROUPING        = "spatial" # "spatial" (start here) or "kmeans"
FREEZE_BACKBONE = False

CKPT_NAME = f"best_region_only_K{NUM_REGIONS}_{GROUPING}.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================================================================
# DATA
# =============================================================================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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

train_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
)
val_loader = DataLoader(
    datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tf),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
)

# =============================================================================
# MODEL
# =============================================================================
model = SwinRegionOnly(
    num_classes=NUM_CLASSES,
    num_regions=NUM_REGIONS,
    grouping=GROUPING,
    pretrained=True,
    freeze_backbone=FREEZE_BACKBONE,
).to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: Swin + Region Grouping only | K={NUM_REGIONS} | grouping={GROUPING}")
print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
print(f"Device: {DEVICE}\n")

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =============================================================================
# TRAINING LOOP
# =============================================================================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    # ---- Train ----
    model.train()
    train_loss = 0.0
    correct_train, total_train = 0, 0

    for imgs, labels in tqdm(train_loader, desc="Train"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        correct_train += (logits.argmax(1) == labels).sum().item()
        total_train   += labels.size(0)

    train_loss /= len(train_loader)
    train_acc   = 100.0 * correct_train / total_train

    # ---- Validation ----
    model.eval()
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Val  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(1)
            correct_val += (preds == labels).sum().item()
            total_val   += labels.size(0)

    val_acc = 100.0 * correct_val / total_val
    scheduler.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"  Val   Acc:  {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(SAVE_DIR, CKPT_NAME)
        torch.save(model.state_dict(), save_path)
        print(f"  Best model saved -> {save_path}")

print(f"\nBest Validation Accuracy: {best_val_acc:.1f}%")
print(f"Config: K={NUM_REGIONS}, grouping={GROUPING}, epochs={EPOCHS}, lr={LR}")
