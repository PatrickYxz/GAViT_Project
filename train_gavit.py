"""
train_gavit.py — Training script for GAViT on NWPU-RESISC45.

Usage:
    python train_gavit.py
    python train_gavit.py --grouping attentive_spatial --num_regions 16 --integration token_feedback
    python train_gavit.py --grouping spatial --num_regions 9 --integration fusion
    python train_gavit.py --grouping kmeans  --num_regions 9 --gat_layers 1

Key hyperparameters are defined in the CONFIG section below and can be
overridden via command-line arguments.
"""

import argparse
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment_identity import (
    assert_clean_git_state,
    assert_fresh_output_paths,
    get_git_state,
    metadata_path_for,
    validate_run_stage,
    validate_run_tag,
    write_checkpoint_metadata,
)
from models.gavit import GAViT
from utils import save_checkpoint, set_seed

# =============================================================================
# ARGS
# =============================================================================
parser = argparse.ArgumentParser(description="Train GAViT on NWPU-RESISC45")
parser.add_argument("--grouping",      type=str,   default="attentive_spatial", choices=["kmeans", "spatial", "attentive_spatial"])
parser.add_argument("--num_regions",   type=int,   default=16)
parser.add_argument("--knn_k",         type=int,   default=5)
parser.add_argument("--gat_hidden",    type=int,   default=256)
parser.add_argument("--gat_heads",     type=int,   default=4)
parser.add_argument("--gat_layers",    type=int,   default=2)
parser.add_argument("--dropout",       type=float, default=0.1)
parser.add_argument("--edge_type",    type=str,   default="knn", choices=["knn", "spatial", "hybrid"])
parser.add_argument("--integration",  type=str,   default="token_feedback", choices=["token_feedback", "fusion"])
parser.add_argument("--epochs",        type=int,   default=30)
parser.add_argument("--lr",            type=float, default=3e-4)
parser.add_argument("--batch_size",    type=int,   default=32)
parser.add_argument("--seed",          type=int,   default=42)
parser.add_argument("--run_stage",     type=str,   required=True,
                    choices=["smoke", "proxy", "formal"])
parser.add_argument("--run_tag",       type=str,   required=True,
                    help="Unique artifact identity, e.g. corrected_knn_smoke")
parser.add_argument("--freeze_backbone", action="store_true")
args = parser.parse_args()

# =============================================================================
# CONFIG
# =============================================================================
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
SAVE_DIR    = "checkpoints"
RUN_STAGE   = validate_run_stage(args.run_stage)
RUN_TAG     = validate_run_tag(args.run_tag)
# Include knn_k and seed so multi-seed / multi-k runs never overwrite each other
CKPT_NAME   = (f"best_gavit_K{args.num_regions}_{args.grouping}_{args.edge_type}"
               f"_k{args.knn_k}_{args.integration}_seed{args.seed}"
               f"_{RUN_STAGE}_{RUN_TAG}.pth")
CKPT_PATH   = os.path.join(SAVE_DIR, CKPT_NAME)

NUM_CLASSES = 45
BATCH_SIZE  = args.batch_size
EPOCHS      = args.epochs
LR          = args.lr
SEED        = args.seed

# GAViT-specific
NUM_REGIONS     = args.num_regions
KNN_K           = args.knn_k
GAT_HIDDEN      = args.gat_hidden
GAT_HEADS       = args.gat_heads
GAT_LAYERS      = args.gat_layers
DROPOUT         = args.dropout
GROUPING        = args.grouping
EDGE_TYPE       = args.edge_type
INTEGRATION     = args.integration
FREEZE_BACKBONE = args.freeze_backbone

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================
set_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)
if RUN_STAGE == "formal":
    assert_clean_git_state(get_git_state())
assert_fresh_output_paths([CKPT_PATH, metadata_path_for(CKPT_PATH)])

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

train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
val_set   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"),   transform=val_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# =============================================================================
# MODEL
# =============================================================================
model = GAViT(
    num_classes=NUM_CLASSES,
    num_regions=NUM_REGIONS,
    knn_k=KNN_K,
    gat_hidden=GAT_HIDDEN,
    gat_heads=GAT_HEADS,
    gat_layers=GAT_LAYERS,
    dropout=DROPOUT,
    grouping=GROUPING,
    edge_type=EDGE_TYPE,
    integration=INTEGRATION,
    pretrained=True,
    freeze_backbone=FREEZE_BACKBONE,
).to(DEVICE)

total_params   = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: GAViT | K={NUM_REGIONS} | grouping={GROUPING} | edge={EDGE_TYPE} | "
      f"integration={INTEGRATION} | GAT {GAT_LAYERS}L×{GAT_HEADS}H | kNN k={KNN_K}")
print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
print(f"Device: {DEVICE}\n")

# =============================================================================
# OPTIMIZER & SCHEDULER
# =============================================================================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =============================================================================
# TRAINING LOOP
# =============================================================================
metadata = {
    "model":   "gavit",
    "dataset": "NWPU-RESISC45",
    "architecture": {
        "num_classes": NUM_CLASSES,
        "num_regions": NUM_REGIONS,
        "knn_k":       KNN_K,
        "gat_hidden":  GAT_HIDDEN,
        "gat_heads":   GAT_HEADS,
        "gat_layers":  GAT_LAYERS,
        "dropout":     DROPOUT,
        "grouping":    GROUPING,
        "edge_type":   EDGE_TYPE,
        "integration": INTEGRATION,
        "freeze_backbone": FREEZE_BACKBONE,
    },
    "training": {
        "seed":        SEED,
        "epochs":      EPOCHS,
        "batch_size":  BATCH_SIZE,
        "lr":          LR,
        "weight_decay": 1e-4,
        "optimizer":   "AdamW",
        "scheduler":   "CosineAnnealingLR",
        "loss":        "CrossEntropyLoss",
    },
    "execution": {
        "run_stage": RUN_STAGE,
        "run_tag": RUN_TAG,
    },
}
write_checkpoint_metadata(CKPT_PATH, metadata)

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

        train_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct_train += (preds == labels).sum().item()
        total_train   += labels.size(0)

    train_loss /= len(train_loader)
    train_acc   = 100.0 * correct_train / total_train

    # ---- Validation ----
    model.eval()
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Val  "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)
            correct_val += (preds == labels).sum().item()
            total_val   += labels.size(0)

    val_acc = 100.0 * correct_val / total_val
    scheduler.step()

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"  Val   Acc:  {val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = CKPT_PATH
        metadata["best"] = {"metric": "val_acc", "value": round(val_acc, 4),
                            "epoch": epoch + 1}
        save_checkpoint(model, save_path, metadata)
        print(f"  Best model saved -> {save_path}")

print(f"\nBest Validation Accuracy: {best_val_acc:.1f}%")
