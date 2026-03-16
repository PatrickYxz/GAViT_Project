"""
test_gavit.py — Evaluate a saved GAViT checkpoint on the test set.

Usage:
    python test_gavit.py
"""

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gavit import GAViT
from utils import set_seed

# =============================================================================
# CONFIG  (must match the training config used to produce the checkpoint)
# =============================================================================
DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
CKPT_PATH = os.path.join("checkpoints", "best_gavit.pth")

NUM_CLASSES     = 45
BATCH_SIZE      = 32
SEED            = 42

NUM_REGIONS     = 9
KNN_K           = 5
GAT_HIDDEN      = 256
GAT_HEADS       = 4
GAT_LAYERS      = 2
DROPOUT         = 0.1
GROUPING        = "kmeans"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
set_seed(SEED)

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
test_set    = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

model = GAViT(
    num_classes=NUM_CLASSES,
    num_regions=NUM_REGIONS,
    knn_k=KNN_K,
    gat_hidden=GAT_HIDDEN,
    gat_heads=GAT_HEADS,
    gat_layers=GAT_LAYERS,
    dropout=DROPOUT,
    grouping=GROUPING,
    pretrained=False,       # weights loaded from checkpoint
    freeze_backbone=False,
).to(DEVICE)

model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print(f"Loaded checkpoint: {CKPT_PATH}")

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.1f}%  ({correct}/{total})")
