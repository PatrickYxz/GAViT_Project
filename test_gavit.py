"""
test_gavit.py — Evaluate a saved GAViT checkpoint on the test set.

Usage:
    python test_gavit.py --ckpt checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth \
                         --grouping attentive_spatial --integration token_feedback --num_regions 16
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gavit import GAViT
from utils import set_seed

# =============================================================================
# Args
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        type=str, default="checkpoints/best_gavit_K16_attentive_spatial_knn_token_feedback.pth")
parser.add_argument("--grouping",    type=str, default="attentive_spatial",
                    choices=["spatial", "kmeans", "attentive_spatial"])
parser.add_argument("--integration", type=str, default="token_feedback",
                    choices=["graph_only", "fusion", "token_feedback"])
parser.add_argument("--edge_type",   type=str, default="knn",
                    choices=["knn", "spatial", "hybrid"])
parser.add_argument("--num_regions", type=int, default=16)
parser.add_argument("--knn_k",       type=int, default=5)
parser.add_argument("--gat_hidden",  type=int, default=256)
parser.add_argument("--gat_heads",   type=int, default=4)
parser.add_argument("--gat_layers",  type=int, default=2)
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--data_root",   type=str,
                    default=os.environ.get("DATA_ROOT",
                        r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"))
args = parser.parse_args()

# =============================================================================
SEED        = 42
NUM_CLASSES = 45
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(SEED)

# =============================================================================
test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
test_set    = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

model = GAViT(
    num_classes=NUM_CLASSES,
    num_regions=args.num_regions,
    knn_k=args.knn_k,
    gat_hidden=args.gat_hidden,
    gat_heads=args.gat_heads,
    gat_layers=args.gat_layers,
    dropout=0.1,
    grouping=args.grouping,
    edge_type=args.edge_type,
    integration=args.integration,
    pretrained=False,
    freeze_backbone=False,
).to(DEVICE)

model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
model.eval()
print(f"Loaded: {args.ckpt}")
print(f"Config: K={args.num_regions} | grouping={args.grouping} | integration={args.integration} | edge={args.edge_type}")

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.1f}%  ({correct}/{total})")