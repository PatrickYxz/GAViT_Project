"""
test_gavit.py — Evaluate a saved GAViT checkpoint on the test set.

The model architecture is reconstructed from the checkpoint's sidecar
metadata (<ckpt>.meta.json) when available; explicit CLI overrides that
conflict with the metadata are rejected. For legacy checkpoints without a
sidecar, architecture falls back to the CLI flags below.

Usage:
    python test_gavit.py --ckpt checkpoints/best_gavit_K16_attentive_spatial_knn_k5_token_feedback_seed42.pth
"""

import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiment_identity import validate_checkpoint_identity
from models.gavit import GAViT
from utils import load_checkpoint_metadata, resolve_arch_config, set_seed

# =============================================================================
# Args (architecture flags default to None -> taken from checkpoint metadata)
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        type=str, required=True)
parser.add_argument("--grouping",    type=str, default=None,
                    choices=["spatial", "kmeans", "attentive_spatial"])
parser.add_argument("--integration", type=str, default=None,
                    choices=["fusion", "token_feedback"])
parser.add_argument("--edge_type",   type=str, default=None,
                    choices=["knn", "spatial", "hybrid", "sparse_hybrid"])
parser.add_argument("--num_regions", type=int, default=None)
parser.add_argument("--knn_k",       type=int, default=None)
parser.add_argument("--gat_hidden",  type=int, default=None)
parser.add_argument("--gat_heads",   type=int, default=None)
parser.add_argument("--gat_layers",  type=int, default=None)
parser.add_argument("--dropout",     type=float, default=None)
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--seed",        type=int, default=42)
parser.add_argument("--data_root",   type=str,
                    default=os.environ.get("DATA_ROOT",
                        r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"))
args = parser.parse_args()

# =============================================================================
NUM_CLASSES = 45
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(args.seed)

# Reconstruct architecture: prefer checkpoint sidecar metadata
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
metadata = load_checkpoint_metadata(args.ckpt)
if metadata is None:
    print(f"WARNING: no sidecar metadata found for {args.ckpt}; "
          "falling back to CLI architecture flags.")
else:
    validate_checkpoint_identity(
        metadata,
        expected_model="gavit",
        expected_dataset="NWPU-RESISC45",
        expected_num_classes=NUM_CLASSES,
    )
arch, arch_source = resolve_arch_config(
    {k: getattr(args, k) for k in ARCH_DEFAULTS if k != "num_classes"},
    metadata,
    ARCH_DEFAULTS,
)
print(f"Architecture source: {arch_source}")

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
print(f"Config: K={arch['num_regions']} | grouping={arch['grouping']} | "
      f"integration={arch['integration']} | edge={arch['edge_type']} | k={arch['knn_k']}")

correct, total = 0, 0
with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Test"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

test_acc = 100.0 * correct / total
print(f"\nTest Accuracy: {test_acc:.1f}%  ({correct}/{total})")
