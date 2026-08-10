import argparse
import os
import sys
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from experiment_identity import (  # noqa: E402
    assert_clean_git_state,
    assert_fresh_output_paths,
    get_git_state,
    metadata_path_for,
    validate_run_stage,
    validate_run_tag,
    write_checkpoint_metadata,
)
from utils import save_checkpoint, set_seed  # noqa: E402

# =========================
# 1. 配置
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_stage", type=str, required=True,
                    choices=["smoke", "proxy", "formal"])
parser.add_argument("--run_tag", type=str, required=True,
                    help="Unique artifact identity, e.g. swin_smoke")
args = parser.parse_args()

DATA_ROOT = os.environ.get(
    "DATA_ROOT",
    r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
)
NUM_CLASSES = 45
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
SEED = args.seed
RUN_STAGE = validate_run_stage(args.run_stage)
RUN_TAG = validate_run_tag(args.run_tag)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(SEED)

SAVE_DIR = "checkpoints"
# Seed in the filename so multi-seed runs never overwrite each other
CKPT_NAME = f"best_swin_seed{SEED}_{RUN_STAGE}_{RUN_TAG}.pth"
CKPT_PATH = os.path.join(SAVE_DIR, CKPT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
if RUN_STAGE == "formal":
    assert_clean_git_state(get_git_state())
assert_fresh_output_paths([CKPT_PATH, metadata_path_for(CKPT_PATH)])

METADATA = {
    "model":   "swin",
    "dataset": "NWPU-RESISC45",
    "architecture": {
        "num_classes": NUM_CLASSES,
        "backbone":    "swin_tiny_patch4_window7_224",
    },
    "training": {
        "seed":        SEED,
        "epochs":      EPOCHS,
        "batch_size":  BATCH_SIZE,
        "lr":          LR,
        "optimizer":   "AdamW",
        "scheduler":   "CosineAnnealingLR",
        "loss":        "CrossEntropyLoss",
    },
    "execution": {
        "run_stage": RUN_STAGE,
        "run_tag": RUN_TAG,
    },
}
write_checkpoint_metadata(CKPT_PATH, METADATA)

# =========================
# 2. 数据
# =========================
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "train"), transform=train_tf)
val_set   = datasets.ImageFolder(os.path.join(DATA_ROOT, "val"), transform=val_tf)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)


# =========================
# 3. 模型（Swin-Tiny）
# =========================
model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=True,
    num_classes=NUM_CLASSES
)
model.to(DEVICE)

# =========================
# 4. 训练组件
# =========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# =========================
# 5. 训练 & 验证
# =========================
best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch [{epoch+1}/{EPOCHS}]")

    # ---- Train ----
    model.train()
    train_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc="Training"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- Validation ----
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Validation"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100.0 * correct / total
    scheduler.step()

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")

    # ---- Save best ----
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        METADATA["best"] = {"metric": "val_acc", "value": round(val_acc, 4),
                            "epoch": epoch + 1}
        save_checkpoint(model, CKPT_PATH, METADATA)
        print(f"✅ Best model saved -> {CKPT_PATH}")

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
