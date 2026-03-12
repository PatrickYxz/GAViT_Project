import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

# =========================
# 1. 配置
# =========================
DATA_ROOT = r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"
NUM_CLASSES = 45
BATCH_SIZE = 32
EPOCHS = 30
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    num_workers=0
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
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
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_swin.pth"))
        print("✅ Best model saved")

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
