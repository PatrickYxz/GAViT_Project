import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

DATA_ROOT = r"C:\Users\Administrator\PycharmProjects\GAViT_Project\datasets\NWPU-RESISC45_split"  # 同样改路径
NUM_CLASSES = 45
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_set = datasets.ImageFolder(os.path.join(DATA_ROOT, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load("checkpoints/best_swin.pth"))
model.to(DEVICE)
model.eval()

correct, total = 0, 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

acc = 100.0 * correct / total
print(f"🎯 Test Accuracy: {acc:.2f}%")
