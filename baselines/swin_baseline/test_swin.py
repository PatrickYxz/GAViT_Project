import os
import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",      default="checkpoints/best_swin.pth")
parser.add_argument("--data_root", default="/home/yang1004/GAViT_Project/datasets/NWPU-RESISC45_split")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

NUM_CLASSES = 45
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_set    = datasets.ImageFolder(os.path.join(args.data_root, "test"), transform=test_tf)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,
    num_classes=NUM_CLASSES
)
model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
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
print(f"Test Accuracy: {acc:.1f}%")
