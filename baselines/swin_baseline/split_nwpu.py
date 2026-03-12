import os
import shutil
import random
from tqdm import tqdm

# =========================
# 配置
# =========================
SRC_DIR = "../../datasets/NWPU-RESISC45"       # 原始数据集路径
DST_DIR = "../../datasets/NWPU-RESISC45_split"  # 划分后的数据集路径

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)  # 保证划分可重复

# =========================
# 开始划分
# =========================
for cls in os.listdir(SRC_DIR):
    cls_path = os.path.join(SRC_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split_name, split_images in splits.items():
        split_cls_dir = os.path.join(DST_DIR, split_name, cls)
        os.makedirs(split_cls_dir, exist_ok=True)

        for img in tqdm(split_images, desc=f"{cls} {split_name}", leave=False):
            shutil.copy(os.path.join(cls_path, img),
                        os.path.join(split_cls_dir, img))

print("Dataset split finished!")
print("Train/Val/Test dataset created at:", os.path.abspath(DST_DIR))
