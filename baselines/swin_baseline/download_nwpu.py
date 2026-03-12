from datasets import load_dataset
import os
from tqdm import tqdm

# =========================
# 配置
# =========================
DATASET_NAME = "jonathan-roberts1/NWPU-RESISC45"
OUTPUT_DIR = "../../datasets/NWPU-RESISC45"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading dataset from Hugging Face...")
dataset = load_dataset(DATASET_NAME, split="train")

# 获取类别名列表
label_names = dataset.features["label"].names

print("Saving images to disk...")

for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
    image = sample["image"]
    label_idx = sample["label"]          # 原来的整数
    label = label_names[label_idx]       # 转成类别名字符串

    class_dir = os.path.join(OUTPUT_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    image_path = os.path.join(class_dir, f"{idx}.jpg")
    image.save(image_path)

print("Done! Dataset saved at:", os.path.abspath(OUTPUT_DIR))
