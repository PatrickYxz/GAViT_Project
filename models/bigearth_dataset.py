"""
BigEarthNet-RGB Dataset for multi-label scene classification.

Assumes BigEarthNet-RGB directory structure:
    BigEarthNet-RGB/
        <patch_name>/
            <patch_name>_B02.tif   (Blue)
            <patch_name>_B03.tif   (Green)
            <patch_name>_B04.tif   (Red)
            <patch_name>_labels_metadata.json

Labels: 19-class multi-label (BigEarthNet-19 standard).
Each sample returns a float32 binary vector of length 19.
"""

import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

# =============================================================================
# BigEarthNet 43-class → 19-class mapping (official)
# =============================================================================
LABEL_CONVERSION = {
    "Continuous urban fabric":                           "Urban fabric",
    "Discontinuous urban fabric":                        "Urban fabric",
    "Green urban areas":                                 "Urban fabric",
    "Sport and leisure facilities":                      "Urban fabric",
    "Industrial or commercial units":                    "Industrial or commercial units",
    "Road and rail networks and associated land":        "Industrial or commercial units",
    "Port areas":                                        "Industrial or commercial units",
    "Airports":                                          "Industrial or commercial units",
    "Mineral extraction sites":                          "Industrial or commercial units",
    "Dump sites":                                        "Industrial or commercial units",
    "Construction sites":                                "Industrial or commercial units",
    "Non-irrigated arable land":                         "Arable land",
    "Permanently irrigated land":                        "Arable land",
    "Rice fields":                                       "Arable land",
    "Vineyards":                                         "Permanent crops",
    "Fruit trees and berry plantations":                 "Permanent crops",
    "Olive groves":                                      "Permanent crops",
    "Annual crops associated with permanent crops":      "Permanent crops",
    "Pastures":                                          "Pastures",
    "Complex cultivation patterns":                      "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
                                                         "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas":                               "Agro-forestry areas",
    "Broad-leaved forest":                               "Broad-leaved forest",
    "Coniferous forest":                                 "Coniferous forest",
    "Mixed forest":                                      "Mixed forest",
    "Natural grassland":                                 "Natural grassland and sparsely vegetated areas",
    "Bare rock":                                         "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas":                          "Natural grassland and sparsely vegetated areas",
    "Burnt areas":                                       "Natural grassland and sparsely vegetated areas",
    "Moors and heathland":                               "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation":                         "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub":                       "Transitional woodland, shrub",
    "Beaches, dunes, sands":                             "Beaches, dunes, sands",
    "Inland marshes":                                    "Inland wetlands",
    "Peatbogs":                                          "Inland wetlands",
    "Salt marshes":                                      "Coastal wetlands",
    "Salines":                                           "Coastal wetlands",
    "Intertidal flats":                                  "Coastal wetlands",
    "Water courses":                                     "Inland waters",
    "Water bodies":                                      "Inland waters",
    "Coastal lagoons":                                   "Marine waters",
    "Estuaries":                                         "Marine waters",
    "Sea and ocean":                                     "Marine waters",
}

# 19 canonical class names (fixed order → index mapping)
CLASSES_19 = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_19)}
NUM_CLASSES  = len(CLASSES_19)  # 19


def parse_labels(json_path: str) -> np.ndarray:
    """
    Read a BigEarthNet patch JSON and return a 19-dim binary label vector.
    Handles both 43-class labels (needs conversion) and 19-class labels directly.
    """
    with open(json_path, "r") as f:
        meta = json.load(f)

    raw_labels = meta.get("labels", [])
    label_vec  = np.zeros(NUM_CLASSES, dtype=np.float32)

    for lbl in raw_labels:
        # Try direct 19-class match first
        if lbl in CLASS_TO_IDX:
            label_vec[CLASS_TO_IDX[lbl]] = 1.0
        # Fall back to 43→19 conversion
        elif lbl in LABEL_CONVERSION:
            mapped = LABEL_CONVERSION[lbl]
            label_vec[CLASS_TO_IDX[mapped]] = 1.0

    return label_vec


def read_rgb_patch(patch_dir: str) -> np.ndarray:
    """
    Read a BigEarthNet-RGB patch.

    Tries (in order):
      1. Pre-merged RGB image: <patch_name>.png or <patch_name>.jpg
      2. Separate band TIFs: B04 (R), B03 (G), B02 (B)

    Returns: uint8 array of shape (H, W, 3)
    """
    patch_name = os.path.basename(patch_dir)

    # Option 1: pre-merged RGB file
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
        rgb_path = os.path.join(patch_dir, patch_name + ext)
        if os.path.exists(rgb_path):
            return np.array(Image.open(rgb_path).convert("RGB"))

    # Option 2: separate band TIFs (B04=R, B03=G, B02=B)
    bands = []
    for band in ("B04", "B03", "B02"):
        tif_path = os.path.join(patch_dir, f"{patch_name}_{band}.tif")
        if not os.path.exists(tif_path):
            raise FileNotFoundError(f"Band file not found: {tif_path}")
        band_img = np.array(Image.open(tif_path))
        # BigEarthNet band values are uint16; normalize to uint8 for display/model
        band_img = np.clip(band_img / 4095.0 * 255, 0, 255).astype(np.uint8)
        bands.append(band_img)

    return np.stack(bands, axis=-1)  # (H, W, 3)


class BigEarthNetDataset(Dataset):
    """
    Multi-label BigEarthNet-RGB dataset.

    Args:
        patch_dirs:  list of absolute paths to patch directories
        labels:      np.ndarray of shape (N, 19), float32 binary label vectors
        transform:   torchvision transform applied to PIL Image
    """

    def __init__(self, patch_dirs: list, labels: np.ndarray, transform=None):
        assert len(patch_dirs) == len(labels)
        self.patch_dirs = patch_dirs
        self.labels     = labels
        self.transform  = transform

    def __len__(self):
        return len(self.patch_dirs)

    def __getitem__(self, idx):
        img_array = read_rgb_patch(self.patch_dirs[idx])
        img       = Image.fromarray(img_array)

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(self.labels[idx])  # float32, shape (19,)
        return img, label


# =============================================================================
# Unit test
# =============================================================================
if __name__ == "__main__":
    print(f"BigEarthNet-19 classes ({NUM_CLASSES}):")
    for i, c in enumerate(CLASSES_19):
        print(f"  {i:2d}  {c}")