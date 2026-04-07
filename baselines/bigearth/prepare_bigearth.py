"""
prepare_bigearth.py — Prepare BigEarthNet-RGB splits.

Downloads official split lists and generates three CSV files:
    datasets/BigEarthNet-RGB_split/train.csv
    datasets/BigEarthNet-RGB_split/val.csv
    datasets/BigEarthNet-RGB_split/test.csv

Each CSV has columns: patch_path, label_0, label_1, ..., label_18
(19 binary columns, one per BigEarthNet-19 class)

Usage:
    python baselines/bigearth/prepare_bigearth.py \
        --data_dir /path/to/BigEarthNet-RGB \
        --out_dir  datasets/BigEarthNet-RGB_split

Official split files (ben-scene-transfer) are downloaded automatically.
If the server has no internet access, manually download and pass --split_dir.
"""

import os
import csv
import json
import random
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.bigearth_dataset import parse_labels, CLASSES_19, NUM_CLASSES

# Official split file URLs (BigEarthNet benchmark splits)
SPLIT_URLS = {
    "train": "https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/train.csv",
    "val":   "https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/val.csv",
    "test":  "https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/raw/master/splits/test.csv",
}


def download_split_lists(split_dir: str):
    """Download official patch-name split lists to split_dir."""
    os.makedirs(split_dir, exist_ok=True)
    for split, url in SPLIT_URLS.items():
        out_path = os.path.join(split_dir, f"{split}_names.csv")
        if os.path.exists(out_path):
            print(f"  [skip] {out_path} already exists")
            continue
        print(f"  Downloading {split} split list...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"  Saved -> {out_path}")
        except Exception as e:
            print(f"  [ERROR] Could not download {split} split: {e}")
            print(f"  Please manually download from:\n    {url}\n  and save to {out_path}")
            raise


def load_patch_names(split_csv: str) -> list:
    """Read patch names from an official split CSV (one name per row, first column)."""
    names = []
    with open(split_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                names.append(row[0].strip())
    return names


def make_random_split(all_patches: list, train=0.7, val=0.15, seed=42):
    """Fallback: random split when official split files are unavailable."""
    random.seed(seed)
    shuffled = all_patches[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train)
    n_val   = int(n * val)
    return (
        shuffled[:n_train],
        shuffled[n_train:n_train + n_val],
        shuffled[n_train + n_val:],
    )


def build_split_csv(patch_names: list, data_dir: str, out_csv: str):
    """
    For each patch name, read its label JSON and write a row to out_csv.
    Skips patches whose directory or JSON is missing.
    """
    header = ["patch_path"] + [f"label_{i}" for i in range(NUM_CLASSES)]
    skipped = 0

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for name in tqdm(patch_names, desc=os.path.basename(out_csv)):
            patch_dir  = os.path.join(data_dir, name)
            json_path  = os.path.join(patch_dir, f"{name}_labels_metadata.json")

            # Skip if directory, JSON, or RGB band TIFs are missing
            b04_path = os.path.join(patch_dir, f"{name}_B04.tif")
            if not os.path.isdir(patch_dir) or not os.path.exists(json_path) \
                    or not os.path.exists(b04_path):
                skipped += 1
                continue

            label_vec = parse_labels(json_path)
            row = [patch_dir] + label_vec.tolist()
            writer.writerow(row)

    print(f"  Written: {out_csv}  (skipped {skipped} missing patches)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",  type=str, required=True,
                        help="Root directory of BigEarthNet-RGB (contains patch folders)")
    parser.add_argument("--out_dir",   type=str,
                        default="datasets/BigEarthNet-RGB_split",
                        help="Output directory for split CSVs")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Directory with pre-downloaded official split CSVs "
                             "(optional; auto-downloaded if not provided)")
    parser.add_argument("--no_official", action="store_true",
                        help="Skip official splits, use random 70/15/15 split instead")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.no_official:
        print("Using random 70/15/15 split...")
        all_patches = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])
        train_names, val_names, test_names = make_random_split(all_patches)
        splits = {"train": train_names, "val": val_names, "test": test_names}
    else:
        split_dir = args.split_dir or os.path.join(args.out_dir, "official_splits")
        print("Downloading official BigEarthNet split lists...")
        download_split_lists(split_dir)
        splits = {
            s: load_patch_names(os.path.join(split_dir, f"{s}_names.csv"))
            for s in ("train", "val", "test")
        }

    print(f"\nSplit sizes: train={len(splits['train'])} | val={len(splits['val'])} | test={len(splits['test'])}")
    print(f"Building split CSVs in: {args.out_dir}\n")

    for split_name, patch_names in splits.items():
        out_csv = os.path.join(args.out_dir, f"{split_name}.csv")
        build_split_csv(patch_names, args.data_dir, out_csv)

    print("\nDone. Class order:")
    from models.bigearth_dataset import CLASSES_19
    for i, c in enumerate(CLASSES_19):
        print(f"  label_{i}: {c}")


if __name__ == "__main__":
    main()