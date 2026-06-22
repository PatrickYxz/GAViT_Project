"""
prepare_bigearth.py — Prepare BigEarthNet splits.

Supports two metadata layouts:
  1. BigEarthNet v1.x patch folders with *_labels_metadata.json files.
  2. BigEarthNet v2.0 nested tile/patch folders plus metadata.parquet.

Generates three CSV files:
    datasets/BigEarthNet-RGB_split/train.csv
    datasets/BigEarthNet-RGB_split/val.csv
    datasets/BigEarthNet-RGB_split/test.csv

Each CSV has columns: patch_path, label_0, label_1, ..., label_18
(19 binary columns, one per BigEarthNet-19 class)

Usage:
    python baselines/bigearth/prepare_bigearth.py \
        --data_dir /path/to/BigEarthNet-RGB \
        --out_dir  datasets/BigEarthNet-RGB_split

    python baselines/bigearth/prepare_bigearth.py \
        --data_dir /path/to/BigEarthNet-S2 \
        --metadata_parquet /path/to/metadata.parquet \
        --out_dir datasets/BigEarthNet-RGB_split

Official split files (ben-scene-transfer) are downloaded automatically.
If the server has no internet access, manually download and pass --split_dir.
"""

import os
import csv
import ast
import random
import argparse
import urllib.request
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable

# BigEarthNet v2.0 metadata.parquet already uses the 19-class nomenclature.
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
NUM_CLASSES = len(CLASSES_19)

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


def label_list_to_vec(labels) -> list:
    """Convert BigEarthNet 19-class label names to a binary list."""
    if labels is None:
        labels = []
    elif isinstance(labels, str):
        try:
            labels = ast.literal_eval(labels)
        except (SyntaxError, ValueError):
            labels = [labels]

    label_vec = [0.0] * NUM_CLASSES
    for label in labels:
        if label in CLASS_TO_IDX:
            label_vec[CLASS_TO_IDX[label]] = 1.0
    return label_vec


def patch_tile_id(patch_id: str) -> str:
    """Return the parent Sentinel-2 tile id from a v2.0 patch id."""
    return patch_id.rsplit("_", 2)[0]


def resolve_patch_dir(data_dir: str, patch_id: str) -> str | None:
    """Resolve either direct v1 layout or nested v2 tile/patch layout."""
    direct = os.path.join(data_dir, patch_id)
    if os.path.isdir(direct):
        return direct

    nested = os.path.join(data_dir, patch_tile_id(patch_id), patch_id)
    if os.path.isdir(nested):
        return nested

    return None


def has_rgb_bands(patch_dir: str, patch_id: str) -> bool:
    return all(
        os.path.exists(os.path.join(patch_dir, f"{patch_id}_{band}.tif"))
        for band in ("B02", "B03", "B04")
    )


def split_name_from_metadata(value: str) -> str | None:
    value = str(value).strip().lower()
    if value in ("train", "training"):
        return "train"
    if value in ("val", "valid", "validation"):
        return "val"
    if value == "test":
        return "test"
    return None


def write_split_csv(out_csv: str, rows: list[list]):
    header = ["patch_path"] + [f"label_{i}" for i in range(NUM_CLASSES)]
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


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


def build_v2_split_csvs_from_records(records, data_dir: str, out_dir: str) -> dict:
    """Build split CSVs from BigEarthNet v2.0 metadata records."""
    os.makedirs(out_dir, exist_ok=True)
    rows_by_split = {"train": [], "val": [], "test": []}
    skipped = {"missing_patch": 0, "missing_bands": 0, "unknown_split": 0}

    for record in tqdm(records, desc="metadata.parquet"):
        patch_id = str(record["patch_id"])
        split_name = split_name_from_metadata(record["split"])
        if split_name is None:
            skipped["unknown_split"] += 1
            continue

        patch_dir = resolve_patch_dir(data_dir, patch_id)
        if patch_dir is None:
            skipped["missing_patch"] += 1
            continue
        if not has_rgb_bands(patch_dir, patch_id):
            skipped["missing_bands"] += 1
            continue

        rows_by_split[split_name].append([patch_dir] + label_list_to_vec(record["labels"]))

    counts = {}
    for split_name, rows in rows_by_split.items():
        out_csv = os.path.join(out_dir, f"{split_name}.csv")
        write_split_csv(out_csv, rows)
        counts[split_name] = len(rows)
        print(f"  Written: {out_csv}  ({len(rows)} rows)")

    print("  Skipped:", skipped)
    return counts


def build_v2_split_csvs(metadata_parquet: str, data_dir: str, out_dir: str) -> dict:
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "Reading metadata.parquet requires pandas and a parquet engine. "
            "Install them with: pip install --user pandas pyarrow"
        ) from e

    df = pd.read_parquet(metadata_parquet)
    required = {"patch_id", "labels", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{metadata_parquet} is missing required columns: {sorted(missing)}")

    return build_v2_split_csvs_from_records(
        df[["patch_id", "labels", "split"]].to_dict("records"),
        data_dir,
        out_dir,
    )


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
    parser.add_argument("--metadata_parquet", type=str, default=None,
                        help="BigEarthNet v2.0 metadata.parquet. When set, this "
                             "replaces v1 JSON/split-list preparation.")
    parser.add_argument("--no_official", action="store_true",
                        help="Skip official splits, use random 70/15/15 split instead")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.metadata_parquet:
        print("Using BigEarthNet v2.0 metadata.parquet...")
        build_v2_split_csvs(args.metadata_parquet, args.data_dir, args.out_dir)
        print("\nDone. Class order:")
        for i, c in enumerate(CLASSES_19):
            print(f"  label_{i}: {c}")
        return

    from models.bigearth_dataset import parse_labels

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
    for i, c in enumerate(CLASSES_19):
        print(f"  label_{i}: {c}")


if __name__ == "__main__":
    main()
