import csv
import tempfile
import unittest
from pathlib import Path

from baselines.bigearth.prepare_bigearth import (
    CLASSES_19,
    CLASS_TO_IDX,
    build_v2_split_csvs_from_records,
    label_list_to_vec,
    resolve_patch_dir,
)


class PrepareBigEarthV2Test(unittest.TestCase):
    def test_label_list_to_vec_uses_bigearthnet_19_order(self):
        vec = label_list_to_vec(["Pastures", "Arable land"])

        self.assertEqual(float(vec[CLASS_TO_IDX["Pastures"]]), 1.0)
        self.assertEqual(float(vec[CLASS_TO_IDX["Arable land"]]), 1.0)
        self.assertEqual(sum(vec), 2.0)

    def test_resolve_patch_dir_supports_v2_nested_tile_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            patch_id = "S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB_84_05"
            tile_id = "S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB"
            patch_dir = data_dir / tile_id / patch_id
            patch_dir.mkdir(parents=True)

            self.assertEqual(resolve_patch_dir(str(data_dir), patch_id), str(patch_dir))

    def test_build_v2_split_csvs_from_records_writes_existing_rgb_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "BigEarthNet-S2"
            out_dir = root / "split"
            patch_id = "S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB_84_05"
            tile_id = "S2A_MSIL2A_20171002T112111_N9999_R037_T29SNB"
            patch_dir = data_dir / tile_id / patch_id
            patch_dir.mkdir(parents=True)
            for band in ("B02", "B03", "B04"):
                (patch_dir / f"{patch_id}_{band}.tif").write_bytes(b"")

            records = [
                {"patch_id": patch_id, "labels": ["Pastures"], "split": "train"},
                {"patch_id": "missing_patch_00_00", "labels": ["Arable land"], "split": "test"},
            ]

            counts = build_v2_split_csvs_from_records(records, str(data_dir), str(out_dir))

            self.assertEqual(counts["train"], 1)
            self.assertEqual(counts["test"], 0)
            with open(out_dir / "train.csv") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["patch_path"], str(patch_dir))
            self.assertEqual(float(rows[0][f"label_{CLASS_TO_IDX['Pastures']}"]), 1.0)


if __name__ == "__main__":
    unittest.main()
