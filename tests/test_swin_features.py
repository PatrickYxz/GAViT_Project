import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from models.swin_features import build_swin_model_kwargs, pool_swin_features


class FakeFeatures:
    def __init__(self, ndim):
        self.ndim = ndim
        self.mean_dim = None

    def mean(self, dim):
        self.mean_dim = dim
        return self


class PoolSwinFeaturesTests(unittest.TestCase):
    def test_pools_both_spatial_dimensions_for_channels_last_features(self):
        features = FakeFeatures(ndim=4)

        pooled = pool_swin_features(features)

        self.assertIs(pooled, features)
        self.assertEqual(features.mean_dim, (1, 2))

    def test_pools_token_dimension_for_sequence_features(self):
        features = FakeFeatures(ndim=3)

        pool_swin_features(features)

        self.assertEqual(features.mean_dim, 1)

    def test_leaves_already_pooled_features_unchanged(self):
        features = FakeFeatures(ndim=2)

        pooled = pool_swin_features(features)

        self.assertIs(pooled, features)
        self.assertIsNone(features.mean_dim)


class BuildSwinModelKwargsTests(unittest.TestCase):
    def test_uses_local_file_as_pretrained_source(self):
        with TemporaryDirectory() as tmp_dir:
            weight_path = Path(tmp_dir) / "model.safetensors"
            weight_path.touch()

            kwargs = build_swin_model_kwargs(
                pretrained=True,
                pretrained_path=str(weight_path),
            )

        self.assertTrue(kwargs["pretrained"])
        self.assertEqual(kwargs["num_classes"], 0)
        self.assertEqual(
            kwargs["pretrained_cfg_overlay"],
            {"file": str(weight_path)},
        )

    def test_rejects_missing_local_pretrained_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "Swin pretrained weights"):
            build_swin_model_kwargs(
                pretrained=True,
                pretrained_path="/missing/model.safetensors",
            )

    def test_preserves_default_timm_configuration_without_local_file(self):
        kwargs = build_swin_model_kwargs(pretrained=False)

        self.assertEqual(kwargs, {"pretrained": False, "num_classes": 0})


if __name__ == "__main__":
    unittest.main()
