"""Real VGG extraction contracts; no MatConvNet numerical-equivalence claim.

Full untrained torchvision VGG16 runs on meta for shape checks. Real CPU
convolutions check pre-ReLU values; FC6 stays on meta to avoid allocating its
102,760,448 weights. No model weights are downloaded by these tests.
"""

import importlib
import io
import unittest

import torch
from torch import nn
from torchvision.models import vgg16


def adapter_from(backbone):
    try:
        module = importlib.import_module("external_baselines.dfagcn.features")
    except ModuleNotFoundError as exc:
        raise AssertionError("DFAGCN feature adapter has not been implemented") from exc
    return module.VGG16FeatureAdapter(backbone)


def meta_vgg16():
    with torch.device("meta"):
        return vgg16(weights=None)


class FeatureAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(cls.previous_threads, 2))
        cls.cpu_adapter = None

    @classmethod
    def tearDownClass(cls):
        cls.cpu_adapter = None
        torch.set_num_threads(cls.previous_threads)

    def meta_adapter(self):
        return adapter_from(meta_vgg16())

    def controlled_cpu_adapter(self):
        if self.__class__.cpu_adapter is None:
            backbone = meta_vgg16()
            backbone.features.to_empty(device="cpu")
            with torch.no_grad():
                for layer in backbone.features:
                    if isinstance(layer, nn.Conv2d):
                        layer.weight.zero_()
                        layer.bias.fill_(1.0)
                backbone.features[14].bias.fill_(-3.0)
                backbone.features[21].bias.fill_(-4.0)
                backbone.features[28].bias.fill_(-5.0)
            self.__class__.cpu_adapter = adapter_from(backbone)
        return self.__class__.cpu_adapter

    def test_complete_torchvision_vgg16_meta_forward_has_paper_layer_shapes(self):
        # Changing branches, layer locations or flattening breaks these shapes.
        outputs = self.meta_adapter()(
            torch.empty(2, 3, 224, 224, device="meta"),
            torch.empty(2, 3, 448, 448, device="meta"),
        )
        self.assertEqual(set(outputs), {"fc", "conv3", "conv4", "conv5"})
        self.assertEqual(outputs["fc"].shape, (2, 4096))
        self.assertEqual(outputs["conv3"].shape, (2, 256, 112, 112))
        self.assertEqual(outputs["conv4"].shape, (2, 512, 56, 56))
        self.assertEqual(outputs["conv5"].shape, (2, 512, 28, 28))
        self.assertTrue(all(value.device.type == "meta" for value in outputs.values()))
        self.assertTrue(all(not value.requires_grad for value in outputs.values()))

    def test_real_cpu_convolutions_preserve_negative_pre_relu_values(self):
        # Missing capture.clone() lets VGG's in-place ReLU erase negatives.
        outputs = self.controlled_cpu_adapter().convolutional_features(
            torch.zeros(1, 3, 448, 448, requires_grad=True)
        )
        self.assertEqual(set(outputs), {"conv3", "conv4", "conv5"})
        for name, expected, shape in (
            ("conv3", -3.0, (1, 256, 112, 112)),
            ("conv4", -4.0, (1, 512, 56, 56)),
            ("conv5", -5.0, (1, 512, 28, 28)),
        ):
            with self.subTest(branch=name):
                self.assertEqual(outputs[name].device.type, "cpu")
                self.assertEqual(outputs[name].shape, shape)
                self.assertTrue(torch.equal(outputs[name], torch.full(shape, expected)))
                self.assertFalse(outputs[name].requires_grad)

    def test_features_stay_in_eval_after_parent_or_backbone_train(self):
        adapter = self.controlled_cpu_adapter()
        adapter.train(True)
        self.assertFalse(adapter.training)
        self.assertFalse(any(layer.training for layer in adapter.backbone.modules()))
        adapter.backbone.train(True)
        adapter.convolutional_features(torch.zeros(1, 3, 448, 448))
        self.assertFalse(any(layer.training for layer in adapter.backbone.modules()))

    def test_fc_method_independently_runs_complete_meta_path_in_eval(self):
        adapter = self.meta_adapter()
        adapter.backbone.train(True)
        output = adapter.fc_features(torch.empty(1, 3, 224, 224, device="meta", requires_grad=True))
        self.assertEqual(output.shape, (1, 4096))
        self.assertFalse(output.requires_grad)
        self.assertFalse(any(layer.training for layer in adapter.backbone.modules()))

    def test_preserves_callers_parameter_gradient_flags(self):
        backbone = meta_vgg16()
        backbone.features[0].weight.requires_grad_(False)
        expected = [parameter.requires_grad for parameter in backbone.parameters()]
        adapter = adapter_from(backbone)
        adapter.fc_features(torch.empty(1, 3, 224, 224, device="meta"))
        self.assertEqual([parameter.requires_grad for parameter in backbone.parameters()], expected)

    def test_cpu_convolution_input_rejects_nonfinite_pixels(self):
        adapter = self.controlled_cpu_adapter()
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                inputs = torch.zeros(1, 3, 448, 448)
                inputs[0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite"):
                    adapter.convolutional_features(inputs)

    def test_cpu_convolution_state_roundtrip_preserves_real_outputs(self):
        # Covers convolution weights only, not FC or MatConvNet parity.
        adapter = self.controlled_cpu_adapter()
        inputs = torch.ones(1, 3, 448, 448)
        expected = adapter.convolutional_features(inputs)
        buffer = io.BytesIO()
        torch.save(adapter.backbone.features.state_dict(), buffer)
        reloaded = meta_vgg16()
        reloaded.features.to_empty(device="cpu")
        buffer.seek(0)
        reloaded.features.load_state_dict(torch.load(buffer, weights_only=True))
        actual = adapter_from(reloaded).convolutional_features(inputs)
        for name in ("conv3", "conv4", "conv5"):
            with self.subTest(branch=name):
                self.assertTrue(torch.equal(actual[name], expected[name]))

    def test_input_contract_rejects_invalid_branch_tensors(self):
        adapter = self.meta_adapter()
        for method, size in (("convolutional_features", 448), ("fc_features", 224)):
            for kind, error, match in (
                ("not_tensor", TypeError, "Tensor"),
                ("rank", ValueError, "shape"),
                ("empty_batch", ValueError, "batch"),
                ("channels", ValueError, "3"),
                ("size", ValueError, "224|448"),
                ("integer", TypeError, "floating"),
                ("complex", TypeError, "floating"),
                ("parameter_dtype", ValueError, "dtype"),
            ):
                with self.subTest(method=method, kind=kind):
                    shape, dtype = (1, 3, size, size), torch.float32
                    if kind == "not_tensor":
                        inputs = [0.0]
                    else:
                        if kind == "rank":
                            shape = (3, size, size)
                        elif kind == "empty_batch":
                            shape = (0, 3, size, size)
                        elif kind == "channels":
                            shape = (1, 4, size, size)
                        elif kind == "size":
                            shape = (1, 3, size - 1, size)
                        elif kind == "integer":
                            dtype = torch.int64
                        elif kind == "complex":
                            dtype = torch.complex64
                        elif kind == "parameter_dtype":
                            dtype = torch.float64
                        inputs = torch.empty(shape, dtype=dtype, device="meta")
                    with self.assertRaisesRegex(error, match):
                        getattr(adapter, method)(inputs)

    def test_forward_rejects_different_batch_sizes(self):
        adapter = self.meta_adapter()
        with self.assertRaisesRegex(ValueError, "batch"):
            adapter(torch.empty(1, 3, 224, 224, device="meta"), torch.empty(2, 3, 448, 448, device="meta"))

    def test_forward_rejects_different_input_devices(self):
        adapter = self.meta_adapter()
        with self.assertRaisesRegex(ValueError, "device"):
            adapter(torch.empty(1, 3, 224, 224, device="meta"), torch.empty(1, 3, 448, 448))

    def test_forward_rejects_different_input_dtypes(self):
        adapter = self.meta_adapter()
        with self.assertRaisesRegex(ValueError, "dtype"):
            adapter(torch.empty(1, 3, 224, 224, device="meta"), torch.empty(1, 3, 448, 448, dtype=torch.float64, device="meta"))

    def test_branch_rejects_input_device_different_from_parameters(self):
        adapter = self.meta_adapter()
        with self.assertRaisesRegex(ValueError, "device"):
            adapter.convolutional_features(torch.zeros(1, 3, 448, 448))

    def test_rejects_incompatible_backbone_layout_or_parameters(self):
        for change in (
            "not_module", "features_missing", "features_short", "conv_type", "conv_channels",
            "conv_kernel", "conv_stride", "conv_weight_shape", "pool_ceil", "avgpool_size",
            "fc_width", "fc_weight_shape", "fc_activation",
        ):
            with self.subTest(change=change):
                backbone = meta_vgg16()
                if change == "not_module":
                    backbone = object()
                elif change == "features_missing":
                    del backbone.features
                elif change == "features_short":
                    backbone.features = backbone.features[:29]
                elif change == "conv_type":
                    backbone.features[14] = nn.ReLU()
                elif change == "conv_channels":
                    backbone.features[14] = nn.Conv2d(256, 128, 3, padding=1, device="meta")
                elif change == "conv_kernel":
                    backbone.features[14] = nn.Conv2d(256, 256, 1, device="meta")
                elif change == "conv_stride":
                    backbone.features[14] = nn.Conv2d(256, 256, 3, stride=2, padding=1, device="meta")
                elif change == "conv_weight_shape":
                    backbone.features[14].weight = nn.Parameter(torch.empty(1, device="meta"))
                elif change == "pool_ceil":
                    backbone.features[4] = nn.MaxPool2d(2, 2, ceil_mode=True)
                elif change == "avgpool_size":
                    backbone.avgpool = nn.AdaptiveAvgPool2d((8, 8))
                elif change == "fc_width":
                    backbone.classifier[0] = nn.Linear(25088, 512, device="meta")
                elif change == "fc_weight_shape":
                    backbone.classifier[0].weight = nn.Parameter(torch.empty(1, device="meta"))
                elif change == "fc_activation":
                    backbone.classifier[1] = nn.Identity()
                with self.assertRaisesRegex((TypeError, ValueError), "backbone|features|avgpool|classifier"):
                    adapter_from(backbone)


if __name__ == "__main__":
    unittest.main()
