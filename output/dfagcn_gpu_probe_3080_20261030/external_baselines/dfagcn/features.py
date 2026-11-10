"""Frozen VGG16 feature interfaces for the DFAGCN independent implementation.

The caller supplies the backbone and already preprocessed inputs. This module
does not download weights, normalize pixels, or claim torchvision/MatConvNet
numerical equivalence. It exports raw convolution maps; graph pooling remains
unspecified until the independent full-model recipe is fixed.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


_CONVOLUTIONS = {
    0: (3, 64), 2: (64, 64), 5: (64, 128), 7: (128, 128),
    10: (128, 256), 12: (256, 256), 14: (256, 256),
    17: (256, 512), 19: (512, 512), 21: (512, 512),
    24: (512, 512), 26: (512, 512), 28: (512, 512),
}
_POOLS = {4, 9, 16, 23, 30}
_CAPTURES = {14: "conv3", 21: "conv4", 28: "conv5"}


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def _validate_backbone(backbone: nn.Module) -> None:
    if not isinstance(backbone, nn.Module):
        raise TypeError("backbone must be a torch.nn.Module")
    features = getattr(backbone, "features", None)
    if not isinstance(features, nn.Sequential) or len(features) != 31:
        raise ValueError("backbone.features must have the 31-layer VGG16-D layout")
    for index, layer in enumerate(features):
        if index in _CONVOLUTIONS:
            incoming, outgoing = _CONVOLUTIONS[index]
            if not isinstance(layer, nn.Conv2d):
                raise TypeError(f"backbone.features[{index}] must be Conv2d")
            valid = (
                layer.in_channels == incoming and layer.out_channels == outgoing
                and _pair(layer.kernel_size) == (3, 3)
                and _pair(layer.stride) == (1, 1)
                and _pair(layer.padding) == (1, 1)
                and _pair(layer.dilation) == (1, 1)
                and layer.groups == 1 and layer.padding_mode == "zeros"
                and tuple(layer.weight.shape) == (outgoing, incoming, 3, 3)
                and layer.bias is not None and tuple(layer.bias.shape) == (outgoing,)
            )
            if not valid:
                raise ValueError(f"backbone.features[{index}] is incompatible with VGG16-D")
        elif index in _POOLS:
            if not isinstance(layer, nn.MaxPool2d):
                raise TypeError(f"backbone.features[{index}] must be MaxPool2d")
            valid = (
                _pair(layer.kernel_size) == (2, 2)
                and _pair(layer.stride if layer.stride is not None else layer.kernel_size) == (2, 2)
                and _pair(layer.padding) == (0, 0)
                and _pair(layer.dilation) == (1, 1)
                and not layer.ceil_mode and not layer.return_indices
            )
            if not valid:
                raise ValueError(f"backbone.features[{index}] has incompatible pooling")
        elif not isinstance(layer, nn.ReLU):
            raise TypeError(f"backbone.features[{index}] must be ReLU")

    avgpool = getattr(backbone, "avgpool", None)
    if not isinstance(avgpool, nn.AdaptiveAvgPool2d) or _pair(avgpool.output_size) != (7, 7):
        raise ValueError("backbone.avgpool must be AdaptiveAvgPool2d((7, 7))")
    classifier = getattr(backbone, "classifier", None)
    if not isinstance(classifier, nn.Sequential) or len(classifier) < 2:
        raise ValueError("backbone.classifier must contain FC6 and its ReLU")
    fc = classifier[0]
    if not isinstance(fc, nn.Linear):
        raise TypeError("backbone.classifier[0] must be Linear")
    if (
        fc.in_features != 25088 or fc.out_features != 4096
        or tuple(fc.weight.shape) != (4096, 25088)
        or fc.bias is None or tuple(fc.bias.shape) != (4096,)
    ):
        raise ValueError("backbone.classifier[0] must implement 25088 -> 4096 FC6")
    if not isinstance(classifier[1], nn.ReLU):
        raise TypeError("backbone.classifier[1] must be the FC6 ReLU")


class VGG16FeatureAdapter(nn.Module):
    """Extract FC6 post-ReLU and Conv3-3/4-3/5-3 pre-ReLU features.

    Extraction always uses eval/no_grad. Parameter requires_grad flags are
    preserved so the caller retains ownership of its backbone. Meta tensors
    support shape-only checks; finite pixel checks require a numerical device.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        _validate_backbone(backbone)
        self.backbone = backbone
        self.train(False)

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode must be a bool")
        return super().train(False)

    def _validate_input(self, images: Tensor, size: int, *, include_fc: bool = False) -> None:
        if not isinstance(images, Tensor):
            raise TypeError("images must be a Tensor")
        if images.ndim != 4:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if images.shape[0] < 1:
            raise ValueError("images batch must be nonempty")
        if images.shape[1] != 3:
            raise ValueError("images must have 3 RGB channels")
        if tuple(images.shape[-2:]) != (size, size):
            raise ValueError(f"images must have spatial size {size} x {size}")
        if not images.is_floating_point():
            raise TypeError("images must have a real floating dtype")
        modules = [self.backbone.features]
        if include_fc:
            modules.append(self.backbone.classifier[0])
        for module in modules:
            for parameter in module.parameters():
                if images.device != parameter.device:
                    raise ValueError("images and branch parameters must share a device")
                if images.dtype != parameter.dtype:
                    raise ValueError("images and branch parameters must share a dtype")
        if images.device.type != "meta" and not torch.isfinite(images).all().item():
            raise ValueError("images must contain only finite pixel values")

    @torch.no_grad()
    def convolutional_features(self, images448: Tensor) -> dict[str, Tensor]:
        self.backbone.eval()
        self._validate_input(images448, 448)
        outputs = {}
        hidden = images448
        for index, layer in enumerate(self.backbone.features):
            hidden = layer(hidden)
            if index in _CAPTURES:
                # VGG's next in-place ReLU otherwise changes the captured map.
                outputs[_CAPTURES[index]] = hidden.clone()
            if index == 28:
                break
        return outputs

    @torch.no_grad()
    def fc_features(self, images224: Tensor) -> Tensor:
        self.backbone.eval()
        self._validate_input(images224, 224, include_fc=True)
        hidden = self.backbone.features(images224)
        hidden = self.backbone.avgpool(hidden).flatten(1)
        hidden = self.backbone.classifier[0](hidden)
        return self.backbone.classifier[1](hidden)

    @torch.no_grad()
    def forward(self, images224: Tensor, images448: Tensor) -> dict[str, Tensor]:
        self._validate_input(images224, 224, include_fc=True)
        self._validate_input(images448, 448)
        if images224.shape[0] != images448.shape[0]:
            raise ValueError("paired input batch sizes must match")
        if images224.device != images448.device:
            raise ValueError("paired inputs must share a device")
        if images224.dtype != images448.dtype:
            raise ValueError("paired inputs must share a dtype")
        return {"fc": self.fc_features(images224), **self.convolutional_features(images448)}
