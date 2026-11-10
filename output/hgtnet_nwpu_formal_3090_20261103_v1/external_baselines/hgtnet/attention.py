"""Visual modules for the project's independent HGTNet candidate.

This is an engineering implementation of the documented local candidate,
not copied author code or a claim to reproduce an unverified original figure.
Swin attention and patch merging are reused directly from torchvision.
Public module inputs and outputs use BCHW; torchvision's BHWC layout stays
inside :class:`TransformerStage`.
"""

from functools import partial
import math
from numbers import Real

import torch
from torch import Tensor, nn
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_real(name: str, value: float, *, probability: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if probability:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between zero and one")
    elif value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_bchw(inputs: Tensor, channels: int, reference: Tensor) -> None:
    if not isinstance(inputs, Tensor):
        raise TypeError("inputs must be a torch.Tensor")
    if inputs.ndim != 4:
        raise ValueError("inputs must have BCHW layout and rank four")
    if any(size <= 0 for size in inputs.shape):
        raise ValueError("all BCHW dimensions must be nonempty")
    if inputs.shape[1] != channels:
        raise ValueError(f"expected {channels} channels, received {inputs.shape[1]}")
    if not inputs.is_floating_point():
        raise TypeError("inputs must use a floating point dtype")
    if inputs.dtype != reference.dtype:
        raise TypeError(f"input dtype {inputs.dtype} differs from module dtype {reference.dtype}")
    if inputs.device != reference.device:
        raise ValueError(f"input device {inputs.device} differs from module device {reference.device}")
    if not torch.isfinite(inputs).all().item():
        raise ValueError("inputs must contain only finite values")


class PatchEmbedding(nn.Module):
    """Nonoverlapping convolutional patches followed by per-patch channel LN."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        patch_size: int = 4,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int("in_channels", in_channels)
        self.embed_dim = _positive_int("embed_dim", embed_dim)
        self.patch_size = _positive_int("patch_size", patch_size)
        norm_eps = _finite_real("norm_eps", norm_eps)
        self.projection = nn.Conv2d(
            self.in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=norm_eps)

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_bchw(inputs, self.in_channels, self.projection.weight)
        if inputs.shape[2] % self.patch_size or inputs.shape[3] % self.patch_size:
            raise ValueError("input height and width must be divisible by patch_size")
        patches = self.projection(inputs).permute(0, 2, 3, 1)
        return self.norm(patches).permute(0, 3, 1, 2).contiguous()


class DualAttention(nn.Module):
    """Parallel channel and spatial gates, added without averaging or extra skip.

    Both global pooling descriptors pass through the very same MLP. The
    spatial gate has three linear dilated convolutions; its only activation
    is the final sigmoid. Thus zero gate parameters give exactly 0.5F+0.5F.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        reduction = _positive_int("reduction", reduction)
        hidden_channels = max(1, self.channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Linear(self.channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, self.channels),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, dilation=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=3, dilation=2, padding=2),
            nn.Conv2d(1, 1, kernel_size=3, dilation=3, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_bchw(inputs, self.channels, self.channel_mlp[0].weight)
        channel_avg = inputs.mean(dim=(2, 3))
        channel_max = inputs.amax(dim=(2, 3))
        channel_gate = torch.sigmoid(self.channel_mlp(channel_avg) + self.channel_mlp(channel_max))
        channel_gate = channel_gate[:, :, None, None]
        spatial_descriptors = torch.cat(
            (inputs.mean(dim=1, keepdim=True), inputs.amax(dim=1, keepdim=True)),
            dim=1,
        )
        spatial_gate = self.spatial(spatial_descriptors)
        return inputs * channel_gate + inputs * spatial_gate


class _Conv2SameChannels(nn.Module):
    """Candidate third-stage downsample; the internal boundary uses BHWC."""

    def __init__(self, channels: int, norm_eps: float) -> None:
        super().__init__()
        self.projection = nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False)
        self.norm = nn.LayerNorm(channels, eps=norm_eps)

    def forward(self, inputs: Tensor) -> Tensor:
        projected = self.projection(inputs.permute(0, 3, 1, 2))
        return self.norm(projected.permute(0, 2, 3, 1))


class TransformerStage(nn.Module):
    """DA, ``stbs`` pairs of real Swin blocks, then the specified downsample.

    The first two stages use torchvision's named PatchMerging unchanged for
    candidate pretraining compatibility. The same-channel Conv2 in the third
    stage is an explicit independent architecture choice.
    """

    def __init__(
        self,
        channels: int,
        stbs: int,
        heads: int,
        *,
        output_channels: int,
        downsample_kind: str,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.channels = _positive_int("channels", channels)
        stbs = _positive_int("stbs", stbs)
        heads = _positive_int("heads", heads)
        self.output_channels = _positive_int("output_channels", output_channels)
        window_size = _positive_int("window_size", window_size)
        mlp_ratio = _finite_real("mlp_ratio", mlp_ratio)
        dropout = _finite_real("dropout", dropout, probability=True)
        attention_dropout = _finite_real("attention_dropout", attention_dropout, probability=True)
        stochastic_depth_prob = _finite_real("stochastic_depth_prob", stochastic_depth_prob, probability=True)
        norm_eps = _finite_real("norm_eps", norm_eps)
        if self.channels % heads:
            raise ValueError("channels must be divisible by heads")
        if int(self.channels * mlp_ratio) < 1:
            raise ValueError("mlp_ratio must produce at least one hidden channel")
        if downsample_kind not in ("patch_merge", "conv2_same_channels"):
            raise ValueError("downsample_kind must be patch_merge or conv2_same_channels")
        expected_channels = 2 * self.channels if downsample_kind == "patch_merge" else self.channels
        if self.output_channels != expected_channels:
            raise ValueError(f"{downsample_kind} requires output_channels={expected_channels}")
        self.downsample_kind = downsample_kind
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.da = DualAttention(self.channels)
        self.blocks = nn.Sequential(
            *[
                SwinTransformerBlock(
                    dim=self.channels,
                    num_heads=heads,
                    window_size=[window_size, window_size],
                    shift_size=[0, 0] if block_index % 2 == 0 else [window_size // 2, window_size // 2],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    stochastic_depth_prob=stochastic_depth_prob,
                    norm_layer=norm_layer,
                )
                for block_index in range(2 * stbs)
            ]
        )
        self.downsample = (
            PatchMerging(self.channels, norm_layer=norm_layer)
            if downsample_kind == "patch_merge"
            else _Conv2SameChannels(self.channels, norm_eps)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        _validate_bchw(inputs, self.channels, self.da.channel_mlp[0].weight)
        if self.downsample_kind == "conv2_same_channels" and min(inputs.shape[2:]) < 2:
            raise ValueError("conv2_same_channels requires height and width of at least two")
        features = self.da(inputs).permute(0, 2, 3, 1)
        features = self.blocks(features)
        return self.downsample(features).permute(0, 3, 1, 2).contiguous()
