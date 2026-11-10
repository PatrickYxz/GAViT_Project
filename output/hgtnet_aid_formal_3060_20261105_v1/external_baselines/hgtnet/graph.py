"""Graph operators for the explicitly independent HGTNet candidate.

Node features use B x N x C.  Graph construction is detached, Euclidean and
deterministic for equal distances.  The relative-mean operator is an isolated
ambiguity probe; the candidate model uses the max-relative operator.
"""

import math
from numbers import Real

import torch
from torch import nn


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _normalization_epsilon(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("norm_eps must be a real number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("norm_eps must be finite and positive")
    return float(value)


def _validate_features(features, *, ndim, channels=None, parameter=None):
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a torch.Tensor")
    if features.layout != torch.strided:
        raise TypeError("features must have a strided layout")
    if features.ndim != ndim or any(size == 0 for size in features.shape):
        raise ValueError(f"features must have {ndim} nonempty dimensions")
    if not features.is_floating_point():
        raise TypeError("features must have a real floating dtype")
    channel_axis = -1 if ndim == 3 else 1
    if channels is not None and features.shape[channel_axis] != channels:
        raise ValueError(f"features must have {channels} channels")
    if parameter is not None:
        if features.dtype != parameter.dtype:
            raise TypeError("feature dtype must match module parameters")
        if features.device != parameter.device:
            raise ValueError("feature device must match module parameters")
    if not torch.isfinite(features).all().item():
        raise ValueError("features must be finite")


def _validate_reduction(reduction):
    if reduction not in ("max", "mean"):
        raise ValueError("reduction must be 'max' or 'mean'")
    return reduction


@torch.no_grad()
def knn_indices(nodes, k, exclude_self=True):
    """Return B x N x k indices, sorted by distance and then node index.

    ``exclude_self=True`` requires N >= k + 1.  Explicitly retaining self
    permits k == N; duplicate features still follow the index tie policy.
    Distances are computed in float64 to avoid overflow for finite fp32
    features.  No silent clipping, random edge selection or batch mixing occurs.
    """
    _validate_features(nodes, ndim=3)
    _positive_integer(k, "k")
    if not isinstance(exclude_self, bool):
        raise TypeError("exclude_self must be a boolean")
    count = nodes.shape[1]
    available = count - int(exclude_self)
    if k > available:
        raise ValueError(f"k={k} requires at least {k + int(exclude_self)} nodes")
    precise = nodes.to(dtype=torch.float64)
    differences = precise.unsqueeze(2) - precise.unsqueeze(1)
    squared_distances = differences.square().sum(dim=-1)
    if not torch.isfinite(squared_distances).all().item():
        raise ValueError("Euclidean distances overflow float64")
    if exclude_self:
        diagonal = torch.eye(count, dtype=torch.bool, device=nodes.device).unsqueeze(0)
        squared_distances.masked_fill_(diagonal, float("inf"))
    return torch.argsort(squared_distances, dim=-1, stable=True)[..., :k]


def relative_aggregate(nodes, neighbours, reduction="max"):
    """Reduce neighbour-query messages independently for each batch/query."""
    _validate_features(nodes, ndim=3)
    _validate_reduction(reduction)
    if not isinstance(neighbours, torch.Tensor):
        raise TypeError("neighbours must be a torch.Tensor")
    if neighbours.layout != torch.strided or neighbours.dtype != torch.long:
        raise TypeError("neighbours must be a strided torch.long tensor")
    if neighbours.device != nodes.device:
        raise ValueError("neighbours and nodes must be on the same device")
    if neighbours.ndim != 3 or neighbours.shape[:2] != nodes.shape[:2] or neighbours.shape[-1] == 0:
        raise ValueError("neighbours must have nonempty shape B x N x k")
    if ((neighbours < 0) | (neighbours >= nodes.shape[1])).any().item():
        raise ValueError("neighbour indices must lie in [0, N)")
    batches = torch.arange(nodes.shape[0], device=nodes.device)[:, None, None]
    messages = nodes[batches, neighbours] - nodes.unsqueeze(2)
    if not torch.isfinite(messages).all().item():
        raise ValueError("relative messages must be finite")
    if reduction == "max":
        return messages.max(dim=2).values
    return messages.mean(dim=2)


class MRConv(nn.Module):
    """Concatenate each center with its reduced relative message, then project."""

    def __init__(self, in_channels, out_channels, reduction="max"):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        self.reduction = _validate_reduction(reduction)
        self.projection = nn.Linear(2 * in_channels, out_channels)

    def forward(self, nodes, neighbours):
        _validate_features(nodes, ndim=3, channels=self.in_channels, parameter=self.projection.weight)
        messages = relative_aggregate(nodes, neighbours, self.reduction)
        return self.projection(torch.cat((nodes, messages), dim=-1))


class GraphBlock(nn.Module):
    """Dynamic max-relative Grapher and FFN, each with an explicit residual."""

    def __init__(self, channels, k, operator="mr_max", ffn_ratio=4, norm_eps=1e-5, exclude_self=True):
        super().__init__()
        self.channels = _positive_integer(channels, "channels")
        self.k = _positive_integer(k, "k")
        _positive_integer(ffn_ratio, "ffn_ratio")
        norm_eps = _normalization_epsilon(norm_eps)
        if operator not in ("mr_max", "relative_mean"):
            raise ValueError("operator must be 'mr_max' or 'relative_mean'")
        if not isinstance(exclude_self, bool):
            raise TypeError("exclude_self must be a boolean")
        self.operator = operator
        self.exclude_self = exclude_self
        self.in_projection = nn.Linear(channels, channels)
        self.in_norm = nn.LayerNorm(channels, eps=norm_eps)
        reduction = "max" if operator == "mr_max" else "mean"
        self.relative_conv = MRConv(channels, 2 * channels, reduction=reduction)
        self.graph_activation = nn.GELU()
        self.out_projection = nn.Linear(2 * channels, channels)
        self.ffn_norm = nn.LayerNorm(channels, eps=norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_ratio * channels),
            nn.GELU(),
            nn.Linear(ffn_ratio * channels, channels),
        )

    def forward(self, nodes, *, return_neighbors=False):
        _validate_features(nodes, ndim=3, channels=self.channels, parameter=self.in_projection.weight)
        if not isinstance(return_neighbors, bool):
            raise TypeError("return_neighbors must be a boolean")
        projected = self.in_norm(self.in_projection(nodes))
        neighbours = knn_indices(projected, self.k, exclude_self=self.exclude_self)
        graph_features = self.relative_conv(projected, neighbours)
        output = nodes + self.out_projection(self.graph_activation(graph_features))
        output = output + self.ffn(self.ffn_norm(output))
        if return_neighbors:
            return output, neighbours
        return output


class DepthwiseProjection(nn.Module):
    """DWConv3 + pointwise Conv1 + per-spatial-node LayerNorm (BCHW)."""

    def __init__(self, in_channels, out_channels, *, stride=1, padding=1, norm_eps=1e-5):
        super().__init__()
        self.in_channels = _positive_integer(in_channels, "in_channels")
        self.out_channels = _positive_integer(out_channels, "out_channels")
        _positive_integer(stride, "stride")
        if isinstance(padding, bool) or not isinstance(padding, int):
            raise TypeError("padding must be an integer")
        if padding < 0:
            raise ValueError("padding must be nonnegative")
        norm_eps = _normalization_epsilon(norm_eps)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(out_channels, eps=norm_eps)

    def forward(self, image):
        _validate_features(image, ndim=4, channels=self.in_channels, parameter=self.depthwise.weight)
        padding = self.depthwise.padding[0]
        if image.shape[-2] + 2 * padding < 3 or image.shape[-1] + 2 * padding < 3:
            raise ValueError("image is smaller than the padded 3x3 depthwise kernel")
        features = self.pointwise(self.depthwise(image))
        return self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
