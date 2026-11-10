"""Three independently supervised graph branches for an explicit DFAGCN recipe.

Widths, pooling and dropout come from a declared independent recipe.  This
module does not claim to recover the paper's unreadable Table I architecture.
Inputs are already pooled frozen VGG features, not raw images.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from .config import validate_recipe
from .fusion import weighted_concat
from .graph import GCNLayer, knn_union_adjacency, mean_readout, normalize_adjacency


_INPUT_WIDTHS = {"conv3": 256, "conv4": 512, "conv5": 512}


class _GraphBranch(nn.Module):
    def __init__(self, input_width: int, recipe: dict, num_classes: int):
        super().__init__()
        self.k, self.metric = recipe["knn_k"], recipe["metric"]
        widths = [input_width, *recipe["gcn_dims"]]
        self.gcn = nn.ModuleList(GCNLayer(left, right, bias=False)
                                 for left, right in zip(widths, widths[1:]))
        self.dropout = nn.Dropout(recipe["dropout"])
        layers = []
        width = widths[-1]
        for index, next_width in enumerate(recipe["fc_dims"]):
            layers.extend((nn.Linear(width, next_width), nn.ReLU()))
            if index < len(recipe["fc_dims"]) - 1:
                layers.append(nn.Dropout(recipe["dropout"]))
            width = next_width
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(width, num_classes)

    def encode(self, nodes: Tensor) -> Tensor:
        adjacency = knn_union_adjacency(nodes, k=self.k, metric=self.metric)
        normalized = normalize_adjacency(adjacency, dtype=nodes.dtype)
        hidden = nodes
        for layer in self.gcn:
            hidden = self.dropout(torch.relu(layer(hidden, normalized)))
        return self.mlp(mean_readout(hidden))

    def forward(self, nodes: Tensor) -> Tensor:
        # CrossEntropyLoss supplies the log-softmax during training.
        return self.head(self.encode(nodes))


class DFAGCNBranches(nn.Module):
    """Three disjoint GCN/MLP/softmax-training heads, exporting 512 each.

    Forward returns raw logits for each branch; summing the three CE losses
    trains disjoint parameter sets.  Evaluation/export use caller-selected
    train/eval mode.  Input pooling is intentionally outside this module.
    """

    def __init__(self, recipe: dict, num_classes: int):
        super().__init__()
        self.recipe = validate_recipe(recipe)
        if type(num_classes) is not int or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2")
        self.num_classes = num_classes
        self.node_count = self.recipe["pooling_hw"][0] * self.recipe["pooling_hw"][1]
        self.branches = nn.ModuleDict({
            name: _GraphBranch(width, self.recipe, num_classes)
            for name, width in _INPUT_WIDTHS.items()
        })

    def _validate_features(self, features: Mapping, *, require_fc: bool = False) -> None:
        if not isinstance(features, Mapping):
            raise TypeError("features must map branch names to tensors")
        required = set(_INPUT_WIDTHS) | ({"fc"} if require_fc else set())
        if not required <= set(features) or set(features) - set(_INPUT_WIDTHS) - {"fc"}:
            raise ValueError("features must contain conv3/conv4/conv5 and optionally fc")
        parameter = next(self.parameters())
        batch = None
        for name, value in features.items():
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{name} must be a real floating Tensor")
            trailing = (4096,) if name == "fc" else (self.node_count, _INPUT_WIDTHS[name])
            if value.ndim != len(trailing) + 1 or tuple(value.shape[1:]) != trailing or value.shape[0] < 1:
                raise ValueError(f"{name} must have nonempty shape B x {trailing}")
            if value.device != parameter.device or value.dtype != parameter.dtype:
                raise ValueError("features and model must share device and dtype")
            if batch is None:
                batch = value.shape[0]
            if value.shape[0] != batch:
                raise ValueError("feature branches must share batch size")
            if not torch.isfinite(value).all():
                raise ValueError("feature branches must be finite")

    def forward(self, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self._validate_features(features)
        return {name: branch(features[name]) for name, branch in self.branches.items()}

    def export_features(self, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        self._validate_features(features, require_fc=True)
        return {**{name: branch.encode(features[name]) for name, branch in self.branches.items()},
                "fc": features["fc"]}

    def fused_features(self, features: Mapping[str, Tensor]) -> Tensor:
        exported = self.export_features(features)
        return weighted_concat(exported["conv5"], exported["conv4"], exported["conv3"],
                               exported["fc"], **self.recipe["fusion"])
