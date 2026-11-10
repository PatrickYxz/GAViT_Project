"""DFAGCN graph primitives, independently implemented from the audited equations.

These operators do not select a pooling size, GCN depth or activation sequence.
The paper's distance metric is unresolved, so callers must choose it explicitly.
Ties use ascending node index; self-neighbors are excluded before the kNN union.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _validate_nodes(nodes: Tensor) -> None:
    if not isinstance(nodes, Tensor):
        raise TypeError("nodes must be a Tensor")
    if not nodes.is_floating_point():
        raise TypeError("nodes must have a real floating dtype")
    if nodes.ndim != 3 or any(size == 0 for size in nodes.shape):
        raise ValueError("nodes must have nonempty shape (batch, nodes, features)")
    if not torch.isfinite(nodes).all():
        raise ValueError("nodes must be finite")


def knn_union_adjacency(nodes: Tensor, k: int = 10, *, metric: str) -> Tensor:
    """Return a detached binary undirected union, independently for each sample.

    Distance calculation promotes half/bfloat16 to float32 for CPU support;
    float64 inputs retain float64 distance precision. The output is boolean.
    """
    _validate_nodes(nodes)
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    if not 1 <= k < nodes.shape[1]:
        raise ValueError("k must satisfy 1 <= k < node count")
    if metric not in ("euclidean", "cosine"):
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    with torch.no_grad():
        work = nodes.detach()
        if work.dtype not in (torch.float32, torch.float64):
            work = work.float()
        if metric == "euclidean":
            distances = torch.cdist(work, work, p=2)
        else:
            norms = torch.linalg.vector_norm(work, dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("cosine distance requires finite nonzero node norms")
            unit = work / norms
            distances = 1 - torch.bmm(unit, unit.transpose(1, 2))
        if not torch.isfinite(distances).all():
            raise ValueError("distance calculation must remain finite")
        distances.diagonal(dim1=1, dim2=2).fill_(float("inf"))
        neighbors = distances.argsort(dim=-1, stable=True)[..., :k]
        directed = torch.zeros_like(distances, dtype=torch.bool)
        directed.scatter_(-1, neighbors, True)
        return directed | directed.transpose(1, 2)


def normalize_adjacency(adjacency: Tensor, *, dtype=None) -> Tensor:
    """Compute D^-1/2 (A + I) D^-1/2 with one loop at each node.

    Input A must be a symmetric binary batch, with optional existing loops.
    Existing diagonal ones are retained rather than incremented. Input storage
    is not changed. Integer/bool inputs default to float32 output.
    """
    if not isinstance(adjacency, Tensor):
        raise TypeError("adjacency must be a Tensor")
    if adjacency.is_complex():
        raise TypeError("adjacency must be real")
    if (
        adjacency.ndim != 3
        or adjacency.shape[1] != adjacency.shape[2]
        or any(size == 0 for size in adjacency.shape)
    ):
        raise ValueError("adjacency must have nonempty shape (batch, nodes, nodes)")
    if not torch.isfinite(adjacency).all():
        raise ValueError("adjacency must be finite")
    if not ((adjacency == 0) | (adjacency == 1)).all():
        raise ValueError("adjacency must be binary")
    if not torch.equal(adjacency, adjacency.transpose(1, 2)):
        raise ValueError("adjacency must be symmetric")
    output_dtype = dtype
    if output_dtype is None:
        output_dtype = adjacency.dtype if adjacency.is_floating_point() else torch.float32
    if not isinstance(output_dtype, torch.dtype) or not output_dtype.is_floating_point:
        raise TypeError("normalization output dtype must be real floating")

    with_loops = adjacency.to(dtype=output_dtype).clone()
    with_loops.diagonal(dim1=1, dim2=2).fill_(1)
    inverse_sqrt_degree = with_loops.sum(dim=-1).rsqrt()
    return (
        inverse_sqrt_degree.unsqueeze(-1)
        * with_loops
        * inverse_sqrt_degree.unsqueeze(-2)
    )


class GCNLayer(nn.Module):
    """A_norm @ X @ W, with optional bias after propagation.

    No implicit activation, dropout or branch architecture is selected here.
    Xavier initialization is an engineering choice for this primitive, not a
    claim about the paper's inaccessible full network configuration.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, nodes: Tensor, normalized_adjacency: Tensor) -> Tensor:
        _validate_nodes(nodes)
        if nodes.shape[-1] != self.in_features:
            raise ValueError("node feature width does not match layer input width")
        if not isinstance(normalized_adjacency, Tensor):
            raise TypeError("normalized_adjacency must be a Tensor")
        if not normalized_adjacency.is_floating_point():
            raise TypeError("normalized_adjacency must have a real floating dtype")
        expected_shape = (nodes.shape[0], nodes.shape[1], nodes.shape[1])
        if normalized_adjacency.shape != expected_shape:
            raise ValueError("normalized_adjacency shape does not match nodes")
        if not torch.isfinite(normalized_adjacency).all():
            raise ValueError("normalized_adjacency must be finite")
        if nodes.dtype != normalized_adjacency.dtype or nodes.dtype != self.weight.dtype:
            raise ValueError("nodes, adjacency and layer must have matching dtype")
        if nodes.device != normalized_adjacency.device or nodes.device != self.weight.device:
            raise ValueError("nodes, adjacency and layer must have matching device")
        output = torch.bmm(normalized_adjacency, nodes) @ self.weight
        return output if self.bias is None else output + self.bias


def mean_readout(nodes: Tensor) -> Tensor:
    """Average node features separately for each image, preserving gradients."""
    _validate_nodes(nodes)
    return nodes.mean(dim=1)
