from typing import Tuple
import math
import torch
import torch.nn.functional as F


def build_spatial_graph(
    num_regions: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build an undirected spatial adjacency graph for a grid of K regions.
    Two regions are connected if they are neighbours in the grid
    (8-connectivity: horizontal, vertical, diagonal).

    Args:
        num_regions: K (must be a perfect square, e.g. 4, 9, 16)
        batch_size:  B
        device:      torch device

    Returns:
        edge_index:  (2, B * E_per_graph)
        edge_weight: (B * E_per_graph,)  — all ones (unweighted)
        batch:       (B * K,)
    """
    K = num_regions
    side = int(math.isqrt(K))
    assert side * side == K, f"num_regions={K} must be a perfect square"

    # Build adjacency for one graph
    edges_src, edges_tgt = [], []
    for i in range(K):
        r, c = divmod(i, side)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side:
                    j = nr * side + nc
                    edges_src.append(i)
                    edges_tgt.append(j)

    src_local = torch.tensor(edges_src, device=device, dtype=torch.long)
    tgt_local = torch.tensor(edges_tgt, device=device, dtype=torch.long)
    E = len(edges_src)

    # Replicate for each batch item with offset
    edge_index_list = []
    for b in range(batch_size):
        offset = b * K
        edge_index_list.append(torch.stack([src_local + offset, tgt_local + offset], dim=0))

    edge_index  = torch.cat(edge_index_list, dim=1)             # (2, B*E)
    edge_weight = torch.ones(batch_size * E, device=device)     # uniform weight
    batch = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, K).reshape(-1)

    return edge_index, edge_weight, batch


def build_knn_graph(
    features: torch.Tensor,
    k: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a directed kNN graph for each sample in a batch using cosine similarity,
    then concatenate into a single batched graph for PyG.

    Args:
        features: (B, K_nodes, D) — region node features
        k:        number of nearest neighbours per node (excluding self)

    Returns:
        edge_index:  (2, B * K_nodes * k) — source/target node indices (batch-offset)
        edge_weight: (B * K_nodes * k,)   — cosine similarity scores
        batch:       (B * K_nodes,)        — batch vector for global_pool in PyG
    """
    B, N, D = features.shape
    k = min(k, N - 1)  # can't have more neighbours than N-1

    # Cosine similarity matrix: (B, N, N)
    norm = F.normalize(features, dim=-1)
    sim = torch.bmm(norm, norm.transpose(1, 2))  # (B, N, N)

    # Mask self-loops
    eye = torch.eye(N, device=features.device, dtype=torch.bool).unsqueeze(0)
    sim = sim.masked_fill(eye, -1.0)

    # Top-k neighbours per node
    topk_vals, topk_idx = sim.topk(k, dim=-1)  # (B, N, k)

    # PyG propagates messages from edge_index[0] to edge_index[1]. Each query
    # node must therefore be the target of edges from its selected neighbours.
    query_local = torch.arange(N, device=features.device).unsqueeze(1).expand(N, k).reshape(-1)  # (N*k,)

    edge_index_list = []
    edge_weight_list = []

    for b in range(B):
        neighbor_local = topk_idx[b].reshape(-1)      # (N*k,)
        weights = topk_vals[b].reshape(-1)            # (N*k,)
        offset = b * N
        src = neighbor_local + offset
        tgt = query_local + offset
        edge_index_list.append(torch.stack([src, tgt], dim=0))  # (2, N*k)
        edge_weight_list.append(weights)

    edge_index  = torch.cat(edge_index_list,  dim=1)  # (2, B*N*k)
    edge_weight = torch.cat(edge_weight_list, dim=0)  # (B*N*k,)

    # Batch vector: which graph each node belongs to
    batch = torch.arange(B, device=features.device).unsqueeze(1).expand(B, N).reshape(-1)  # (B*N,)

    return edge_index, edge_weight, batch


def build_sparse_hybrid_graph(
    features: torch.Tensor,
    feature_k: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the fixed-budget K=16 sparse spatial-plus-feature graph.

    Each query receives every four-neighbor region in the 4x4 grid plus its
    two most similar non-spatial regions. Edges use PyG's
    ``selected_neighbor -> query`` message direction. Spatial edges are
    emitted before feature edges for each graph so diagnostics can separate
    the 48/32 edge components without changing GAT computation.
    """
    if features.ndim != 3:
        raise ValueError(
            f"features must have shape (B, 16, D); observed ndim={features.ndim}"
        )

    batch_size, num_regions, _ = features.shape
    if num_regions != 16:
        raise ValueError(
            "sparse_hybrid requires the approved 4x4 grid; "
            f"observed num_regions={num_regions}"
        )
    if feature_k != 2:
        raise ValueError(
            "sparse_hybrid uses the approved top-2 feature budget; "
            f"observed feature_k={feature_k}"
        )

    spatial_sources: list[int] = []
    spatial_targets: list[int] = []
    spatial_by_query: list[list[int]] = []
    for query in range(num_regions):
        row, col = divmod(query, 4)
        neighbors = []
        for delta_row, delta_col in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            neighbor_row = row + delta_row
            neighbor_col = col + delta_col
            if 0 <= neighbor_row < 4 and 0 <= neighbor_col < 4:
                source = neighbor_row * 4 + neighbor_col
                neighbors.append(source)
                spatial_sources.append(source)
                spatial_targets.append(query)
        spatial_by_query.append(neighbors)

    spatial_src = torch.tensor(
        spatial_sources, device=features.device, dtype=torch.long
    )
    spatial_tgt = torch.tensor(
        spatial_targets, device=features.device, dtype=torch.long
    )

    normalized = F.normalize(features, dim=-1)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    candidate_mask = torch.ones(
        (num_regions, num_regions), device=features.device, dtype=torch.bool
    )
    for query, spatial_neighbors in enumerate(spatial_by_query):
        candidate_mask[query, query] = False
        candidate_mask[query, spatial_neighbors] = False
    ranked_similarity = similarity.masked_fill(
        ~candidate_mask.unsqueeze(0), float("-inf")
    )
    # Stable sorting gives the required lower-index tie break because the
    # candidate axis is already in ascending region-index order.
    feature_sources = torch.argsort(
        ranked_similarity, dim=-1, descending=True, stable=True
    )[:, :, :feature_k]
    feature_weights = torch.gather(
        similarity, dim=-1, index=feature_sources
    )
    feature_targets = (
        torch.arange(num_regions, device=features.device)
        .unsqueeze(1)
        .expand(num_regions, feature_k)
    )

    edge_index_list = []
    edge_weight_list = []
    for batch_index in range(batch_size):
        offset = batch_index * num_regions
        spatial_edges = torch.stack(
            [spatial_src + offset, spatial_tgt + offset], dim=0
        )
        feature_edges = torch.stack(
            [
                feature_sources[batch_index].reshape(-1) + offset,
                feature_targets.reshape(-1) + offset,
            ],
            dim=0,
        )
        edge_index_list.append(torch.cat([spatial_edges, feature_edges], dim=1))
        edge_weight_list.append(
            torch.cat(
                [
                    torch.ones(48, device=features.device, dtype=features.dtype),
                    feature_weights[batch_index].reshape(-1),
                ]
            )
        )

    edge_index = torch.cat(edge_index_list, dim=1)
    edge_weight = torch.cat(edge_weight_list, dim=0)
    batch = (
        torch.arange(batch_size, device=features.device)
        .unsqueeze(1)
        .expand(batch_size, num_regions)
        .reshape(-1)
    )
    return edge_index, edge_weight, batch
