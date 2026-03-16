from typing import Tuple
import torch
import torch.nn.functional as F


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

    # Source indices: same for every batch item (before offset)
    src_local = torch.arange(N, device=features.device).unsqueeze(1).expand(N, k).reshape(-1)  # (N*k,)

    edge_index_list = []
    edge_weight_list = []

    for b in range(B):
        tgt_local = topk_idx[b].reshape(-1)         # (N*k,)
        weights    = topk_vals[b].reshape(-1)         # (N*k,)
        offset = b * N
        src = src_local + offset
        tgt = tgt_local + offset
        edge_index_list.append(torch.stack([src, tgt], dim=0))  # (2, N*k)
        edge_weight_list.append(weights)

    edge_index  = torch.cat(edge_index_list,  dim=1)  # (2, B*N*k)
    edge_weight = torch.cat(edge_weight_list, dim=0)  # (B*N*k,)

    # Batch vector: which graph each node belongs to
    batch = torch.arange(B, device=features.device).unsqueeze(1).expand(B, N).reshape(-1)  # (B*N,)

    return edge_index, edge_weight, batch
