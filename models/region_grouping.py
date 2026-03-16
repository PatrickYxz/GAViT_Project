import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KMeansGrouping(nn.Module):
    """
    Groups N tokens into K region nodes via iterative k-means in feature space.

    The k-means assignment step is detached from the computation graph (argmax
    is not differentiable), but gradients flow back to the backbone through
    the differentiable cluster-averaging step.

    Args:
        num_regions: K, number of region nodes
        num_iters:   number of k-means iterations per forward pass
    """

    def __init__(self, num_regions: int = 9, num_iters: int = 10):
        super().__init__()
        self.K = num_regions
        self.num_iters = num_iters

    @torch.no_grad()
    def _assign(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D)  — detached
        returns: assignments (B, N) with values in [0, K)
        """
        B, N, D = tokens.shape
        K = self.K

        # Random initialization: pick K distinct token indices per sample
        rand_idx = torch.rand(B, N, device=tokens.device).argsort(dim=1)[:, :K]  # (B, K)
        centroids = tokens[torch.arange(B, device=tokens.device).unsqueeze(1), rand_idx]  # (B, K, D)

        for _ in range(self.num_iters):
            # Cosine similarity between tokens and centroids
            t_norm = F.normalize(tokens, dim=-1)      # (B, N, D)
            c_norm = F.normalize(centroids, dim=-1)   # (B, K, D)
            sim = torch.bmm(t_norm, c_norm.transpose(1, 2))  # (B, N, K)
            assignments = sim.argmax(dim=-1)           # (B, N)

            # Update centroids: mean of assigned tokens
            # One-hot encode assignments: (B, N, K)
            one_hot = torch.zeros(B, N, K, device=tokens.device)
            one_hot.scatter_(2, assignments.unsqueeze(-1), 1.0)
            counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, K)
            centroids = torch.bmm(one_hot.transpose(1, 2), tokens) / counts.transpose(1, 2)
            # centroids: (B, K, D)

        return assignments  # (B, N)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, N, D)
        Returns:
            region_features: (B, K, D)  — averaged token features per cluster
            assignments:     (B, N)     — cluster index for each token
        """
        B, N, D = tokens.shape
        K = self.K

        # Detach tokens for assignment step (k-means not differentiable)
        assignments = self._assign(tokens.detach())  # (B, N)

        # Differentiable averaging: use one-hot weight matrix
        one_hot = torch.zeros(B, N, K, device=tokens.device, dtype=tokens.dtype)
        one_hot.scatter_(2, assignments.unsqueeze(-1), 1.0)          # (B, N, K)
        counts = one_hot.sum(dim=1, keepdim=True).clamp(min=1)       # (B, 1, K)
        weight = one_hot / counts                                     # (B, N, K)

        # (B, K, D) = einsum over N: weight^T @ tokens
        region_features = torch.bmm(weight.transpose(1, 2), tokens)  # (B, K, D)

        return region_features, assignments


class SpatialGrouping(nn.Module):
    """
    Groups 7×7 patch tokens into K spatial macro-regions by dividing the grid.

    Deterministic and fully differentiable. K must be a perfect square
    (e.g. 1, 4, 9, 16, 49).

    Args:
        num_regions: K  (must be a perfect square)
        grid_size:   H = W of the token grid (7 for Swin-T)
    """

    def __init__(self, num_regions: int = 9, grid_size: int = 7):
        super().__init__()
        self.K = num_regions
        self.grid_size = grid_size

        k_side = int(math.sqrt(num_regions))
        assert k_side * k_side == num_regions, (
            f"SpatialGrouping requires K to be a perfect square (1, 4, 9, 16, 49), got {num_regions}"
        )

        # Build assignment map once
        G = grid_size
        assignment = torch.zeros(G * G, dtype=torch.long)
        for i in range(G):
            for j in range(G):
                region_r = min(int(i / G * k_side), k_side - 1)
                region_c = min(int(j / G * k_side), k_side - 1)
                assignment[i * G + j] = region_r * k_side + region_c

        self.register_buffer("assignment", assignment)    # (N,)
        # Precompute weight matrix (N, K): one-hot / count_per_cluster
        N = G * G
        one_hot = torch.zeros(N, num_regions)
        one_hot.scatter_(1, assignment.unsqueeze(1), 1.0)
        counts = one_hot.sum(dim=0, keepdim=True).clamp(min=1)  # (1, K)
        weight = one_hot / counts                                # (N, K)
        self.register_buffer("weight", weight)                  # (N, K)

    def forward(self, tokens: torch.Tensor):
        """
        Args:
            tokens: (B, N, D) where N = grid_size^2
        Returns:
            region_features: (B, K, D)
            assignments:     (B, N)
        """
        B, N, D = tokens.shape
        # (B, K, D) = (B, N, D) @ (N, K)  via einsum
        region_features = torch.einsum("bnd,nk->bkd", tokens, self.weight)  # (B, K, D)
        assignments = self.assignment.unsqueeze(0).expand(B, -1)             # (B, N)
        return region_features, assignments
