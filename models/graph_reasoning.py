import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GraphReasoning(nn.Module):
    """
    Multi-layer GAT that refines region node features via neighbour aggregation.

    Design:
      - Input is projected from backbone_dim (768) to hidden_dim * num_heads
        so that residual connections have consistent shape throughout all layers.
      - Each GAT layer: GATConv(concat=True) → LayerNorm(residual) → GELU → Dropout

    Args:
        in_dim:     input node feature dimension (768 for Swin-T)
        hidden_dim: per-head output dimension for each GAT layer
        num_heads:  number of attention heads (output dim = hidden_dim * num_heads)
        num_layers: number of stacked GAT layers
        dropout:    dropout applied after each layer
    """

    def __init__(
        self,
        in_dim:     int = 768,
        hidden_dim: int = 256,
        num_heads:  int = 4,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()

        self.out_dim = hidden_dim * num_heads

        # Project backbone features to GAT width (enables residual from layer 1 onward)
        self.input_proj = nn.Linear(in_dim, self.out_dim)

        self.gat_layers = nn.ModuleList([
            GATConv(self.out_dim, hidden_dim, heads=num_heads,
                    dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(self.out_dim) for _ in range(num_layers)
        ])
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:           torch.Tensor,
        edge_index:  torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x:           (B*K, in_dim)  — batched node features
            edge_index:  (2, E)         — batched edge indices
            edge_weight: (E,)           — optional edge weights (unused by GATConv,
                                          kept for API consistency / future use)
        Returns:
            x: (B*K, out_dim)
        """
        x = self.input_proj(x)  # (B*K, out_dim)

        for gat, norm in zip(self.gat_layers, self.norms):
            residual = x
            x = gat(x, edge_index)   # (B*K, out_dim)
            x = norm(x + residual)
            x = self.act(x)
            x = self.dropout(x)

        return x
