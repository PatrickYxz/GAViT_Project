import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.swin_backbone      import SwinBackbone
from models.region_grouping    import KMeansGrouping, SpatialGrouping
from models.graph_construction import build_knn_graph, build_spatial_graph
from models.graph_reasoning    import GraphReasoning


class GAViT(nn.Module):
    """
    Graph-Augmented Vision Transformer (GAViT).

    Pipeline:
        Image → Swin-T backbone → 49 tokens (7×7, 768-dim)
              ├→ Mean Pool        → global backbone descriptor (768-dim)
              └→ Region Grouping  → K region nodes
                → kNN Graph       → edge_index
                → GAT Reasoning   → refined node features
                → Mean Pool       → graph descriptor (gat_hidden * gat_heads)
              → Fusion (concat)   → (768 + gat_hidden * gat_heads)-dim
              → FC Classifier     → num_classes logits

    Args:
        num_classes:     number of output classes (45 for NWPU-RESISC45)
        num_regions:     K, number of region nodes (try 4, 9, 16)
        knn_k:           k for kNN graph construction (only used when edge_type='knn')
        gat_hidden:      per-head hidden dim in GAT layers
        gat_heads:       number of GAT attention heads
        gat_layers:      number of stacked GAT layers
        dropout:         dropout rate in GAT and classifier
        grouping:        'kmeans' or 'spatial'
        edge_type:       'knn' (cosine kNN), 'spatial' (grid adjacency), or 'hybrid' (both)
        pretrained:      whether to load ImageNet weights for Swin-T
        freeze_backbone: freeze Swin-T weights (faster training, may hurt accuracy)
    """

    def __init__(
        self,
        num_classes:     int   = 45,
        num_regions:     int   = 9,
        knn_k:           int   = 5,
        gat_hidden:      int   = 256,
        gat_heads:       int   = 4,
        gat_layers:      int   = 2,
        dropout:         float = 0.1,
        grouping:        str   = "kmeans",
        edge_type:       str   = "knn",
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
    ):
        super().__init__()

        # --- Backbone ---
        self.backbone = SwinBackbone(pretrained=pretrained, freeze=freeze_backbone)
        backbone_dim = self.backbone.hidden_dim  # 768

        # --- Region Grouping ---
        if grouping == "spatial":
            self.region_grouping = SpatialGrouping(num_regions=num_regions)
        else:
            self.region_grouping = KMeansGrouping(num_regions=num_regions)

        self.num_regions = num_regions
        self.knn_k = min(knn_k, num_regions - 1)
        self.edge_type = edge_type

        # --- Graph Reasoning ---
        self.graph_reasoning = GraphReasoning(
            in_dim=backbone_dim,
            hidden_dim=gat_hidden,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        # --- Classifier (fusion: backbone global + graph-refined) ---
        graph_out_dim = self.graph_reasoning.out_dim
        fused_dim = backbone_dim + graph_out_dim  # 768 + 1024 = 1792
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        K = self.num_regions

        # 1. Swin-T backbone → patch tokens
        tokens = self.backbone(x)                           # (B, 49, 768)

        # 2. Global backbone feature (preserve original Swin representation)
        backbone_global = tokens.mean(dim=1)                # (B, 768)

        # 3. Region grouping → K region node features
        region_features, _ = self.region_grouping(tokens)  # (B, K, 768)

        # 4. Build batched graph
        if self.edge_type == "spatial":
            edge_index, edge_weight, batch = build_spatial_graph(
                K, B, x.device
            )
        elif self.edge_type == "hybrid":
            ei_knn, ew_knn, batch = build_knn_graph(region_features, k=self.knn_k)
            ei_sp, ew_sp, _      = build_spatial_graph(K, B, x.device)
            edge_index  = torch.cat([ei_knn, ei_sp], dim=1)
            edge_weight = torch.cat([ew_knn, ew_sp], dim=0)
        else:  # "knn" (default)
            edge_index, edge_weight, batch = build_knn_graph(
                region_features, k=self.knn_k
            )

        # 5. Flatten nodes for PyG: (B*K, 768)
        x_nodes = region_features.reshape(B * K, -1)

        # 6. GAT reasoning → updated node features
        x_out = self.graph_reasoning(x_nodes, edge_index, edge_weight)  # (B*K, out_dim)

        # 7. Graph-level pooling → (B, out_dim)
        graph_global = global_mean_pool(x_out, batch)       # (B, 1024)

        # 8. Fusion: concat backbone global + graph-refined features
        fused = torch.cat([backbone_global, graph_global], dim=1)  # (B, 1792)

        # 9. Classify
        logits = self.classifier(fused)                     # (B, num_classes)
        return logits
