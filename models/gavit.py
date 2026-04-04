import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.swin_backbone      import SwinBackbone
from models.region_grouping    import KMeansGrouping, SpatialGrouping, AttentiveSpatialGrouping
from models.graph_construction import build_knn_graph, build_spatial_graph
from models.graph_reasoning    import GraphReasoning


class GAViT(nn.Module):
    """
    Graph-Augmented Vision Transformer (GAViT).

    Supports two integration modes:

    1. "fusion" (legacy): classifier uses concat of backbone global + graph global
       Image → Swin-T → tokens
             ├→ Mean Pool → backbone global (768)
             └→ Region Grouping → GAT → Mean Pool → graph global (1024)
             → Concat (1792) → FC → logits

    2. "token_feedback" (new, per Prof Wang's guidance):
       GAT-refined region features are mapped back to token space,
       updating each token with its region's relational context.
       Classification is performed on the updated tokens.
       Image → Swin-T → tokens (B, 49, 768)
             → Region Grouping → K region nodes
             → GAT → refined region features (B, K, out_dim)
             → Project back to 768-dim → map to tokens via assignments
             → Updated tokens = original tokens + region feedback
             → Mean Pool → FC → logits

    Args:
        num_classes:     number of output classes (45 for NWPU-RESISC45)
        num_regions:     K, number of region nodes
        knn_k:           k for kNN graph construction
        gat_hidden:      per-head hidden dim in GAT layers
        gat_heads:       number of GAT attention heads
        gat_layers:      number of stacked GAT layers
        dropout:         dropout rate
        grouping:        'kmeans', 'spatial', or 'attentive_spatial'
        edge_type:       'knn', 'spatial', or 'hybrid'
        integration:     'token_feedback' or 'fusion'
        pretrained:      whether to load ImageNet weights for Swin-T
        freeze_backbone: freeze Swin-T weights
    """

    def __init__(
        self,
        num_classes:     int   = 45,
        num_regions:     int   = 16,
        knn_k:           int   = 5,
        gat_hidden:      int   = 256,
        gat_heads:       int   = 4,
        gat_layers:      int   = 2,
        dropout:         float = 0.1,
        grouping:        str   = "attentive_spatial",
        edge_type:       str   = "knn",
        integration:     str   = "token_feedback",
        pretrained:      bool  = True,
        freeze_backbone: bool  = False,
    ):
        super().__init__()
        self.integration = integration

        # --- Backbone ---
        self.backbone = SwinBackbone(pretrained=pretrained, freeze=freeze_backbone)
        backbone_dim = self.backbone.hidden_dim  # 768

        # --- Region Grouping ---
        if grouping == "attentive_spatial":
            self.region_grouping = AttentiveSpatialGrouping(
                num_regions=num_regions, feat_dim=backbone_dim
            )
        elif grouping == "spatial":
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

        graph_out_dim = self.graph_reasoning.out_dim  # gat_hidden * gat_heads = 1024

        # --- Integration-specific layers ---
        if integration == "token_feedback":
            # Project GAT output back to backbone dim for token-level residual
            self.feedback_proj = nn.Sequential(
                nn.Linear(graph_out_dim, backbone_dim),
                nn.LayerNorm(backbone_dim),
            )
            # Classifier operates on updated token features (backbone_dim)
            self.classifier = nn.Sequential(
                nn.LayerNorm(backbone_dim),
                nn.Dropout(dropout),
                nn.Linear(backbone_dim, num_classes),
            )
        else:  # "fusion" (legacy)
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
        tokens = self.backbone(x)  # (B, 49, 768)

        # 2. Region grouping → K region node features
        region_features, assignments = self.region_grouping(tokens)  # (B, K, 768), (B, N)

        # 3. Build batched graph
        if self.edge_type == "spatial":
            edge_index, edge_weight, batch = build_spatial_graph(K, B, x.device)
        elif self.edge_type == "hybrid":
            ei_knn, ew_knn, batch = build_knn_graph(region_features, k=self.knn_k)
            ei_sp, ew_sp, _      = build_spatial_graph(K, B, x.device)
            edge_index  = torch.cat([ei_knn, ei_sp], dim=1)
            edge_weight = torch.cat([ew_knn, ew_sp], dim=0)
        else:  # "knn"
            edge_index, edge_weight, batch = build_knn_graph(
                region_features, k=self.knn_k
            )

        # 4. Flatten nodes for PyG: (B*K, 768)
        x_nodes = region_features.reshape(B * K, -1)

        # 5. GAT reasoning → refined node features: (B*K, graph_out_dim)
        x_refined = self.graph_reasoning(x_nodes, edge_index, edge_weight)

        # 6. Integration
        if self.integration == "token_feedback":
            # Map refined region features back to token space
            x_refined_2d = x_refined.reshape(B, K, -1)           # (B, K, graph_out_dim)
            region_feedback = self.feedback_proj(x_refined_2d)    # (B, K, 768)

            # Each token receives its region's refined feature
            # assignments: (B, N) with values in [0, K)
            token_feedback = region_feedback.gather(
                1,
                assignments.unsqueeze(-1).expand(-1, -1, region_feedback.size(-1))
            )  # (B, N, 768)

            # Residual update: original tokens + relational context
            updated_tokens = tokens + token_feedback  # (B, N, 768)

            # Global pool over updated tokens → classify
            pooled = updated_tokens.mean(dim=1)       # (B, 768)
            logits = self.classifier(pooled)           # (B, num_classes)

        else:  # "fusion" (legacy)
            backbone_global = tokens.mean(dim=1)                      # (B, 768)
            graph_global = global_mean_pool(x_refined, batch)         # (B, 1024)
            fused = torch.cat([backbone_global, graph_global], dim=1) # (B, 1792)
            logits = self.classifier(fused)

        return logits


# ---------------------------------------------------------------------------
# Unit test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.randn(2, 3, 224, 224, device=device)

    # Test token_feedback mode (new default)
    print("=== GAViT v2: token_feedback, attentive_spatial K=16 ===")
    model = GAViT(
        num_classes=45,
        num_regions=16,
        grouping="attentive_spatial",
        integration="token_feedback",
        pretrained=False,
    ).to(device)
    out = model(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {total:,}")

    # Test fusion mode (legacy)
    print("\n=== GAViT v1: fusion, spatial K=9 ===")
    model2 = GAViT(
        num_classes=45,
        num_regions=9,
        grouping="spatial",
        integration="fusion",
        pretrained=False,
    ).to(device)
    out2 = model2(dummy)
    total2 = sum(p.numel() for p in model2.parameters())
    print(f"  Output shape: {out2.shape}")
    print(f"  Parameters:   {total2:,}")