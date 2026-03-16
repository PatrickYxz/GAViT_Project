import torch
import torch.nn as nn

from models.swin_backbone   import SwinBackbone
from models.region_grouping import KMeansGrouping, SpatialGrouping


class SwinRegionOnly(nn.Module):
    """
    Ablation model: Swin-T + Region Grouping, NO graph reasoning.

    Pipeline:
        Image → Swin-T → 49 tokens
              → Region Grouping → K region features (B, K, 768)
              → Mean Pool       → (B, 768)
              → FC Classifier   → num_classes logits

    Purpose:
        Isolates the contribution of region grouping alone (without GNN).
        Compare against:
          - Swin-T baseline (no grouping, no GNN)     → does grouping help?
          - Full GAViT (grouping + GNN)                → does GNN add on top of grouping?

    Args:
        num_classes:     number of output classes
        num_regions:     K, number of region nodes (try 1, 4, 9, 16)
        grouping:        'spatial' (deterministic grid) or 'kmeans' (feature-space)
        pretrained:      load ImageNet weights for Swin-T
        freeze_backbone: freeze Swin-T weights
    """

    def __init__(
        self,
        num_classes:     int  = 45,
        num_regions:     int  = 9,
        grouping:        str  = "spatial",
        pretrained:      bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = SwinBackbone(pretrained=pretrained, freeze=freeze_backbone)
        feat_dim = self.backbone.hidden_dim  # 768

        if grouping == "spatial":
            self.region_grouping = SpatialGrouping(num_regions=num_regions)
        else:
            self.region_grouping = KMeansGrouping(num_regions=num_regions)

        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            logits: (B, num_classes)
        """
        tokens = self.backbone(x)                           # (B, 49, 768)
        region_features, _ = self.region_grouping(tokens)  # (B, K, 768)
        pooled = region_features.mean(dim=1)               # (B, 768)
        return self.classifier(pooled)                     # (B, num_classes)


if __name__ == "__main__":
    # Quick sanity check
    model = SwinRegionOnly(num_classes=45, num_regions=9, grouping="spatial", pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # expect (2, 45)
    assert out.shape == (2, 45), "Shape mismatch"
    print("Sanity check passed.")
