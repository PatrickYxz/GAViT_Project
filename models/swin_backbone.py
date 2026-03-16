import torch
import torch.nn as nn
import timm


class SwinBackbone(nn.Module):
    """
    Swin-T backbone that outputs spatial patch tokens instead of a class prediction.

    timm's swin_tiny_patch4_window7_224:
      - Input:  (B, 3, 224, 224)
      - Output of forward_features: (B, 7, 7, 768)  [last stage, before global pool]
      - We reshape to (B, 49, 768) — 49 tokens, each 768-dim
    """

    HIDDEN_DIM = 768  # swin_tiny last-stage channel count

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        super().__init__()
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,   # removes the classification head
        )
        self.hidden_dim = self.HIDDEN_DIM

        if freeze:
            for param in self.swin.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224)
        Returns:
            tokens: (B, 49, 768)
        """
        features = self.swin.forward_features(x)

        # timm may return (B, H, W, C) or (B, L, C) depending on version
        if features.dim() == 4:
            B, H, W, C = features.shape
            return features.reshape(B, H * W, C)
        else:
            # already (B, L, C)
            return features
