from pathlib import Path


def build_swin_model_kwargs(pretrained=True, pretrained_path=None):
    """Build timm arguments, optionally using a local pretrained file."""
    kwargs = {"pretrained": pretrained, "num_classes": 0}
    if pretrained_path:
        weight_path = Path(pretrained_path)
        if not weight_path.is_file():
            raise FileNotFoundError(
                f"Swin pretrained weights not found: {weight_path}"
            )
        kwargs["pretrained"] = True
        kwargs["pretrained_cfg_overlay"] = {"file": str(weight_path)}
    return kwargs


def pool_swin_features(features):
    """Pool timm Swin features to one vector per image."""
    if features.ndim == 4:
        return features.mean(dim=(1, 2))
    if features.ndim == 3:
        return features.mean(dim=1)
    return features
