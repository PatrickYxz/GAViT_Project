"""Existing architectures with a shared single-label input/training recipe."""
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.swin_backbone import SwinBackbone


def architecture(model, num_classes):
    if model == 'swin':
        return {'num_classes': num_classes, 'backbone': 'swin_tiny_patch4_window7_224', 'dropout': 0.1}
    if model != 'gavit':
        raise ValueError(f'Unknown model: {model}')
    return {'num_classes': num_classes, 'num_regions': 16, 'knn_k': 5,
            'gat_hidden': 256, 'gat_heads': 4, 'gat_layers': 2, 'dropout': 0.1,
            'grouping': 'attentive_spatial', 'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'}


class SwinClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=bool(pretrained_path), pretrained_path=pretrained_path)
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.1), nn.Linear(768, num_classes))

    def forward(self, images):
        return self.classifier(self.backbone(images).mean(dim=1))


def build_model(model, num_classes, pretrained_path=None):
    config = architecture(model, num_classes)
    if model == 'swin':
        return SwinClassifier(num_classes, pretrained_path)
    from models.gavit import GAViT
    return GAViT(**config, pretrained=bool(pretrained_path), pretrained_path=pretrained_path)


def transform_identity():
    return {'resize': [224, 224], 'interpolation': 'bilinear', 'antialias': True,
            'horizontal_flip_probability': 0.5, 'vertical_flip_probability': 0.5,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0},
            'normalization_mean': [0.485, 0.456, 0.406], 'normalization_std': [0.229, 0.224, 0.225],
            'eval_random_augmentation': False}


def image_transform(training):
    operations = [transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR, antialias=True)]
    if training:
        operations += [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5),
                       transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0)]
    return transforms.Compose(operations + [transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def capture_graph_once(model):
    """Check actual graph-reasoning input, including spatial/feature edge semantics."""
    result = {}

    def check(_module, inputs):
        if result:
            return
        nodes, edge_index, _weights = inputs
        batch = nodes.shape[0] // 16
        if tuple(edge_index.shape) != (2, batch * 80):
            raise ValueError('Unexpected sparse-hybrid edge count')
        for i in range(batch):
            edges = edge_index[:, i * 80:(i + 1) * 80] - i * 16
            src, dst = edges
            distance = (src // 4 - dst // 4).abs() + (src % 4 - dst % 4).abs()
            if (int(edges.min()) < 0 or int(edges.max()) >= 16
                    or not bool((distance[:48] == 1).all())
                    or not bool((distance[48:] > 1).all())
                    or torch.unique(edges, dim=1).shape[1] != 80
                    or not bool((torch.bincount(dst[48:], minlength=16) == 2).all())):
                raise ValueError('Invalid spatial/feature graph topology')
        result.update(edges_per_image=80, spatial_edges=48, feature_edges=32,
                      actual_forward_checked=True)

    handle = model.graph_reasoning.register_forward_pre_hook(check) if hasattr(model, 'graph_reasoning') else None
    return result, handle
