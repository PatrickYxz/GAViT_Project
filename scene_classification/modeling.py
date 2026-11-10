"""Existing architectures with a shared single-label input/training recipe."""
import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.swin_backbone import SwinBackbone
from grouping_identity import complete_grouping_architecture, soft_assignment_identity, spatial_prior_name
from models.token_refinement import ResidualTokenFFN, refinement_identity, validate_refinement


def architecture(model, num_classes, grouping=None, token_refinement=None, spatial_prior=None):
    extra = {}
    if spatial_prior is not None:
        if model != 'gavit' or grouping != 'soft_prototype' or token_refinement is not None:
            raise ValueError('Spatial prior requires soft GAViT without token refinement')
        extra['soft_assignment'] = soft_assignment_identity(spatial_prior)
    if token_refinement is not None:
        if (token_refinement != 'residual_ffn_v1'
                or (model == 'gavit' and grouping != 'soft_prototype')):
            raise ValueError('Unsupported token refinement candidate')
        extra['token_refinement'] = refinement_identity()
    if model == 'swin':
        if grouping is not None:
            raise ValueError('Swin does not accept grouping')
        return {'num_classes': num_classes, 'backbone': 'swin_tiny_patch4_window7_224', 'dropout': 0.1, **extra}
    if model != 'gavit':
        raise ValueError(f'Unknown model: {model}')
    if grouping not in (None, 'attentive_spatial', 'soft_prototype'):
        raise ValueError(f'Unsupported scene grouping: {grouping}')
    return complete_grouping_architecture({'num_classes': num_classes, 'num_regions': 16, 'knn_k': 5,
            'gat_hidden': 256, 'gat_heads': 4, 'gat_layers': 2, 'dropout': 0.1,
            'grouping': grouping or 'attentive_spatial', 'edge_type': 'sparse_hybrid', 'integration': 'token_feedback', **extra})


class SwinClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_path=None, token_refinement=None):
        super().__init__()
        self.backbone = SwinBackbone(pretrained=bool(pretrained_path), pretrained_path=pretrained_path)
        self.classifier = nn.Sequential(nn.LayerNorm(768), nn.Dropout(0.1), nn.Linear(768, num_classes))
        self.token_refinement = ResidualTokenFFN(token_refinement) if token_refinement is not None else None

    def forward(self, images):
        tokens = self.backbone(images)
        if self.token_refinement is not None:
            tokens = self.token_refinement(tokens)
        return self.classifier(tokens.mean(dim=1))


def refinement_name(config):
    if 'token_refinement' not in config:
        return None
    return validate_refinement(config['token_refinement'])['version']


def build_model(model, num_classes, pretrained_path=None, *, architecture_config=None):
    config = architecture(model, num_classes) if architecture_config is None else complete_grouping_architecture(architecture_config, checkpoint=True)
    if config != architecture(model, num_classes, grouping=config.get('grouping'),
                              token_refinement=refinement_name(config), spatial_prior=spatial_prior_name(config)):
        raise ValueError('Unsupported scene architecture configuration')
    if model == 'swin':
        return SwinClassifier(num_classes, pretrained_path, config.get('token_refinement'))
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
