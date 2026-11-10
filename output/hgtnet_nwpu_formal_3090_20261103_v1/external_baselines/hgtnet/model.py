"""Full, nonofficial HGTNet candidate; architectural assumptions live in config."""
from __future__ import annotations

import hashlib
from collections import Counter

import torch
from torch import nn

from .attention import PatchEmbedding, TransformerStage
from .config import config_sha256, validate_config
from .graph import DepthwiseProjection, GraphBlock


class HGTNet(nn.Module):
    """224px RGB -> DA/Swin -> dynamic graph -> summed GAP -> raw logits.

    Construction never loads or downloads weights. This is an independently
    specified candidate, not verified author code or an exact reproduction.
    """

    implementation = 'hgtnet_independent_v0'

    def __init__(self, config):
        super().__init__()
        cfg = self.config = validate_config(config)
        self.config_sha256 = config_sha256(cfg)
        self.pretraining_loaded = False
        self.patch_embed = PatchEmbedding(cfg['in_channels'], cfg['stage_channels'][0],
                                          cfg['patch_size'], cfg['norm_eps'])
        self.stages = nn.ModuleList([
            TransformerStage(channels, stbs, heads, output_channels=out,
                             downsample_kind=kind, window_size=cfg['window_size'],
                             mlp_ratio=cfg['mlp_ratio'], dropout=cfg['dropout'],
                             attention_dropout=cfg['attention_dropout'],
                             stochastic_depth_prob=cfg['drop_path'], norm_eps=cfg['norm_eps'])
            for channels, stbs, heads, out, kind in zip(
                cfg['stage_channels'], cfg['stb_counts'], cfg['heads'],
                cfg['stage_output_channels'], cfg['stage_downsamples'])
        ])
        self.graph_transition1 = DepthwiseProjection(512, 256,
            stride=cfg['graph_transition_strides'][0], padding=cfg['graph_transition_paddings'][0],
            norm_eps=cfg['norm_eps'])
        self.graph_transition2 = DepthwiseProjection(256, 512,
            stride=cfg['graph_transition_strides'][1], padding=cfg['graph_transition_paddings'][1],
            norm_eps=cfg['norm_eps'])
        def blocks(channels, depth):
            return nn.ModuleList([GraphBlock(channels, cfg['graph_k'], cfg['graph_operator'],
                cfg['graph_ffn_ratio'], cfg['norm_eps'], cfg['graph_exclude_self']) for _ in range(depth)])
        self.graph_blocks1 = blocks(256, cfg['graph_depths'][0])
        self.graph_blocks2 = blocks(512, cfg['graph_depths'][1])
        self.head = nn.Linear(512, cfg['num_classes'])

    def assert_identity(self):
        if config_sha256(self.config) != self.config_sha256:
            raise ValueError('Model config changed after construction')

    def _validate_input(self, image):
        self.assert_identity()
        if not isinstance(image, torch.Tensor) or image.layout != torch.strided:
            raise TypeError('Input must be a strided torch.Tensor')
        if image.ndim != 4 or image.shape[0] < 1 or tuple(image.shape[1:]) != (3, 224, 224):
            raise ValueError('HGTNet candidate requires nonempty B x 3 x 224 x 224 input')
        if not image.is_floating_point() or image.dtype != self.patch_embed.projection.weight.dtype:
            raise TypeError('Input must have the same floating dtype as model parameters')
        if image.device != self.patch_embed.projection.weight.device or image.is_meta:
            raise ValueError('Input must be on the real model device; meta is not execution')
        if not torch.isfinite(image).all().item():
            raise ValueError('Input must be finite')

    @staticmethod
    def _graph_group(features, blocks, traced):
        batch, channels, height, width = features.shape
        nodes = features.flatten(2).transpose(1, 2)
        graphs = []
        for block in blocks:
            if traced:
                nodes, indices = block(nodes, return_neighbors=True)
                # This digest describes actual selected indices, not an
                # individual edge's contribution or an explanatory score.
                values = indices.detach().cpu().contiguous()
                graphs.append({'indices_shape': list(values.shape),
                               'indices_sha256': hashlib.sha256(values.numpy().tobytes()).hexdigest(),
                               'self_edges': int((values == torch.arange(height * width)[None, :, None]).sum()),
                               'k': block.k})
            else:
                nodes = block(nodes)
        return nodes.transpose(1, 2).reshape(batch, channels, height, width), graphs

    def _run(self, image, traced):
        self._validate_input(image)
        features = self.patch_embed(image)
        trace = {'input': list(image.shape), 'patch': list(features.shape),
                 'stage_outputs': [], 'graph_outputs': [], 'graph_indices': []}
        for stage in self.stages:
            features = stage(features)
            if traced:
                trace['stage_outputs'].append(list(features.shape))
        visual = features.mean(dim=(2, 3))
        features = self.graph_transition1(features)
        features, graphs = self._graph_group(features, self.graph_blocks1, traced)
        if traced:
            trace['graph_outputs'].append(list(features.shape))
            trace['graph_indices'].append(graphs)
        features = self.graph_transition2(features)
        features, graphs = self._graph_group(features, self.graph_blocks2, traced)
        if traced:
            trace['graph_outputs'].append(list(features.shape))
            trace['graph_indices'].append(graphs)
        fused = visual + features.mean(dim=(2, 3))
        logits = self.head(fused)
        if not torch.isfinite(logits).all().item():
            raise ValueError('Model produced nonfinite logits')
        if traced:
            trace.update(fusion=list(fused.shape), logits=list(logits.shape))
            return logits, trace
        return logits

    def forward(self, image):
        return self._run(image, False)

    def forward_with_trace(self, image):
        """Return logits and JSON-safe shapes/digests from the actual forward."""
        return self._run(image, True)


def parameter_counts(model):
    """Count disjoint architectural groups, excluding nonparameter buffers."""
    groups = Counter()
    for name, value in model.named_parameters():
        if name.startswith('patch_embed.'):
            group = 'patch_embed'
        elif name.startswith('stages.'):
            group = 'dual_attention' if '.da.' in name else 'transformer'
        elif name.startswith('graph_'):
            group = 'graph'
        elif name.startswith('head.'):
            group = 'head'
        else:
            raise ValueError(f'Unclassified parameter: {name}')
        groups[group] += value.numel()
    return {'total': sum(groups.values()), 'trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'groups': dict(sorted(groups.items()))}
