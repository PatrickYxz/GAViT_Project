"""Local, image-dependent soft regions on the existing 7x7 -> 4x4 anchors."""
import torch
from torch import nn
from torch.nn import functional as F

from grouping_identity import soft_assignment_identity, validate_soft_assignment
from .region_grouping import AttentiveSpatialGrouping


class SoftPrototypeGrouping(AttentiveSpatialGrouping):
    def __init__(self, num_regions=16, grid_size=7, feat_dim=768, soft_assignment=None):
        if (num_regions, grid_size, feat_dim) != (16, 7, 768):
            raise ValueError('soft_prototype requires num_regions=16, grid_size=7, feat_dim=768')
        self.config = validate_soft_assignment(
            soft_assignment_identity() if soft_assignment is None else soft_assignment)
        super().__init__(num_regions, grid_size, feat_dim)
        # Preserve the old attention initialization and the caller's RNG stream.
        # All parameters are constructed on CPU, then moved with the model.
        with torch.random.fork_rng(devices=[]):
            seed = (torch.initial_seed() + 0x534F4654) % (2**63 - 1)
            torch.random.default_generator.manual_seed(seed)
            self.assignment_projection = nn.Sequential(
                nn.LayerNorm(feat_dim, eps=self.config['assignment_ln_eps']),
                nn.Linear(feat_dim, self.config['assignment_dim'], bias=False))
            nn.init.xavier_uniform_(self.assignment_projection[1].weight)
        regions = torch.arange(num_regions)
        distance = torch.maximum(
            (self.assignment[:, None] // 4 - regions[None] // 4).abs(),
            (self.assignment[:, None] % 4 - regions[None] % 4).abs())
        self.register_buffer('candidate_mask', distance <= self.config['assignment_radius'])
        self.assignment_observer = None

    def _validate_tokens(self, tokens):
        if (tokens.ndim != 3 or tokens.shape[0] < 1 or tuple(tokens.shape[1:]) != (49, 768)
                or not tokens.is_floating_point() or not torch.isfinite(tokens).all()):
            raise ValueError('soft_prototype requires finite floating tokens [B>0,49,768]')

    def _assignment(self, tokens, scores):
        self._validate_tokens(tokens)
        if scores.shape != (*tokens.shape[:2], 1) or not torch.isfinite(scores).all():
            raise ValueError('Nonfinite or invalid token attention scores')
        w0 = scores.expand(-1, -1, self.K).masked_fill(
            self.membership[None] == 0, -torch.inf).softmax(dim=1)
        projected = self.assignment_projection(tokens)
        if not torch.isfinite(projected).all():
            raise ValueError('Nonfinite assignment projection')
        z = F.normalize(projected, dim=-1, eps=self.config['assignment_norm_eps'])
        raw = torch.einsum('bnk,bnd->bkd', w0, z)
        prototypes = F.normalize(raw, dim=-1, eps=self.config['assignment_norm_eps'])
        logits = torch.einsum('bnd,bkd->bnk', z, prototypes) / self.config['assignment_temperature']
        log_s = logits.masked_fill(~self.candidate_mask[None], -torch.inf).log_softmax(-1)
        return log_s, w0, raw.norm(dim=-1), prototypes

    def _pool(self, tokens, scores, log_s):
        w = (log_s + scores).softmax(dim=1)
        regions = torch.einsum('bnk,bnd->bkd', w, tokens)
        if not torch.isfinite(regions).all() or not torch.isfinite(w).all():
            raise ValueError('Nonfinite soft region pooling')
        return regions, w

    def forward(self, tokens):
        self._validate_tokens(tokens)
        scores = self.attn(tokens)
        log_s, w0, norms, prototypes = self._assignment(tokens, scores)
        regions, w = self._pool(tokens, scores, log_s)
        s = log_s.exp()
        if self.assignment_observer is not None:
            self.assignment_observer({key: value.detach() for key, value in {
                'S': s, 'W': w, 'W0': w0, 'prototype_norms': norms,
                'prototypes': prototypes, 'token_norm': tokens.norm(dim=-1)}.items()})
        return regions, s
