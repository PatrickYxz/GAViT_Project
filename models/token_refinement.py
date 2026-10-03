"""One frozen residual FFN candidate for the NWPU calibration experiment."""
import copy

import torch
from torch import nn


def refinement_identity():
    return {'version': 'residual_ffn_v1', 'position': 'before_mean_pool',
            'input_dim': 768, 'hidden_dim': 192, 'bias': True, 'ln_eps': 1e-5,
            'activation': 'gelu_exact', 'dropout': 0.0,
            'init': 'fc1_xavier_uniform_bias_zero_fc2_zero', 'init_seed': 42003}


def validate_refinement(config):
    expected = refinement_identity()
    if (not isinstance(config, dict) or set(config) != set(expected)
            or any(type(config[k]) is not type(v) or config[k] != v for k, v in expected.items())):
        raise ValueError('Unsupported token refinement identity')
    return copy.deepcopy(config)


class ResidualTokenFFN(nn.Module):
    """Shared independently at each token; no token mixing or additional dropout."""
    def __init__(self, config):
        super().__init__()
        self.config = validate_refinement(config)
        # Include constructors in the isolated CPU RNG scope. Do not seed CUDA.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.config['init_seed'])
            self.norm = nn.LayerNorm(768, eps=1e-5)
            self.fc1 = nn.Linear(768, 192)
            self.activation = nn.GELU(approximate='none')
            self.fc2 = nn.Linear(192, 768)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, tokens):
        return tokens + self.fc2(self.activation(self.fc1(self.norm(tokens))))
