"""Versioned soft-grouping identity, usable without importing Torch."""
from copy import deepcopy

_SOFT = {
    'assignment_version': 'local_prototype_cosine_v1',
    'assignment_dim': 128, 'assignment_radius': 1,
    'assignment_metric': 'chebyshev', 'assignment_temperature': 0.1,
    'assignment_iterations': 1,
    'assignment_projection': 'shared_ln_linear_no_bias',
    'assignment_init': 'xavier_uniform',
    'assignment_norm_eps': 1e-6, 'assignment_ln_eps': 1e-5,
    'prototype_pooling': 'fixed_region_attention',
    'region_pooling': 'attentive_log_membership',
    'feedback_mapping': 'soft_membership',
    'graph_node_geometry': 'anchor_index_prior',
}


def soft_assignment_identity() -> dict:
    return dict(_SOFT)


def validate_soft_assignment(config: dict) -> dict:
    if not isinstance(config, dict) or config.keys() != _SOFT.keys():
        raise ValueError('soft_assignment must contain the complete versioned configuration')
    for key, expected in _SOFT.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f'Unsupported soft_assignment {key}={config[key]!r}; expected {expected!r}')
    return dict(config)


def complete_grouping_architecture(arch: dict, *, checkpoint: bool = False) -> dict:
    if not isinstance(arch, dict):
        raise ValueError('Grouping architecture must be a dictionary')
    result = deepcopy(arch)
    if result.get('grouping') != 'soft_prototype':
        if 'soft_assignment' in result:
            raise ValueError('soft_assignment is only valid for grouping=soft_prototype')
        return result
    for key, expected in {'num_regions': 16, 'edge_type': 'sparse_hybrid',
                          'integration': 'token_feedback'}.items():
        if type(result.get(key)) is not type(expected) or result.get(key) != expected:
            raise ValueError(f'soft_prototype requires {key}={expected!r}')
    if 'soft_assignment' not in result:
        if checkpoint:
            raise ValueError('Checkpoint is missing soft_assignment configuration')
        result['soft_assignment'] = soft_assignment_identity()
    result['soft_assignment'] = validate_soft_assignment(result['soft_assignment'])
    return result
