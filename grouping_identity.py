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


_SPATIAL_SOFT = {
    **_SOFT,
    'assignment_version': 'local_prototype_cosine_spatial_v1',
    'spatial_prior': 'anchor_gaussian_v1',
    'spatial_prior_metric': 'squared_euclidean',
    'spatial_prior_centers': 'fixed_membership_centroids',
    'spatial_prior_scale_token_units': 1.75,
    'spatial_prior_strength': 0.5,
    'spatial_prior_position': 'post_temperature',
}


def soft_assignment_identity(spatial_prior=None) -> dict:
    if spatial_prior is None:
        return dict(_SOFT)
    if spatial_prior == 'anchor_gaussian_v1':
        return dict(_SPATIAL_SOFT)
    raise ValueError(f'Unsupported spatial_prior: {spatial_prior!r}')


def validate_soft_assignment(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError('soft_assignment must contain the complete versioned configuration')
    expected_config = (_SPATIAL_SOFT if config.get('assignment_version') ==
                       _SPATIAL_SOFT['assignment_version'] else _SOFT)
    if config.keys() != expected_config.keys():
        raise ValueError('soft_assignment must contain the complete versioned configuration')
    for key, expected in expected_config.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f'Unsupported soft_assignment {key}={config[key]!r}; expected {expected!r}')
    return dict(config)


def spatial_prior_name(architecture_config: dict):
    config = architecture_config.get('soft_assignment')
    return None if config is None else validate_soft_assignment(config).get('spatial_prior')


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
