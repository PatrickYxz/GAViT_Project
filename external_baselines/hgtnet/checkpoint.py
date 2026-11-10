"""Strict, source-bound checkpoints for local HGTNet engineering checks.

These files contain model tensors and descriptive provenance only. They are not
optimizer resume snapshots or evidence of remote-sensing dataset performance.
No destination is overwritten and no untrusted pickle classes are loaded.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import re

import torch
from torch import nn

from .config import config_sha256, validate_config


SOURCE_FILES = ('attention.py', 'config.py', 'graph.py', 'model.py', 'checkpoint.py', 'pretrained.py')
IDENTITY = {
    'schema': 'hgtnet-local-checkpoint-v1',
    'implementation': 'hgtnet_independent_v0',
    'official': False,
    'scope': 'local_engineering_only',
    'synthetic_inputs': True,
    'formal_training_started': False,
    'accuracy_metrics': None,
}
PAYLOAD_KEYS = set(IDENTITY) | {'config', 'config_sha256', 'source_identity', 'pretraining', 'state_dict'}
PRETRAINING_KEYS = {
    'schema_version', 'candidate_kind', 'is_original_hgtnet_weights',
    'source_identity_verified', 'mode', 'compatible', 'missing', 'unexpected',
    'shape_mismatch', 'dtype_mismatch', 'nonfinite', 'invalid',
    'intentionally_initialized', 'source_intentionally_skipped', 'coverage',
    'can_apply', 'loaded', 'source_validation', 'source',
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_identity() -> dict[str, str]:
    """Hash all local model-defining modules; missing files fail explicitly."""
    folder = Path(__file__).resolve().parent
    return {name: _sha256_bytes((folder / name).read_bytes()) for name in SOURCE_FILES}


def _full_sha256(value, name='expected_sha256') -> str:
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(name + ' must be a full lowercase hexadecimal SHA256')
    return value


def _json_provenance(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('pretraining provenance must be a JSON object or None')
    if set(value) - PRETRAINING_KEYS:
        raise ValueError('pretraining provenance has unknown report fields')
    if 'is_original_hgtnet_weights' in value and value['is_original_hgtnet_weights'] is not False:
        raise ValueError('candidate initialization is not original HGTNet weights')
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError('provenance JSON object keys must be strings')
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif item is not None and type(item) not in (str, int, float, bool):
            raise ValueError('provenance must contain JSON values only')
    try:
        check(value)
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError('pretraining provenance must be finite JSON data') from exc


def _model_config(model):
    if not isinstance(model, nn.Module):
        raise TypeError('model must be a torch.nn.Module')
    config = validate_config(getattr(model, 'config', None))
    identity = config_sha256(config)
    if getattr(model, 'config_sha256', None) != identity:
        raise ValueError('model config changed after initialization')
    return config, identity


def _state_mapping(state, *, expected=None):
    if not isinstance(state, Mapping) or any(not isinstance(key, str) for key in state):
        raise ValueError('state_dict must be a string-keyed mapping')
    if expected is not None and set(state) != set(expected):
        raise ValueError('state_dict has missing or unexpected keys')
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError('state_dict values must be tensors: ' + key)
        if value.is_meta or value.layout != torch.strided or value.is_quantized:
            raise ValueError('state_dict tensors must be actual dense values: ' + key)
        if not torch.isfinite(value).all().item():
            raise ValueError('state_dict contains nonfinite values: ' + key)
        if expected is not None:
            destination = expected[key]
            if value.shape != destination.shape:
                raise ValueError('state_dict shape mismatch: ' + key)
            if value.dtype != destination.dtype:
                raise ValueError('state_dict dtype mismatch: ' + key)
            if destination.is_meta or destination.layout != torch.strided or destination.is_quantized:
                raise ValueError('target must have actual dense tensors: ' + key)


def _report(path, digest, payload):
    return {
        **IDENTITY,
        'path': str(Path(path).resolve()),
        'sha256': digest,
        'config_sha256': payload['config_sha256'],
        'source_identity': dict(payload['source_identity']),
        'pretraining': _json_provenance(payload['pretraining']),
        'state_tensors': len(payload['state_dict']),
        'state_numel': sum(value.numel() for value in payload['state_dict'].values()),
    }


def save_checkpoint(model, path, *, pretraining=None):
    """Exclusively create a new CPU checkpoint after validating every tensor."""
    config, identity = _model_config(model)
    provenance = _json_provenance(pretraining)
    sources = source_identity()
    state = model.state_dict()
    _state_mapping(state)
    # Clone all values before opening the output; a failed copy cannot create a
    # partial file. CPU tensors are portable and do not alias the live model.
    snapshot = {key: value.detach().to(device='cpu').clone() for key, value in state.items()}
    payload = {
        **IDENTITY, 'config': config, 'config_sha256': identity,
        'source_identity': sources, 'pretraining': provenance, 'state_dict': snapshot,
    }
    target = Path(path)
    created = False
    try:
        with target.open('xb') as handle:
            created = True
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if created:
            target.unlink(missing_ok=True)
        raise
    digest = _sha256_bytes(target.read_bytes())
    report = _report(target, digest, payload)
    report.update(mode='saved', loaded=False, saved_tensors=len(payload['state_dict']))
    return report


def load_checkpoint(model, path, *, expected_sha256):
    """Validate identity and all values before copying any checkpoint tensor.

    Exact bytes are hashed before safe deserialization. The entire state is
    checked and converted before load_state_dict; malformed metadata, missing
    keys, invalid tensors and allocation errors cannot partially load a model.
    """
    expected_digest = _full_sha256(expected_sha256)
    config, identity = _model_config(model)
    raw = Path(path).read_bytes()
    actual_digest = _sha256_bytes(raw)
    if actual_digest != expected_digest:
        raise ValueError('checkpoint file SHA256 mismatch')
    try:
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError('checkpoint is not a safe tensor checkpoint') from exc
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError('checkpoint schema fields are missing or unknown')
    for key, expected in IDENTITY.items():
        if type(payload[key]) is not type(expected) or payload[key] != expected:
            raise ValueError('checkpoint identity mismatch: ' + key)
    incoming_config = validate_config(payload['config'])
    if incoming_config != config or payload['config_sha256'] != identity:
        raise ValueError('checkpoint and model configs differ')
    if config_sha256(incoming_config) != payload['config_sha256']:
        raise ValueError('checkpoint config SHA256 mismatch')
    if payload['source_identity'] != source_identity():
        raise ValueError('checkpoint source identity differs from current implementation')
    _json_provenance(payload['pretraining'])
    destination = model.state_dict()
    _state_mapping(payload['state_dict'], expected=destination)
    copied = {
        key: value.detach().to(device=destination[key].device).clone()
        for key, value in payload['state_dict'].items()
    }
    model.load_state_dict(copied, strict=True)
    report = _report(path, actual_digest, payload)
    report['mode'] = 'loaded'
    report['loaded'] = True
    report['loaded_tensors'] = len(payload['state_dict'])
    return report
