"""Exclusive, streamed float32 feature files with verified declared provenance.

Only NumPy .npy numeric arrays and JSON are used.  These files contain frozen
features, not labels; training labels remain in the separately checked manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .identity import cache_key


SCHEMA = 'dfagcn-pooled-cache-v1'
_CHANNELS = {'conv3': 256, 'conv4': 512, 'conv5': 512, 'fc': 4096}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _pool_hw(identity):
    cache_key(identity)  # Validate the complete preexisting identity schema.
    values = identity['feature_spec'].get('pooling_hw')
    if (not isinstance(values, (list, tuple)) or len(values) != 2
            or any(type(x) is not int or x < 1 for x in values)):
        raise ValueError('feature_spec.pooling_hw must contain two positive integers')
    return tuple(values)


def _shapes(identity):
    height, width = _pool_hw(identity)
    count = len(identity['samples'])
    return {name: (count, channels) if name == 'fc' else (count, height * width, channels)
            for name, channels in _CHANNELS.items()}


def _finite_chunks(values):
    return all(np.isfinite(values[start:start + 64]).all()
               for start in range(0, values.shape[0], 64))


def write_feature_cache(path, identity, batches):
    """Stream batches into new files; an unsuccessful directory stays incomplete.

    The final index is written only after every declared sample is present and
    each numeric file has been flushed and hashed. Existing output is refused.
    """
    shapes = _shapes(identity)
    canonical_identity = json.loads(json.dumps(identity, allow_nan=False))
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {name: np.lib.format.open_memmap(output / (name + '.npy'), mode='w+',
                                             dtype=np.float32, shape=shape)
              for name, shape in shapes.items()}
    written = 0
    try:
        for batch in batches:
            if not isinstance(batch, dict) or set(batch) != set(_CHANNELS):
                raise ValueError('each batch must contain exactly conv3, conv4, conv5 and fc')
            checked = {}
            for name, value in batch.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype != torch.float32 or value.device.type == 'meta':
                        raise TypeError('cache tensors must be numerical float32')
                    value = value.detach().cpu().numpy()
                if not isinstance(value, np.ndarray) or value.dtype != np.dtype('float32'):
                    raise TypeError('cache arrays must have float32 dtype')
                if value.ndim != len(shapes[name]) or value.shape[1:] != shapes[name][1:]:
                    raise ValueError(name + ' batch shape does not match extraction identity')
                if value.shape[0] < 1 or not np.isfinite(value).all():
                    raise ValueError('empty or nonfinite feature batch')
                checked[name] = value
            count = checked['fc'].shape[0]
            if any(value.shape[0] != count for value in checked.values()):
                raise ValueError('batch branches have different sample counts')
            if written + count > shapes['fc'][0]:
                raise ValueError('more samples than declared in cache identity')
            for name, value in checked.items():
                arrays[name][written:written + count] = value
            written += count
        if written != shapes['fc'][0]:
            raise ValueError('fewer samples than declared in cache identity')
        for array in arrays.values():
            array.flush()
    finally:
        # Partial numeric files are retained if a batch or generator fails.
        for array in arrays.values():
            array.flush()
    index = {'schema': SCHEMA, 'identity': canonical_identity,
             'key': cache_key(canonical_identity), 'arrays': {}}
    for name, shape in shapes.items():
        filename = name + '.npy'
        index['arrays'][name] = {'file': filename, 'shape': list(shape), 'dtype': 'float32',
                                 'sha256': sha256_file(output / filename)}
    with (output / 'index.json').open('x', encoding='utf-8') as stream:
        json.dump(index, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


class FeatureCache:
    """Read-only memory mapped cache; each returned sample owns writable storage.

    ``key`` binds declared extraction identity; ``content_sha256`` additionally
    binds this index and its numeric file digests for fitted-artifact identity.
    """
    def __init__(self, path, *, expected_identity):
        output = Path(path)
        raw = (output / 'index.json').read_bytes()
        index = json.loads(raw)
        if (not isinstance(index, dict) or set(index) != {'schema', 'identity', 'key', 'arrays'}
                or index['schema'] != SCHEMA):
            raise ValueError('unsupported or incomplete feature cache index')
        expected_key = cache_key(expected_identity)
        if index['key'] != expected_key or cache_key(index['identity']) != expected_key:
            raise ValueError('cache identity does not match the expected ordered samples/provenance')
        shapes = _shapes(expected_identity)
        if not isinstance(index['arrays'], dict) or set(index['arrays']) != set(shapes):
            raise ValueError('cache index does not declare all expected arrays')
        arrays = {}
        for name, shape in shapes.items():
            declared = index['arrays'][name]
            if (not isinstance(declared, dict)
                    or set(declared) != {'file', 'shape', 'dtype', 'sha256'}
                    or declared['file'] != name + '.npy'
                    or declared['shape'] != list(shape) or declared['dtype'] != 'float32'):
                raise ValueError('cache array metadata is inconsistent with identity')
            file_path = output / declared['file']
            if not file_path.is_file() or file_path.is_symlink():
                raise ValueError('cache array must be an existing regular file')
            if sha256_file(file_path) != declared['sha256']:
                raise ValueError('cache array SHA256 mismatch: ' + name)
            values = np.load(file_path, mmap_mode='r', allow_pickle=False)
            if values.shape != shape or values.dtype != np.dtype('float32'):
                raise ValueError('cache numeric array has incompatible shape or dtype')
            if not _finite_chunks(values):
                raise ValueError('cache numeric array contains nonfinite values')
            arrays[name] = values
        self.path = output
        self.identity = json.loads(json.dumps(index['identity']))
        self.key = expected_key
        self.content_sha256 = hashlib.sha256(raw).hexdigest()
        self.arrays = arrays

    def __len__(self):
        return len(self.identity['samples'])

    def __getitem__(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError('feature cache sample index out of range')
        return {name: torch.from_numpy(np.array(values[index], copy=True))
                for name, values in self.arrays.items()}
