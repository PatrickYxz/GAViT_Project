"""Deterministic independent VGG preprocessing and pooled feature extraction.

The RGB/ImageNet preprocessing below is an explicit independent recipe. It is
not a claim of numerical equivalence to the paper's MatConvNet checkpoint.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import vgg16
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .cache import FeatureCache, _pool_hw, sha256_file, write_feature_cache
from .features import VGG16FeatureAdapter
from .identity import cache_key


PREPROCESSING = {'resize_mode': 'bilinear_stretch', 'channels': 'RGB',
                 'mean': [.485, .456, .406], 'std': [.229, .224, .225],
                 'fc_hw': 224, 'conv_hw': 448, 'augmentation': False}


def random_vgg_adapter(seed):
    """A real numerical VGG for engineering smoke; never a pretrained result."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = vgg16(weights=None)
    backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:2])
    return VGG16FeatureAdapter(backbone)


def adapter_sha256(adapter):
    """Hash actual extraction tensors, including their names, shapes and dtypes.

    This semantic tensor fingerprint is only the random-smoke identity. Formal
    input weight files use their separate file-byte SHA256.
    """
    if not isinstance(adapter, VGG16FeatureAdapter):
        raise TypeError('adapter must be VGG16FeatureAdapter')
    digest = hashlib.sha256()
    for name, tensor in sorted(adapter.state_dict().items()):
        if tensor.device.type == 'meta':
            raise ValueError('cannot fingerprint meta weights')
        values = tensor.detach().cpu().contiguous().numpy()
        descriptor = json.dumps([name, list(values.shape), str(values.dtype)], separators=(',', ':'))
        digest.update(descriptor.encode('utf-8'))
        digest.update(b'\0')
        digest.update(memoryview(values).cast('B'))
    return digest.hexdigest()


def load_vgg_adapter(weights_path, expected_sha256, *, device='cpu'):
    """Load used layers from a supplied torchvision-layout VGG state dictionary.

    Safe tensor-only deserialization is used. Unused FC7/FC8 from the standard
    checkpoint are accepted and excluded; arbitrary extra keys are rejected.
    There is no download or implied ImageNet/MatConvNet source authentication.
    """
    path = Path(weights_path)
    if sha256_file(path) != expected_sha256:
        raise ValueError('VGG weight file SHA256 does not match expected_sha256')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError('VGG checkpoint must be a plain state dictionary')
    with torch.device('meta'):
        backbone = vgg16(weights=None)
    backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:2])
    expected = set(backbone.state_dict())
    allowed_unused = {'classifier.3.weight', 'classifier.3.bias',
                      'classifier.6.weight', 'classifier.6.bias'}
    if set(state) - expected - allowed_unused or not expected.issubset(state):
        raise ValueError('VGG checkpoint has missing or incompatible extraction keys')
    used = {key: state[key] for key in expected}
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
           or not torch.isfinite(value).all() for value in used.values()):
        raise ValueError('VGG checkpoint extraction parameters must be finite float32 tensors')
    backbone.load_state_dict(used, strict=True, assign=True)
    return VGG16FeatureAdapter(backbone).to(device)


def _preprocess(image, size):
    image = TF.resize(image, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), PREPROCESSING['mean'], PREPROCESSING['std'])


def extract_cache(adapter, root, records, identity, output, *, pool_hw, batch_size=1, device='cpu'):
    """Verify ordered source files and stream raw frozen VGG features to disk."""
    cache_key(identity)
    if tuple(pool_hw) != _pool_hw(identity):
        raise ValueError('pool_hw differs from extraction identity')
    preprocessing = identity['preprocessing']
    if any(preprocessing.get(name) != value for name, value in PREPROCESSING.items()):
        raise ValueError('preprocessing declaration differs from implemented RGB recipe')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    selected = [{'path': row['path'], 'sha256': row['sha256']} for row in records]
    if selected != identity['samples']:
        raise ValueError('record order or file digests differ from extraction identity')
    root = Path(root).resolve()
    paths = []
    for record in records:
        path = root / record['path']
        if not path.resolve().is_relative_to(root) or sha256_file(path) != record['sha256']:
            raise ValueError('source image path or SHA256 differs from declared record')
        paths.append(path)
    if not isinstance(adapter, VGG16FeatureAdapter):
        raise TypeError('adapter must be a real VGG16FeatureAdapter')
    adapter = adapter.to(device)

    def batches():
        for start in range(0, len(paths), batch_size):
            fc_images, conv_images = [], []
            for offset, path in enumerate(paths[start:start + batch_size], start):
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != records[offset]['sha256']:
                    raise ValueError('source image SHA256 changed during extraction')
                # Decode the exact bytes just hashed, rather than reopening a
                # pathname which could have changed after preflight verification.
                with Image.open(io.BytesIO(raw)) as image:
                    if image.mode != 'RGB':
                        raise ValueError('source images must have RGB mode')
                    image.load()
                    fc_images.append(_preprocess(image, 224))
                    conv_images.append(_preprocess(image, 448))
            with torch.no_grad():
                features = adapter(torch.stack(fc_images).to(device), torch.stack(conv_images).to(device))
                result = {'fc': features['fc'].cpu()}
                for name in ('conv3', 'conv4', 'conv5'):
                    pooled = F.adaptive_avg_pool2d(features[name], tuple(pool_hw))
                    result[name] = pooled.flatten(2).transpose(1, 2).contiguous().cpu()
            yield result
    write_feature_cache(output, identity, batches())
    return FeatureCache(output, expected_identity=identity)
