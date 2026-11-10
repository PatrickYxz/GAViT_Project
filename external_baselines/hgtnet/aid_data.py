"""Frozen AID train240/val60 inputs; imports remain standard-library-only.

The original manifest supplies assignments and duplicate groups. No split is
recomputed, no image is deduplicated, and test image bytes are never opened.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys


def _load_inspector():
    """Load the bundled trusted file, unaffected by third-party tools packages."""
    path = Path(__file__).resolve().parents[2] / 'tools' / 'inspect_hgtnet_aid_inputs.py'
    name = '_gavit_hgtnet_aid_stdlib_inspector'
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != path:
            raise RuntimeError('AID inspector cache points to a different source')
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError('Bundled AID inspector cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


inspector = _load_inspector()


MANIFEST_SHA = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
MANIFEST = '/home/featurize/work/aid_protocol_20260920_03.json'
DATA_ROOT = '/home/featurize/data/AID/AID'


def _records(records):
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('Nonempty selected AID records required')
    checked, seen, labels, names, pixels = [], set(), {}, {}, {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('Selected AID records must be objects')
        path, label, split = row.get('path'), row.get('label'), row.get('split')
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if (not isinstance(path, str) or not path or '\\' in path or '\x00' in path
                or path.startswith('/') or len(parts) < 2
                or any(part in ('.', '..') for part in parts)
                or PurePosixPath(path).as_posix() != path or path in seen
                or type(label) is not int or not 0 <= label < 30
                or split not in ('train', 'val')
                or not inspector._valid_sha(row.get('sha256'))
                or not inspector._valid_sha(row.get('pixel_sha256'))):
            raise ValueError('Unsafe AID path/label/hash/split; test image reads are forbidden')
        name = parts[0]
        if (label in labels and labels[label] != name) or (name in names and names[name] != label):
            raise ValueError('Selected AID class/label mapping differs')
        pixel = row['pixel_sha256']
        if pixel in pixels and pixels[pixel] != (label, split):
            raise ValueError('AID pixel group crosses classes or splits')
        labels[label], names[name], pixels[pixel] = name, label, (label, split)
        seen.add(path)
        checked.append(deepcopy(row))
    return checked


def prepare_inputs(manifest_path, *, expected_sha256=MANIFEST_SHA):
    """Verify the original bytes/metadata and copy its sorted 8/2 per class."""
    if not inspector._valid_sha(expected_sha256):
        raise ValueError('A complete expected AID manifest SHA256 is required')
    with inspector.open_regular_readonly(manifest_path) as stream:
        raw = stream.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Original AID manifest SHA256 mismatch')
    try:
        value = json.loads(raw, object_pairs_hook=inspector._unique_json_object,
                           parse_constant=inspector._reject_json_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Original AID manifest is not strict JSON') from exc
    errors, by_class, _ = inspector.validate_manifest(value)
    if errors:
        raise ValueError('Original AID manifest metadata invalid: ' + '; '.join(errors))
    selected = {}
    for split, count in [('train', 8), ('val', 2)]:
        rows = [row for label in range(30)
                for row in sorted(by_class[label][split], key=lambda row: row['path'])[:count]]
        selected[split] = _records(sorted(rows, key=lambda row: row['path']))
    selected['pool'] = _records(sorted(deepcopy(selected['train'] + selected['val']),
                                       key=lambda row: row['path']))
    return {'class_names': deepcopy(value['class_names']), 'selected': selected,
            'counts': {'train': 240, 'val': 60, 'test': 0}, 'manifest_sha256': actual}


def _verified_bytes(root, row):
    # Do not resolve: doing so would hide a symlink before the no-follow opener.
    path = inspector.absolute_path(root) / row['path']
    with inspector.open_regular_readonly(path) as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Selected AID image SHA256 mismatch: ' + row['path'])
    return raw


def validate_selected(root, records):
    """Hash exactly the supplied safe train/validation rows, raising on failure."""
    rows = _records(records)
    total_bytes = 0
    for row in rows:
        total_bytes += len(_verified_bytes(root, row))
    return {'checked_count': len(rows), 'verified_count': len(rows),
            'test_image_bytes_read': 0, 'verified_bytes': total_bytes}


def make_dataset(root, records):
    """Decode the same no-follow-opened bytes that passed the image byte SHA.

    Original RGB600 images are resized bilinearly to 224 and normalized as
    float32 ImageNet tensors. Exposed identities are copies, preserving order.
    """
    rows = _records(records)
    root = inspector.absolute_path(root)
    validate_selected(root, rows)

    import torch
    from PIL import Image

    mean = torch.tensor([.485, .456, .406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([.229, .224, .225], dtype=torch.float32).view(3, 1, 1)

    class SelectedAidDataset(torch.utils.data.Dataset):
        @property
        def records(self):
            return deepcopy(rows)

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            raw = _verified_bytes(root, row)
            with Image.open(io.BytesIO(raw)) as source:
                if source.mode != 'RGB' or source.size != (600, 600):
                    raise ValueError('Original AID images must be RGB600x600: ' + row['path'])
                resized = source.resize((224, 224), resample=Image.Resampling.BILINEAR)
                tensor = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            tensor = tensor.reshape(224, 224, 3).permute(2, 0, 1).to(dtype=torch.float32).div_(255)
            tensor = (tensor - mean) / std
            if tensor.shape != (3, 224, 224) or not torch.isfinite(tensor).all().item():
                raise ValueError('Selected AID preprocessing produced invalid RGB224')
            return tensor, torch.tensor(row['label'], dtype=torch.long)

    return SelectedAidDataset()
