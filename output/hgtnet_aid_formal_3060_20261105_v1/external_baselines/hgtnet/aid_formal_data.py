"""Original full AID assignments, byte-safe decoding and explicit final-test gate."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import PurePosixPath

from .aid_data import inspector, MANIFEST, MANIFEST_SHA, DATA_ROOT

COUNTS = dict(train=4000, val=1000, pool=5000, test=5000)


def _records(records, *, allow_test=False):
    if type(allow_test) is not bool:
        raise ValueError('Explicit boolean final-test authorization required')
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError('Nonempty AID records required')
    checked, paths, labels, names, pixels = [], set(), {}, {}, {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('AID records must be objects')
        path, label, split = row.get('path'), row.get('label'), row.get('split')
        parts = PurePosixPath(path).parts if isinstance(path, str) else ()
        if (not isinstance(path, str) or not path or '\\' in path or '\x00' in path
                or path.startswith('/') or len(parts) != 2
                or any(part in ('.', '..') for part in parts)
                or PurePosixPath(path).as_posix() != path or path in paths
                or type(label) is not int or not 0 <= label < 30
                or split not in ('train', 'val', 'test') or (split == 'test' and not allow_test)
                or not inspector._valid_sha(row.get('sha256'))
                or not inspector._valid_sha(row.get('pixel_sha256'))):
            raise ValueError('Unsafe AID path/label/hash or unauthorized test record')
        name, pixel = parts[0], row['pixel_sha256']
        if (label in labels and labels[label] != name) or (name in names and names[name] != label):
            raise ValueError('AID class/label mapping changed')
        if pixel in pixels and pixels[pixel] != (label, split):
            raise ValueError('AID pixel group crosses classes or splits')
        labels[label], names[name], pixels[pixel] = name, label, (label, split)
        paths.add(path)
        checked.append(deepcopy(row))
    return checked


def prepare_inputs(manifest_path, *, expected_sha256=MANIFEST_SHA):
    if not inspector._valid_sha(expected_sha256):
        raise ValueError('Complete original AID manifest SHA required')
    with inspector.open_regular_readonly(manifest_path) as stream:
        raw = stream.read()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('Original AID manifest SHA256 mismatch')
    try:
        value = json.loads(raw, object_pairs_hook=inspector._unique_json_object,
                           parse_constant=inspector._reject_json_constant)
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Original AID manifest must be strict JSON') from exc
    if not isinstance(value, dict):
        raise ValueError('Original AID manifest must be an object')
    errors, _, _ = inspector.validate_manifest(value)
    if errors:
        raise ValueError('Original AID manifest metadata invalid: ' + '; '.join(errors))
    rows = _records(sorted(value['records'], key=lambda row: row['path']), allow_test=True)
    selected = {split: [deepcopy(row) for row in rows if row['split'] == split]
                for split in ('train', 'val', 'test')}
    selected['pool'] = [deepcopy(row) for row in rows if row['split'] in ('train', 'val')]
    if {name: len(group) for name, group in selected.items()} != COUNTS:
        raise ValueError('Original AID full split counts changed')
    return dict(class_names=deepcopy(value['class_names']), counts=dict(COUNTS),
                selected=selected, manifest_sha256=actual)


def _verified_bytes(root, row):
    # Retain lexical parents so no-follow traversal cannot hide a symlink.
    path = inspector.absolute_path(root) / row['path']
    with inspector.open_regular_readonly(path) as stream:
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('AID image SHA256 mismatch: ' + row['path'])
    return raw


def validate_selected(root, records, *, allow_test=False):
    rows = _records(records, allow_test=allow_test)
    total_bytes = test_bytes = 0
    for row in rows:
        count = len(_verified_bytes(root, row))
        total_bytes += count
        if row['split'] == 'test':
            test_bytes += count
    return dict(checked_count=len(rows), verified_count=len(rows),
                test_image_bytes_read=test_bytes, verified_bytes=total_bytes)


def make_dataset(root, records, training=False, *, allow_test=False):
    if type(training) is not bool:
        raise ValueError('training must be boolean')
    rows = _records(records, allow_test=allow_test)
    if training and any(row['split'] == 'test' for row in rows):
        raise ValueError('Test images cannot enter training or refit')
    root = inspector.absolute_path(root)
    validate_selected(root, rows, allow_test=allow_test)
    import torch
    from PIL import Image
    from torchvision import transforms

    pipeline = [transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)]
    pipeline += ([transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=.5)]
                 if training else [transforms.CenterCrop(224)])
    pipeline += [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])]
    transform = transforms.Compose(pipeline)

    class FormalAidDataset(torch.utils.data.Dataset):
        paths = tuple(row['path'] for row in rows)
        labels = tuple(row['label'] for row in rows)

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
                image = transform(source)
            if image.shape != (3, 224, 224) or image.dtype != torch.float32 or not torch.isfinite(image).all().item():
                raise ValueError('AID formal transform produced invalid FP32 RGB224')
            return image, torch.tensor(row['label'], dtype=torch.long)

    return FormalAidDataset()
