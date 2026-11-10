"""Frozen 360/90 NWPU inputs for the independent HGTNet engineering smoke.

Inspection, subset selection, byte verification, and ZIP recovery use only the
standard library.  The immutable existing NWPU helper is loaded by file location;
its command-line entry point is never invoked.  Image dependencies are imported
only when a real dataset is requested.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MANIFEST = Path('/home/featurize/work/nwpu_protocol_20260927_02.json')
MANIFEST_SHA = 'ce0cfef201c6aee507bc978a9d319e42b5d7d4ba3d461d61ffb74f7c2259840e'
DATA_ROOT = Path('/home/featurize/data/NWPU-RESISC45')
ARCHIVE = Path('/home/featurize/work/NWPU-RESISC45_a826b44.zip')
ARCHIVE_SHA = 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53'

_HELPER = None
_HELPER_NAME = '_gavit_hgtnet_nwpu_stdlib_helper'


def _helper():
    """Reuse one safely registered module without importing a training package."""
    global _HELPER
    if _HELPER is None:
        path = Path(__file__).resolve().parents[2] / 'tools' / 'run_dfagcn_nwpu_smoke.py'
        existing = sys.modules.get(_HELPER_NAME)
        if existing is not None:
            if Path(existing.__file__).resolve() != path:
                raise RuntimeError('NWPU helper module cache points to a different source file')
            _HELPER = existing
        else:
            spec = importlib.util.spec_from_file_location(_HELPER_NAME, path)
            if spec is None or spec.loader is None:
                raise ImportError('the frozen stdlib NWPU helper cannot be loaded')
            module = importlib.util.module_from_spec(spec)
            sys.modules[_HELPER_NAME] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(_HELPER_NAME, None)
                raise
            _HELPER = module
    return _HELPER


def _records(records):
    """Copy and check selected records before any image is read or recovered."""
    helper = _helper()
    checked = []
    paths = set()
    labels = {}
    classes = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError('selected NWPU records must be mappings')
        try:
            relative = helper._relative(row['path'])
            label, split = row['label'], row['split']
            helper._sha(row['sha256'])
        except (KeyError, TypeError) as exc:
            raise ValueError('selected NWPU record is incomplete') from exc
        if (len(relative.parts) != 2 or type(label) is not int or not 0 <= label < 45
                or split not in ('train', 'val') or row['path'] in paths):
            raise ValueError('selected NWPU path/label/split is invalid; test images are forbidden')
        name = relative.parts[0]
        if (label in labels and labels[label] != name) or (name in classes and classes[name] != label):
            raise ValueError('selected NWPU class/label mapping is inconsistent')
        labels[label] = name
        classes[name] = label
        paths.add(row['path'])
        checked.append(dict(row))
    if not checked:
        raise ValueError('selected NWPU records must not be empty')
    return checked


def prepare_inputs(manifest_path, *, expected_sha256=MANIFEST_SHA):
    """Validate all 31,500 manifest rows, then select fixed train360/val90."""
    helper = _helper()
    manifest = helper.load_manifest(manifest_path, expected_sha256=expected_sha256)
    selected = helper.select_smoke(manifest)
    _records(selected['pool'])
    return {'class_names': list(manifest['class_names']), 'selected': selected,
            'counts': {'train': 360, 'val': 90, 'pool': 450, 'test': 0},
            'manifest_sha256': expected_sha256}


def validate_selected(root, records):
    """Validate selected train/validation paths and their full image digests."""
    return _helper().validate_selected(root, _records(records))


def recover_selected(archive, records, destination, *, expected_sha256=ARCHIVE_SHA):
    """Recover selected train/validation images to a fresh, verified directory."""
    return _helper().recover_selected(archive, _records(records), destination,
                                      expected_sha256=expected_sha256)


def make_dataset(root, records):
    """Read verified real RGB images with fixed bilinear224/ImageNet transforms.

    The input records are copied, preserving their order.  Bytes are verified
    before construction and on each access, so a changed image cannot silently
    enter a training/validation batch after preflight.
    """
    rows = _records(records)
    root = Path(root).resolve()
    _helper().validate_selected(root, rows)

    import torch
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class SelectedNwpuDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            _helper().validate_selected(root, [row])
            with Image.open(root / row['path']) as source:
                image = transform(source.convert('RGB'))
            if image.shape != (3, 224, 224) or not bool(torch.isfinite(image).all()):
                raise ValueError('NWPU image transform produced an invalid tensor')
            return image, torch.tensor(row['label'], dtype=torch.long)

    return SelectedNwpuDataset()
