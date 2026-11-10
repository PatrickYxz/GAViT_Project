"""Frozen full-NWPU data for calibration/refit/test, with lazy image imports.

This module may select test identities without reading image bytes. The caller
defers test recovery/dataset creation until the frozen refit model is ready.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
import zipfile

from .nwpu_data import _helper, MANIFEST, MANIFEST_SHA, DATA_ROOT, ARCHIVE, ARCHIVE_SHA

COUNTS = {'train': 5040, 'val': 1260, 'pool': 6300, 'test': 25200}


def _records(records):
    helper = _helper()
    checked, paths, labels, classes = [], set(), {}, {}
    for row in records:
        if not isinstance(row, dict): raise ValueError('Formal NWPU records must be mappings')
        try:
            relative = helper._relative(row['path'])
            label, split = row['label'], row['split']
            helper._sha(row['sha256'])
        except (KeyError, TypeError) as exc:
            raise ValueError('Incomplete formal NWPU record') from exc
        if (len(relative.parts) != 2 or type(label) is not int or not 0 <= label < 45
                or split not in ('train', 'val', 'test') or row['path'] in paths):
            raise ValueError('Invalid or duplicate formal NWPU path/label/split')
        name = relative.parts[0]
        if (label in labels and labels[label] != name) or (name in classes and classes[name] != label):
            raise ValueError('Inconsistent formal NWPU class/label mapping')
        labels[label] = name; classes[name] = label; paths.add(row['path'])
        checked.append(copy.deepcopy(row))
    if not checked: raise ValueError('Formal NWPU selection must not be empty')
    return checked


def prepare_inputs(manifest_path, *, expected_sha256=MANIFEST_SHA):
    """Check all31500 identities, returning independent ordered split snapshots."""
    manifest = _helper().load_manifest(manifest_path, expected_sha256=expected_sha256)
    rows = _records(sorted(manifest['records'], key=lambda row: row['path']))
    selected = {split: [copy.deepcopy(row) for row in rows if row['split'] == split]
                for split in ('train', 'val', 'test')}
    selected['pool'] = [copy.deepcopy(row) for row in rows if row['split'] in ('train', 'val')]
    if {key: len(value) for key, value in selected.items()} != COUNTS:
        raise ValueError('Formal frozen split counts changed')
    return dict(class_names=list(manifest['class_names']), counts=dict(COUNTS), selected=selected,
                manifest_sha256=expected_sha256)


def _image_path(root, row):
    relative = _helper()._relative(row['path'])
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink(): raise ValueError('Formal NWPU image path contains a symlink: ' + row['path'])
    if not path.is_file() or not path.resolve().is_relative_to(root):
        raise ValueError('Selected image missing or outside data root: ' + row['path'])
    return path


def _root(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('Formal data root must be an actual directory')
    return root.resolve()


def _read_image(root, row):
    raw = _image_path(root, row).read_bytes()
    if hashlib.sha256(raw).hexdigest() != row['sha256']:
        raise ValueError('Formal image SHA256 mismatch: ' + row['path'])
    return raw


def validate_selected(root, records):
    """Read and hash only supplied selected paths; supports explicit test phase."""
    rows, root = _records(records), _root(root)
    for row in rows: _read_image(root, row)
    return True


def _receipt(records, expected_sha256):
    identities = [{key: row[key] for key in ('path', 'label', 'split', 'sha256')} for row in records]
    selection = json.dumps(identities, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    counts = dict(Counter(row['split'] for row in records))
    return dict(schema='hgtnet-formal-recovery-v1', archive_sha256=expected_sha256,
                selection_sha256=hashlib.sha256(selection).hexdigest(), selected_images=len(records),
                split_counts=counts, test_images_extracted=counts.get('test', 0), complete=True)


def recover_selected(archive, records, destination, *, expected_sha256=ARCHIVE_SHA, resume=False):
    """Recover exact selected members; reuse an owned verified destination only on resume."""
    if type(resume) is not bool: raise ValueError('resume must be boolean')
    rows = _records(records); _helper()._sha(expected_sha256)
    archive, destination = Path(archive), Path(destination)
    existing = destination.exists() or destination.is_symlink()
    if existing and not resume: raise FileExistsError(destination)
    if destination.is_symlink(): raise ValueError('Recovered destination must not be a symlink')
    receipt = _receipt(rows, expected_sha256)
    if archive.is_symlink() or not archive.is_file() or _helper().sha256_file(archive) != expected_sha256:
        raise ValueError('Persistent NWPU ZIP SHA256 mismatch or invalid file')
    if existing:
        recorded = destination / 'recovery.json'
        if not destination.is_dir() or recorded.is_symlink() or not recorded.is_file():
            raise ValueError('Existing recovery directory lacks an owned complete receipt')
        try: actual = json.loads(recorded.read_text())
        except (OSError, ValueError) as exc: raise ValueError('Invalid recovery receipt') from exc
        if actual != receipt: raise ValueError('Recovery receipt selection/source/count mismatch')
        validate_selected(destination, rows)
        return destination
    wanted = {row['path']: row for row in rows}
    with zipfile.ZipFile(archive) as source:
        members, seen = {}, set()
        for info in source.infolist():
            if info.is_dir():
                _helper()._relative(info.filename.rstrip('/')); continue
            name = str(_helper()._relative(info.filename))
            if name in seen or ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Duplicate or symlink ZIP member')
            seen.add(name)
            relative = name[len('NWPU-RESISC45/'):] if name.startswith('NWPU-RESISC45/') else name
            if relative in wanted:
                if relative in members or info.flag_bits & 1:
                    raise ValueError('Ambiguous or encrypted selected ZIP member')
                members[relative] = info
        if set(members) != set(wanted): raise ValueError('Persistent ZIP lacks selected formal images')
        destination.mkdir(parents=True, exist_ok=False)
        for relative, row in wanted.items():
            raw = source.read(members[relative])
            if hashlib.sha256(raw).hexdigest() != row['sha256']:
                raise ValueError('Selected archive image SHA256 mismatch: ' + relative)
            path = destination / relative; path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream: stream.write(raw)
    validate_selected(destination, rows)
    recorded = destination / 'recovery.json'
    temporary = recorded.with_name(recorded.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, recorded)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def make_dataset(root, records, training=False):
    """Fixed engineering crop/flip policy; hash and decode the identical raw bytes."""
    if type(training) is not bool: raise ValueError('training must be boolean')
    rows, root = _records(records), _root(root)
    if training and any(row['split'] == 'test' for row in rows):
        raise ValueError('Test records cannot enter training or refit')
    validate_selected(root, rows)
    import torch
    from PIL import Image
    from torchvision import transforms
    pipeline = [transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR)]
    pipeline += ([transforms.RandomCrop(224), transforms.RandomHorizontalFlip(p=0.5)]
                 if training else [transforms.CenterCrop(224)])
    pipeline += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    transform = transforms.Compose(pipeline)

    class FormalNwpuDataset(torch.utils.data.Dataset):
        paths = tuple(row['path'] for row in rows)
        labels = tuple(row['label'] for row in rows)

        @property
        def records(self): return copy.deepcopy(rows)

        def __len__(self): return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            raw = _read_image(root, row)
            with Image.open(io.BytesIO(raw)) as source: image = transform(source.convert('RGB'))
            if image.shape != (3, 224, 224) or image.dtype != torch.float32 or not bool(torch.isfinite(image).all()):
                raise ValueError('Formal image transform produced invalid float32 RGB224')
            return image, torch.tensor(row['label'], dtype=torch.long)

    return FormalNwpuDataset()
