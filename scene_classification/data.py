"""Deterministic, relocatable image manifests. No Torch dependency."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re

from PIL import Image

from experiment_identity import sha256_file


AID_CLASSES = set('airport bareland baseballfield beach bridge center church commercial '
                  'denseresidential desert farmland forest industrial meadow mediumresidential '
                  'mountain park parking playground pond port railwaystation resort river school '
                  'sparseresidential square stadium storagetanks viaduct'.split())
PROFILES = {'AID': (30, 10000, 0.5, 600), 'NWPU-RESISC45': (45, 31500, 0.2, 256),
            'synthetic': (None, None, 0.5, None)}
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if not replace:
        with path.open('x', encoding='utf-8') as f:
            f.write(text)
    else:
        temp = path.with_name(path.name + '.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)


def _profile_check(dataset, names, total):
    if dataset not in PROFILES:
        raise ValueError(f'Unknown dataset: {dataset}')
    classes, count, _, _ = PROFILES[dataset]
    if len(names) < 2 or len(names) != len(set(names)):
        raise ValueError('At least two unique class names required')
    if classes and (len(names) != classes or total != count):
        raise ValueError(f'{dataset}: expected {classes} classes / {count} images; got {len(names)} / {total}')
    if dataset == 'AID':
        normalized = {re.sub('[^a-z]', '', name.lower()) for name in names}
        if normalized != AID_CLASSES:
            raise ValueError('AID class names do not match the 30-class taxonomy')


def _image_identity(path, dataset):
    try:
        with Image.open(path) as im:
            im.load()
            if im.mode != 'RGB':
                raise ValueError(f'expected RGB, got {im.mode}')
            size = PROFILES[dataset][3]
            if size and im.size != (size, size):
                raise ValueError(f'expected original {size}x{size}, got {im.size}')
            pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise ValueError(f'Invalid image {path}: {exc}') from exc
    return {'sha256': sha256_file(str(path)), 'pixel_sha256': pixels}


def _assign(paths_by_class, protocol):
    assignments = {}
    for label, paths in enumerate(paths_by_class):
        paths = sorted(paths)
        random.Random(protocol['split_seed'] + label).shuffle(paths)
        n_pool = int(len(paths) * protocol['pool_ratio'])
        pool, test = paths[:n_pool], paths[n_pool:]
        random.Random(protocol['val_split_seed'] + label).shuffle(pool)
        n_val = int(n_pool * protocol['val_fraction_of_pool'])
        train, val = pool[n_val:], pool[:n_val]
        if not train or not val or not test:
            raise ValueError('Each class needs nonempty train/val/test (at least 10 images for synthetic/AID)')
        for split, members in [('train', train), ('val', val), ('test', test)]:
            for path in members:
                assignments[path] = (label, split)
    return assignments


def prepare_manifest(root, output, *, dataset, split_seed=42, val_split_seed=4242):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    by_class = [[p.relative_to(root).as_posix() for p in sorted((root / name).rglob('*'))
                 if p.is_file() and p.suffix.lower() in EXTENSIONS] for name in names]
    _profile_check(dataset, names, sum(map(len, by_class)))
    protocol = {'version': 'scene-pool-v1', 'pool_ratio': PROFILES[dataset][2],
                'val_fraction_of_pool': 0.2, 'split_seed': split_seed,
                'val_split_seed': val_split_seed, 'rounding': 'floor_per_class',
                'assignment': 'sorted_paths_python_random_seed_plus_class_index'}
    assignment = _assign(by_class, protocol)
    records, seen_pixels = [], {}
    for relative in sorted(assignment):
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Image path escapes data root: {relative}')
        identity = _image_identity(path, dataset)
        digest = identity['pixel_sha256']
        if digest in seen_pixels:
            raise ValueError(f'duplicate image pixels: {relative} and {seen_pixels[digest]}')
        seen_pixels[digest] = relative
        label, split = assignment[relative]
        records.append({'path': relative, 'label': label, 'split': split, **identity})
    manifest = {'schema_version': 1, 'dataset': dataset, 'class_names': names,
                'protocol': protocol, 'counts': dict(Counter(r['split'] for r in records)),
                'records': records}
    write_json(output, manifest)
    return manifest


def load_manifest(path, root):
    """Validate every record and image before any training; no path-root binding."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported manifest version')
    dataset, names, records = manifest['dataset'], manifest['class_names'], manifest['records']
    _profile_check(dataset, names, len(records))
    if names != sorted(names):
        raise ValueError('Class map must be sorted')
    protocol = manifest['protocol']
    if (protocol.get('version') != 'scene-pool-v1'
            or protocol.get('pool_ratio') != PROFILES[dataset][2]
            or protocol.get('val_fraction_of_pool') != 0.2
            or protocol.get('rounding') != 'floor_per_class'
            or protocol.get('assignment') != 'sorted_paths_python_random_seed_plus_class_index'
            or any(type(protocol.get(k)) is not int for k in ('split_seed', 'val_split_seed'))):
        raise ValueError('Invalid split protocol')
    root = Path(root).resolve()
    paths, pixels, by_class = set(), set(), [[] for _ in names]
    for record in records:
        relative, label = record['path'], record['label']
        rel = Path(relative)
        if (rel.is_absolute() or '..' in rel.parts or rel.as_posix() != relative
                or type(label) is not int or not 0 <= label < len(names)
                or len(rel.parts) < 2 or rel.parts[0] != names[label]
                or relative in paths or record['split'] not in ('train', 'val', 'test')):
            raise ValueError(f'Invalid/duplicate manifest record: {relative}')
        full = root / rel
        if not full.resolve().is_relative_to(root):
            raise ValueError(f'Image path escapes data root: {relative}')
        identity = _image_identity(full, dataset)
        if any(record.get(key) != digest for key, digest in identity.items()):
            raise ValueError(f'Image changed since manifest: {relative}')
        if identity['pixel_sha256'] in pixels:
            raise ValueError('duplicate image pixels')
        paths.add(relative)
        pixels.add(identity['pixel_sha256'])
        by_class[label].append(relative)
    expected = _assign(by_class, protocol)
    if any(expected[r['path']] != (r['label'], r['split']) for r in records):
        raise ValueError('Split assignment changed from frozen seeds')
    if dict(Counter(r['split'] for r in records)) != manifest['counts']:
        raise ValueError('Manifest counts mismatch')
    return manifest


def select_records(manifest, split, *, per_class=None):
    if split not in ('train', 'val', 'test', 'pool'):
        raise ValueError(f'Unknown split: {split}')
    if per_class is not None and (type(per_class) is not int or per_class < 1):
        raise ValueError('per_class must be positive')
    selected, counts = [], Counter()
    for row in sorted(manifest['records'], key=lambda r: r['path']):
        if row['split'] not in (('train', 'val') if split == 'pool' else (split,)):
            continue
        if per_class is None or counts[row['label']] < per_class:
            selected.append(row)
            counts[row['label']] += 1
    return selected
