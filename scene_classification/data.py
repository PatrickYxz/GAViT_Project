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


def _take_exact(groups, count):
    """First seeded subset reaching an exact image count, without splitting groups."""
    reachable = {0: None}
    for i, group in enumerate(groups):
        for size in sorted(reachable, reverse=True):
            total = size + len(group)
            if total <= count and total not in reachable:
                reachable[total] = (size, i)
        if count in reachable:
            break
    if count not in reachable:
        raise ValueError('Cannot form exact split sizes while keeping duplicate groups intact')
    chosen = set()
    while count:
        count, i = reachable[count]
        chosen.add(i)
    return ([g for i, g in enumerate(groups) if i in chosen],
            [g for i, g in enumerate(groups) if i not in chosen])


def _assign(paths_by_class, protocol, pixel_hashes=None):
    assignments = {}
    for label, paths in enumerate(paths_by_class):
        groups = {}
        for path in sorted(paths):
            key = pixel_hashes[path] if pixel_hashes is not None else path
            groups.setdefault(key, []).append(path)
        groups = list(groups.values())
        random.Random(protocol['split_seed'] + label).shuffle(groups)
        n_pool = int(len(paths) * protocol['pool_ratio'])
        pool, test = _take_exact(groups, n_pool)
        random.Random(protocol['val_split_seed'] + label).shuffle(pool)
        n_val = int(n_pool * protocol['val_fraction_of_pool'])
        val, train = _take_exact(pool, n_val)
        if not train or not val or not test:
            raise ValueError('Each class needs nonempty train/val/test (at least 10 images for synthetic/AID)')
        for split, members in [('train', train), ('val', val), ('test', test)]:
            for group in members:
                for path in group:
                    assignments[path] = (label, split)
    return assignments


def _duplicate_report(records, *, check_splits=True):
    pixels = {}
    for record in records:
        pixels.setdefault(record['pixel_sha256'], []).append(record)
    groups = []
    for digest, rows in pixels.items():
        if len(rows) < 2:
            continue
        paths = sorted(r['path'] for r in rows)
        if len({r['label'] for r in rows}) != 1:
            raise ValueError(f'duplicate image pixels with different class labels: {paths}')
        if check_splits and len({r['split'] for r in rows}) != 1:
            raise ValueError(f'duplicate image pixels cross split boundaries: {paths}')
        groups.append({'pixel_sha256': digest, 'label': rows[0]['label'],
                       'paths': paths, 'split': rows[0].get('split')})
    return {'unique_images': len(pixels), 'extra_copies': len(records) - len(pixels),
            'groups': sorted(groups, key=lambda g: g['paths'])}


def prepare_manifest(root, output, *, dataset, split_seed=42, val_split_seed=4242,
                     duplicate_policy='reject'):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if duplicate_policy not in ('reject', 'group'):
        raise ValueError('Unknown duplicate policy')
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    by_class = [[p.relative_to(root).as_posix() for p in sorted((root / name).rglob('*'))
                 if p.is_file() and p.suffix.lower() in EXTENSIONS] for name in names]
    _profile_check(dataset, names, sum(map(len, by_class)))
    protocol = {'version': 'scene-pool-v1', 'pool_ratio': PROFILES[dataset][2],
                'val_fraction_of_pool': 0.2, 'split_seed': split_seed,
                'val_split_seed': val_split_seed, 'rounding': 'floor_per_class',
                'assignment': 'sorted_paths_python_random_seed_plus_class_index'}
    if duplicate_policy == 'group':
        protocol.update(version='scene-pool-group-v2', duplicate_policy='group_within_class',
                        assignment='sorted_pixel_groups_seeded_exact_subset_v1')
    labels = {path: label for label, paths in enumerate(by_class) for path in paths}
    records = []
    for relative in sorted(labels):
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Image path escapes data root: {relative}')
        identity = _image_identity(path, dataset)
        records.append({'path': relative, 'label': labels[relative], **identity})
    report = _duplicate_report(records, check_splits=False)
    if duplicate_policy == 'reject' and report['groups']:
        raise ValueError(f'duplicate image pixels: {report["groups"][0]["paths"]}; '
                         'use --duplicate-policy group to keep same-class duplicates in one split')
    pixel_hashes = {r['path']: r['pixel_sha256'] for r in records} if duplicate_policy == 'group' else None
    assignment = _assign(by_class, protocol, pixel_hashes)
    for record in records:
        record['split'] = assignment[record['path']][1]
    manifest = {'schema_version': 2 if duplicate_policy == 'group' else 1,
                'dataset': dataset, 'class_names': names,
                'protocol': protocol, 'counts': dict(Counter(r['split'] for r in records)),
                'records': records}
    if duplicate_policy == 'group':
        manifest['duplicates'] = _duplicate_report(records)
    write_json(output, manifest)
    return manifest


def load_manifest(path, root):
    """Validate every record and image before any training; no path-root binding."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    version = manifest.get('schema_version')
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported manifest version')
    dataset, names, records = manifest['dataset'], manifest['class_names'], manifest['records']
    _profile_check(dataset, names, len(records))
    if names != sorted(names):
        raise ValueError('Class map must be sorted')
    protocol = manifest['protocol']
    expected_protocol = (('scene-pool-v1', 'sorted_paths_python_random_seed_plus_class_index')
                         if version == 1 else
                         ('scene-pool-group-v2', 'sorted_pixel_groups_seeded_exact_subset_v1'))
    if (protocol.get('version') != expected_protocol[0]
            or protocol.get('pool_ratio') != PROFILES[dataset][2]
            or protocol.get('val_fraction_of_pool') != 0.2
            or protocol.get('rounding') != 'floor_per_class'
            or protocol.get('assignment') != expected_protocol[1]
            or (version == 2 and protocol.get('duplicate_policy') != 'group_within_class')
            or any(type(protocol.get(k)) is not int for k in ('split_seed', 'val_split_seed'))):
        raise ValueError('Invalid split protocol')
    root = Path(root).resolve()
    paths, by_class = set(), [[] for _ in names]
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
        paths.add(relative)
        by_class[label].append(relative)
    report = _duplicate_report(records)
    if version == 1 and report['groups']:
        raise ValueError('duplicate image pixels')
    if version == 2 and manifest.get('duplicates') != report:
        raise ValueError('Duplicate report mismatch')
    pixel_hashes = {r['path']: r['pixel_sha256'] for r in records} if version == 2 else None
    expected = _assign(by_class, protocol, pixel_hashes)
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
