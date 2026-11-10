#!/usr/bin/env python3
"""Read-only inventory of frozen AID inputs; no training entry or ML imports.

Only the 240 train / 60 validation files selected from a trusted original manifest
are opened. Test records receive metadata validation only. Pixel hashes are
checked for grouping consistency, never recomputed without an image decoder.
"""
import argparse
from collections import Counter, defaultdict
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys

MANIFEST_SHA256 = '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'
WEIGHTS_SHA256 = '68c6b09e73c66f27db19af59df563497fac4fc7889e42d82a0858b32c451a893'
DEFAULT_MANIFEST = '/home/featurize/work/aid_protocol_20260920_03.json'
DEFAULT_DATA_ROOT = '/home/featurize/data/AID/AID'
DEFAULT_WEIGHTS = '/home/featurize/work/swin_b-68c6b09e.pth'
AID_CLASSES = frozenset('airport bareland baseballfield beach bridge center church commercial '
                        'denseresidential desert farmland forest industrial meadow mediumresidential '
                        'mountain park parking playground pond port railwaystation resort river school '
                        'sparseresidential square stadium storagetanks viaduct'.split())
PROTOCOL = dict(version='scene-pool-group-v2', pool_ratio=0.5, val_fraction_of_pool=0.2,
                split_seed=42, val_split_seed=4242, rounding='floor_per_class',
                assignment='sorted_pixel_groups_seeded_exact_subset_v1', duplicate_policy='group_within_class')
COUNTS = dict(train=4000, val=1000, pool=5000, test=5000)
SHA_PATTERN = re.compile(r'[0-9a-f]{64}\Z')
CHUNK_BYTES = 1024 * 1024


class UnsafePathError(ValueError):
    """A path contains a symlink or cannot be read as a regular file safely."""


def absolute_path(path):
    """Normalize an input path without resolving or following symlinks."""
    return Path(os.path.abspath(os.fspath(path)))


def open_regular_readonly(path):
    """Open each component relative to its parent with O_NOFOLLOW.

    Directory descriptors keep ancestor checks effective during the final open.
    No fallback that follows symlinks is used on unsupported platforms.
    """
    path = absolute_path(path)
    if os.name != 'posix' or not all(hasattr(os, name) for name in ('O_NOFOLLOW', 'O_DIRECTORY')):
        raise UnsafePathError('safe_nofollow_open_unavailable')
    current = None
    try:
        current = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            os.close(current)
            current = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise UnsafePathError('not_a_regular_file')
        return os.fdopen(descriptor, 'rb')
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise UnsafePathError('symlink_or_non_directory_component') from exc
        raise
    finally:
        if current is not None:
            os.close(current)


def hash_stream(stream):
    digest, size = hashlib.sha256(), 0
    while True:
        block = stream.read(CHUNK_BYTES)
        if not block:
            return digest.hexdigest(), size
        digest.update(block)
        size += len(block)


def _unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate_json_key:' + key)
        value[key] = item
    return value


def _reject_json_constant(value):
    raise ValueError('non_finite_json_constant:' + value)


def _valid_sha(value):
    return isinstance(value, str) and SHA_PATTERN.fullmatch(value) is not None


def validate_manifest(value):
    """Validate frozen protocol metadata without reading any image bytes."""
    errors, by_class = [], defaultdict(lambda: defaultdict(list))
    if not isinstance(value, dict):
        return ['manifest_not_an_object'], {}, {}
    if type(value.get('schema_version')) is not int or value.get('schema_version') != 2:
        errors.append('schema_version_must_be_2')
    if value.get('dataset') != 'AID':
        errors.append('dataset_must_be_AID')
    names = value.get('class_names')
    if (not isinstance(names, list) or len(names) != 30
            or any(not isinstance(name, str) or not name for name in names)):
        return errors + ['class_names_invalid'], {}, {}
    normalized = {re.sub('[^a-z]', '', name.lower()) for name in names}
    if len(set(names)) != 30 or names != sorted(names) or normalized != AID_CLASSES:
        errors.append('class_names_must_be_sorted_AID_taxonomy_30')
    protocol = value.get('protocol')
    if not isinstance(protocol, dict) or any(protocol.get(key) != expected for key, expected in PROTOCOL.items()):
        errors.append('frozen_protocol_mismatch')
    elif any(type(protocol[key]) is not int for key in ('split_seed', 'val_split_seed')):
        errors.append('seed_must_be_integer')
    counts = value.get('counts')
    if (not isinstance(counts, dict)
            or any(type(counts.get(key)) is not int or counts.get(key) != COUNTS[key] for key in ('train', 'val', 'test'))
            or ('pool' in counts and (type(counts['pool']) is not int or counts['pool'] != 5000))
            or set(counts) - set(COUNTS)):
        errors.append('declared_counts_mismatch')
    rows = value.get('records')
    if not isinstance(rows, list) or len(rows) != 10000:
        return errors + ['records_must_contain_10000_rows'], {}, {}
    seen, actual_counts, pixels = set(), Counter(), defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f'record_{index}_not_an_object')
            continue
        path, label, split = row.get('path'), row.get('label'), row.get('split')
        valid_path = isinstance(path, str) and bool(path) and '\\' not in path and '\x00' not in path
        parts = PurePosixPath(path).parts if valid_path else ()
        if (not valid_path or path.startswith('/') or len(parts) < 2
                or any(part in ('.', '..') for part in parts)
                or PurePosixPath(path).as_posix() != path
                or type(label) is not int or not 0 <= label < 30
                or parts[0] != names[label] or path in seen):
            errors.append(f'record_{index}_unsafe_duplicate_path_or_label')
            continue
        seen.add(path)
        if split not in ('train', 'val', 'test'):
            errors.append(f'record_{index}_invalid_split')
            continue
        if not _valid_sha(row.get('sha256')) or not _valid_sha(row.get('pixel_sha256')):
            errors.append(f'record_{index}_invalid_or_missing_hash')
            continue
        by_class[label][split].append(row)
        actual_counts[split] += 1
        pixels[row['pixel_sha256']].append(row)
    if dict(actual_counts) != {key: COUNTS[key] for key in ('train', 'val', 'test')}:
        errors.append('actual_split_counts_mismatch')
    for label in range(30):
        groups = by_class[label]
        size = sum(len(rows) for rows in groups.values())
        # AID support differs by class; its own floor-per-class ratios apply.
        pool = int(size * 0.5)
        val = int(pool * 0.2)
        if (len(groups['train']) != pool - val or len(groups['val']) != val
                or len(groups['test']) != size - pool):
            errors.append(f'class_{label}_split_ratios_mismatch')
        if len(groups['train']) < 8 or len(groups['val']) < 2:
            errors.append(f'class_{label}_insufficient_smoke_support')
    duplicates = []
    for digest, group in pixels.items():
        if len(group) < 2:
            continue
        if len({row['label'] for row in group}) != 1:
            errors.append('pixel_group_crosses_classes:' + digest)
        if len({row['split'] for row in group}) != 1:
            errors.append('pixel_group_crosses_splits:' + digest)
        duplicates.append(dict(pixel_sha256=digest, label=group[0]['label'],
                               paths=sorted(row['path'] for row in group), split=group[0]['split']))
    duplicate_report = dict(unique_images=len(pixels), extra_copies=len(rows) - len(pixels),
                            groups=sorted(duplicates, key=lambda group: group['paths']))
    if value.get('duplicates') != duplicate_report:
        errors.append('duplicate_report_missing_or_mismatch')
    return errors, by_class, duplicate_report


def inspect_selected_images(selected, data_root):
    """Hash only preselected train/val records; no image decoding or test reads."""
    result = dict(checked_count=0, verified_count=0, missing_count=0, invalid_count=0,
                  missing=[], invalid=[])
    root = absolute_path(data_root)
    for row in selected:
        try:
            with open_regular_readonly(root / row['path']) as stream:
                digest, _ = hash_stream(stream)
            result['checked_count'] += 1
            if digest != row['sha256']:
                result['invalid'].append(dict(path=row['path'], reason='sha256_mismatch',
                                              expected_sha256=row['sha256'], actual_sha256=digest))
            else:
                result['verified_count'] += 1
        except FileNotFoundError:
            result['missing'].append(row['path'])
        except UnsafePathError as exc:
            result['invalid'].append(dict(path=row['path'], reason='unsafe_path', detail=str(exc)))
        except OSError as exc:
            result['invalid'].append(dict(path=row['path'], reason='unreadable', detail=str(exc)))
    result['missing_count'] = len(result['missing'])
    result['invalid_count'] = len(result['invalid'])
    return result


def inspect_manifest(path, data_root, *, expected_sha256=MANIFEST_SHA256):
    """Inventory a manifest. SHA injection exists solely for internal fixtures."""
    result = dict(path=str(absolute_path(path)), expected_sha256=expected_sha256,
                  actual_sha256=None, size_bytes=None, status='missing', identity_verified=False,
                  structure_verified=False, errors=[], counts=None, class_names=None,
                  selected_counts=dict(train=0, val=0, test=0), selected_paths=dict(train=[], val=[]),
                  image_checks=dict(checked_count=0, verified_count=0, missing_count=0, invalid_count=0,
                                    missing=[], invalid=[]),
                  test_image_bytes_read=0, pixel_hashes_recomputed=False,
                  split_assignment_recomputed=False)
    try:
        with open_regular_readonly(path) as stream:
            content = stream.read(64 * 1024 * 1024 + 1)
        result['size_bytes'] = len(content)
        result['actual_sha256'] = hashlib.sha256(content).hexdigest()
        if len(content) > 64 * 1024 * 1024:
            raise ValueError('manifest_exceeds_64_MiB_limit')
        if result['actual_sha256'] != expected_sha256:
            raise ValueError('manifest_sha256_mismatch')
        result['identity_verified'] = True
        value = json.loads(content, object_pairs_hook=_unique_json_object, parse_constant=_reject_json_constant)
        errors, by_class, duplicates = validate_manifest(value)
        if errors:
            result['status'] = 'invalid'
            result['errors'] = errors
            return result
        result['structure_verified'] = True
        result['counts'] = dict(COUNTS)
        result['class_names'] = value['class_names']
        result['duplicate_groups'] = duplicates
        result['per_class_counts'] = [dict(label=label, name=name,
                                          **{split: len(by_class[label][split]) for split in ('train', 'val', 'test')})
                                      for label, name in enumerate(value['class_names'])]
        selected = []
        for split, count in [('train', 8), ('val', 2)]:
            rows = sorted((row for label in range(30)
                           for row in sorted(by_class[label][split], key=lambda row: row['path'])[:count]),
                          key=lambda row: row['path'])
            selected.extend(rows)
            result['selected_paths'][split] = [row['path'] for row in rows]
            result['selected_counts'][split] = len(rows)
        result['image_checks'] = inspect_selected_images(selected, data_root)
        checks = result['image_checks']
        result['status'] = ('invalid_images' if checks['invalid_count'] else
                            'missing_images' if checks['missing_count'] else 'verified')
    except FileNotFoundError:
        result['errors'] = ['manifest_missing']
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        result['status'] = 'invalid'
        result['errors'] = [str(exc)]
    return result


def inspect_weights(path):
    result = dict(path=str(absolute_path(path)), expected_sha256=WEIGHTS_SHA256,
                  actual_sha256=None, size_bytes=None, status='missing', identity_verified=False,
                  role='candidate_partial_Swin_B_initialization_not_trained_NWPU_checkpoint')
    try:
        with open_regular_readonly(path) as stream:
            digest, size = hash_stream(stream)
        result.update(actual_sha256=digest, size_bytes=size, identity_verified=digest == WEIGHTS_SHA256,
                      status='verified' if digest == WEIGHTS_SHA256 else 'invalid')
        if digest != WEIGHTS_SHA256:
            result['error'] = 'weights_sha256_mismatch'
    except FileNotFoundError:
        result['error'] = 'weights_missing'
    except (OSError, ValueError) as exc:
        result.update(status='invalid', error=str(exc))
    return result


def inspect_gpu():
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits']
    result = dict(status='unavailable', command=command, timeout_seconds=10, devices=[])
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
        result['returncode'] = completed.returncode
        if completed.returncode:
            result['detail'] = completed.stderr.strip() or completed.stdout.strip()
            return result
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 5 or not parts[0].isdigit() or not parts[3].isdigit():
                result.update(status='invalid_response', detail=completed.stdout)
                result['devices'] = []
                return result
            result['devices'].append(dict(index=int(parts[0]), uuid=parts[1], name=parts[2],
                                          memory_total_mib=int(parts[3]), driver_version=parts[4]))
        result['status'] = 'inventoried' if result['devices'] else 'no_devices_reported'
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        result['detail'] = str(exc)
    return result


def inspect_versions():
    versions = {}
    for package in ('torch', 'torchvision', 'timm', 'numpy', 'Pillow'):
        try:
            versions[package] = dict(status='metadata_found', version=importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:
            versions[package] = dict(status='not_installed', version=None)
        except (OSError, ValueError) as exc:
            versions[package] = dict(status='metadata_unavailable', version=None, detail=str(exc))
    return dict(method='importlib.metadata_only_no_ML_imports', python=platform.python_version(), distributions=versions)


def inspect_disk(paths):
    results = []
    for role, supplied in paths.items():
        requested, probe = absolute_path(supplied), absolute_path(supplied)
        try:
            # Disk usage of a missing asset refers explicitly to its nearest
            # existing ancestor, never to an invented server filesystem.
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            usage = shutil.disk_usage(probe)
            results.append(dict(role=role, requested_path=str(requested), requested_path_exists=requested.exists(),
                                measured_path=str(probe), status='inventoried', total_bytes=usage.total,
                                used_bytes=usage.used, free_bytes=usage.free))
        except OSError as exc:
            results.append(dict(role=role, requested_path=str(requested), status='unavailable', detail=str(exc)))
    return results


def inventory(manifest, data_root, weights):
    manifest_result = inspect_manifest(manifest, data_root)
    weights_result = inspect_weights(weights)
    assets_verified = manifest_result['status'] == 'verified' and weights_result['status'] == 'verified'
    return dict(schema='hgtnet-aid-input-inventory-v1',
                status='INPUT_ASSETS_VERIFIED_SMOKE_PREPARATION_PENDING' if assets_verified else 'INPUT_CHECK_INCOMPLETE',
                input_assets_verified=assets_verified, training_started=False, training_entry_available=False,
                purpose='read_only_input_inventory_for_future_AID_smoke_preparation',
                manifest=manifest_result, data_root=str(absolute_path(data_root)), weights=weights_result,
                gpu=inspect_gpu(), package_versions=inspect_versions(),
                disk=inspect_disk(dict(manifest=manifest, data_root=data_root, weights=weights)),
                boundaries=dict(image_decoding_performed=False, model_or_CUDA_computation_performed=False,
                                test_images_opened=False, original_AID_zip_identity_known=False,
                                new_split_created=False, downloads_performed=False,
                                training_readiness_claimed=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only frozen AID input inventory; no training entry.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    command = subparsers.add_parser('inspect', help='Print input inventory JSON to stdout without writing files.')
    command.add_argument('--manifest', default=DEFAULT_MANIFEST)
    command.add_argument('--data-root', default=DEFAULT_DATA_ROOT)
    command.add_argument('--weights', default=DEFAULT_WEIGHTS)
    arguments = parser.parse_args(argv)
    result = inventory(arguments.manifest, arguments.data_root, arguments.weights)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
