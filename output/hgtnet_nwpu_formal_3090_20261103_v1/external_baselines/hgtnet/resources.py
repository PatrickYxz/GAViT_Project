"""Stdlib resource and runtime gates for the single-epoch NWPU smoke.

Inspect reports distribution metadata without importing any ML package. Actual
CUDA/runtime values are collected by the run entry point and checked separately.
"""
from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


TARGET_NAME = 'NVIDIA GeForce RTX 3080'
TARGET_MEMORY_MIB = 10240
TARGETS = {TARGET_NAME: TARGET_MEMORY_MIB, 'NVIDIA GeForce RTX 3090': 24576}
WORK_BYTES = 2 * 1024 ** 3
SCRATCH_BYTES = 1024 ** 3
_VERSIONS = {
    'python': {'3.11.8'},
    'torch': {'2.2.2', '2.2.2+cu121'},
    'torchvision': {'0.17.2', '0.17.2+cu121'},
    'numpy': {'1.26.4'},
    'Pillow': {'10.3.0'},
}


def _uuid(value):
    return isinstance(value, str) and re.fullmatch(r'GPU-[A-Za-z0-9-]+', value) is not None


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


def _target_memory(target_name):
    if not isinstance(target_name, str) or target_name not in TARGETS:
        raise ValueError('unsupported NVIDIA GPU target')
    return TARGETS[target_name]


def _validate_gpu(gpu):
    if (not isinstance(gpu, dict) or not _uuid(gpu.get('uuid'))
            or not isinstance(gpu.get('name'), str) or not gpu['name'].strip()
            or not isinstance(gpu.get('driver_version'), str) or not gpu['driver_version'].strip()
            or not _integer(gpu.get('memory_total_mib'), minimum=1)
            or not _integer(gpu.get('memory_free_mib'))
            or gpu['memory_free_mib'] > gpu['memory_total_mib']):
        raise ValueError('invalid NVIDIA GPU inventory data')


def _validate_process(process):
    if (not isinstance(process, dict) or not _integer(process.get('pid'), minimum=1)
            or not _uuid(process.get('gpu_uuid'))
            or not isinstance(process.get('name'), str) or not process['name'].strip()
            or not _integer(process.get('used_memory_mib'))):
        raise ValueError('invalid NVIDIA compute-process inventory data')


def _query(fields):
    result = subprocess.run(
        ['nvidia-smi', fields, '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip() or 'NVIDIA query failed')
    return csv.reader(result.stdout.splitlines(), strict=True)


def inventory_gpu():
    """Return NVIDIA inventory; an error is never interpreted as an idle GPU."""
    result = {'gpus': [], 'processes': [], 'error': None}
    try:
        for raw in _query('--query-gpu=name,uuid,memory.total,memory.free,driver_version'):
            if len(raw) != 5:
                raise ValueError('NVIDIA GPU query must contain five CSV columns')
            row = [value.strip() for value in raw]
            gpu = {'name': row[0], 'uuid': row[1], 'memory_total_mib': int(row[2]),
                   'memory_free_mib': int(row[3]), 'driver_version': row[4]}
            _validate_gpu(gpu)
            result['gpus'].append(gpu)
        for raw in _query('--query-compute-apps=pid,gpu_uuid,process_name,used_memory'):
            if len(raw) != 4:
                raise ValueError('NVIDIA process query must contain four CSV columns')
            row = [value.strip() for value in raw]
            process = {'pid': int(row[0]), 'gpu_uuid': row[1], 'name': row[2],
                       'used_memory_mib': int(row[3])}
            _validate_process(process)
            result['processes'].append(process)
    except (OSError, ValueError, csv.Error, subprocess.SubprocessError) as exc:
        result['error'] = str(exc)
    return result


def select_gpu(requested_uuid, inventory, *, target_name=TARGET_NAME):
    """Bind a physical UUID, or select the sole idle card for the exact target."""
    target_memory = _target_memory(target_name)
    if (not isinstance(inventory, dict) or inventory.get('error')
            or not isinstance(inventory.get('gpus'), list)
            or not isinstance(inventory.get('processes'), list)):
        raise ValueError('working NVIDIA GPU and process queries are required')
    gpus, processes = inventory['gpus'], inventory['processes']
    for gpu in gpus:
        _validate_gpu(gpu)
    if len({gpu['uuid'] for gpu in gpus}) != len(gpus):
        raise ValueError('duplicate physical GPU UUID in NVIDIA inventory')
    for process in processes:
        _validate_process(process)
    if requested_uuid is not None and not _uuid(requested_uuid):
        raise ValueError('explicit physical GPU UUID is invalid')
    busy = {row['gpu_uuid'] for row in processes}
    candidates = [gpu for gpu in gpus
                  if gpu['name'] == target_name
                  and gpu['memory_total_mib'] == target_memory
                  and gpu['memory_free_mib'] > 0 and gpu['uuid'] not in busy
                  and (requested_uuid is None or gpu['uuid'] == requested_uuid)]
    if len(candidates) != 1:
        raise ValueError('exactly one selected idle ' + target_name + ' with '
                         + str(target_memory) + ' MiB and valid free memory is required')
    return dict(candidates[0])


def dependency_inventory():
    """Locate distributions without testing imports or claiming CUDA works."""
    result = {'python': {'located': True,
                         'version': '.'.join(map(str, sys.version_info[:3])),
                         'import_tested': False}}
    for name, module in [('torch', 'torch'), ('torchvision', 'torchvision'),
                         ('numpy', 'numpy'), ('Pillow', 'PIL')]:
        try:
            located = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            located = False
        try:
            version = importlib.metadata.version(name) if located else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        result[name] = {'located': located, 'version': version, 'import_tested': False}
    return result


def dependency_blockers(inventory):
    """Version gates for metadata only; run still verifies the actual runtime."""
    blockers = []
    for name, accepted in _VERSIONS.items():
        item = inventory.get(name, {}) if isinstance(inventory, dict) else {}
        if not isinstance(item, dict) or item.get('located') is not True:
            blockers.append('dependency:' + name)
        elif item.get('version') not in accepted:
            blockers.append('version:' + name)
    return blockers


def check_runtime(runtime, gpu_uuid, *, target_name=TARGET_NAME):
    """Reject actual imported runtime/device/settings that differ from the gate."""
    target_memory = _target_memory(target_name)
    if not isinstance(runtime, dict) or not _uuid(gpu_uuid):
        raise ValueError('runtime data and an explicit physical GPU UUID are required')
    python_version = runtime.get('python')
    if not isinstance(python_version, str) or not re.match(r'^3\.11\.8(?:\s|$)', python_version):
        raise ValueError('runtime Python must be exactly 3.11.8')
    for key, name in [('torch', 'torch'), ('torchvision', 'torchvision'),
                      ('numpy', 'numpy'), ('pillow', 'Pillow')]:
        if runtime.get(key) not in _VERSIONS[name]:
            raise ValueError('actual runtime version mismatch: ' + key)
    required = {
        'cuda': '12.1', 'cudnn': 8902, 'physical_gpu_uuid': gpu_uuid,
        'device': 'cuda:0', 'cpu_threads': 8, 'amp': False,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'dtype': 'float32', 'seed': 42,
    }
    for key, expected in required.items():
        value = runtime.get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError('actual runtime setting mismatch: ' + key)
    gpu = runtime.get('gpu')
    if (not isinstance(gpu, dict) or gpu.get('uuid') != gpu_uuid
            or gpu.get('name') != target_name
            or type(gpu.get('memory_total_mib')) is not int
            or gpu['memory_total_mib'] != target_memory):
        raise ValueError('actual runtime physical GPU identity mismatch')
    return runtime


def require_new_paths(output, scratch):
    """Check independent new destinations, including aliases and dangling links."""
    paths = [Path(output), Path(scratch)]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise ValueError('destination already exists: ' + str(path))
    first, second = [path.resolve() for path in paths]
    if first == second or first in second.parents or second in first.parents:
        raise ValueError('output and scratch must be distinct non-containing paths')


def _capacity_path(path, required_bytes):
    path = Path(path).absolute()
    ancestor = path
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:
            raise OSError('no existing ancestor for ' + str(path))
        ancestor = parent
    ancestor = ancestor.resolve()
    if not ancestor.is_dir():
        raise OSError('nearest existing ancestor is not a directory: ' + str(ancestor))
    free = shutil.disk_usage(ancestor)[2]
    if not _integer(free):
        raise ValueError('disk free capacity is invalid')
    return {'path': str(path), 'ancestor': str(ancestor),
            'device': os.stat(ancestor).st_dev, 'free_bytes': free,
            'required_bytes': required_bytes}


def capacity(output, scratch):
    """Report work/scratch budgets, adding them when they share one filesystem."""
    result = {'ready': False, 'blockers': [], 'output': None, 'scratch': None,
              'same_volume': None}
    for name, path, budget in [('output', output, WORK_BYTES), ('scratch', scratch, SCRATCH_BYTES)]:
        try:
            result[name] = _capacity_path(path, budget)
        except (OSError, ValueError, RuntimeError) as exc:
            result['blockers'].append('disk_capacity:' + name)
            result[name] = {'path': str(path), 'error': str(exc), 'required_bytes': budget}
    if result['blockers']:
        return result
    work, local = result['output'], result['scratch']
    result['same_volume'] = work['device'] == local['device']
    if result['same_volume']:
        if min(work['free_bytes'], local['free_bytes']) < WORK_BYTES + SCRATCH_BYTES:
            result['blockers'].append('disk_capacity:combined')
    else:
        for name in ('output', 'scratch'):
            if result[name]['free_bytes'] < result[name]['required_bytes']:
                result['blockers'].append('disk_capacity:' + name)
    result['ready'] = not result['blockers']
    return result
