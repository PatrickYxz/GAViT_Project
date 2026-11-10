"""Guard the smoke runtime/resource gates without importing ML packages."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from external_baselines.hgtnet import resources


GPU = {
    'name': 'NVIDIA GeForce RTX 3080', 'uuid': 'GPU-current-fixture',
    'memory_total_mib': 10240, 'memory_free_mib': 10000,
    'driver_version': '550.54.15',
}


def inventory(*gpus, processes=()):
    return {'gpus': list(gpus), 'processes': list(processes), 'error': None}


def runtime():
    return {
        'python': '3.11.8 (main, Feb 12 2024, 12:00:00) [GCC 11.4.0]',
        'torch': '2.2.2+cu121', 'torchvision': '0.17.2+cu121',
        'numpy': '1.26.4', 'pillow': '10.3.0', 'cuda': '12.1', 'cudnn': 8902,
        'gpu': copy.deepcopy(GPU), 'physical_gpu_uuid': GPU['uuid'],
        'device': 'cuda:0', 'cpu_threads': 8, 'amp': False,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'dtype': 'float32', 'seed': 42,
    }


class ResourcesTests(unittest.TestCase):
    def test_unique_idle_current_gpu_is_selected_without_prior_uuid(self):
        selected = resources.select_gpu(None, inventory(copy.deepcopy(GPU)))
        self.assertEqual(selected['uuid'], 'GPU-current-fixture')

    def test_explicit_uuid_can_select_one_of_two_idle_cards(self):
        other = dict(GPU, uuid='GPU-other-fixture')
        self.assertEqual(resources.select_gpu(GPU['uuid'], inventory(GPU, other)), GPU)
        with self.assertRaises(ValueError):
            resources.select_gpu(None, inventory(GPU, other))

    def test_busy_wrong_card_and_unknown_uuid_are_rejected(self):
        cases = [
            (None, inventory(dict(GPU, name='NVIDIA GeForce RTX 3090'))),
            (None, inventory(dict(GPU, memory_total_mib=12288))),
            ('GPU-stale-fixture', inventory(GPU)),
            (None, inventory(GPU, processes=[{'pid': 123, 'gpu_uuid': GPU['uuid'],
                                             'name': 'python', 'used_memory_mib': 500}])),
            (None, inventory(GPU, dict(GPU))),
            (None, dict(inventory(GPU), error='query timed out')),
        ]
        for requested, observed in cases:
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                resources.select_gpu(requested, observed)

    def test_invalid_free_memory_and_unparseable_processes_are_rejected(self):
        for free in (0, -1, 10241, None, True, '10000'):
            with self.subTest(free=free), self.assertRaises(ValueError):
                resources.select_gpu(None, inventory(dict(GPU, memory_free_mib=free)))
        for process in ({}, {'pid': 'N/A', 'gpu_uuid': 'GPU-other', 'name': 'python',
                              'used_memory_mib': 42}):
            with self.subTest(process=process), self.assertRaises(ValueError):
                resources.select_gpu(None, inventory(GPU, processes=[process]))

    def test_nvidia_csv_is_parsed_into_numeric_inventory(self):
        results = [
            subprocess.CompletedProcess([], 0, 'NVIDIA GeForce RTX 3080, GPU-current-fixture, 10240, 10000, 550.54.15\n', ''),
            subprocess.CompletedProcess([], 0, '567, GPU-other, /usr/bin/python, 132\n', ''),
        ]
        with mock.patch.object(resources.subprocess, 'run', side_effect=results):
            observed = resources.inventory_gpu()
        self.assertIsNone(observed['error'])
        self.assertEqual(observed['gpus'], [GPU])
        self.assertEqual(observed['processes'], [{'pid': 567, 'gpu_uuid': 'GPU-other',
                                                 'name': '/usr/bin/python', 'used_memory_mib': 132}])

    def test_query_failures_and_malformed_csv_never_become_idle_gpu(self):
        valid = subprocess.CompletedProcess([], 0, 'NVIDIA GeForce RTX 3080, GPU-current-fixture, 10240, 10000, 550.54.15\n', '')
        cases = [
            [subprocess.CompletedProcess([], 1, '', 'NVIDIA failed')],
            [subprocess.CompletedProcess([], 0, 'GPU-current-fixture, 10240\n', '')],
            [valid, subprocess.CompletedProcess([], 1, '', 'permission denied')],
            [valid, subprocess.CompletedProcess([], 0, '123, GPU-other, python\n', '')],
            [valid, subprocess.CompletedProcess([], 0, 'N/A, GPU-other, python, 5\n', '')],
            [valid, subprocess.CompletedProcess([], 0, '123, GPU-other, python, N/A\n', '')],
        ]
        for results in cases:
            with self.subTest(results=results), mock.patch.object(resources.subprocess, 'run', side_effect=results):
                observed = resources.inventory_gpu()
            self.assertTrue(observed['error'])
            with self.assertRaises(ValueError):
                resources.select_gpu(None, observed)
        with mock.patch.object(resources.subprocess, 'run', side_effect=FileNotFoundError('nvidia-smi')):
            self.assertTrue(resources.inventory_gpu()['error'])

    def test_no_compute_process_lines_are_valid_but_blank_inventory_is_not_ready(self):
        results = [subprocess.CompletedProcess([], 0, '', ''), subprocess.CompletedProcess([], 0, '', '')]
        with mock.patch.object(resources.subprocess, 'run', side_effect=results):
            observed = resources.inventory_gpu()
        self.assertIsNone(observed['error'])
        with self.assertRaises(ValueError):
            resources.select_gpu(None, observed)

    def test_dependency_metadata_is_explicitly_unimported(self):
        observed = resources.dependency_inventory()
        self.assertEqual(set(observed), {'python', 'torch', 'torchvision', 'numpy', 'Pillow'})
        for value in observed.values():
            self.assertEqual(set(value), {'located', 'version', 'import_tested'})
            self.assertIs(value['import_tested'], False)
        self.assertEqual(observed['python']['version'], '.'.join(map(str, sys.version_info[:3])))

    def test_metadata_accepts_base_or_cu121_but_rejects_cpu_or_wrong_versions(self):
        dependencies = {name: {'located': True, 'version': version, 'import_tested': False}
                        for name, version in [('python', '3.11.8'), ('torch', '2.2.2'),
                                              ('torchvision', '0.17.2'), ('numpy', '1.26.4'), ('Pillow', '10.3.0')]}
        self.assertEqual(resources.dependency_blockers(dependencies), [])
        dependencies['torch']['version'] = '2.2.2+cu121'
        dependencies['torchvision']['version'] = '0.17.2+cu121'
        self.assertEqual(resources.dependency_blockers(dependencies), [])
        for name, wrong in [('python', '3.12.13'), ('torch', '2.2.2+cpu'),
                            ('torchvision', '0.17.2+cu118'), ('numpy', '2.0.0'), ('Pillow', '10.3.1')]:
            broken = copy.deepcopy(dependencies)
            broken[name]['version'] = wrong
            self.assertIn('version:' + name, resources.dependency_blockers(broken))
        dependencies['torch']['located'] = False
        self.assertIn('dependency:torch', resources.dependency_blockers(dependencies))

    def test_resource_module_and_inventory_work_with_ml_imports_forbidden(self):
        code = '''
import builtins, json
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'PIL'}:
        raise AssertionError('ML import attempted: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from external_baselines.hgtnet.resources import dependency_inventory
print(json.dumps(dependency_inventory(), sort_keys=True))
'''
        process = subprocess.run([sys.executable, '-c', code], text=True, capture_output=True,
                                 cwd=Path(__file__).resolve().parents[1], check=False)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('import_tested', json.loads(process.stdout)['torch'])

    def test_actual_runtime_is_verified_independently_from_distribution_metadata(self):
        self.assertEqual(resources.check_runtime(runtime(), GPU['uuid']), runtime())
        actual = runtime()
        actual['torch'] = '2.2.2'
        actual['torchvision'] = '0.17.2'
        self.assertEqual(resources.check_runtime(actual, GPU['uuid']), actual)
        changes = {'python': '3.11.80', 'torch': '2.2.2+cu118', 'torchvision': '0.17.2+cpu',
                   'cuda': '11.8', 'cudnn': 8901, 'numpy': '2.0.0', 'pillow': '10.4.0',
                   'physical_gpu_uuid': 'GPU-stale', 'device': 'cpu', 'amp': True,
                   'cpu_threads': 16, 'matmul_tf32': True, 'cudnn_tf32': True,
                   'cudnn_benchmark': True, 'cudnn_deterministic': False,
                   'dtype': 'float16', 'seed': 43}
        for key, wrong in changes.items():
            broken = runtime()
            broken[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, GPU['uuid'])
        for key, wrong in [('uuid', 'GPU-stale'), ('name', 'NVIDIA GeForce RTX 3090'),
                           ('memory_total_mib', 24576)]:
            broken = runtime()
            broken['gpu'][key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, GPU['uuid'])

    def test_runtime_missing_fields_and_boolean_numeric_aliases_are_rejected(self):
        for key in runtime():
            broken = runtime()
            del broken[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, GPU['uuid'])
        for key, wrong in [('amp', 0), ('cudnn_deterministic', 1), ('seed', 42.0), ('cpu_threads', 8.0)]:
            broken = runtime()
            broken[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, GPU['uuid'])

    def test_new_paths_are_checked_by_real_path_without_creating_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            output, scratch = base / 'new/work/output', base / 'new/scratch'
            resources.require_new_paths(output, scratch)
            self.assertFalse((base / 'new').exists())
            (base / 'alias').symlink_to(base, target_is_directory=True)
            cases = [(base, scratch), (output, output), (output, output / 'child'),
                     (base / 'new', output), (base / 'alias/new/work/output', output)]
            (base / 'dangling').symlink_to(base / 'absent')
            cases.append((base / 'dangling', scratch))
            for first, second in cases:
                with self.subTest(first=first, second=second), self.assertRaises(ValueError):
                    resources.require_new_paths(first, second)

    def test_same_volume_budget_is_combined_and_nearest_existing_ancestor_reported(self):
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            output, scratch = base / 'work/new/output', base / 'scratch/new'
            with mock.patch.object(resources.shutil, 'disk_usage', return_value=(10 * gib, 8 * gib, 3 * gib)):
                report = resources.capacity(output, scratch)
            self.assertTrue(report['ready'])
            self.assertTrue(report['same_volume'])
            self.assertEqual(report['blockers'], [])
            self.assertEqual(report['output']['ancestor'], str(base.resolve()))
            self.assertEqual(report['output']['required_bytes'], 2 * gib)
            self.assertEqual(report['scratch']['required_bytes'], gib)
            self.assertFalse((base / 'work').exists())
            with mock.patch.object(resources.shutil, 'disk_usage', return_value=(10 * gib, 8 * gib, 3 * gib - 1)):
                report = resources.capacity(output, scratch)
            self.assertFalse(report['ready'])
            self.assertIn('disk_capacity:combined', report['blockers'])

    def test_separate_volume_budgets_and_disk_errors_are_blockers(self):
        gib = 1024 ** 3
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            (base / 'work').mkdir()
            (base / 'scratch').mkdir()
            original_stat = resources.os.stat
            def fake_stat(path, *args, **kwargs):
                observed = original_stat(path, *args, **kwargs)
                if Path(path) == base / 'scratch':
                    values = list(observed)
                    values[2] += 1
                    return os.stat_result(values)
                return observed
            def free(path):
                return (10 * gib, 0, (2 * gib if Path(path) == base / 'work' else gib))
            with mock.patch.object(resources.os, 'stat', side_effect=fake_stat), mock.patch.object(resources.shutil, 'disk_usage', side_effect=free):
                report = resources.capacity(base / 'work/new', base / 'scratch/new')
            self.assertTrue(report['ready'])
            self.assertFalse(report['same_volume'])
            with mock.patch.object(resources.shutil, 'disk_usage', side_effect=OSError('unavailable')):
                report = resources.capacity(base / 'work/new', base / 'scratch/new')
            self.assertFalse(report['ready'])
            self.assertTrue(report['blockers'])


if __name__ == '__main__':
    unittest.main()
