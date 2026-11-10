"""Formal resource regressions using file fixtures and stdlib inventories only.

These tests do not import Torch, initialize CUDA, or start any training process.
"""
import builtins
import contextlib
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import tempfile
import types
import unittest
from unittest import mock
import zipfile

import test_dfagcn_formal_runner as original


CURRENT_UUID = 'GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed'
OTHER_UUID = 'GPU-22222222-2222-2222-2222-222222222222'
WORK_FREE = 17029922816
WORK_REQUIRED = 16508521324
SCRATCH_REQUIRED = 42949672960


class FormalResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.module = original.api()
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(original.manifest_fixture()))
        weights = self.root / 'vgg16-397923af.pth'
        weights.write_bytes(b'resource-only-weight-identity-fixture')
        metadata = self.root / 'weights.json'
        metadata.write_text(json.dumps({'sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}))
        recipe = self.root / 'recipe.json'
        shutil.copyfile(original.ROOT / 'configs/dfagcn_independent_v1.json', recipe)
        archive = self.root / 'data.zip'
        with zipfile.ZipFile(archive, 'w') as zipped:
            zipped.writestr('placeholder.txt', 'resource inventory fixture')
        for binary_name in ('train', 'predict'):
            binary = self.root / binary_name
            binary.write_text('#!/bin/sh\nexit 0\n')
            binary.chmod(0o700)
        self.args = types.SimpleNamespace(
            action='inspect', manifest=manifest, data_root=self.root / 'missing-data',
            archive=archive, weights=weights, weights_metadata=metadata,
            weights_sha256=None, recipe=recipe, train_binary=self.root / 'train',
            predict_binary=self.root / 'predict', output=self.root / 'formal-run',
            background_log=self.root / 'background.log', scratch_root=self.root / 'scratch',
            gpu_uuid=CURRENT_UUID, cpu_threads=8,
        )
        self.versions = {
            'python': '3.11.8',
            'torch': {'located': True, 'version': '2.2.2', 'import_tested': False},
            'torchvision': {'located': True, 'version': '0.17.2', 'import_tested': False},
            'numpy': {'located': True, 'version': '1.26.4', 'import_tested': False},
            'Pillow': {'located': True, 'version': '10.3.0', 'import_tested': False},
        }
        self.inventory = {
            'gpus': [self.device(CURRENT_UUID)], 'processes': [], 'error': None,
        }
        self.work_free = WORK_FREE
        self.scratch_free = 349 * 2**30
        self.work_device = 11
        self.scratch_device = 22
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(
            self.module, 'MANIFEST_SHA', hashlib.sha256(manifest.read_bytes()).hexdigest()))
        stack.enter_context(mock.patch.object(
            self.module.GPU, 'WEIGHTS_SHA', hashlib.sha256(weights.read_bytes()).hexdigest()))
        stack.enter_context(mock.patch.object(
            self.module, 'dependency_inventory', side_effect=lambda: copy.deepcopy(self.versions)))
        stack.enter_context(mock.patch.object(
            self.module.GPU, 'gpu_inventory', side_effect=lambda: copy.deepcopy(self.inventory)))
        stack.enter_context(mock.patch.object(self.module, '_disk', side_effect=self.disk_inventory))

    @staticmethod
    def device(uuid):
        return {
            'name': 'NVIDIA GeForce RTX 3080', 'uuid': uuid,
            'memory_total_mib': 10240, 'memory_free_mib': 9874,
            'driver_version': '610.57.04',
        }

    def disk_inventory(self, path, *, minimum_gate_bytes=SCRATCH_REQUIRED):
        scratch = Path(path) == self.args.scratch_root
        return {
            'checked_path': str(path),
            'free_bytes': self.scratch_free if scratch else self.work_free,
            'minimum_gate_bytes': minimum_gate_bytes,
            'feature_cache_bytes': 10102579200,
            'device_id': self.scratch_device if scratch else self.work_device,
            'scope': 'controlled filesystem inventory, not a cost measurement',
        }

    def inspect(self):
        return self.module.inspect_resources(self.args)

    def test_reported_base_versions_and_distinct_filesystems_pass_without_importing_ml(self):
        real_import = builtins.__import__

        def stdlib_only(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'torchvision', 'numpy', 'PIL'}:
                raise AssertionError('inspect imported ML runtime: ' + name)
            return real_import(name, *args, **kwargs)

        with mock.patch.object(builtins, '__import__', side_effect=stdlib_only):
            report = self.inspect()
        self.assertTrue(report['ready'], report)
        self.assertEqual(report['disk']['free_bytes'], 349 * 2**30)
        self.assertEqual(report['disk']['minimum_gate_bytes'], SCRATCH_REQUIRED)
        self.assertEqual(report['persistent_disk']['free_bytes'], WORK_FREE)
        self.assertEqual(report['persistent_disk']['minimum_gate_bytes'], WORK_REQUIRED)
        self.assertFalse(self.args.output.exists())
        self.assertFalse(self.args.scratch_root.exists())

    def test_other_cuda_builds_and_base_versions_are_rejected(self):
        for package, version in (
            ('torch', '2.2.2+cu118'), ('torch', '2.3.0'),
            ('torchvision', '0.17.2+cu118'), ('torchvision', '0.18.0'),
        ):
            with self.subTest(package=package, version=version):
                old = self.versions[package]['version']
                self.versions[package]['version'] = version
                try:
                    report = self.inspect()
                    self.assertFalse(report['ready'])
                    self.assertIn('version:' + package, report['blockers'])
                    self.assertNotIn('run_command', report)
                finally:
                    self.versions[package]['version'] = old

    def test_unique_current_gpu_is_bound_explicitly_in_generated_command(self):
        self.args.gpu_uuid = None
        report = self.inspect()
        self.assertTrue(report['ready'], report)
        arguments = shlex.split(report['run_command'])
        position = arguments.index('--gpu-uuid')
        self.assertEqual(arguments[position + 1], CURRENT_UUID)
        scratch_position = arguments.index('--scratch-root')
        self.assertEqual(arguments[scratch_position + 1], str(self.args.scratch_root))
        self.assertEqual(report['selected_gpu']['uuid'], CURRENT_UUID)

    def test_two_eligible_gpus_require_an_explicit_choice(self):
        self.inventory['gpus'].append(self.device(OTHER_UUID))
        self.args.gpu_uuid = None
        report = self.inspect()
        self.assertFalse(report['ready'])
        self.assertNotIn('run_command', report)
        self.args.gpu_uuid = OTHER_UUID
        explicit = self.inspect()
        self.assertTrue(explicit['ready'], explicit)
        self.assertEqual(explicit['selected_gpu']['uuid'], OTHER_UUID)

    def test_runtime_accepts_base_metadata_but_keeps_cuda_and_physical_binding_strict(self):
        runtime = {
            'python': '3.11.8 (runtime fixture)', 'torch': '2.2.2',
            'torchvision': '0.17.2', 'numpy': '1.26.4', 'pillow': '10.3.0',
            'cuda': '12.1', 'cudnn': 8902, 'device': 'cuda:0',
            'physical_gpu_uuid': CURRENT_UUID, 'cpu_threads': 8, 'seed': 42,
            'dtype': 'float32', 'amp': False, 'matmul_tf32': False,
            'cudnn_tf32': False, 'cudnn_benchmark': False, 'cudnn_deterministic': True,
        }
        self.module.require_runtime(runtime, expected_gpu_uuid=CURRENT_UUID)
        for field, wrong in (
            ('torch', '2.2.2+cu118'), ('cuda', '11.8'), ('cudnn', 8901),
            ('physical_gpu_uuid', OTHER_UUID), ('amp', True), ('matmul_tf32', True),
        ):
            with self.subTest(field=field, wrong=wrong), self.assertRaises(ValueError):
                self.module.require_runtime(dict(runtime, **{field: wrong}),
                                            expected_gpu_uuid=CURRENT_UUID)

    def test_same_filesystem_requires_the_sum_of_both_budgets(self):
        self.work_device = self.scratch_device = 11
        self.work_free = self.scratch_free = 50 * 2**30
        report = self.inspect()
        self.assertFalse(report['ready'], report)
        self.assertNotIn('run_command', report)
        self.work_free = self.scratch_free = 60 * 2**30
        enough = self.inspect()
        self.assertTrue(enough['ready'], enough)

    def test_existing_scratch_is_rejected_and_its_files_remain_unchanged(self):
        self.args.scratch_root.mkdir()
        sentinel = self.args.scratch_root / 'prior-evidence.txt'
        sentinel.write_bytes(b'preserve existing user data\n')
        report = self.inspect()
        self.assertFalse(report['ready'], report)
        self.assertNotIn('run_command', report)
        self.assertEqual(sentinel.read_bytes(), b'preserve existing user data\n')
        self.assertFalse(self.args.output.exists())

    def test_disk_report_uses_existing_destination_ancestor_and_its_device(self):
        # A second module keeps this real filesystem boundary outside the
        # controlled _disk inventory used by the resource scenarios above.
        actual = original.api()
        destination = self.root / 'not-created' / 'cache'
        with mock.patch.object(actual.shutil, 'disk_usage', return_value=types.SimpleNamespace(
                total=100 * 2**30, used=20 * 2**30, free=80 * 2**30)):
            report = actual._disk(destination, minimum_gate_bytes=WORK_REQUIRED)
        self.assertEqual(Path(report['checked_path']), self.root)
        self.assertEqual(report['device_id'], self.root.stat().st_dev)
        self.assertEqual(report['minimum_gate_bytes'], WORK_REQUIRED)
        self.assertFalse(destination.parent.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
