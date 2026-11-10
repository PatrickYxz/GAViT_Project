"""Behavior checks for the standalone, read-only AID inventory tool."""
import builtins
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1] / 'tools' / 'inspect_hgtnet_aid_inputs.py'
NAMES = sorted('Airport BareLand BaseballField Beach Bridge Center Church Commercial DenseResidential Desert Farmland Forest Industrial Meadow MediumResidential Mountain Park Parking Playground Pond Port RailwayStation Resort River School SparseResidential Square Stadium StorageTanks Viaduct'.split())
CONTENT = b'AID-smoke-fixture\n'
DIGEST = hashlib.sha256(CONTENT).hexdigest()


def manifest_fixture():
    rows = []
    for label, name in enumerate(NAMES):
        size = (200, 300, 500)[label // 10]
        for index in range(size):
            split = 'train' if index < size * 4 // 10 else ('val' if index < size // 2 else 'test')
            relative = f'{name}/{index:05d}.jpg'
            rows.append(dict(path=relative, label=label, split=split, sha256=DIGEST,
                             pixel_sha256=hashlib.sha256(relative.encode()).hexdigest()))
    return dict(schema_version=2, dataset='AID', class_names=NAMES[:],
                protocol=dict(version='scene-pool-group-v2', pool_ratio=0.5, val_fraction_of_pool=0.2,
                              split_seed=42, val_split_seed=4242, rounding='floor_per_class',
                              assignment='sorted_pixel_groups_seeded_exact_subset_v1', duplicate_policy='group_within_class'),
                counts=dict(train=4000, val=1000, test=5000), records=rows,
                duplicates=dict(unique_images=10000, extra_copies=0, groups=[]))


class AidInputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = manifest_fixture()

    def setUp(self):
        self.assertTrue(SOURCE.is_file(), 'Standalone read-only AID inspector must exist')
        spec = importlib.util.spec_from_file_location('aid_inputs_under_test', SOURCE)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / 'manifest.json'
        self.data = self.root / 'data'
        self.data.mkdir()

    def inspect(self, value=None, expected=None):
        raw = json.dumps(self.fixture if value is None else value, sort_keys=True).encode()
        self.manifest.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest() if expected is None else expected
        return self.tool.inspect_manifest(self.manifest, self.data, expected_sha256=sha)

    def write_selected(self, value=None):
        value = self.fixture if value is None else value
        for name in NAMES:
            for split, count in [('train', 8), ('val', 2)]:
                rows = sorted((r for r in value['records'] if r['path'].startswith(name + '/') and r['split'] == split), key=lambda r: r['path'])[:count]
                for row in rows:
                    target = self.data / row['path']
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(CONTENT)

    def test_selects_240_60_from_unequal_classes_and_reads_no_test_bytes(self):
        self.write_selected()
        test_paths = {str(self.data / r['path']) for r in self.fixture['records'] if r['split'] == 'test'}
        real_open = self.tool.open_regular_readonly
        def guarded_open(path, *args, **kwargs):
            if str(path) in test_paths:
                raise AssertionError('Test image bytes must not be read')
            return real_open(path, *args, **kwargs)
        with mock.patch.object(self.tool, 'open_regular_readonly', side_effect=guarded_open):
            result = self.inspect()
        self.assertEqual(result['status'], 'verified')
        self.assertEqual(result['selected_counts'], {'train': 240, 'val': 60, 'test': 0})
        self.assertEqual(result['image_checks']['verified_count'], 300)
        self.assertEqual(result['counts'], {'train': 4000, 'val': 1000, 'pool': 5000, 'test': 5000})
        self.assertEqual(result['selected_paths']['train'][:2], ['Airport/00000.jpg', 'Airport/00001.jpg'])
        self.assertEqual(result['test_image_bytes_read'], 0)

    def test_missing_manifest_and_bad_sha_do_not_read_images(self):
        missing = self.tool.inspect_manifest(self.manifest, self.data)
        self.assertEqual(missing['status'], 'missing')
        self.assertFalse(missing['identity_verified'])
        with mock.patch.object(self.tool, 'inspect_selected_images', side_effect=AssertionError('Untrusted manifest used')):
            bad = self.inspect(expected='0' * 64)
        self.assertEqual(bad['status'], 'invalid')
        self.assertIn('manifest_sha256_mismatch', bad['errors'])

    def test_invalid_metadata_paths_labels_hashes_counts_and_pixel_groups(self):
        cases = []
        for field, bad in [('path', '../escape.jpg'), ('path', '/escape.jpg'), ('path', 'Airport//a.jpg'),
                           ('path', 'Airport/./a.jpg'), ('path', 'Airport\\a.jpg'), ('label', True),
                           ('label', 30), ('label', 1), ('split', 'pool'), ('sha256', 'not-a-hash'),
                           ('pixel_sha256', 'f' * 63)]:
            value = copy.deepcopy(self.fixture)
            value['records'][0][field] = bad
            cases.append((f'{field}:{bad}', value))
        value = copy.deepcopy(self.fixture); value['records'][1]['path'] = value['records'][0]['path']; cases.append(('duplicate_path', value))
        value = copy.deepcopy(self.fixture); value['class_names'][0] = 'Unknown'; cases.append(('taxonomy', value))
        value = copy.deepcopy(self.fixture); value['class_names'].reverse(); cases.append(('unsorted_classes', value))
        value = copy.deepcopy(self.fixture); value['protocol']['split_seed'] = 43; cases.append(('seed', value))
        value = copy.deepcopy(self.fixture); value['protocol']['pool_ratio'] = 0.2; cases.append(('ratio', value))
        value = copy.deepcopy(self.fixture); value['counts']['train'] = 3999; cases.append(('counts', value))
        value = copy.deepcopy(self.fixture); value['records'].pop(); cases.append(('total', value))
        value = copy.deepcopy(self.fixture); value['records'][200]['pixel_sha256'] = value['records'][0]['pixel_sha256']; cases.append(('crossclass_pixels', value))
        value = copy.deepcopy(self.fixture); value['records'][100]['pixel_sha256'] = value['records'][0]['pixel_sha256']; cases.append(('crosssplit_pixels', value))
        for field in ['schema_version', 'dataset', 'class_names', 'protocol', 'counts', 'records', 'duplicates']:
            value = copy.deepcopy(self.fixture); value.pop(field); cases.append(('missing_' + field, value))
        value = copy.deepcopy(self.fixture); value['records'][0].pop('pixel_sha256'); cases.append(('missing_record_field', value))
        for name, value in cases:
            with self.subTest(name=name):
                result = self.inspect(value)
                self.assertEqual(result['status'], 'invalid')
                self.assertFalse(result['structure_verified'])
                self.assertEqual(result['image_checks']['checked_count'], 0)

    def test_same_class_same_split_duplicate_group_is_preserved(self):
        value = copy.deepcopy(self.fixture)
        rows = [r for r in value['records'] if r['path'].startswith('Industrial/')][:2]
        rows[1]['pixel_sha256'] = rows[0]['pixel_sha256']
        value['duplicates'] = dict(unique_images=9999, extra_copies=1, groups=[dict(pixel_sha256=rows[0]['pixel_sha256'], label=rows[0]['label'], paths=[r['path'] for r in rows], split='train')])
        self.write_selected(value)
        self.assertEqual(self.inspect(value)['status'], 'verified')

    def test_missing_images_are_reported_without_claiming_verification(self):
        result = self.inspect()
        self.assertEqual(result['status'], 'missing_images')
        self.assertEqual(result['image_checks']['missing_count'], 300)
        self.assertEqual(result['image_checks']['verified_count'], 0)
        self.assertTrue(result['structure_verified'])

    def test_changed_image_and_symlinks_are_rejected(self):
        self.write_selected()
        target = self.data / 'Airport/00000.jpg'
        target.write_bytes(b'changed')
        result = self.inspect()
        self.assertEqual(result['status'], 'invalid_images')
        self.assertEqual(result['image_checks']['invalid'][0]['reason'], 'sha256_mismatch')
        target.unlink(); target.symlink_to(self.root / 'outside.jpg')
        (self.root / 'outside.jpg').write_bytes(CONTENT)
        result = self.inspect()
        self.assertEqual(result['image_checks']['invalid'][0]['reason'], 'unsafe_path')
        target.unlink(); target.write_bytes(CONTENT)
        airport = self.data / 'Airport'; airport.rename(self.root / 'Airport')
        airport.symlink_to(self.root / 'Airport', target_is_directory=True)
        self.assertEqual(self.inspect()['status'], 'invalid_images')

    def test_manifest_and_weights_symlinks_are_not_followed(self):
        self.inspect()
        linked = self.root / 'linked.json'; linked.symlink_to(self.manifest)
        self.assertEqual(self.tool.inspect_manifest(linked, self.data)['status'], 'invalid')
        weights = self.root / 'weights.pth'; weights.write_bytes(CONTENT)
        linked_weights = self.root / 'linked.pth'; linked_weights.symlink_to(weights)
        self.assertEqual(self.tool.inspect_weights(linked_weights)['status'], 'invalid')

    def test_malformed_json_and_non_regular_manifest_return_inventory_errors(self):
        for raw in [b'{broken', b'{"schema_version":2,"schema_version":2}', b'NaN', b'[]']:
            with self.subTest(raw=raw):
                self.manifest.write_bytes(raw)
                result = self.tool.inspect_manifest(self.manifest, self.data, expected_sha256=hashlib.sha256(raw).hexdigest())
                self.assertEqual(result['status'], 'invalid')
                self.assertEqual(result['image_checks']['checked_count'], 0)
        self.assertEqual(self.tool.inspect_manifest(self.data, self.data)['status'], 'invalid')

    def test_gpu_inventory_is_bounded_and_preserves_actual_uuid(self):
        raw = '0, GPU-observed-example, Example GPU, 24576, 555.42\n'
        with mock.patch.object(self.tool.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, raw, '')) as run:
            result = self.tool.inspect_gpu()
        self.assertEqual(result['status'], 'inventoried')
        self.assertEqual(result['devices'][0]['uuid'], 'GPU-observed-example')
        self.assertEqual(result['devices'][0]['memory_total_mib'], 24576)
        self.assertEqual(run.call_args.args[0], ['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version', '--format=csv,noheader,nounits'])
        self.assertEqual(run.call_args.kwargs['timeout'], 10)
        with mock.patch.object(self.tool.subprocess, 'run', side_effect=subprocess.TimeoutExpired('nvidia-smi', 10)):
            result = self.tool.inspect_gpu()
        self.assertEqual(result['status'], 'unavailable')
        self.assertEqual(result['devices'], [])

    def test_weights_stream_full_hash_and_report_missing_or_mismatch(self):
        weights = self.root / 'weights.pth'
        self.assertEqual(self.tool.inspect_weights(weights)['status'], 'missing')
        weights.write_bytes(CONTENT * 100000)
        result = self.tool.inspect_weights(weights)
        self.assertEqual(result['status'], 'invalid')
        self.assertEqual(result['actual_sha256'], hashlib.sha256(CONTENT * 100000).hexdigest())
        self.assertEqual(result['size_bytes'], len(CONTENT) * 100000)

    def test_cli_is_stdout_json_with_inventory_even_when_assets_missing(self):
        result = subprocess.run([sys.executable, '-B', str(SOURCE), 'inspect', '--manifest', str(self.manifest), '--data-root', str(self.data), '--weights', str(self.root / 'weights.pth')], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertFalse(value['training_started'])
        self.assertFalse(value['training_entry_available'])
        self.assertEqual(value['status'], 'INPUT_CHECK_INCOMPLETE')
        self.assertEqual(value['manifest']['status'], 'missing')
        self.assertEqual(value['weights']['status'], 'missing')
        self.assertIn('gpu', value); self.assertIn('disk', value); self.assertIn('package_versions', value)
        invalid = subprocess.run([sys.executable, '-B', str(SOURCE), 'train'], capture_output=True, text=True)
        self.assertNotEqual(invalid.returncode, 0)

    def test_cli_rejects_fixture_sha_and_has_no_override_flag(self):
        self.inspect()
        result = subprocess.run([sys.executable, '-B', str(SOURCE), 'inspect', '--manifest', str(self.manifest), '--data-root', str(self.data)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        value = json.loads(result.stdout)
        self.assertFalse(value['manifest']['identity_verified'])
        self.assertIn('manifest_sha256_mismatch', value['manifest']['errors'])
        override = subprocess.run([sys.executable, '-B', str(SOURCE), 'inspect', '--expected-sha256', '0' * 64], capture_output=True, text=True)
        self.assertNotEqual(override.returncode, 0)

    def test_import_and_main_are_stdlib_only_and_do_not_write(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        real_import = builtins.__import__
        real_open = builtins.open
        real_os_open = os.open
        def guarded_import(name, *args, **kwargs):
            top = name.split('.')[0]
            if top in {'torch', 'torchvision', 'PIL', 'numpy', 'timm', 'scipy', 'external_baselines'}:
                raise AssertionError('ML import forbidden: ' + name)
            return real_import(name, *args, **kwargs)
        def guarded_open(path, mode='r', *args, **kwargs):
            if any(c in mode for c in 'wax+'):
                raise AssertionError('File writes forbidden')
            return real_open(path, mode, *args, **kwargs)
        def guarded_os_open(path, flags, *args, **kwargs):
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError('File writes forbidden')
            return real_os_open(path, flags, *args, **kwargs)
        output = io.StringIO()
        with mock.patch('builtins.__import__', side_effect=guarded_import), mock.patch('builtins.open', side_effect=guarded_open), mock.patch('os.open', side_effect=guarded_os_open), contextlib.redirect_stdout(output):
            spec = importlib.util.spec_from_file_location('guarded_aid_inspector', SOURCE)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            code = module.main(['inspect', '--manifest', str(self.manifest), '--data-root', str(self.data), '--weights', str(self.root / 'absent.pth')])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue())['status'], 'INPUT_CHECK_INCOMPLETE')
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))


if __name__ == '__main__':
    unittest.main()
