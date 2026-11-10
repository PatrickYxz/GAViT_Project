"""AID input boundaries; all files here are explicit temporary CPU fixtures."""
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tests.test_hgtnet_aid_inputs import manifest_fixture


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.aid_data')
    except ModuleNotFoundError as exc:
        raise AssertionError('AID data interface must be implemented') from exc


class AidDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepare(self, value=None, raw=None, expected=None):
        raw = json.dumps(manifest_fixture() if value is None else value).encode() if raw is None else raw
        path = self.root / 'manifest.json'
        path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest() if expected is None else expected
        return api().prepare_inputs(path, expected_sha256=digest)

    def image(self, *, mode='RGB', size=(600, 600), split='train'):
        from PIL import Image
        stream = io.BytesIO()
        Image.new(mode, size, color=255 if mode == 'L' else (255, 0, 0)).save(stream, format='PNG')
        raw = stream.getvalue()
        path = self.root / 'Airport' / 'a.png'
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
        row = dict(path='Airport/a.png', label=0, split=split, sha256=hashlib.sha256(raw).hexdigest(),
                   pixel_sha256='1' * 64)
        return path, row

    def test_selection_preserves_unequal_classes_and_industrial_duplicate_group(self):
        value = manifest_fixture()
        group = [r for r in value['records'] if r['path'].startswith('Industrial/')][:2]
        group[1]['pixel_sha256'] = group[0]['pixel_sha256']
        value['duplicates'] = dict(unique_images=9999, extra_copies=1, groups=[dict(
            pixel_sha256=group[0]['pixel_sha256'], label=group[0]['label'],
            paths=[r['path'] for r in group], split='train')])
        prepared = self.prepare(value)
        self.assertEqual(prepared['counts'], {'train': 240, 'val': 60, 'test': 0})
        self.assertEqual(len(prepared['selected']['pool']), 300)
        self.assertEqual([r['path'] for r in prepared['selected']['train'][:2]],
                         ['Airport/00000.jpg', 'Airport/00001.jpg'])
        self.assertEqual([r['path'] for r in prepared['selected']['val'][:2]],
                         ['Airport/00080.jpg', 'Airport/00081.jpg'])
        selected_group = [r for r in prepared['selected']['train'] if r['path'] in {r['path'] for r in group}]
        self.assertEqual(len(selected_group), 2)
        self.assertEqual(selected_group[0]['pixel_sha256'], selected_group[1]['pixel_sha256'])
        prepared['selected']['train'][0]['label'] = 29
        self.assertEqual(prepared['selected']['pool'][0]['label'], 0)

    def test_bad_manifest_identity_json_labels_paths_splits_and_pixel_groups_fail_closed(self):
        with self.assertRaises(ValueError):
            self.prepare(expected='0' * 64)
        for raw in [b'{"dataset":"AID","dataset":"NWPU"}', b'{"bad":NaN}', b'not json']:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.prepare(raw=raw)
        mutations = [('label', True), ('label', 30), ('path', '../a.jpg'), ('split', 'pool'),
                     ('pixel_sha256', 'bad')]
        for key, bad in mutations:
            value = manifest_fixture(); value['records'][0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(value)
        value = manifest_fixture()
        value['records'][100]['pixel_sha256'] = value['records'][0]['pixel_sha256']
        with self.assertRaises(ValueError):
            self.prepare(value)

    def test_validation_reads_only_supplied_train_val_rows_and_rejects_test_before_open(self):
        path, row = self.image()
        result = api().validate_selected(self.root, [row])
        self.assertEqual(result['verified_count'], 1)
        self.assertEqual(result['test_image_bytes_read'], 0)
        path.unlink()
        test = dict(row, split='test')
        with self.assertRaises(ValueError):
            api().validate_selected(self.root, [test])

    def test_changed_bytes_outside_paths_and_symlink_components_are_rejected(self):
        path, row = self.image()
        for relative in ['../escape.png', '/outside.png', 'Airport//a.png', 'Airport/./a.png', 'Airport\\a.png']:
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                api().validate_selected(self.root, [dict(row, path=relative)])
        path.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            api().validate_selected(self.root, [row])
        path.unlink(); path.symlink_to(self.root / 'missing')
        with self.assertRaises(ValueError):
            api().validate_selected(self.root, [row])
        path.unlink(); path.parent.rename(self.root / 'real_class')
        path.parent.symlink_to(self.root / 'real_class', target_is_directory=True)
        with self.assertRaises(ValueError):
            api().validate_selected(self.root, [row])

    def test_dataset_uses_verified_rgb600_bytes_and_fixed_normalization(self):
        import torch
        path, row = self.image()
        records = [copy.deepcopy(row)]
        dataset = api().make_dataset(self.root, records)
        records[0]['label'] = 29
        exposed = dataset.records; exposed[0]['label'] = 29
        image, label = dataset[0]
        self.assertEqual(dataset.records, [row])
        self.assertEqual(tuple(image.shape), (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(label.item(), 0)
        self.assertTrue(torch.allclose(image[:, 0, 0], torch.tensor([
            (1.0 - .485) / .229, -.456 / .224, -.406 / .225]), atol=1e-6))
        path.write_bytes(b'changed after construction')
        with self.assertRaises(ValueError):
            dataset[0]

    def test_dataset_decodes_same_hashed_bytes_instead_of_reopening_path(self):
        from PIL import Image
        path, row = self.image()
        dataset = api().make_dataset(self.root, [row])
        actual_open = Image.open
        def guarded_open(source, *args, **kwargs):
            self.assertIsInstance(source, io.BytesIO)
            path.write_bytes(b'changed between hash and decode')
            return actual_open(source, *args, **kwargs)
        with mock.patch.object(Image, 'open', side_effect=guarded_open):
            image, label = dataset[0]
        self.assertEqual(label.item(), 0)
        self.assertEqual(tuple(image.shape), (3, 224, 224))

    def test_non_rgb_or_wrong_original_size_cannot_enter_dataset(self):
        for kwargs in [dict(mode='L'), dict(size=(224, 224))]:
            with self.subTest(kwargs=kwargs):
                _, row = self.image(**kwargs)
                dataset = api().make_dataset(self.root, [row])
                with self.assertRaises(ValueError):
                    dataset[0]

    def test_new_modules_import_with_ml_forbidden(self):
        script = '''import builtins, sys, types\nsys.modules['tools'] = types.ModuleType('tools')\noriginal = builtins.__import__\ndef guarded(name, *a, **k):\n if name.split('.')[0] in {'torch','torchvision','numpy','PIL','timm'}: raise AssertionError(name)\n return original(name,*a,**k)\nbuiltins.__import__ = guarded\nfrom external_baselines.hgtnet import aid_data, aid_engine\nassert aid_data.MANIFEST_SHA == '30ac5760bf0c002c9a7fe788a0d19fc4caf0b6f7cccb7d71d959ee0d2903aa22'\n'''
        run = subprocess.run([sys.executable, '-c', script], cwd=Path(__file__).resolve().parents[1],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)


if __name__ == '__main__':
    unittest.main()
