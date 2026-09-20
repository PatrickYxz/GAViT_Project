import importlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image


def module(name):
    try:
        return importlib.import_module('scene_classification.' + name)
    except ModuleNotFoundError as exc:
        raise AssertionError('scene_classification.' + name + ' not implemented') from exc


def fixture(root, count=10):
    for label, cls in enumerate(('alpha', 'beta')):
        (root / cls).mkdir(parents=True)
        for i in reversed(range(count)):
            Image.new('RGB', (16, 16), (label * 80, i * 10, 33)).save(root / cls / f'{i:02}.png')


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'images'
        fixture(self.root)
        self.out = Path(self.tmp.name) / 'manifest.json'

    def prepare(self):
        return module('data').prepare_manifest(self.root, self.out, dataset='synthetic')

    def test_deterministic_disjoint_relative_manifest(self):
        m = self.prepare()
        second = Path(self.tmp.name) / 'again.json'
        module('data').prepare_manifest(self.root, second, dataset='synthetic')
        self.assertEqual(self.out.read_bytes(), second.read_bytes())
        self.assertEqual(m['counts'], {'train': 8, 'val': 2, 'test': 10})
        self.assertEqual(len({r['path'] for r in m['records']}), 20)
        self.assertTrue(all(not Path(r['path']).is_absolute() for r in m['records']))
        self.assertEqual(m['class_names'], ['alpha', 'beta'])
        self.assertEqual(m['protocol']['split_seed'], 42)
        self.assertEqual(m['protocol']['val_split_seed'], 4242)

    def test_relocated_root_is_valid(self):
        self.prepare()
        new_root = self.root.with_name('relocated')
        self.root.rename(new_root)
        m = module('data').load_manifest(self.out, new_root)
        self.assertEqual(len(m['records']), 20)

    def test_output_is_never_overwritten(self):
        self.prepare()
        before = self.out.read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(self.out.read_bytes(), before)

    def test_duplicate_pixels_rejected_even_different_encoding(self):
        with Image.open(self.root / 'alpha/00.png') as im:
            im.save(self.root / 'beta/copy.bmp')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_corrupt_image_rejected(self):
        (self.root / 'alpha/00.png').write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError, 'image'):
            self.prepare()

    def test_changed_image_or_missing_file_rejected(self):
        self.prepare()
        Image.new('RGB', (16, 16), (1, 2, 3)).save(self.root / 'alpha/00.png')
        with self.assertRaisesRegex(ValueError, 'changed'):
            module('data').load_manifest(self.out, self.root)
        (self.root / 'alpha/00.png').unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            module('data').load_manifest(self.out, self.root)

    def test_wrong_label_or_path_escape_rejected(self):
        m = self.prepare()
        m['records'][0]['label'] = 1
        self.out.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            module('data').load_manifest(self.out, self.root)
        m['records'][0]['path'] = '../elsewhere.png'
        self.out.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            module('data').load_manifest(self.out, self.root)

    def test_wrong_count_or_duplicate_record_rejected(self):
        m = self.prepare()
        m['records'].append(m['records'][0])
        self.out.write_text(json.dumps(m))
        with self.assertRaises(ValueError):
            module('data').load_manifest(self.out, self.root)

    def test_toy_cannot_masquerade_as_aid(self):
        with self.assertRaisesRegex(ValueError, 'AID'):
            module('data').prepare_manifest(self.root, self.out, dataset='AID')

    def test_fixed_smoke_samples_never_include_test(self):
        m = self.prepare()
        sample = module('data').select_records(m, 'train', per_class=2)
        self.assertEqual(len(sample), 4)
        self.assertTrue(all(r['split'] == 'train' for r in sample))
        self.assertEqual(sample, module('data').select_records(m, 'train', per_class=2))
        pool = module('data').select_records(m, 'pool')
        self.assertEqual(len(pool), 10)
        self.assertTrue(all(r['split'] != 'test' for r in pool))


class MetricTests(unittest.TestCase):
    def test_metrics_include_absent_classes_and_correct_orientation(self):
        # Rows=true, columns=predicted; class2 has zero support and remains in macro-F1.
        result = module('metrics').classification_metrics([0, 0, 1, 1], [0, 1, 1, 1], ['a', 'b', 'c'])
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 2, 0], [0, 0, 0]])
        self.assertEqual(result['oa_percent'], 75.0)
        self.assertAlmostEqual(result['macro_f1_percent'], (2/3 + 4/5) / 3 * 100)
        self.assertEqual(result['per_class'][0]['accuracy_percent'], 50.0)
        self.assertIsNone(result['per_class'][2]['accuracy_percent'])
        self.assertEqual(result['per_class'][2]['support'], 0)

    def test_metric_input_errors_are_rejected(self):
        for truth, pred in [([], []), ([0], []), ([3], [0]), ([0], [-1]), ([0.5], [0])]:
            with self.subTest(truth=truth, pred=pred), self.assertRaises(ValueError):
                module('metrics').classification_metrics(truth, pred, ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
