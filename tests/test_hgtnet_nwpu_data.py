"""Frozen NWPU selection, safe recovery, and real deterministic image tensors."""
from collections import Counter
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def adapter():
    try:
        return importlib.import_module('external_baselines.hgtnet.nwpu_data')
    except ModuleNotFoundError as exc:
        if exc.name == 'external_baselines.hgtnet.nwpu_data':
            raise AssertionError('HGTNet frozen NWPU data adapter is not implemented') from exc
        raise


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_fixture():
    names = [f'class_{label:02d}' for label in range(45)]
    rows = []
    for label, name in enumerate(names):
        for split, count in [('train', 112), ('val', 28), ('test', 560)]:
            rows.extend({'path': f'{name}/{split}_{index:03d}.jpg', 'label': label,
                         'split': split, 'sha256': '1' * 64} for index in range(count))
    return {'schema_version': 2, 'dataset': 'NWPU-RESISC45', 'class_names': names,
            'counts': {'train': 5040, 'val': 1260, 'test': 25200}, 'records': rows}


def image_record(path, raw, *, label=0, split='train'):
    return {'path': path, 'label': label, 'split': split,
            'sha256': hashlib.sha256(raw).hexdigest()}


class NwpuPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_manifest(self, value=None):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(value if value is not None else manifest_fixture()))
        return path

    def test_complete_manifest_selects_fixed_disjoint_360_90_in_path_order(self):
        value = adapter()
        path = self.write_manifest()
        prepared = value.prepare_inputs(path, expected_sha256=digest(path))
        self.assertEqual(prepared['counts'], {'train': 360, 'val': 90, 'pool': 450, 'test': 0})
        self.assertEqual(prepared['manifest_sha256'], digest(path))
        self.assertEqual(prepared['class_names'], [f'class_{i:02d}' for i in range(45)])
        selected = prepared['selected']
        self.assertEqual(selected['train'][0]['path'], 'class_00/train_000.jpg')
        self.assertEqual(selected['train'][-1]['path'], 'class_44/train_007.jpg')
        self.assertEqual(selected['val'][0]['path'], 'class_00/val_000.jpg')
        self.assertEqual(selected['val'][-1]['path'], 'class_44/val_001.jpg')
        self.assertEqual(Counter(row['label'] for row in selected['train']), dict.fromkeys(range(45), 8))
        self.assertEqual(Counter(row['label'] for row in selected['val']), dict.fromkeys(range(45), 2))
        self.assertEqual(len({row['path'] for row in selected['pool']}), 450)
        self.assertTrue(all(row['split'] in ('train', 'val') for row in selected['pool']))
        self.assertEqual([r['path'] for r in selected['pool']], sorted(r['path'] for r in selected['pool']))
        self.assertEqual(value.prepare_inputs(path, expected_sha256=digest(path)), prepared)

    def test_manifest_default_digest_rejects_fixture_and_changed_bytes(self):
        value = adapter()
        path = self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'SHA'):
            value.prepare_inputs(path)
        expected = digest(path)
        path.write_text(path.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            value.prepare_inputs(path, expected_sha256=expected)

    def test_manifest_rejects_invalid_counts_path_label_classmap_or_duplicate(self):
        value = adapter()
        for problem in ('counts', 'path', 'label', 'classmap', 'duplicate', 'per_class'):
            document = manifest_fixture()
            if problem == 'counts':
                document['counts']['train'] = 5039
            elif problem == 'path':
                document['records'][0]['path'] = '../outside.jpg'
            elif problem == 'label':
                document['records'][0]['label'] = True
            elif problem == 'classmap':
                document['records'][0]['label'] = 1
            elif problem == 'duplicate':
                document['records'][1] = copy.deepcopy(document['records'][0])
            else:
                document['records'][0]['split'] = 'val'
            path = self.write_manifest(document)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                value.prepare_inputs(path, expected_sha256=digest(path))

    def test_preparation_and_validation_import_no_machine_learning_libraries(self):
        adapter()
        path = self.write_manifest()
        script = '''import hashlib, importlib.abc, sys, tempfile, zipfile
from pathlib import Path
class BlockML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','torchvision','numpy','PIL'}:
            raise AssertionError('ML import during stdlib preparation: '+fullname)
sys.meta_path.insert(0, BlockML())
from external_baselines.hgtnet.nwpu_data import prepare_inputs, validate_selected, recover_selected
result=prepare_inputs(sys.argv[1], expected_sha256=sys.argv[2])
assert result['counts']=={'train':360,'val':90,'pool':450,'test':0}
with tempfile.TemporaryDirectory() as folder:
    root=Path(folder)
    source=root/'source'
    (source/'class_00').mkdir(parents=True)
    (source/'class_00/a.jpg').write_bytes(b'a')
    rows=[{'path':'class_00/a.jpg','label':0,'split':'train','sha256':hashlib.sha256(b'a').hexdigest()}]
    assert validate_selected(source,rows)
    archive=root/'fixture.zip'
    with zipfile.ZipFile(archive,'w') as stream:
        stream.writestr('class_00/a.jpg',b'a')
    recovered=recover_selected(archive,rows,root/'recovered',expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    assert (recovered/'class_00/a.jpg').read_bytes()==b'a'
print('stdlib preparation passed')
'''
        completed = subprocess.run([sys.executable, '-c', script, str(path), digest(path)],
                                   cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('stdlib preparation passed', completed.stdout)

    def test_selected_validation_rejects_test_duplicate_bad_label_and_parent_symlink(self):
        value = adapter()
        folder = self.root / 'class_00'
        folder.mkdir()
        image = folder / 'a.jpg'
        image.write_bytes(b'fixture')
        row = image_record('class_00/a.jpg', b'fixture')
        self.assertTrue(value.validate_selected(self.root, [row]))
        for records in ([dict(row, split='test')], [row, row], [dict(row, label=True)],
                        [dict(row, label=45)], [dict(row, path='../a.jpg')]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                value.validate_selected(self.root, records)
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'a.jpg').write_bytes(b'fixture')
        linked_root = self.root / 'linked_root'
        linked_root.mkdir()
        (linked_root / 'class_00').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            value.validate_selected(linked_root, [row])

    def test_recovery_reads_only_train_val_and_never_overwrites(self):
        value = adapter()
        archive = self.root / 'images.zip'
        rows = [image_record('class_00/a.jpg', b'a'),
                image_record('class_01/b.jpg', b'b', label=1, split='val')]
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('NWPU-RESISC45/class_00/a.jpg', b'a')
            stream.writestr('NWPU-RESISC45/class_01/b.jpg', b'b')
            stream.writestr('NWPU-RESISC45/class_00/test.jpg', b'test')
        output = value.recover_selected(archive, rows, self.root / 'recovered',
                                        expected_sha256=digest(archive))
        self.assertEqual(sorted(p.relative_to(output).as_posix() for p in output.rglob('*.jpg')),
                         ['class_00/a.jpg', 'class_01/b.jpg'])
        self.assertEqual(json.loads((output / 'recovery.json').read_text())['test_images_extracted'], 0)
        with self.assertRaises(FileExistsError):
            value.recover_selected(archive, rows, output, expected_sha256=digest(archive))
        with self.assertRaises(ValueError):
            value.recover_selected(archive, [dict(rows[0], split='test')], self.root / 'test_output',
                                   expected_sha256=digest(archive))
        self.assertFalse((self.root / 'test_output').exists())

    def test_recovery_rejects_bad_zip_hash_unsafe_members_or_image_hash(self):
        value = adapter()
        for problem in ('archive_hash', 'traversal', 'ambiguous', 'symlink', 'image_hash'):
            archive = self.root / f'{problem}.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('class_00/a.jpg', b'bad' if problem == 'image_hash' else b'a')
                if problem == 'traversal':
                    stream.writestr('../outside.jpg', b'x')
                elif problem == 'ambiguous':
                    stream.writestr('NWPU-RESISC45/class_00/a.jpg', b'a')
                elif problem == 'symlink':
                    entry = zipfile.ZipInfo('class_00/link.jpg')
                    entry.create_system = 3
                    entry.external_attr = 0o120777 << 16
                    stream.writestr(entry, 'a.jpg')
            output = self.root / problem
            expected = '0' * 64 if problem == 'archive_hash' else digest(archive)
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                value.recover_selected(archive, [image_record('class_00/a.jpg', b'a')], output,
                                       expected_sha256=expected)
            self.assertFalse((output / 'recovery.json').exists())
        self.assertFalse((self.root / 'outside.jpg').exists())


class RealImageDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def jpeg(self, name, mode, color):
        from PIL import Image
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new(mode, (37, 19), color).save(path, format='JPEG', quality=100, subsampling=0)
        return image_record(name, path.read_bytes(), label=int(name[6:8]))

    def test_real_rgb_and_grayscale_jpeg_are_deterministic_imagenet_tensors(self):
        value = adapter()
        import torch
        rows = [self.jpeg('class_00/rgb.jpg', 'RGB', (64, 128, 192)),
                self.jpeg('class_01/gray.jpg', 'L', 128)]
        dataset = value.make_dataset(self.root, rows)
        self.assertEqual(len(dataset), 2)
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(image).all()))
        self.assertEqual(label.dtype, torch.long)
        self.assertEqual(label.item(), 0)
        self.assertTrue(torch.equal(dataset[0][0], image))
        expected = torch.tensor([(64 / 255 - .485) / .229, (128 / 255 - .456) / .224,
                                 (192 / 255 - .406) / .225])
        self.assertTrue(torch.allclose(image[:, 100, 100], expected, atol=1e-6))
        gray, gray_label = dataset[1]
        expected_gray = torch.tensor([(128 / 255 - .485) / .229, (128 / 255 - .456) / .224,
                                      (128 / 255 - .406) / .225])
        self.assertTrue(torch.allclose(gray[:, 80, 80], expected_gray, atol=1e-6))
        self.assertEqual(gray_label.item(), 1)

    def test_image_resize_uses_bilinear_and_preserves_record_order(self):
        value = adapter()
        from PIL import Image
        import numpy as np
        import torch
        path = self.root / 'class_00' / 'pattern.jpg'
        path.parent.mkdir()
        pixels = np.zeros((7, 11, 3), dtype=np.uint8)
        pixels[:, ::2] = 255
        Image.fromarray(pixels).save(path, quality=100, subsampling=0)
        row = image_record('class_00/pattern.jpg', path.read_bytes())
        other = self.jpeg('class_01/other.jpg', 'L', 64)
        dataset = value.make_dataset(self.root, [other, row])
        self.assertEqual(dataset[0][1].item(), 1)
        actual = dataset[1][0]
        with Image.open(path) as opened:
            resized = opened.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
            expected = torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255
        expected = (expected - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        self.assertTrue(torch.equal(actual, expected))

    def test_dataset_checks_hash_before_use_and_owns_record_snapshot(self):
        value = adapter()
        row = self.jpeg('class_00/a.jpg', 'RGB', (64, 128, 192))
        rows = [row]
        dataset = value.make_dataset(self.root, rows)
        row['label'] = 30
        rows.clear()
        self.assertEqual(dataset[0][1].item(), 0)
        path = self.root / 'class_00/a.jpg'
        path.write_bytes(b'not a JPEG and different bytes')
        with self.assertRaisesRegex(ValueError, 'SHA'):
            dataset[0]
        with self.assertRaisesRegex(ValueError, 'SHA'):
            value.make_dataset(self.root, [dict(row, label=0)])

    def test_dataset_rejects_test_rows_and_inconsistent_class_labels(self):
        value = adapter()
        row = self.jpeg('class_00/a.jpg', 'L', 128)
        other = self.jpeg('class_00/b.jpg', 'L', 64)
        for records in ([dict(row, split='test')], [row, dict(other, label=1)],
                        [dict(row, label=-1)], [dict(row, sha256='bad')]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                value.make_dataset(self.root, records)


if __name__ == '__main__':
    unittest.main()
