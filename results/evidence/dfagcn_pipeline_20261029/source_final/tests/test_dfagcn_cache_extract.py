"""Real file/cache checks and a numerical VGG CPU extraction integration test."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from external_baselines.dfagcn.identity import build_cache_identity


def api(name):
    try:
        return importlib.import_module('external_baselines.dfagcn.' + name)
    except ModuleNotFoundError as exc:
        raise AssertionError(name + ' is not implemented') from exc


def identity(samples=None):
    return build_cache_identity(manifest_sha256='a' * 64, class_names=['one', 'two'],
        samples=samples or [{'path': 'one/1.png', 'sha256': 'b' * 64},
                            {'path': 'two/2.png', 'sha256': 'c' * 64}],
        weights_sha256='d' * 64, preprocessing={'resize_mode': 'test_fixture'},
        feature_spec={'pooling_hw': [3, 4]}, source_sha256='e' * 64,
        stage='frozen_pretrained')


def batch(count=2):
    return {key: np.full((count, 4096) if key == 'fc' else (count, 12, channels), value,
                         dtype=np.float32)
            for key, channels, value in [('conv3', 256, -3), ('conv4', 512, -4),
                                         ('conv5', 512, -5), ('fc', 4096, 6)]}


class FeatureCacheTests(unittest.TestCase):
    def test_streamed_batches_reload_as_independent_tensor_rows(self):
        module = api('cache')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'cache'
            module.write_feature_cache(output, identity(), [batch(1), batch(1)])
            cache = module.FeatureCache(output, expected_identity=identity())
            self.assertEqual(len(cache), 2)
            self.assertTrue(all(isinstance(value, np.memmap) for value in cache.arrays.values()))
            first = cache[0]
            self.assertEqual(set(first), {'conv3', 'conv4', 'conv5', 'fc'})
            torch.testing.assert_close(first['conv3'], torch.full((12, 256), -3.0))
            torch.testing.assert_close(first['fc'], torch.full((4096,), 6.0))
            first['fc'][0] = 99
            self.assertEqual(cache[0]['fc'][0].item(), 6.0)
            with self.assertRaises(FileExistsError):
                module.write_feature_cache(output, identity(), [batch()])

    def test_cache_rejects_identity_order_mismatch_and_altered_file(self):
        module = api('cache')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'cache'
            module.write_feature_cache(output, identity(), [batch()])
            wrong = identity()
            wrong['samples'].reverse()
            with self.assertRaises(ValueError):
                module.FeatureCache(output, expected_identity=wrong)
            target = output / 'fc.npy'
            with target.open('r+b') as stream:
                stream.seek(-4, 2)
                stream.write(np.float32(44).tobytes())
            with self.assertRaises(ValueError):
                module.FeatureCache(output, expected_identity=identity())

    def test_incomplete_bad_batch_keeps_directory_without_complete_index(self):
        module = api('cache')
        mutations = [('missing', lambda x: x.pop('fc')),
                     ('dtype', lambda x: x.update(fc=x['fc'].astype(np.float64))),
                     ('shape', lambda x: x.update(conv3=np.ones((2, 12, 255), np.float32))),
                     ('nonfinite', lambda x: x['fc'].__setitem__((0, 0), np.nan))]
        with tempfile.TemporaryDirectory() as temp:
            for name, mutate in mutations:
                with self.subTest(name=name):
                    output = Path(temp) / name
                    values = batch()
                    mutate(values)
                    with self.assertRaises((TypeError, ValueError)):
                        module.write_feature_cache(output, identity(), [values])
                    self.assertTrue(output.exists())
                    self.assertFalse((output / 'index.json').exists())
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        module.FeatureCache(output, expected_identity=identity())

    def test_exact_sample_count_and_pooling_contract_are_enforced(self):
        module = api('cache')
        with tempfile.TemporaryDirectory() as temp:
            for count in (1, 3):
                with self.subTest(count=count), self.assertRaises(ValueError):
                    module.write_feature_cache(Path(temp) / str(count), identity(), [batch(count)])
            invalid = identity()
            invalid['feature_spec']['pooling_hw'] = [0, 4]
            with self.assertRaises(ValueError):
                module.write_feature_cache(Path(temp) / 'badpool', invalid, [batch()])

    def test_cache_rejects_forged_array_metadata_even_with_intact_data(self):
        module = api('cache')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'cache'
            module.write_feature_cache(output, identity(), [batch()])
            index_path = output / 'index.json'
            index = json.loads(index_path.read_text())
            index['arrays']['fc']['file'] = '../fc.npy'
            index_path.write_text(json.dumps(index))
            with self.assertRaises(ValueError):
                module.FeatureCache(output, expected_identity=identity())


class ExtractionContractTests(unittest.TestCase):
    def test_changed_file_and_record_order_rejected_before_output(self):
        extraction = api('extraction')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'one').mkdir()
            image_path = root / 'one/1.png'
            Image.new('RGB', (16, 16), (20, 30, 40)).save(image_path)
            records = [{'path': 'one/1.png', 'sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()}]
            declared = identity(records)
            declared['preprocessing'] = copy.deepcopy(extraction.PREPROCESSING)
            image_path.write_bytes(image_path.read_bytes() + b'changed')
            with self.assertRaises(ValueError):
                extraction.extract_cache(None, root, records, declared, root / 'output', pool_hw=[3, 4])
            self.assertFalse((root / 'output').exists())
            other = copy.deepcopy(records)
            other[0]['path'] = 'one/wrong.png'
            with self.assertRaises(ValueError):
                extraction.extract_cache(None, root, other, declared, root / 'output', pool_hw=[3, 4])

    def test_pool_and_preprocessing_mismatch_rejected(self):
        extraction = api('extraction')
        declared = identity()
        for pool, preprocessing in [([4, 4], extraction.PREPROCESSING), ([3, 4], {'channels': 'BGR'})]:
            declared['preprocessing'] = copy.deepcopy(preprocessing)
            with tempfile.TemporaryDirectory() as temp, self.assertRaises(ValueError):
                extraction.extract_cache(None, temp, declared['samples'], declared, Path(temp) / 'cache', pool_hw=pool)

    def test_weight_loader_rejects_wrong_sha_before_deserializing(self):
        extraction = api('extraction')
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'weights.pt'
            path.write_bytes(b'not a checkpoint')
            with self.assertRaises(ValueError):
                extraction.load_vgg_adapter(path, 'f' * 64)

    def test_supplied_real_vgg_weight_file_roundtrip_preserves_cpu_fc6(self):
        extraction = api('extraction')
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(min(previous_threads, 2))
        try:
            adapter = extraction.random_vgg_adapter(73)
            values = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
            expected = adapter.fc_features(values).clone()
            self.assertGreater(expected.abs().sum().item(), 0)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'local_random_vgg_state.pt'
                torch.save(adapter.backbone.state_dict(), path)
                digest = api('cache').sha256_file(path)
                del adapter
                loaded = extraction.load_vgg_adapter(path, digest)
                actual = loaded.fc_features(values)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            torch.set_num_threads(previous_threads)

    def test_real_vgg_cpu_fc6_and_pooled_convolutions_are_cached(self):
        extraction = api('extraction')
        old_threads = torch.get_num_threads()
        torch.set_num_threads(min(old_threads, 2))
        try:
            adapter = extraction.random_vgg_adapter(42)
            fingerprint = extraction.adapter_sha256(adapter)
            self.assertEqual(len(fingerprint), 64)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'one').mkdir()
                path = root / 'one/1.png'
                Image.new('RGB', (32, 32), (20, 30, 40)).save(path)
                records = [{'path': 'one/1.png', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}]
                declared = identity(records)
                declared['weights_sha256'] = fingerprint
                declared['preprocessing'] = copy.deepcopy(extraction.PREPROCESSING)
                declared['preprocessing']['weights_kind'] = 'random_smoke'
                cache = extraction.extract_cache(adapter, root, records, declared, root / 'cache', pool_hw=[3, 4])
                self.assertEqual(cache[0]['fc'].shape, (4096,))
                self.assertGreater(cache[0]['fc'].abs().sum().item(), 0)
                self.assertTrue(torch.all(cache[0]['fc'] >= 0))
                self.assertEqual(cache[0]['conv3'].shape, (12, 256))
                self.assertEqual(cache[0]['conv4'].shape, (12, 512))
                self.assertEqual(cache[0]['conv5'].shape, (12, 512))
                self.assertTrue(all(torch.isfinite(v).all() for v in cache[0].values()))
                original_writer = api('cache').write_feature_cache
                def changed_source_writer(*args, **kwargs):
                    Image.new('RGB', (32, 32), (90, 80, 70)).save(path)
                    return original_writer(*args, **kwargs)
                with patch.object(extraction, 'write_feature_cache', side_effect=changed_source_writer):
                    with self.assertRaisesRegex(ValueError, 'SHA256'):
                        extraction.extract_cache(adapter, root, records, declared,
                                                 root / 'changed_during_extract', pool_hw=[3, 4])
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
