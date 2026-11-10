"""Checkpoint identity and failure atomicity, using an inexpensive real module."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from external_baselines.hgtnet.config import config_sha256, load_config


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.checkpoint')
    except ModuleNotFoundError as exc:
        raise AssertionError('HGTNet checkpoint API is missing') from exc


class TinyCandidate(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = load_config('configs/hgtnet_independent_v0.json')
        self.config_sha256 = config_sha256(self.config)
        self.projection = nn.Linear(3, 2)
        self.register_buffer('counter', torch.tensor(7, dtype=torch.int64))

    def forward(self, x):
        return self.projection(x)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'new.pt'

    def identity(self):
        return {name: hashlib.sha256(name.encode()).hexdigest() for name in (
            'attention.py', 'config.py', 'graph.py', 'model.py', 'checkpoint.py', 'pretrained.py')}

    def sources(self):
        return patch.object(api(), 'source_identity', return_value=self.identity())

    def saved(self):
        with self.sources():
            model = TinyCandidate()
            report = api().save_checkpoint(model, self.path, pretraining={'mode': 'applied', 'is_original_hgtnet_weights': False})
        return model, report

    def altered(self, change):
        _, report = self.saved()
        content = torch.load(self.path, map_location='cpu', weights_only=True)
        change(content)
        torch.save(content, self.path)
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def assert_load_fails_without_mutation(self, digest):
        model = TinyCandidate()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        with self.sources(), self.assertRaises((ValueError, TypeError)):
            api().load_checkpoint(model, self.path, expected_sha256=digest)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]), key)

    def test_roundtrip_copies_parameters_buffers_and_json_identity(self):
        original, saved = self.saved()
        restored = TinyCandidate()
        with self.sources():
            loaded = api().load_checkpoint(restored, self.path, expected_sha256=saved['sha256'])
        self.assertEqual(saved['sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(saved['config_sha256'], original.config_sha256)
        self.assertEqual(loaded['source_identity'], self.identity())
        self.assertEqual(loaded['loaded_tensors'], len(original.state_dict()))
        self.assertEqual(loaded['pretraining']['mode'], 'applied')
        self.assertTrue(loaded['loaded'])
        self.assertEqual(loaded['mode'], 'loaded')
        self.assertFalse(saved['loaded'])
        self.assertFalse(loaded['official'])
        self.assertEqual(loaded['scope'], 'local_engineering_only')
        self.assertFalse(loaded['formal_training_started'])
        json.dumps(saved, allow_nan=False)
        json.dumps(loaded, allow_nan=False)
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
        inputs = torch.tensor([[1., 2., 3.]])
        self.assertTrue(torch.equal(original(inputs), restored(inputs)))

    def test_existing_file_or_symlink_is_never_overwritten(self):
        _, saved = self.saved()
        previous = self.path.read_bytes()
        with self.sources(), self.assertRaises(FileExistsError):
            api().save_checkpoint(TinyCandidate(), self.path)
        self.assertEqual(self.path.read_bytes(), previous)
        target = Path(self.temp.name) / 'link.pt'
        target.symlink_to(self.path)
        with self.sources(), self.assertRaises(FileExistsError):
            api().save_checkpoint(TinyCandidate(), target)
        self.assertEqual(self.path.read_bytes(), previous)

    def test_full_hash_is_required_and_wrong_hash_does_not_load(self):
        _, saved = self.saved()
        for digest in ('', '68c6b09e', 'x' * 64, None, '0' * 64):
            with self.subTest(digest=digest):
                self.assert_load_fails_without_mutation(digest)

    def test_metadata_source_and_config_changes_fail_before_copy(self):
        mutations = (
            lambda p: p.update(official=True),
            lambda p: p.update(scope='formal_training'),
            lambda p: p.update(schema='unrelated'),
            lambda p: p.update(formal_training_started=True),
            lambda p: p.update(synthetic_inputs=False),
            lambda p: p.update(accuracy=99),
            lambda p: p['source_identity'].update({'model.py': '0' * 64}),
            lambda p: p.update(config_sha256='0' * 64),
            lambda p: p['config'].update(num_classes=30),
        )
        for index, change in enumerate(mutations):
            with self.subTest(index=index):
                self.path = Path(self.temp.name) / f'bad_{index}.pt'
                self.assert_load_fails_without_mutation(self.altered(change))

    def test_missing_unknown_shape_dtype_sparse_and_nonfinite_reject_atomically(self):
        def missing(payload):
            payload['state_dict'].pop('projection.bias')
        def extra(payload):
            payload['state_dict']['unexpected'] = torch.zeros(1)
        def shape(payload):
            payload['state_dict']['projection.weight'] = torch.zeros(1)
        def dtype(payload):
            payload['state_dict']['projection.bias'] = payload['state_dict']['projection.bias'].double()
        def sparse(payload):
            payload['state_dict']['projection.weight'] = payload['state_dict']['projection.weight'].to_sparse()
        def nonfinite(payload):
            payload['state_dict']['projection.bias'][0] = float('nan')
        for index, change in enumerate((missing, extra, shape, dtype, sparse, nonfinite)):
            with self.subTest(index=index):
                self.path = Path(self.temp.name) / f'bad_tensor_{index}.pt'
                self.assert_load_fails_without_mutation(self.altered(change))

    def test_mutated_model_config_rejected_for_save_and_load(self):
        model, saved = self.saved()
        model.config['num_classes'] = 30
        with self.sources(), self.assertRaises(ValueError):
            api().save_checkpoint(model, Path(self.temp.name) / 'mutated.pt')
        self.assertFalse((Path(self.temp.name) / 'mutated.pt').exists())
        with self.sources(), self.assertRaises(ValueError):
            api().load_checkpoint(model, self.path, expected_sha256=saved['sha256'])

    def test_new_class_count_or_operator_identity_cannot_load(self):
        _, saved = self.saved()
        for key, value in [('num_classes', 30), ('graph_operator', 'relative_mean')]:
            target = TinyCandidate()
            target.config[key] = value
            target.config_sha256 = config_sha256(target.config)
            with self.subTest(key=key), self.sources(), self.assertRaises(ValueError):
                api().load_checkpoint(target, self.path, expected_sha256=saved['sha256'])

    def test_save_rejects_bad_values_or_non_json_provenance_without_file(self):
        model = TinyCandidate()
        with torch.no_grad():
            model.projection.weight[0, 0] = float('inf')
        with self.sources(), self.assertRaises(ValueError):
            api().save_checkpoint(model, self.path)
        self.assertFalse(self.path.exists())
        for value in ({'coverage': {'nan': float('nan')}}, {'coverage': {'tensor': torch.zeros(1)}},
                      ['wrong root'], {'unknown': 'report'}, {'is_original_hgtnet_weights': True}):
            with self.subTest(value=str(value)), self.sources(), self.assertRaises((ValueError, TypeError)):
                api().save_checkpoint(TinyCandidate(), self.path, pretraining=value)
            self.assertFalse(self.path.exists())

    def test_load_uses_safe_weights_only_mode(self):
        _, saved = self.saved()
        real_load = torch.load
        calls = []
        def observed(*args, **kwargs):
            calls.append(kwargs)
            return real_load(*args, **kwargs)
        with self.sources(), patch.object(api().torch, 'load', side_effect=observed):
            api().load_checkpoint(TinyCandidate(), self.path, expected_sha256=saved['sha256'])
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]['weights_only'])
        self.assertEqual(calls[0]['map_location'], 'cpu')

    def test_actual_source_identity_hashes_exact_model_modules(self):
        identity = api().source_identity()
        self.assertEqual(set(identity), set(self.identity()))
        folder = Path(api().__file__).resolve().parent
        for name, digest in identity.items():
            self.assertEqual(digest, hashlib.sha256((folder / name).read_bytes()).hexdigest())

    def test_serialization_error_removes_only_new_partial_file(self):
        def broken_save(payload, handle):
            handle.write(b'incomplete checkpoint')
            raise OSError('simulated writer failure')
        with self.sources(), patch.object(api().torch, 'save', side_effect=broken_save), self.assertRaises(OSError):
            api().save_checkpoint(TinyCandidate(), self.path)
        self.assertFalse(self.path.exists())

    def test_truncated_or_unsafe_files_reject_before_mutation(self):
        for index, contents in enumerate((b'not a checkpoint', b'PK\x03\x04incomplete zip')):
            self.path = Path(self.temp.name) / f'truncated_{index}.pt'
            self.path.write_bytes(contents)
            with self.subTest(index=index):
                self.assert_load_fails_without_mutation(hashlib.sha256(contents).hexdigest())


if __name__ == '__main__':
    unittest.main()
