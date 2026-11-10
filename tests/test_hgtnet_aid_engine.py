"""Actual finite CPU tensor updates/reload; never real AID or GPU evidence."""
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from external_baselines.hgtnet.config import config_sha256, load_config


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.aid_engine')
    except ModuleNotFoundError as exc:
        raise AssertionError('AID engine interface must be implemented') from exc


class MiniHGT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(3, 4)
        stage = nn.Module(); stage.transformer = nn.Linear(4, 4); stage.da = nn.Linear(4, 4)
        self.stages = nn.ModuleList([stage])
        self.graph_branch = nn.Linear(4, 4)
        self.head = nn.Linear(4, 30)

    def forward(self, x):
        x = torch.tanh(self.patch_embed(x.mean((2, 3))))
        x = torch.tanh(self.stages[0].transformer(x) + self.stages[0].da(x))
        return self.head(torch.tanh(self.graph_branch(x)))


class AidEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def binding(self, train=4, val=2):
        cfg = load_config('configs/hgtnet_independent_v0.json'); cfg['num_classes'] = 30
        return dict(schema='hgtnet-aid-smoke-binding-v1', input_scope='cpu_fixture', config=cfg,
                    config_sha256=config_sha256(cfg), counts=dict(train=train, val=val, test=0),
                    class_names=[f'class{i:02}' for i in range(30)], runtime=dict(device='cpu'),
                    manifest_sha256='0' * 64, weights_sha256='1' * 64, source_identity={'fixture': '2' * 64},
                    val_paths=[f'class{i % 30:02}/v{i}.png' for i in range(val)],
                    val_labels=[i % 30 for i in range(val)])

    def loaders(self, train=4, val=2, *, shuffle=True, seed=42, bad_index=None):
        generator = torch.Generator().manual_seed(9)
        images = torch.randn(max(train, val), 3, 1, 1, generator=generator).expand(-1, -1, 224, 224)
        labels = torch.arange(max(train, val)) % 30
        if bad_index is not None:
            images = images.clone(); images[bad_index, 0, 0, 0] = float('nan')
        return (DataLoader(TensorDataset(images[:train], labels[:train]), batch_size=2,
                           shuffle=shuffle, generator=torch.Generator().manual_seed(seed)),
                DataLoader(TensorDataset(images[:val], labels[:val]), batch_size=2))

    def run_smoke(self, output, *, binding=None, model=None, loaders=None, callback=None, factory=MiniHGT):
        return api().run_epoch_smoke(MiniHGT() if model is None else model,
                                    *(self.loaders() if loaders is None else loaders), output,
                                    binding=self.binding() if binding is None else binding, device='cpu',
                                    model_factory=factory, on_progress=callback)

    def test_actual_update_all_five_groups_full_validation_and_independent_reload(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            model = MiniHGT(); before = copy.deepcopy(model.state_dict()); events = []
            record = self.run_smoke(Path(tmp), model=model, callback=events.append)
            self.assertEqual(record['schema'], 'hgtnet-aid-smoke-result-v1')
            self.assertTrue(record['completed']); self.assertTrue(record['training_started'])
            self.assertFalse(record['formal_training_started'])
            self.assertEqual(record['input_scope'], 'cpu_fixture')
            self.assertEqual(record['counts'], {'train': 4, 'val': 2, 'test': 0})
            self.assertEqual(record['optimizer_steps'], 2)
            self.assertEqual(record['checkpoint']['reload_max_abs_error'], 0.0)
            self.assertEqual(len(record['val_metrics']['confusion_matrix']), 30)
            self.assertEqual(sum(map(sum, record['val_metrics']['confusion_matrix'])), 2)
            self.assertFalse(torch.equal(model.head.weight, before['head.weight']))
            self.assertEqual(set(events[0]['updated_groups']),
                             {'patch_embed', 'transformer', 'dual_attention', 'graph', 'head'})
            self.assertTrue(all(events[0]['updated_groups'].values()))
            self.assertEqual(json.loads(Path(tmp, 'optimizer_progress.json').read_text())['optimizer_steps'], 2)
            prediction = json.loads(Path(tmp, 'val_predictions.json').read_text())
            self.assertEqual(prediction['labels'], [0, 1]); self.assertEqual(len(prediction['predictions']), 2)

    def test_full_240_60_fixture_performs_120_updates_without_claiming_real_input(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            record = self.run_smoke(Path(tmp), binding=self.binding(240, 60), loaders=self.loaders(240, 60))
            self.assertEqual(record['optimizer_steps'], 120)
            self.assertEqual(record['input_scope'], 'cpu_fixture')
            self.assertEqual(sum(map(sum, record['val_metrics']['confusion_matrix'])), 60)

    def test_binding_rejects_old_dataset_45classes_test_usage_and_fake_cuda(self):
        for change in [dict(schema='hgtnet-nwpu-smoke-binding-v1'), dict(input_scope='real_nwpu'),
                       dict(input_scope='real_aid'), dict(counts=dict(train=4, val=2, test=1)),
                       dict(runtime=dict(device='cuda:0')), dict(val_labels=[True, 1]),
                       dict(val_paths=['a.png', 'a.png'])]:
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                binding = self.binding(); binding.update(change)
                with self.subTest(change=change), self.assertRaises(ValueError):
                    self.run_smoke(Path(tmp), binding=binding)
                self.assertFalse(Path(tmp, 'first_step.json').exists())
        binding = self.binding(); binding['config']['num_classes'] = 45
        binding['config_sha256'] = config_sha256(binding['config'])
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, self.assertRaises(ValueError):
            self.run_smoke(Path(tmp), binding=binding)

    def test_sampler_seed_order_and_dropped_records_are_rejected(self):
        for loaders in [self.loaders(shuffle=False), self.loaders(seed=43)]:
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, self.assertRaises(ValueError):
                self.run_smoke(Path(tmp), loaders=loaders)
        train, val = self.loaders()
        shuffled_val = DataLoader(val.dataset, batch_size=2, shuffle=True)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, self.assertRaises(ValueError):
            self.run_smoke(Path(tmp), loaders=(train, shuffled_val))

    def test_nonfinite_first_input_or_gradient_does_not_claim_training_started(self):
        for bad_input in [False, True]:
            model = MiniHGT()
            if not bad_input:
                model.head.weight.register_hook(lambda gradient: gradient * float('nan'))
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                with self.subTest(bad_input=bad_input), self.assertRaises((ValueError, RuntimeError)):
                    self.run_smoke(Path(tmp), model=model, loaders=self.loaders(bad_index=0) if bad_input else None)
                self.assertFalse(Path(tmp, 'first_step.json').exists())
                self.assertFalse(Path(tmp, 'smoke_epoch1.pt').exists())

    def test_missing_architectural_gradient_group_fails_before_update(self):
        model = MiniHGT(); model.graph_branch.requires_grad_(False)
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, self.assertRaises(ValueError):
            self.run_smoke(Path(tmp), model=model)
        class HeadOnly(nn.Module):
            def __init__(self): super().__init__(); self.head = nn.Linear(3, 30)
            def forward(self, x): return self.head(x.mean((2, 3)))
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp, self.assertRaises(ValueError):
            self.run_smoke(Path(tmp), model=HeadOnly(), factory=HeadOnly)

    def test_failure_after_update_retains_actual_progress_and_first_step(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            binding = self.binding(); binding['val_labels'] = [1, 0]
            with self.assertRaises(ValueError) as failure:
                self.run_smoke(Path(tmp), binding=binding)
            self.assertEqual(failure.exception.optimizer_steps, 2)
            self.assertTrue(failure.exception.training_started)
            self.assertEqual(json.loads(Path(tmp, 'optimizer_progress.json').read_text())['optimizer_steps'], 2)
            self.assertTrue(json.loads(Path(tmp, 'first_step.json').read_text())['training_started'])
            self.assertFalse(Path(tmp, 'smoke_epoch1.pt').exists())

    def test_nonfinite_parameter_after_actual_step_is_recorded_as_started(self):
        original = torch.optim.Adam.step
        def corrupt(optimizer, *args, **kwargs):
            result = original(optimizer, *args, **kwargs)
            with torch.no_grad(): optimizer.param_groups[0]['params'][0].view(-1)[0] = float('inf')
            return result
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            with mock.patch.object(torch.optim.Adam, 'step', corrupt), self.assertRaises(ValueError) as failure:
                self.run_smoke(Path(tmp))
            self.assertEqual(failure.exception.optimizer_steps, 1)
            self.assertTrue(json.loads(Path(tmp, 'first_step.json').read_text())['training_started'])
            self.assertFalse(json.loads(Path(tmp, 'first_step.json').read_text())['diagnostic_passed'])

    def test_existing_artifacts_cannot_be_overwritten(self):
        for name in ['first_step.json', 'smoke_epoch1.pt', 'history.json', 'val_predictions.json', 'optimizer_progress.json']:
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                path = Path(tmp, name); path.write_bytes(b'user data')
                with self.subTest(name=name), self.assertRaises(FileExistsError):
                    self.run_smoke(Path(tmp))
                self.assertEqual(path.read_bytes(), b'user data')

    def test_checkpoint_wrong_sha_binding_schema_keys_dtype_shape_or_values_never_copy(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            path = Path(tmp, 'model.pt'); binding = self.binding(); model = MiniHGT()
            saved = api().save_smoke_checkpoint(model, path, binding=binding)
            raw = path.read_bytes(); payload = torch.load(io.BytesIO(raw), weights_only=True)
            self.assertEqual(payload['schema'], 'hgtnet-aid-smoke-checkpoint-v1')
            target = MiniHGT()
            api().reload_smoke_checkpoint(target, path, expected_sha256=saved['sha256'], binding=binding)
            self.assertTrue(torch.equal(target.head.weight, model.head.weight))
            changed = copy.deepcopy(binding); changed['manifest_sha256'] = 'f' * 64
            for expected, identity in [('0' * 64, binding), (saved['sha256'], changed)]:
                before = copy.deepcopy(target.state_dict())
                with self.assertRaises(ValueError):
                    api().reload_smoke_checkpoint(target, path, expected_sha256=expected, binding=identity)
                self.assertTrue(all(torch.equal(v, target.state_dict()[k]) for k, v in before.items()))
            mutations = []
            value = copy.deepcopy(payload); value['schema'] = 'hgtnet-real-image-smoke-checkpoint-v1'; mutations.append(value)
            value = copy.deepcopy(payload); value['state_dict']['unknown'] = torch.ones(1); mutations.append(value)
            value = copy.deepcopy(payload); value['state_dict']['head.weight'] = torch.ones(45, 4); mutations.append(value)
            value = copy.deepcopy(payload); value['state_dict']['head.weight'] = value['state_dict']['head.weight'].double(); mutations.append(value)
            value = copy.deepcopy(payload); value['state_dict']['head.weight'][0, 0] = float('nan'); mutations.append(value)
            value = copy.deepcopy(payload); value['protocol']['epochs'] = True; mutations.append(value)
            value = copy.deepcopy(payload); value['best_epoch'] = True; mutations.append(value)
            value = copy.deepcopy(payload); value['state_dict']['head.weight'] = 'not a tensor'; mutations.append(value)
            for index, value in enumerate(mutations):
                torch.save(value, path); digest = hashlib.sha256(path.read_bytes()).hexdigest()
                before = copy.deepcopy(target.state_dict())
                with self.subTest(index=index), self.assertRaises(ValueError):
                    api().reload_smoke_checkpoint(target, path, expected_sha256=digest, binding=binding)
                self.assertTrue(all(torch.equal(v, target.state_dict()[k]) for k, v in before.items()))

    def test_checkpoint_save_rejects_45class_nonfinite_or_non_fp32_state(self):
        for defect in ['45class', 'nan', 'double']:
            model = MiniHGT()
            if defect == '45class': model.head = nn.Linear(4, 45)
            elif defect == 'double': model.double()
            else:
                with torch.no_grad(): model.head.weight[0, 0] = float('nan')
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                path = Path(tmp, 'bad.pt')
                with self.subTest(defect=defect):
                    with self.assertRaises(ValueError):
                        api().save_smoke_checkpoint(model, path, binding=self.binding())
                    self.assertFalse(path.exists())

    def test_checkpoint_reload_checks_destination_config_identity_before_copy(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
            path = Path(tmp, 'model.pt'); binding = self.binding(); source = MiniHGT()
            saved = api().save_smoke_checkpoint(source, path, binding=binding)
            destination = MiniHGT(); destination.config_sha256 = 'f' * 64
            before = copy.deepcopy(destination.state_dict())
            with self.assertRaises(ValueError):
                api().reload_smoke_checkpoint(destination, path, expected_sha256=saved['sha256'], binding=binding)
            self.assertTrue(all(torch.equal(value, destination.state_dict()[key]) for key, value in before.items()))

    def test_reload_factory_cannot_reuse_trained_model_or_shared_parameter_storage(self):
        for shared_storage in [False, True]:
            model = MiniHGT()
            def factory():
                if not shared_storage: return model
                fresh = MiniHGT()
                fresh.head.weight = nn.Parameter(model.head.weight.detach())
                return fresh
            with tempfile.TemporaryDirectory(dir='/private/tmp') as tmp:
                with self.subTest(shared_storage=shared_storage):
                    with self.assertRaises(ValueError) as failure:
                        self.run_smoke(Path(tmp), model=model, factory=factory)
                    self.assertTrue(failure.exception.training_started)
                    self.assertEqual(failure.exception.optimizer_steps, 2)


if __name__ == '__main__':
    unittest.main()
