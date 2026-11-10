"""Bounded AID capacity behavior, using only tiny generated CPU tensors."""
import importlib
import inspect
import copy
import math
import statistics
import unittest
from unittest.mock import patch

import torch
from torch import nn


class TinyFiveGroupModel(nn.Module):
    """All five semantic groups, without a complete HGTNet backbone."""
    def __init__(self, classes=30):
        super().__init__()
        self.patch_embed = nn.Linear(3, 3)
        stage = nn.Module()
        stage.transform = nn.Linear(3, 3)
        stage.da = nn.Linear(3, 3)
        self.stages = nn.ModuleList([stage])
        self.graph_block = nn.Linear(3, 3)
        self.head = nn.Linear(3, classes)

    def forward(self, images):
        features = torch.tanh(self.patch_embed(images.mean((2, 3))))
        features = torch.tanh(self.stages[0].transform(features))
        features = torch.tanh(self.stages[0].da(features))
        return self.head(torch.tanh(self.graph_block(features)))


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.aid_batch_probe_engine')
    except ModuleNotFoundError as exc:
        raise AssertionError('Independent AID batch16 engine is missing') from exc


class AidBatchProbeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        generator = torch.Generator().manual_seed(42)
        return torch.randn(16, 3, 224, 224, generator=generator), torch.arange(16, dtype=torch.long)

    def assert_rejected_without_update(self, model, images, labels, **kwargs):
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        events = []
        with self.assertRaises((ValueError, RuntimeError)):
            api().run_batch_probe(model, images, labels, device='cpu', on_progress=events.append, **kwargs)
        self.assertEqual(events, [])
        for name, value in model.named_parameters():
            self.assertTrue(torch.allclose(value, before[name], equal_nan=True), name)

    def test_independent_entry_accepts_frozen_probe_interface(self):
        engine = api()
        self.assertTrue(callable(engine.run_batch_probe))
        self.assertEqual(str(inspect.signature(engine.run_batch_probe)),
                         '(model, images, labels, *, device, batch_size=16, '
                         'warmup_steps=1, measured_steps=3, on_progress=None)')

    def test_thirty_classes_make_four_real_adam_updates_with_five_groups(self):
        torch.manual_seed(42)
        model = TinyFiveGroupModel()
        images, labels = self.inputs()
        images_before, labels_before = images.clone(), labels.clone()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        reference = copy.deepcopy(model)
        reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-5, weight_decay=5e-5,
                                               betas=(0.9, 0.999), eps=1e-8,
                                               amsgrad=False, foreach=None, fused=None)
        for _ in range(4):
            reference_optimizer.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(reference(images), labels).backward()
            reference_optimizer.step()
        events = []
        record = api().run_batch_probe(model, images, labels, device='cpu', on_progress=events.append)
        self.assertEqual(record['schema'], 'hgtnet-aid-batch16-capacity-probe-v1')
        self.assertTrue(record['completed'])
        self.assertTrue(record['training_started'])
        self.assertFalse(record['formal_training_started'])
        self.assertEqual((record['unique_images'], record['processed_images'], record['optimizer_steps']), (16, 64, 4))
        self.assertEqual((record['warmup_steps'], record['measured_steps']), (1, 3))
        self.assertEqual([event['optimizer_steps'] for event in events], [1, 2, 3, 4])
        self.assertEqual([event['phase'] for event in events], ['warmup', 'measured', 'measured', 'measured'])
        self.assertEqual([event['samples_seen'] for event in events], [16, 32, 48, 64])
        self.assertTrue(all(event['optimizer_step_completed'] and event['post_update_checks_passed'] for event in events))
        self.assertEqual(events[0]['gradients'], record['first_step_gradients'])
        self.assertEqual(record['first_step_updated_groups'],
                         {'patch_embed': True, 'transformer': True, 'dual_attention': True, 'graph': True, 'head': True})
        report = record['first_step_gradients']
        self.assertTrue(report['passed'])
        self.assertEqual(report['missing'], [])
        self.assertEqual(report['nonfinite'], [])
        self.assertEqual(sum(group['tensors'] for group in report['groups'].values()), 10)
        self.assertTrue(all(math.isfinite(group['l2']) and group['l2'] > 0 for group in report['groups'].values()))
        for name, value in model.named_parameters():
            self.assertFalse(torch.equal(value, before[name]), name)
            self.assertTrue(torch.equal(value, dict(reference.named_parameters())[name]), name)
            self.assertIsNone(value.grad)
        self.assertTrue(torch.equal(images, images_before))
        self.assertTrue(torch.equal(labels, labels_before))
        self.assertEqual(len(record['step_seconds']), 3)
        self.assertEqual(record['step_seconds'], [event['elapsed_seconds'] for event in events[1:]])
        self.assertEqual(record['warmup_seconds'], events[0]['elapsed_seconds'])
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in record['step_seconds']))
        self.assertEqual(record['median_step_seconds'], statistics.median(record['step_seconds']))
        self.assertAlmostEqual(record['compute_images_per_second'], 16 / record['median_step_seconds'])
        self.assertEqual(record['input_scope'], 'caller_provided_tensors')
        self.assertIsNone(record['cuda_peak_allocated_bytes'])
        self.assertIsNone(record['cuda_peak_reserved_bytes'])
        self.assertIsNone(record['accuracy_metrics'])
        self.assertFalse(record['checkpoint_saved'])
        self.assertFalse(record['gradient_clipping'])
        self.assertEqual({key: record['optimizer'][key] for key in ('name', 'lr', 'weight_decay', 'betas', 'eps')},
                         {'name': 'Adam', 'lr': 1e-5, 'weight_decay': 5e-5, 'betas': [0.9, 0.999], 'eps': 1e-8})

    def test_old_forty_five_class_head_and_out_of_aid_labels_are_rejected(self):
        images, labels = self.inputs()
        self.assert_rejected_without_update(TinyFiveGroupModel(classes=45), images, labels)
        self.assert_rejected_without_update(TinyFiveGroupModel(), images, labels + 30)

    def test_invalid_batch_shape_dtype_or_budget_cannot_update(self):
        images, labels = self.inputs()
        for x, y in [(images[:15], labels[:15]), (images.double(), labels), (images, labels.float()),
                     (images[:, :, :112, :112], labels), (images * float('nan'), labels)]:
            with self.subTest(shape=x.shape, dtype=x.dtype):
                self.assert_rejected_without_update(TinyFiveGroupModel(), x, y)
        for kwargs in [{'batch_size': 2}, {'batch_size': True}, {'warmup_steps': 0},
                       {'warmup_steps': 2}, {'measured_steps': 100}, {'measured_steps': True}]:
            with self.subTest(kwargs=kwargs):
                self.assert_rejected_without_update(TinyFiveGroupModel(), images, labels, **kwargs)

    def test_all_five_groups_are_required_even_when_remaining_gradients_pass(self):
        images, labels = self.inputs()
        model = TinyFiveGroupModel()
        model.graph_block = nn.Identity()
        self.assert_rejected_without_update(model, images, labels)

    def test_unclassified_parameter_cannot_hide_in_five_group_model(self):
        images, labels = self.inputs()
        model = TinyFiveGroupModel()
        model.unclassified = nn.Parameter(torch.ones(1))
        self.assert_rejected_without_update(model, images, labels)

    def test_missing_nonfinite_or_zero_group_gradients_block_first_update(self):
        images, labels = self.inputs()
        missing = TinyFiveGroupModel()
        missing.graph_unused = nn.Linear(3, 3)
        self.assert_rejected_without_update(missing, images, labels)
        for bad in (float('nan'), float('inf'), 0.0):
            model = TinyFiveGroupModel()
            for value in model.graph_block.parameters():
                value.register_hook(lambda gradient, bad=bad: gradient * bad)
            with self.subTest(gradient=bad):
                self.assert_rejected_without_update(model, images, labels)

    def test_frozen_double_or_initially_nonfinite_model_is_rejected(self):
        images, labels = self.inputs()
        for mode in ('frozen', 'double', 'nan'):
            model = TinyFiveGroupModel()
            if mode == 'frozen':
                model.patch_embed.weight.requires_grad_(False)
            elif mode == 'double':
                model.double()
            else:
                with torch.no_grad():
                    model.head.weight[0, 0] = float('nan')
            with self.subTest(mode=mode):
                self.assert_rejected_without_update(model, images, labels)

    def test_later_forward_failure_keeps_only_observed_first_update(self):
        class LaterFailure(TinyFiveGroupModel):
            calls = 0
            def forward(self, images):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError('second forward failed')
                return super().forward(images)
        images, labels = self.inputs()
        events = []
        with self.assertRaisesRegex(RuntimeError, 'second forward failed'):
            api().run_batch_probe(LaterFailure(), images, labels, device='cpu', on_progress=events.append)
        self.assertEqual([event['optimizer_steps'] for event in events], [1])
        self.assertTrue(events[0]['post_update_checks_passed'])

    def test_later_nonfinite_gradient_cannot_be_counted_as_an_update(self):
        images, labels = self.inputs()
        model = TinyFiveGroupModel()
        calls = []
        def fail_second_gradient(gradient):
            calls.append(True)
            return gradient if len(calls) == 1 else gradient * float('nan')
        model.graph_block.weight.register_hook(fail_second_gradient)
        events = []
        with self.assertRaises((ValueError, RuntimeError)):
            api().run_batch_probe(model, images, labels, device='cpu', on_progress=events.append)
        self.assertEqual([event['optimizer_steps'] for event in events], [1])

    def test_real_update_then_nonfinite_parameter_failure_preserves_callback(self):
        engine = api()
        model = TinyFiveGroupModel()
        images, labels = self.inputs()
        before = model.patch_embed.weight.detach().clone()
        adam = torch.optim.Adam
        class CorruptAfterActualUpdate(adam):
            def step(self, closure=None):
                result = super().step(closure)
                with torch.no_grad():
                    model.head.weight[0, 0] = float('nan')
                return result
        events = []
        with patch.object(engine.torch.optim, 'Adam', CorruptAfterActualUpdate):
            with self.assertRaisesRegex(ValueError, 'Updated probe parameters nonfinite'):
                engine.run_batch_probe(model, images, labels, device='cpu', on_progress=events.append)
        self.assertFalse(torch.equal(model.patch_embed.weight, before))
        self.assertEqual([event['optimizer_steps'] for event in events], [1])
        self.assertTrue(events[0]['optimizer_step_completed'])
        self.assertFalse(events[0]['post_update_checks_passed'])

    def test_zero_actual_group_update_fails_and_preserves_observed_step(self):
        engine = api()
        model = TinyFiveGroupModel()
        images, labels = self.inputs()
        adam = torch.optim.Adam
        class RestoreOneGroupAfterUpdate(adam):
            def step(self, closure=None):
                originals = [parameter.detach().clone() for parameter in model.graph_block.parameters()]
                result = super().step(closure)
                with torch.no_grad():
                    for parameter, original in zip(model.graph_block.parameters(), originals):
                        parameter.copy_(original)
                return result
        events = []
        with patch.object(engine.torch.optim, 'Adam', RestoreOneGroupAfterUpdate):
            with self.assertRaisesRegex(ValueError, 'did not update every gradient group'):
                engine.run_batch_probe(model, images, labels, device='cpu', on_progress=events.append)
        self.assertEqual([event['optimizer_steps'] for event in events], [1])
        self.assertFalse(events[0]['updated_groups']['graph'])
        self.assertFalse(events[0]['post_update_checks_passed'])


if __name__ == '__main__':
    unittest.main()
