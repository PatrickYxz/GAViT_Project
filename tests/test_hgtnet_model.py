"""Full architecture wiring; numerical CPU execution is in the local smoke."""
import importlib
import unittest

import torch
from external_baselines.hgtnet.config import load_config


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.model')
    except ModuleNotFoundError as exc:
        raise AssertionError('HGTNet model API is missing') from exc


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config('configs/hgtnet_independent_v0.json')

    def model(self):
        with torch.device('meta'):
            return api().HGTNet(self.cfg)

    def test_real_full_architecture_not_a_swin_t_or_tiny_surrogate(self):
        m = self.model()
        self.assertEqual([len(s.blocks) for s in m.stages], [2, 2, 18])
        self.assertEqual([s.blocks[0].attn.num_heads for s in m.stages], [4, 8, 16])
        self.assertEqual([len(m.graph_blocks1), len(m.graph_blocks2)], [6, 2])
        self.assertEqual(m.head.in_features, 512)
        self.assertEqual(m.head.out_features, 45)
        self.assertEqual(m.graph_transition2.depthwise.padding, (0, 0))
        self.assertTrue(all(b.k == 20 and b.exclude_self for b in list(m.graph_blocks1) + list(m.graph_blocks2)))
        self.assertGreater(sum(p.numel() for p in m.parameters()), 70_000_000)
        self.assertLess(sum(p.numel() for p in m.parameters()), 90_000_000)

    def test_parameter_groups_account_for_every_parameter_once(self):
        m = self.model()
        groups = api().parameter_counts(m)
        self.assertEqual(groups['total'], sum(p.numel() for p in m.parameters()))
        self.assertEqual(sum(groups['groups'].values()), groups['total'])
        for name in ['patch_embed', 'transformer', 'dual_attention', 'graph', 'head']:
            self.assertGreater(groups['groups'][name], 0)
        self.assertEqual(groups['groups']['graph'], 14_066_432)

    def test_model_owns_config_without_caller_alias(self):
        m = self.model()
        cfg = load_config('configs/hgtnet_independent_v0.json')
        with torch.device('meta'):
            other = api().HGTNet(cfg)
        cfg['choices']['identity'] = 'mutated'
        self.assertEqual(m.config, other.config)
        other.config['seed'] += 1
        with self.assertRaises(ValueError):
            other.assert_identity()

    def test_class30_changes_only_head_shape(self):
        cfg = dict(self.cfg, num_classes=30)
        with torch.device('meta'):
            m = api().HGTNet(cfg)
        self.assertEqual(m.head.out_features, 30)
        self.assertEqual(m.patch_embed.projection.weight.shape, (128, 3, 4, 4))

    def test_bad_inputs_fail_before_attention_even_on_meta_model(self):
        m = self.model()
        for shape in [(1, 3, 112, 112), (1, 3, 224, 223), (1, 4, 224, 224), (0, 3, 224, 224), (3, 224, 224)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                m(torch.empty(shape, device='meta'))
        with self.assertRaises(TypeError):
            m(torch.empty((1, 3, 224, 224), dtype=torch.int64, device='meta'))

    def test_trace_uses_real_computation_shapes_and_graph_indices(self):
        # Meta cannot validate finite tensors or compute kNN. It must fail, not
        # emit a fabricated success trace using the static config helper.
        m = self.model()
        with self.assertRaises(ValueError):
            m.forward_with_trace(torch.empty(1, 3, 224, 224, device='meta'))

    def test_no_implicit_online_or_pretrained_loading(self):
        m = self.model()
        self.assertFalse(m.pretraining_loaded)
        self.assertEqual(m.implementation, 'hgtnet_independent_v0')


if __name__ == '__main__':
    unittest.main()
