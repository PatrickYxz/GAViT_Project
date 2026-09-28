"""A new checkpoint must not silently reconstruct a different grouping."""
import copy
import importlib
import unittest


class GroupingIdentityTests(unittest.TestCase):
    def api(self):
        try:
            return importlib.import_module('grouping_identity')
        except ImportError as exc:
            self.fail(f'Grouping identity implementation missing: {exc}')

    def architecture(self):
        return {'num_classes': 19, 'num_regions': 16, 'grouping': 'soft_prototype',
                'edge_type': 'sparse_hybrid', 'integration': 'token_feedback'}

    def test_training_defaults_are_independent_and_checkpoint_is_strict(self):
        api = self.api()
        original = self.architecture()
        with self.assertRaisesRegex(ValueError, 'soft_assignment'):
            api.complete_grouping_architecture(original, checkpoint=True)
        result = api.complete_grouping_architecture(original)
        self.assertNotIn('soft_assignment', original)
        self.assertEqual(result['soft_assignment']['assignment_temperature'], 0.1)
        self.assertEqual(result['soft_assignment']['assignment_version'], 'local_prototype_cosine_v1')
        self.assertEqual(api.complete_grouping_architecture(result, checkpoint=True), result)
        result['soft_assignment']['assignment_temperature'] = 99.0
        self.assertEqual(api.soft_assignment_identity()['assignment_temperature'], 0.1)

    def test_every_missing_or_changed_assignment_field_is_rejected(self):
        api = self.api()
        arch = api.complete_grouping_architecture(self.architecture())
        for key in arch['soft_assignment']:
            for operation in ('remove', 'change'):
                with self.subTest(key=key, operation=operation):
                    bad = copy.deepcopy(arch)
                    if operation == 'remove':
                        del bad['soft_assignment'][key]
                    else:
                        bad['soft_assignment'][key] = 'wrong'
                    with self.assertRaises(ValueError):
                        api.complete_grouping_architecture(bad, checkpoint=True)
        for bad_value in (None, [], True, float('nan'), float('inf')):
            bad = copy.deepcopy(arch)
            bad['soft_assignment'] = bad_value
            with self.assertRaises(ValueError):
                api.complete_grouping_architecture(bad, checkpoint=True)
        for key in ('assignment_radius', 'assignment_iterations'):
            bad = copy.deepcopy(arch)
            bad['soft_assignment'][key] = True
            with self.assertRaises(ValueError):
                api.complete_grouping_architecture(bad, checkpoint=True)
        bad = copy.deepcopy(arch)
        bad['soft_assignment']['extra'] = 'unexpected'
        with self.assertRaises(ValueError):
            api.complete_grouping_architecture(bad, checkpoint=True)

    def test_old_architectures_and_input_dictionary_are_unchanged(self):
        api = self.api()
        for grouping in ('attentive_spatial', 'spatial', 'kmeans'):
            old = {**self.architecture(), 'grouping': grouping}
            self.assertEqual(api.complete_grouping_architecture(old, checkpoint=True), old)
            with self.assertRaises(ValueError):
                api.complete_grouping_architecture({**old, 'soft_assignment': api.soft_assignment_identity()})

    def test_invalid_combinations_are_rejected(self):
        api = self.api()
        for key, value in [('num_regions', 9), ('num_regions', True),
                           ('edge_type', 'knn'), ('integration', 'fusion')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                api.complete_grouping_architecture({**self.architecture(), key: value})

    def test_metadata_resolution_preserves_new_identity_and_rejects_fallback(self):
        from utils import resolve_arch_config
        api = self.api()
        defaults = {**self.architecture(), 'grouping': 'attentive_spatial'}
        arch = api.complete_grouping_architecture(self.architecture())
        resolved, source = resolve_arch_config({}, {'architecture': arch}, defaults)
        self.assertEqual(resolved, arch)
        self.assertEqual(source, 'metadata')
        for metadata in (None, {'architecture': self.architecture()}):
            with self.assertRaisesRegex(ValueError, 'soft_assignment|metadata'):
                resolve_arch_config({'grouping': 'soft_prototype'}, metadata, defaults)
        with self.assertRaisesRegex(ValueError, 'conflict'):
            resolve_arch_config({'grouping': 'attentive_spatial'}, {'architecture': arch}, defaults)


if __name__ == '__main__':
    unittest.main()
