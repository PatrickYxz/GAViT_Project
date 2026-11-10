"""Explicit RTX 3090 target gates using the user's observed GPU inventory."""
import copy
import inspect
import unittest

from external_baselines.hgtnet import resources


NAME_3080 = 'NVIDIA GeForce RTX 3080'
NAME_3090 = 'NVIDIA GeForce RTX 3090'
USER_GPU = {
    'name': NAME_3090,
    'uuid': 'GPU-4f625a02-7dfa-e0eb-de8d-1dcb4ba3dcb1',
    'memory_total_mib': 24576,
    'memory_free_mib': 24123,
    'driver_version': '610.57.04',
}
OLD_GPU = dict(USER_GPU, name=NAME_3080, uuid='GPU-3080-fixture',
               memory_total_mib=10240, memory_free_mib=10000)


def inventory(*gpus, processes=()):
    return {'gpus': list(gpus), 'processes': list(processes), 'error': None}


def runtime(gpu=USER_GPU):
    return {
        'python': '3.11.8 (main, Feb 12 2024, 12:00:00) [GCC 11.4.0]',
        'torch': '2.2.2+cu121', 'torchvision': '0.17.2+cu121',
        'numpy': '1.26.4', 'pillow': '10.3.0', 'cuda': '12.1', 'cudnn': 8902,
        'gpu': copy.deepcopy(gpu), 'physical_gpu_uuid': gpu['uuid'],
        'device': 'cuda:0', 'cpu_threads': 8, 'amp': False,
        'matmul_tf32': False, 'cudnn_tf32': False, 'cudnn_benchmark': False,
        'cudnn_deterministic': True, 'dtype': 'float32', 'seed': 42,
    }


class TargetResourcesTests(unittest.TestCase):
    def setUp(self):
        # An unsupported API must fail as an assertion, never as a TypeError
        # from passing a keyword the old implementation does not accept.
        self.assertTrue(hasattr(resources, 'TARGETS'),
                        'missing explicit RTX 3080/3090 target support')
        for function in (resources.select_gpu, resources.check_runtime):
            self.assertIn('target_name', inspect.signature(function).parameters,
                          'missing explicit target_name resource API')

    def test_exact_target_allowlist_and_backward_compatible_keyword_defaults(self):
        self.assertEqual(resources.TARGETS, {NAME_3080: 10240, NAME_3090: 24576})
        self.assertEqual(resources.TARGET_NAME, NAME_3080)
        self.assertEqual(resources.TARGET_MEMORY_MIB, 10240)
        for function in (resources.select_gpu, resources.check_runtime):
            parameter = inspect.signature(function).parameters['target_name']
            self.assertEqual(parameter.default, NAME_3080)
            self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)

    def test_user_observed_idle_3090_inventory_selects_current_physical_uuid(self):
        observed = inventory(copy.deepcopy(USER_GPU))
        for requested in (None, USER_GPU['uuid']):
            with self.subTest(requested=requested):
                selected = resources.select_gpu(requested, observed, target_name=NAME_3090)
                self.assertEqual(selected, USER_GPU)
                self.assertIsNot(selected, observed['gpus'][0])

    def test_user_3090_runtime_accepts_exact_target_and_current_uuid(self):
        actual = runtime()
        self.assertIs(resources.check_runtime(actual, USER_GPU['uuid'],
                                             target_name=NAME_3090), actual)

    def test_default_3080_gate_still_rejects_the_users_3090(self):
        with self.assertRaises(ValueError):
            resources.select_gpu(USER_GPU['uuid'], inventory(USER_GPU))
        with self.assertRaises(ValueError):
            resources.check_runtime(runtime(), USER_GPU['uuid'])
        self.assertEqual(resources.select_gpu(None, inventory(OLD_GPU)), OLD_GPU)
        self.assertEqual(resources.check_runtime(runtime(OLD_GPU), OLD_GPU['uuid']),
                         runtime(OLD_GPU))

    def test_target_must_be_allowlisted_before_inventory_or_runtime_checks(self):
        for target in ('3090', 'NVIDIA GeForce RTX 3090 Ti', 'NVIDIA A100', '',
                       None, True, 3090, []):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, 'target'):
                    resources.select_gpu(None, None, target_name=target)
                with self.assertRaisesRegex(ValueError, 'target'):
                    resources.check_runtime(None, None, target_name=target)

    def test_exact_target_name_and_memory_prevent_swapped_or_lookalike_cards(self):
        cases = [(NAME_3090, OLD_GPU), (NAME_3080, USER_GPU)]
        cases.extend((NAME_3090, dict(USER_GPU, **changes)) for changes in (
            {'name': NAME_3080}, {'name': 'NVIDIA GeForce RTX 3090 Ti'},
            {'name': 'NVIDIA A100'}, {'memory_total_mib': 10240,
                                    'memory_free_mib': 10000},
            {'memory_total_mib': 24575, 'memory_free_mib': 24000},
            {'memory_total_mib': True, 'memory_free_mib': 1},
        ))
        for target, gpu in cases:
            with self.subTest(target=target, gpu=gpu):
                with self.assertRaises(ValueError):
                    resources.select_gpu(gpu['uuid'], inventory(gpu), target_name=target)
                with self.assertRaises(ValueError):
                    resources.check_runtime(runtime(gpu), gpu['uuid'], target_name=target)

    def test_explicit_3090_target_checks_current_uuid_and_ambiguity(self):
        other = dict(USER_GPU, uuid='GPU-second-3090')
        self.assertEqual(resources.select_gpu(USER_GPU['uuid'], inventory(USER_GPU, other),
                                              target_name=NAME_3090), USER_GPU)
        for requested, observed in (
            ('GPU-stale', inventory(USER_GPU)), (True, inventory(USER_GPU)),
            (None, inventory(USER_GPU, other)), (None, inventory(USER_GPU, USER_GPU)),
        ):
            with self.subTest(requested=requested, observed=observed), self.assertRaises(ValueError):
                resources.select_gpu(requested, observed, target_name=NAME_3090)
        for key in ('physical_gpu_uuid', 'gpu'):
            broken = runtime()
            if key == 'gpu':
                broken[key]['uuid'] = 'GPU-stale'
            else:
                broken[key] = 'GPU-stale'
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, USER_GPU['uuid'], target_name=NAME_3090)

    def test_3090_target_does_not_relax_idle_query_or_free_memory_gates(self):
        process = {'pid': 123, 'gpu_uuid': USER_GPU['uuid'],
                   'name': 'python', 'used_memory_mib': 1}
        cases = [inventory(USER_GPU, processes=[process]),
                 dict(inventory(USER_GPU), error='process query failed'),
                 inventory(USER_GPU, processes=[{}])]
        cases.extend(inventory(dict(USER_GPU, memory_free_mib=free))
                     for free in (0, -1, 24577, None, True, '24123'))
        for observed in cases:
            with self.subTest(observed=observed), self.assertRaises(ValueError):
                resources.select_gpu(None, observed, target_name=NAME_3090)

    def test_3090_runtime_keeps_software_precision_and_numeric_type_gates(self):
        changes = {'python': '3.12.13', 'torch': '2.2.2+cpu',
                   'torchvision': '0.17.2+cu118', 'numpy': '2.0.0',
                   'pillow': '10.4.0', 'cuda': '12.2', 'cudnn': 8901,
                   'device': 'cpu', 'amp': True, 'matmul_tf32': True,
                   'cudnn_tf32': True, 'cudnn_benchmark': True,
                   'cudnn_deterministic': False, 'dtype': 'float16',
                   'seed': True, 'cpu_threads': True}
        for key, wrong in changes.items():
            broken = runtime()
            broken[key] = wrong
            with self.subTest(key=key), self.assertRaises(ValueError):
                resources.check_runtime(broken, USER_GPU['uuid'], target_name=NAME_3090)


if __name__ == '__main__':
    unittest.main()
