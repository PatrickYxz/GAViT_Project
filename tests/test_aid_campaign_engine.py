"""AID campaign refusal boundaries and complete-transaction reuse."""
import json
from pathlib import Path
import tempfile
import unittest

from scene_classification import aid_campaign_engine as engine
from scene_classification.aid_campaign_state import atomic_json, file_digest


class CampaignEngineTests(unittest.TestCase):
    def test_formal_refuses_cpu_before_importing_training_dependencies(self):
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            engine.validate_settings({'mode': 'formal', 'device': 'cpu', 'variant': 'soft', 'seed': 42})

    def test_formal_refuses_historical_fixed_swin_seeds(self):
        for variant in ('fixed', 'swin'):
            with self.subTest(variant=variant), self.assertRaisesRegex(ValueError, 'historical'):
                engine.validate_settings({'mode': 'formal', 'device': 'cuda:0',
                                          'variant': variant, 'seed': 42})

    def test_formal_refuses_changed_recipe(self):
        for key, value in [('batch_size', 16), ('workers', 0)]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'recipe'):
                engine.validate_settings({'mode': 'formal', 'device': 'cuda:0', 'variant': 'soft',
                                          'seed': 42, key: value})

    def test_manifest_hash_and_actual_counts_are_both_required(self):
        bad = {'dataset': 'AID', 'class_names': [str(i) for i in range(30)],
               'counts': {'train': 4000, 'val': 1000, 'test': 5000}, 'records': []}
        with self.assertRaisesRegex(ValueError, 'manifest'):
            engine.validate_manifest_identity(bad, 'wrong', formal=True)
        with self.assertRaisesRegex(ValueError, 'counts'):
            engine.validate_manifest_identity(bad, engine.AID_MANIFEST_SHA256, formal=True)

    def test_completed_job_is_verified_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / 'metrics.json').write_text('{"oa_percent": 50}\n')
            (out / 'run.exit').write_text('exit_code=0\n')
            atomic_json(out / 'completion.json', {'status': 'complete', 'identity': {'seed': 42},
                'summary': {'status': 'complete', 'seed': 42},
                'artifact_sha256': {'metrics.json': file_digest(out / 'metrics.json'),
                                    'run.exit': file_digest(out / 'run.exit')}})
            (out / 'COMPLETED').write_text(file_digest(out / 'completion.json') + '\n')
            result = engine.completed_summary(out, {'seed': 42})
            self.assertEqual(result, {'status': 'complete', 'seed': 42, 'reused': True})
            (out / 'metrics.json').write_text('corrupted')
            with self.assertRaisesRegex(ValueError, 'hash'):
                engine.completed_summary(out, {'seed': 42})

    def test_changed_completion_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            atomic_json(out / 'completion.json', {'status': 'complete', 'identity': {'seed': 42}})
            (out / 'COMPLETED').write_text('0' * 64 + '\n')
            with self.assertRaisesRegex(ValueError, 'receipt hash'):
                engine.completed_summary(out, {'seed': 42})

    def test_cuda_uuid_fallback_requires_independent_runtime_measurement(self):
        measured = {'cuda_device_uuid': 'GPU-abcd', 'visible_cuda_devices': 1,
                    'device': 'cuda:0', 'verification_method': 'cudaDeviceGetUuid'}
        self.assertEqual(engine.resolve_gpu_uuid(None, 'GPU-abcd', measured), 'GPU-abcd')
        with self.assertRaisesRegex(ValueError, 'UUID'):
            engine.resolve_gpu_uuid(None, 'GPU-abcd', {**measured, 'cuda_device_uuid': 'GPU-else'})
        with self.assertRaisesRegex(ValueError, 'UUID'):
            engine.resolve_gpu_uuid(None, 'GPU-abcd', {**measured, 'verification_method': 'guess'})

    def test_weight_decay_alone_does_not_count_as_a_live_core_group(self):
        rows = [{'parameter': 'head.weight', 'group': 'classifier', 'numel': 4,
                 'gradient_finite': True, 'gradient_nonzero': False,
                 'parameter_finite': True, 'max_abs_update': 0.001}]
        with self.assertRaisesRegex(ValueError, 'gradient'):
            engine.summarize_parameter_updates(rows, ['classifier'])

    def test_first_update_requires_every_core_group_and_records_zero_delta_tensors(self):
        rows = [{'parameter': 'head.weight', 'group': 'classifier', 'numel': 4,
                 'gradient_finite': True, 'gradient_nonzero': True,
                 'parameter_finite': True, 'max_abs_update': 0.001},
                {'parameter': 'head.bias', 'group': 'classifier', 'numel': 2,
                 'gradient_finite': True, 'gradient_nonzero': False,
                 'parameter_finite': True, 'max_abs_update': 0.0}]
        report = engine.summarize_parameter_updates(rows, ['classifier'])
        self.assertEqual(report['classifier']['changed_tensors'], 1)
        self.assertEqual(report['classifier']['parameter_tensors'], 2)
        self.assertEqual(report['classifier']['zero_delta_parameters'], ['head.bias'])
        with self.assertRaisesRegex(ValueError, 'Missing core'):
            engine.summarize_parameter_updates(rows, ['classifier', 'soft_assignment'])


if __name__ == '__main__':
    unittest.main()
