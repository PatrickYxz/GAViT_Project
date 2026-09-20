import copy
import gc
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scene_classification.data import prepare_manifest
from test_scene_data_metrics import fixture


def runtime():
    try:
        return importlib.import_module('scene_classification.runner')
    except ModuleNotFoundError as exc:
        raise AssertionError('scene_classification.runner not implemented') from exc


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'images'
        fixture(self.root)
        self.manifest = Path(self.tmp.name) / 'manifest.json'
        prepare_manifest(self.root, self.manifest, dataset='synthetic')

    def config(self, **changes):
        kwargs = dict(model='swin', manifest=str(self.manifest), data_root=str(self.root),
                      output=str(Path(self.tmp.name) / 'run'), phase='smoke', device='cpu',
                      random_init=True, batch_size=2, workers=0, train_per_class=1, val_per_class=1)
        kwargs.update(changes)
        return runtime().RunConfig(**kwargs)

    def test_real_models_train_save_reload_and_evaluate_validation(self):
        for name in ('swin', 'gavit'):
            with self.subTest(model=name):
                cfg = self.config(model=name, output=str(Path(self.tmp.name) / name))
                report = runtime().run_training(cfg)
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(report['final_epoch'], 1)
                self.assertLessEqual(report['reload_max_logit_diff'], 1e-5)
                self.assertEqual(report['counts'], {'train': 2, 'val': 2})
                out = Path(cfg.output)
                self.assertEqual((out / 'run.exit').read_text(), 'exit_code=0\n')
                metadata = json.loads((out / 'best.meta.json').read_text())
                self.assertEqual(metadata['dataset'], 'synthetic')
                self.assertEqual(metadata['training']['weight_decay'], 1e-4)
                self.assertEqual(metadata['training']['scheduler_t_max'], 30)
                self.assertEqual(metadata['architecture']['num_classes'], 2)
                self.assertTrue(metadata['pretrained']['random_init'])
                self.assertEqual(metadata['git'], report['code']['git'])
                self.assertTrue(metadata['git']['commit'])
                self.assertIn('TRAIN FIRST BATCH OK', (out / 'run.log').read_text())
                if name == 'gavit':
                    self.assertEqual(metadata['graph_topology']['total_directed_edges'], 80)
                    self.assertEqual(report['graph_capture']['edges_per_image'], 80)
                eval_out = out / 'val_eval'
                result = runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                    str(self.root), str(eval_out), split='val', device='cpu', batch_size=2, workers=0)
                self.assertEqual(result['samples'], 2)
                self.assertTrue((eval_out / 'predictions.csv').is_file())
                self.assertEqual(len(result['confusion_matrix']), 2)
                with patch('scene_classification.runner._code_identity', return_value={'files_sha256': {'changed': 'code'}}), \
                        patch('scene_classification.runner._restore') as restore:
                    with self.assertRaisesRegex(ValueError, 'code identity'):
                        runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                            str(self.root), str(out / 'changed_code'), split='val', device='cpu', workers=0)
                    restore.assert_not_called()
                original_report = (out / 'run.json').read_text()
                changed_report = json.loads(original_report)
                changed_report['status'] = 'failed'
                (out / 'run.json').write_text(json.dumps(changed_report))
                with self.assertRaisesRegex(ValueError, 'completion'):
                    runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                        str(self.root), str(out / 'failed_source'), split='val', device='cpu')
                (out / 'run.json').write_text(original_report)
                with self.assertRaises(FileExistsError):
                    runtime().run_training(cfg)
                with self.assertRaisesRegex(ValueError, 'smoke'):
                    runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                        str(self.root), str(out / 'forbidden_test'), split='test', device='cpu')
                # Sidecar substitution and corrupted checkpoint must be detected before inference.
                meta_path = out / 'best.meta.json'
                original = meta_path.read_text()
                metadata['class_names'] = ['wrong', 'classes']
                meta_path.write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, 'identity'):
                    runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                        str(self.root), str(out / 'wrong_identity'), split='val', device='cpu')
                meta_path.write_text(original)
                with (out / 'best.pth').open('ab') as f:
                    f.write(b'corrupt')
                with self.assertRaisesRegex(ValueError, 'hash'):
                    runtime().evaluate_checkpoint(str(out / 'best.pth'), str(self.manifest),
                        str(self.root), str(out / 'corrupt'), split='val', device='cpu')
                gc.collect()

    def test_formal_random_init_and_synthetic_are_rejected(self):
        for phase in ('calibrate', 'refit'):
            cfg = self.config(phase=phase, output=str(Path(self.tmp.name) / phase))
            with self.assertRaises(ValueError):
                runtime().run_training(cfg)
            out = Path(cfg.output)
            self.assertEqual((out / 'run.exit').read_text(), 'exit_code=1\n')
            self.assertEqual(json.loads((out / 'run.json').read_text())['status'], 'failed')
            self.assertFalse((out / 'best.pth').exists())

    def test_training_config_is_model_independent(self):
        a, b = self.config(model='swin'), self.config(model='gavit')
        self.assertEqual(runtime().training_identity(a), runtime().training_identity(b))

    def test_same_local_weight_file_initializes_both_backbones(self):
        import timm
        from scene_classification.modeling import build_model
        # Locally generated weights verify file loading and exact tensor equality;
        # they are not real ImageNet weights or a pretrained-performance claim.
        source = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
        path = Path(self.tmp.name) / 'local_weights.pth'
        torch.save(source.state_dict(), path)
        expected = {k: v for k, v in source.state_dict().items() if not k.startswith('head.')}
        for name in ('swin', 'gavit'):
            with self.subTest(model=name):
                model = build_model(name, 30, str(path))
                actual = model.backbone.swin.state_dict()
                self.assertEqual(set(actual), set(expected))
                self.assertTrue(all(torch.equal(actual[key], value) for key, value in expected.items()))
                del model

    def test_calibration_selection_cannot_be_reused_for_another_seed_or_manifest(self):
        source = {'status': 'complete', 'phase': 'calibrate', 'model': 'swin', 'seed': 42,
                  'manifest_sha256': 'abc', 'pretrained_sha256': 'def', 'best_epoch': 9,
                  'final_epoch': 30, 'training': runtime().training_identity(self.config()),
                  'architecture': runtime().architecture('swin', 30)}
        expected = {key: source[key] for key in ('model', 'seed', 'manifest_sha256',
                                                'pretrained_sha256', 'training', 'architecture')}
        self.assertEqual(runtime().validate_calibration(source, expected), 9)
        for key, value in [('seed', 43), ('manifest_sha256', 'other'), ('status', 'failed'),
                           ('phase', 'smoke'), ('best_epoch', 0), ('best_epoch', 31), ('final_epoch', 29)]:
            wrong = copy.deepcopy(source)
            wrong[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                runtime().validate_calibration(wrong, expected)

    def test_completed_report_must_match_checkpoint_sidecar(self):
        meta = {'model': 'swin', 'dataset': 'AID', 'training': {'seed': 42},
                'architecture': {'num_classes': 30}, 'checkpoint_sha256': 'weights',
                'data': {'manifest_sha256': 'manifest'},
                'pretrained': {'sha256': 'pretrained'},
                'execution': {'phase': 'calibrate'}, 'final_epoch': 30,
                'best': {'epoch': 9}}
        source = {'status': 'complete', 'phase': 'calibrate', 'model': 'swin', 'dataset': 'AID',
                  'seed': 42, 'training': {'seed': 42}, 'architecture': {'num_classes': 30},
                  'checkpoint_sha256': 'weights', 'manifest_sha256': 'manifest',
                  'pretrained_sha256': 'pretrained', 'best_epoch': 9, 'final_epoch': 30}
        runtime().validate_source_run(source, meta, 'exit_code=0\n')
        for key, value in [('seed', 43), ('checkpoint_sha256', 'other'), ('best_epoch', 8),
                           ('phase', 'refit'), ('status', 'running'), ('final_epoch', 29)]:
            changed = copy.deepcopy(source)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'completion'):
                runtime().validate_source_run(changed, meta, 'exit_code=0\n')
        with self.assertRaisesRegex(ValueError, 'completion'):
            runtime().validate_source_run(source, meta, 'exit_code=1\n')


if __name__ == '__main__':
    unittest.main()
