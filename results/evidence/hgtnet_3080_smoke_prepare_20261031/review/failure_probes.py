"""Independent small CPU failure probes. No production or GPU code changes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from external_baselines.hgtnet.config import config_sha256, load_config
from external_baselines.hgtnet import nwpu_engine as engine


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(3, 45)
        self.calls = 0

    def forward(self, image):
        self.calls += 1
        return self.head(image.mean((2, 3)))


def binding():
    cfg = load_config(ROOT / 'configs/hgtnet_independent_v0.json')
    return {'schema': 'hgtnet-nwpu-smoke-binding-v1', 'input_scope': 'cpu_fixture',
            'config': cfg, 'config_sha256': config_sha256(cfg),
            'counts': {'train': 4, 'val': 2, 'test': 0},
            'class_names': [f'class{i:02}' for i in range(45)],
            'manifest_sha256': '0' * 64, 'source_identity': {'fixture': '1' * 64},
            'pretraining': None, 'runtime': {'device': 'cpu'},
            'val_paths': ['class00/a.jpg', 'class01/b.jpg'], 'val_labels': [0, 1]}


def loaders():
    image = torch.randn(4, 3, 224, 224, generator=torch.Generator().manual_seed(42))
    return (DataLoader(TensorDataset(image, torch.tensor([0, 1, 2, 3])), batch_size=2),
            DataLoader(TensorDataset(image[:2], torch.tensor([0, 1])), batch_size=2))


class FailureProbes(unittest.TestCase):
    def test_second_training_batch_failure_preserves_true_first_step(self):
        class FailSecond(Tiny):
            def forward(self, image):
                if self.calls == 1:
                    raise RuntimeError('independent second-batch injection')
                return super().forward(image)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'second-batch'):
                engine.run_epoch_smoke(FailSecond(), *loaders(), output,
                    binding=binding(), device='cpu', model_factory=Tiny)
            self.assertTrue(json.loads((output/'first_step.json').read_text())['optimizer_step_succeeded'])
            self.assertFalse((output/'smoke_epoch1.pt').exists())
            self.assertFalse((output/'history.json').exists())

    def test_reload_fidelity_failure_never_returns_complete(self):
        class WrongReload(Tiny):
            def forward(self, image):
                return super().forward(image) + 1
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'reload differs'):
                engine.run_epoch_smoke(Tiny(), *loaders(), output,
                    binding=binding(), device='cpu', model_factory=WrongReload)
            self.assertTrue((output/'smoke_epoch1.pt').exists())
            self.assertTrue((output/'first_step.json').exists())
            self.assertFalse((output/'history.json').exists())

    def test_wrapper_preserves_observed_cpu_first_step_on_later_failure(self):
        spec = importlib.util.spec_from_file_location('review_hgtnet_runner', ROOT/'tools/run_hgtnet_nwpu_smoke.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as tmp:
            args = runner.parser().parse_args(['run', '--gpu-uuid', 'GPU-review-fixture',
                '--output', str(Path(tmp)/'output'), '--scratch-root', str(Path(tmp)/'scratch')])
            def injected_runtime(*_):
                engine.run_epoch_smoke(Tiny(), *loaders(), args.output,
                    binding=binding(), device='cpu', model_factory=Tiny)
                raise RuntimeError('independent post-CPU-engine injection')
            preflight = {'ready': True, 'selected_gpu': {'uuid': 'GPU-review-fixture'}}
            with patch.object(runner, 'inspect_resources', return_value=preflight), \
                 patch.object(runner, '_runtime_and_torch', side_effect=injected_runtime):
                self.assertEqual(runner.run_smoke(args), 1)
            value = json.loads((args.output/'summary.json').read_text())
            self.assertFalse(value['completed'])
            self.assertTrue(value['training_started'])
            self.assertFalse(value['formal_training_started'])
            self.assertEqual((args.output/'run.exit').read_text(), '1\n')
            self.assertEqual(runner.status(args.output)['state'], 'failed')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
