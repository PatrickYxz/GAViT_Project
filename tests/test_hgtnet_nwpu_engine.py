"""Actual train/validation/checkpoint behavior, separate from unavailable CUDA."""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from external_baselines.hgtnet.config import config_sha256, load_config


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.nwpu_engine')
    except ModuleNotFoundError as exc:
        raise AssertionError('HGTNet NWPU engine API missing') from exc


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.head = nn.Linear(3, 45)
    def forward(self, x):
        return self.head(x.mean(dim=(2, 3)))


class EngineTests(unittest.TestCase):
    def binding(self):
        cfg = load_config('configs/hgtnet_independent_v0.json')
        return {'schema': 'hgtnet-nwpu-smoke-binding-v1', 'input_scope': 'cpu_fixture',
                'config': cfg, 'config_sha256': config_sha256(cfg), 'counts': {'train': 4, 'val': 2, 'test': 0},
                'class_names': [f'class{i:02}' for i in range(45)],
                'manifest_sha256': '0' * 64, 'source_identity': {'fixture': '1' * 64},
                'pretraining': None, 'runtime': {'device': 'cpu'},
                'val_paths': ['class00/a.jpg', 'class01/b.jpg'], 'val_labels': [0, 1]}

    def loaders(self, bad=False):
        torch.manual_seed(42)
        x = torch.randn(4, 3, 224, 224)
        if bad: x[0, 0, 0, 0] = float('nan')
        return (DataLoader(TensorDataset(x, torch.tensor([0, 1, 2, 3])), batch_size=2),
                DataLoader(TensorDataset(x[:2].clone(), torch.tensor([0, 1])), batch_size=2))

    def test_real_cpu_step_val_and_independent_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            events=[]; m=Tiny(); before=m.head.weight.detach().clone()
            record=api().run_epoch_smoke(m,*self.loaders(),Path(tmp),binding=self.binding(),
                                        device='cpu',model_factory=Tiny,on_progress=events.append)
            self.assertTrue(record['completed'])
            self.assertEqual(record['input_scope'],'cpu_fixture')
            self.assertFalse(record['formal_training_started'])
            self.assertEqual(record['counts'],{'train':4,'val':2,'test':0})
            self.assertEqual(record['optimizer_steps'],2)
            self.assertEqual(record['best_epoch'],1)
            self.assertEqual(record['final_epoch'],1)
            self.assertEqual(record['checkpoint']['reload_max_abs_error'],0.0)
            self.assertFalse(torch.equal(before,m.head.weight))
            self.assertEqual(sum(map(sum,record['val_metrics']['confusion_matrix'])),2)
            self.assertTrue(Path(tmp,'first_step.json').is_file())
            self.assertEqual(events[0]['stage'],'first_optimizer_step')
            self.assertIn('updated_groups',events[0])
            self.assertEqual(events[0]['updated_groups'],{'head':True})
            self.assertTrue(Path(tmp,'smoke_epoch1.pt').is_file())

    def test_bad_input_cannot_leave_a_successful_first_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                api().run_epoch_smoke(Tiny(),*self.loaders(True),Path(tmp),binding=self.binding(),
                                      device='cpu',model_factory=Tiny)
            self.assertFalse(Path(tmp,'first_step.json').exists())
            self.assertFalse(Path(tmp,'smoke_epoch1.pt').exists())

    def test_val_order_or_counts_do_not_silently_change_metrics(self):
        for key,value in [('val_labels',[1,0]),('counts',{'train':360,'val':90,'test':0})]:
            with tempfile.TemporaryDirectory() as tmp:
                binding=self.binding();binding[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    api().run_epoch_smoke(Tiny(),*self.loaders(),Path(tmp),binding=binding,
                                          device='cpu',model_factory=Tiny)
                self.assertFalse(Path(tmp,'smoke_epoch1.pt').exists())

    def test_existing_artifacts_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp,'smoke_epoch1.pt');path.write_bytes(b'user checkpoint')
            with self.assertRaises(FileExistsError):
                api().run_epoch_smoke(Tiny(),*self.loaders(),Path(tmp),binding=self.binding(),
                                      device='cpu',model_factory=Tiny)
            self.assertEqual(path.read_bytes(),b'user checkpoint')

    def test_safe_checkpoint_requires_exact_sha_metadata_and_all_shapes_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp,'x.pt'); m=Tiny(); binding=self.binding()
            saved=api().save_smoke_checkpoint(m,path,binding=binding)
            target=Tiny();before=target.head.weight.detach().clone()
            with self.assertRaises(ValueError):
                api().reload_smoke_checkpoint(target,path,expected_sha256='0'*64,binding=binding)
            self.assertTrue(torch.equal(before,target.head.weight))
            loaded=api().reload_smoke_checkpoint(target,path,expected_sha256=saved['sha256'],binding=binding)
            self.assertTrue(loaded['loaded'])
            self.assertTrue(torch.equal(m.head.weight,target.head.weight))
            other=self.binding();other['manifest_sha256']='2'*64
            with self.assertRaises(ValueError):
                api().reload_smoke_checkpoint(target,path,expected_sha256=saved['sha256'],binding=other)

    def test_fixture_cannot_claim_real_gpu_scope_or_test_usage(self):
        for changed in [{'input_scope':'real_nwpu'}, {'counts':{'train':4,'val':2,'test':1}}]:
            with tempfile.TemporaryDirectory() as tmp:
                binding=self.binding();binding.update(changed)
                with self.assertRaises(ValueError):
                    api().run_epoch_smoke(Tiny(),*self.loaders(),Path(tmp),binding=binding,
                                          device='cpu',model_factory=Tiny)


if __name__ == '__main__': unittest.main()
