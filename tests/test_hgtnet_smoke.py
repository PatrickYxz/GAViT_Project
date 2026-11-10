"""Guard local synthetic engineering checks from destructive or false success."""
import importlib
import tempfile
from pathlib import Path
import unittest

import torch
from torch import nn


def api():
    try:
        return importlib.import_module('external_baselines.hgtnet.smoke')
    except ModuleNotFoundError as exc:
        raise AssertionError('HGTNet smoke API is missing') from exc


class SmokeTests(unittest.TestCase):
    def test_existing_output_cannot_be_overwritten_or_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp, 'keep.txt'); marker.write_text('user data')
            with self.assertRaises(FileExistsError):
                api().prepare_output(tmp)
            self.assertEqual(marker.read_text(), 'user data')

    def test_new_output_does_not_allow_path_through_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp, 'link'); link.symlink_to(Path(tmp, 'missing'))
            with self.assertRaises(FileExistsError):
                api().prepare_output(link)

    def test_gradients_report_rejects_missing_and_nonfinite(self):
        m = nn.Module(); m.head = nn.Linear(2, 2)
        report = api().gradient_report(m)
        self.assertFalse(report['passed'])
        self.assertIn('head.weight', report['missing'])
        for p in m.parameters():
            p.grad = torch.ones_like(p)
        report = api().gradient_report(m)
        self.assertTrue(report['passed'])
        self.assertGreater(report['groups']['head']['l2'], 0)
        m.head.bias.grad[0] = float('inf')
        self.assertFalse(api().gradient_report(m)['passed'])

    def test_finite_zero_gradient_cannot_pass_as_learning(self):
        m = nn.Module(); m.head = nn.Linear(2, 2)
        for p in m.parameters():
            p.grad = torch.zeros_like(p)
        self.assertFalse(api().gradient_report(m)['passed'])

    def test_bad_cli_never_claims_complete(self):
        module = importlib.import_module('external_baselines.hgtnet.__main__')
        with self.assertRaises(SystemExit) as caught:
            module.main(['smoke', '--output', '/unused', '--weights', '/fake'])
        self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
