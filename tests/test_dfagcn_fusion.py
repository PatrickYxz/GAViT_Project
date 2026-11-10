"""Order/dimension/gradient checks for the published concatenation equation."""
import importlib
import unittest

import torch


def api():
    try:
        return importlib.import_module('external_baselines.dfagcn.fusion')
    except ModuleNotFoundError as exc:
        raise AssertionError('DFAGCN fusion is not implemented') from exc


def branches(dtype=torch.float64):
    return [torch.full((2, dim), value, dtype=dtype, requires_grad=True)
            for dim, value in ((512, 5.0), (512, 4.0), (512, 3.0), (4096, 2.0))]


class FusionTests(unittest.TestCase):
    def test_order_dimensions_and_gradient_match_equation(self):
        parts = branches()
        out = api().weighted_concat(*parts, alpha=0.7, beta=0.3, gamma=0.4)
        self.assertEqual(out.shape, (2, 5632))
        expected = torch.cat([parts[0], parts[1] * 0.7, parts[2] * 0.3, parts[3] * 0.4], dim=1)
        torch.testing.assert_close(out, expected)
        out.sum().backward()
        for part, coefficient in zip(parts, (1.0, 0.7, 0.3, 0.4)):
            torch.testing.assert_close(part.grad, torch.full_like(part, coefficient))

    def test_zero_weights_keep_slots_without_normalization(self):
        parts = branches()
        out = api().weighted_concat(*parts, alpha=0.0, beta=0.0, gamma=0.0)
        torch.testing.assert_close(out[:, :512], parts[0])
        torch.testing.assert_close(out[:, 512:], torch.zeros(2, 512 * 2 + 4096, dtype=torch.float64))

    def test_paper_coefficients_explicit_by_dataset(self):
        self.assertEqual(api().paper_coefficients('AID'), dict(alpha=0.4, beta=0.3, gamma=1.1))
        self.assertEqual(api().paper_coefficients('NWPU'), dict(alpha=0.7, beta=0.3, gamma=0.4))
        with self.assertRaises(ValueError):
            api().paper_coefficients('BigEarthNet')
        with self.assertRaises(TypeError):
            api().weighted_concat(*branches())

    def test_rejects_invalid_coefficients(self):
        module = api()
        for coefficient in (float('nan'), float('inf'), -0.1, True, '0.3'):
            with self.subTest(coefficient=coefficient), self.assertRaises((TypeError, ValueError)):
                module.weighted_concat(*branches(), alpha=coefficient, beta=0.3, gamma=0.4)

    def test_rejects_invalid_or_mismatched_branch(self):
        module = api()
        invalid = [torch.ones(2, 511), torch.ones(1, 512), torch.ones(2, 512, 1),
                   torch.ones(2, 512, dtype=torch.long), torch.full((2, 512), float('nan')),
                   torch.empty(0, 512), [0.0]]
        for bad in invalid:
            with self.subTest(shape=getattr(bad, 'shape', None)):
                parts = branches(dtype=torch.float32)
                parts[1] = bad
                with self.assertRaises((TypeError, ValueError)):
                    module.weighted_concat(*parts, alpha=0.7, beta=0.3, gamma=0.4)

    def test_rejects_mixed_dtype_and_device(self):
        module = api()
        for bad in (torch.ones(2, 4096, dtype=torch.float64), torch.empty(2, 4096, device='meta')):
            parts = branches(dtype=torch.float32)
            parts[3] = bad
            with self.subTest(dtype=bad.dtype, device=bad.device), self.assertRaises(ValueError):
                module.weighted_concat(*parts, alpha=0.7, beta=0.3, gamma=0.4)

    def test_rejects_finite_inputs_whose_weighted_output_overflows(self):
        parts = branches(dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            api().weighted_concat(*parts, alpha=1e10, beta=0.3, gamma=0.4)


if __name__ == '__main__':
    unittest.main()
