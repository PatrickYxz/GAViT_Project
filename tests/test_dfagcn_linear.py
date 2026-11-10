"""Actual official LIBLINEAR CLI boundary checks, not a substitute classifier."""
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


def api():
    try:
        return importlib.import_module('external_baselines.dfagcn.linear')
    except ModuleNotFoundError as exc:
        raise AssertionError('DFAGCN LIBLINEAR adapter is not implemented') from exc


BIN_ROOT = Path(os.environ.get(
    'GAVIT_LIBLINEAR_ROOT',
    str(Path(__file__).resolve().parents[1] / 'results/evidence/dfagcn_pipeline_20261029/liblinear/bin')))


def fixture():
    # Three axis-separated classes; fourth column intentionally entirely zero.
    values = np.array([[-3., 0., 0., 0.], [-2., 0., 0., 0.], [-4., 0., 0., 0.],
                       [0., 3., 0., 0.], [0., 2., 0., 0.], [0., 4., 0., 0.],
                       [3., 0., 0., 0.], [2., 0., 0., 0.], [4., 0., 0., 0.]])
    return values, np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)


class LiblinearBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def train(self, *, destination='fit', **kwargs):
        x, y = fixture()
        return api().train_linear(x, y, self.root / destination,
                                  train_binary=BIN_ROOT / 'train', C=10.0, **kwargs)

    def test_actual_multiclass_cli_preserves_labels_and_trailing_zero_width(self):
        model = self.train()
        metadata = json.loads((model.parent / 'metadata.json').read_text())
        self.assertEqual(metadata['nr_feature'], 4)
        self.assertEqual(metadata['class_labels'], [0, 1, 2])
        self.assertEqual(metadata['solver_type'], 'L2R_L2LOSS_SVC')
        self.assertEqual(metadata['bias'], 1.0)
        for row in (model.parent / 'train.svm').read_text().splitlines():
            self.assertTrue(row.endswith('4:0'))
            self.assertFalse(any(item.startswith('0:') for item in row.split()[1:]))
        probes = np.array([[-5., 0., 0., 0.], [0., 5., 0., 0.], [5., 0., 0., 0.]])
        actual = api().predict_linear(probes, model, self.root / 'pred', predict_binary=BIN_ROOT / 'predict')
        self.assertEqual(actual, [0, 1, 2])
        self.assertEqual(metadata['return_code'], 0)
        self.assertEqual(len(metadata['binary_sha256']), 64)
        self.assertEqual(len(metadata['model_sha256']), 64)
        self.assertNotIn('accuracy', json.loads((self.root / 'pred' / 'metadata.json').read_text()))

    def test_fit_and_predict_output_directories_are_exclusive(self):
        model = self.train()
        with self.assertRaises(FileExistsError):
            self.train()
        x, _ = fixture()
        api().predict_linear(x, model, self.root / 'pred', predict_binary=BIN_ROOT / 'predict')
        with self.assertRaises(FileExistsError):
            api().predict_linear(x, model, self.root / 'pred', predict_binary=BIN_ROOT / 'predict')

    def test_modified_model_bytes_are_rejected_before_prediction(self):
        model = self.train()
        model.write_text(model.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'model|SHA|identity'):
            api().predict_linear(fixture()[0], model, self.root / 'pred', predict_binary=BIN_ROOT / 'predict')
        self.assertFalse((self.root / 'pred').exists())

    def test_prediction_rejects_feature_width_mismatch(self):
        model = self.train()
        for width in (3, 5):
            with self.subTest(width=width), self.assertRaises(ValueError):
                api().predict_linear(np.ones((2, width)), model, self.root / 'pred',
                                     predict_binary=BIN_ROOT / 'predict')

    def test_fit_rejects_nonfinite_empty_or_nonfloating_features(self):
        module = api()
        invalid = [np.empty((0, 2)), np.empty((2, 0)), np.ones(2), np.ones((2, 2, 2)),
                   np.ones((2, 2), dtype=np.int64), np.ones((2, 2), dtype=np.complex64),
                   np.array([[np.nan, 1.], [1., 0.]]), np.array([[np.inf, 1.], [1., 0.]])]
        for x in invalid:
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises((TypeError, ValueError)):
                module.train_linear(x, [0, 1], self.root / 'fit', train_binary=BIN_ROOT / 'train', C=1.0)
        self.assertFalse((self.root / 'fit').exists())

    def test_fit_rejects_invalid_labels_and_hyperparameters(self):
        module = api()
        x = np.ones((2, 2))
        for labels in ([0], [0., 1.], [-1, 0], [0, 2], [1, 2], [0, 0], [False, True]):
            with self.subTest(labels=labels), self.assertRaises((TypeError, ValueError)):
                module.train_linear(x, labels, self.root / 'fit', train_binary=BIN_ROOT / 'train', C=1.)
        for value in (0., -1., np.nan, np.inf, True, '1'):
            with self.subTest(C=value), self.assertRaises((TypeError, ValueError)):
                module.train_linear(x, [0, 1], self.root / 'fit', train_binary=BIN_ROOT / 'train', C=value)
        for solver in (True, 2.0, 11, 21):
            with self.subTest(solver=solver), self.assertRaises((TypeError, ValueError)):
                module.train_linear(x, [0, 1], self.root / 'fit', train_binary=BIN_ROOT / 'train', C=1., solver=solver)
        for bias in (np.nan, np.inf, True, '1'):
            with self.subTest(bias=bias), self.assertRaises((TypeError, ValueError)):
                module.train_linear(x, [0, 1], self.root / 'fit', train_binary=BIN_ROOT / 'train', C=1., bias=bias)

    def test_missing_binary_and_failed_cli_do_not_create_success_metadata(self):
        module = api()
        with self.assertRaises((FileNotFoundError, ValueError)):
            module.train_linear(*fixture(), self.root / 'missing', train_binary=self.root / 'absent', C=1.)
        with self.assertRaises(RuntimeError):
            module.train_linear(*fixture(), self.root / 'failed', train_binary='/usr/bin/false', C=1.)
        self.assertTrue((self.root / 'failed' / 'stderr.log').exists())
        self.assertFalse((self.root / 'failed' / 'metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
