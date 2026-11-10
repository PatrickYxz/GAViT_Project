"""Offline evidence checks; no ML imports, model execution or source images."""
import copy
import importlib.util
from pathlib import Path
import unittest


def api():
    path = Path(__file__).resolve().parents[1] / 'tools/analyze_hgtnet_aid_results.py'
    if not path.is_file():
        raise AssertionError('offline AID analysis entry is not implemented')
    spec = importlib.util.spec_from_file_location('aid_results_analysis', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AIDResultsAnalysisTests(unittest.TestCase):
    def test_metrics_keep_zero_support_class_in_macro_average(self):
        # Dropping the unused third class would incorrectly report macro-F1 66.67%.
        result = api().reconstruct_metrics([0, 0, 1], [0, 1, 1], ['a', 'b', 'c'])
        self.assertEqual(result['confusion_matrix'], [[1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.assertAlmostEqual(result['accuracy_percent'], 200 / 3)
        self.assertAlmostEqual(result['macro_f1_percent'], 400 / 9)
        self.assertEqual([r['support'] for r in result['per_class']], [2, 1, 0])
        self.assertEqual(result['per_class'][2]['f1_percent'], 0.0)
        self.assertEqual(result['per_class'][2]['accuracy_percent'], 0.0)
        self.assertAlmostEqual(result['per_class'][0]['precision_percent'], 100.0)
        self.assertAlmostEqual(result['per_class'][1]['precision_percent'], 50.0)

    def test_prediction_order_and_labels_must_equal_original_manifest(self):
        # Sorting or aligning a changed packet would hide evidence corruption.
        manifest = {'class_names': ['a', 'b'], 'records': [
            {'path': 'b/2.jpg', 'label': 1, 'split': 'test'},
            {'path': 'a/1.jpg', 'label': 0, 'split': 'test'},
            {'path': 'a/3.jpg', 'label': 0, 'split': 'val'}]}
        packet = {'paths': ['a/1.jpg', 'b/2.jpg'], 'labels': [0, 1], 'predictions': [1, 1]}
        rows = api().validate_predictions(packet, manifest, 'test')
        self.assertEqual([r['path'] for r in rows], ['a/1.jpg', 'b/2.jpg'])
        changed = copy.deepcopy(packet)
        changed['paths'].reverse()
        changed['labels'].reverse()
        changed['predictions'].reverse()
        with self.assertRaises(ValueError):
            api().validate_predictions(changed, manifest, 'test')
        changed = copy.deepcopy(packet)
        changed['labels'] = [1, 0]
        with self.assertRaises(ValueError):
            api().validate_predictions(changed, manifest, 'test')
        changed = copy.deepcopy(packet)
        changed['predictions'] = [1, 2]
        with self.assertRaises(ValueError):
            api().validate_predictions(changed, manifest, 'test')

    def test_metric_reconstruction_rejects_invalid_or_truncated_predictions(self):
        for labels, predictions in [([0, 1], [0]), ([0], [True]), ([0], [2]), ([], [])]:
            with self.subTest(labels=labels, predictions=predictions), self.assertRaises(ValueError):
                api().reconstruct_metrics(labels, predictions, ['a', 'b'])

    def test_validation_ties_select_earliest_epoch_and_require_complete_order(self):
        # Using >= instead of > would choose epoch 3 rather than epoch 2.
        history = [{'epoch': 1, 'val_correct': 8}, {'epoch': 2, 'val_correct': 9},
                   {'epoch': 3, 'val_correct': 9}, {'epoch': 4, 'val_correct': 8}]
        self.assertEqual(api().earliest_best_epoch(history), 2)
        with self.assertRaises(ValueError):
            api().earliest_best_epoch([history[0], history[2]])


if __name__ == '__main__':
    unittest.main()
