"""Checks for case selection, captured topology and diagnostic reports."""
import csv
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest


def diagnostic():
    try:
        return importlib.import_module('tools.diagnose_bigearth_feedback')
    except ModuleNotFoundError as exc:
        raise AssertionError('feedback diagnostic is not implemented') from exc


class FeedbackDiagnosticTests(unittest.TestCase):
    def fixture(self, root):
        root = Path(root)
        (root / 'images').mkdir()
        (root / 'images/row_8.jpg').write_bytes(b'fixture for file hashing only')
        (root / 'model.py').write_text('# source identity fixture\n')
        run = dict(status='complete', scope='validation_subset_diagnostic_not_formal_test',
                   source_sha256={'model.py': diagnostic().digest(root / 'model.py')}, diagnostic_sha256={})
        (root / 'run.json').write_text(json.dumps(run))
        (root / 'cases.csv').write_text('class_index,bucket,label,row_index,image\n14,both_wrong,1,8,images/row_8.jpg\n')
        row = dict(row_index='8', patch_path='/not-extracted/patch_8')
        row.update({f'label_{c}': int(c == 14) for c in range(19)})
        row.update({f'gavit_score_{c}': .2 for c in range(19)})
        with (root / 'predictions.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, list(row)); writer.writeheader(); writer.writerow(row)
        return SimpleNamespace(source=root, repo=root, pair_tools=root, input_mode='gallery')

    def test_gallery_preparation_works_without_extracted_data_or_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            _, cases = diagnostic().prepare(args)
            self.assertEqual(cases[0]['patch_path'], '/not-extracted/patch_8')
            self.assertEqual(cases[0]['input_sha256'], diagnostic().digest(args.source / 'images/row_8.jpg'))

    def test_stops_for_missing_gallery_file_and_changed_model_source(self):
        for failure in ('image', 'source'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(tmp)
                if failure == 'image':
                    (args.source / 'images/row_8.jpg').unlink()
                else:
                    (args.repo / 'model.py').write_text('# modified\n')
                with self.assertRaises((FileNotFoundError, ValueError)):
                    diagnostic().prepare(args)

    def test_rejects_nonfinite_probability_and_nonbinary_cached_label(self):
        for replacement in ('nan', '2'):
            with self.subTest(value=replacement), tempfile.TemporaryDirectory() as tmp:
                args = self.fixture(tmp)
                path = args.source / 'predictions.csv'
                with path.open(newline='') as f:
                    row = next(csv.DictReader(f))
                row['gavit_score_0' if replacement == 'nan' else 'label_0'] = replacement
                with path.open('w', newline='') as f:
                    writer = csv.DictWriter(f, list(row)); writer.writeheader(); writer.writerow(row)
                with self.assertRaises(ValueError):
                    diagnostic().prepare(args)

    def test_original_mode_fingerprints_the_premerged_input_used_by_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            args.input_mode = 'original'
            patch = args.source / 'patch_8'; patch.mkdir()
            (patch / 'patch_8.png').write_bytes(b'premerged RGB fixture')
            path = args.source / 'predictions.csv'
            path.write_text(path.read_text().replace('/not-extracted/patch_8', str(patch)))
            _, cases = diagnostic().prepare(args)
            self.assertEqual(cases[0]['input_sha256'], {'patch_8.png': diagnostic().digest(patch / 'patch_8.png')})

    def test_cli_retains_source_identity_and_refuses_output_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(tmp)
            out = args.source / 'output'
            command = [sys.executable, diagnostic().__file__, '--source', tmp, '--repo', tmp,
                       '--pair-tools', tmp, '--out', str(out), '--prepare-only']
            result = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = json.loads((out / 'run.json').read_text())
            self.assertEqual(run['status'], 'prepared_only')
            self.assertEqual(run['diagnostic_sha256'], diagnostic().digest(diagnostic().__file__))
            again = subprocess.run(command, text=True, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(json.loads((out / 'run.json').read_text()), run)

    def test_selects_first_fixed_case_per_stratum_and_shares_image(self):
        cases = [
            dict(class_index='14', bucket='both_wrong', label='1', row_index='8', image='images/row_8.jpg'),
            dict(class_index='14', bucket='both_wrong', label='1', row_index='3', image='images/row_3.jpg'),
            dict(class_index='16', bucket='swin_only_correct', label='1', row_index='8', image='images/row_8.jpg'),
            dict(class_index='11', bucket='gavit_only_correct', label='0', row_index='4', image='images/row_4.jpg'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cases.csv'
            with path.open('w', newline='') as f:
                writer = csv.DictWriter(f, list(cases[0])); writer.writeheader(); writer.writerows(cases)
            selected = diagnostic().select_cases(path)
        self.assertEqual([r['row_index'] for r in selected], [4, 8])
        self.assertEqual([r['class_index'] for r in selected[1]['strata']], [14, 16])

    def test_rejects_gallery_path_outside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['../other.jpg', '/tmp/other.jpg']:
                with self.subTest(name=name), self.assertRaises(ValueError):
                    diagnostic().gallery_path(Path(tmp), name)

    def graph(self):
        assignments = [(y // 2) * 4 + x // 2 for y in range(7) for x in range(7)]
        edges = []
        for q in range(16):
            y, x = divmod(q, 4)
            for s in range(16):
                a, b = divmod(s, 4)
                if abs(a-y) + abs(b-x) == 1:
                    edges.append([s, q])
        edges += [[(q+d) % 16, q] for q in range(16) for d in (5, 10)]
        return assignments, edges

    def test_classifies_captured_edges_and_actual_region_sizes(self):
        assignment, edges = self.graph()
        counts, kinds = diagnostic().validate_topology(assignment, edges)
        self.assertEqual(counts, [4,4,4,2,4,4,4,2,4,4,4,2,2,2,2,1])
        self.assertEqual(kinds.count('spatial'), 48)
        self.assertEqual(kinds.count('feature'), 32)

    def test_rejects_duplicate_self_cross_graph_and_wrong_incoming_budget(self):
        assignment, edges = self.graph()
        for replacement in [[0,0], [16,0], edges[0], [5,1]]:
            broken = [*edges[:-1], replacement]
            with self.subTest(edge=replacement), self.assertRaises(ValueError):
                diagnostic().validate_topology(assignment, broken)

    def test_svg_contains_captured_edges_not_a_rebuilt_knn_graph(self):
        assignment, edges = self.graph()
        _, kinds = diagnostic().validate_topology(assignment, edges)
        svg = diagnostic().graph_svg(assignment, edges, kinds, [1.0]*16)
        self.assertEqual(svg.count('class="edge spatial"'), 48)
        self.assertEqual(svg.count('class="edge feature"'), 32)
        self.assertIn('data-source="5" data-target="0"', svg)
        self.assertEqual(svg.count('class="token"'), 49)

    def test_summary_respects_true_label_and_does_not_turn_enriched_cases_into_ap(self):
        records = [dict(row_index=1, strata=[dict(class_index=14, label=1)],
                        normal_scores=[.1]*14+[.8]+[.1]*4,
                        zero_scores=[.1]*14+[.2]+[.1]*4),
                   dict(row_index=2, strata=[dict(class_index=14, label=0)],
                        normal_scores=[.1]*14+[.8]+[.1]*4,
                        zero_scores=[.1]*14+[.2]+[.1]*4)]
        report = diagnostic().summarize(records)
        self.assertEqual(report['14']['feedback_corrects'], 1)
        self.assertEqual(report['14']['feedback_breaks'], 1)
        self.assertEqual(report['14']['n'], 2)
        self.assertNotIn('AP', report['14'])


if __name__ == '__main__':
    unittest.main()
