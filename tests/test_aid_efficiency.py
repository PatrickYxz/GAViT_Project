"""Offline checks for a GPU probe; these do not establish CUDA performance."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TOOL = Path(__file__).resolve().parents[1] / 'tools/benchmark_aid.py'


class EfficiencyTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), 'E06 benchmark implementation is missing')
        spec = importlib.util.spec_from_file_location('aid_efficiency', TOOL)
        self.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.probe)

    def test_async_work_is_finished_before_stopping_timer(self):
        now, pending = [0.0], [0.0]
        costs = iter([8.0, 9.0, .002, .004])
        def forward():
            pending[0] += next(costs)
        def synchronize():
            now[0] += pending[0]
            pending[0] = 0.0
        observed = self.probe.measure(forward, synchronize, lambda: now[0], 2, 2)
        self.assertAlmostEqual(observed[0], 2.0)
        self.assertAlmostEqual(observed[1], 4.0)
        self.assertEqual(pending[0], 0.0)

    def test_throughput_uses_total_time_and_batch_size(self):
        result = self.probe.aggregate([[10, 30], [10, 30]], 32)
        self.assertEqual(result['batch_ms_mean'], 20)
        self.assertEqual(result['images_per_second'], 1600)
        self.assertEqual(result['amortized_ms_per_image'], .625)
        self.assertEqual(result['repeat_batch_ms_means'], [20, 20])
        for samples in ([[0]], [[float('nan')]], [], [[]]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                self.probe.aggregate(samples, 32)

    def test_other_compute_process_is_rejected(self):
        self.probe.check_process_rows('123\n', own_pid=123)
        self.probe.check_process_rows('', own_pid=123)
        for rows in ('123\n456\n', 'N/A\n'):
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.probe.check_process_rows(rows, own_pid=123)

    def test_stage_failure_stops_queue_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fake = root / 'worker.py'
            fake.write_text('import pathlib,sys\np=pathlib.Path(sys.argv[1]); p.open("a").write(sys.argv[-3]+"\\n"); sys.exit(7 if len(p.read_text().splitlines())==2 else 0)\n')
            events = root / 'events'
            with self.assertRaises(subprocess.CalledProcessError):
                self.probe.run_stages(root / 'out', [sys.executable, str(fake), str(events)])
            self.assertEqual(len(events.read_text().splitlines()), 2)
            self.assertEqual((root / 'out/swin_b1.exit').read_text(), 'exit_code=0\n')
            self.assertEqual((root / 'out/gavit_b1.exit').read_text(), 'exit_code=7\n')
            with self.assertRaises(FileExistsError):
                self.probe.run_stages(root / 'out', [])

    def test_mismatched_pair_or_incomplete_results_are_rejected(self):
        reports = []
        for model, batch in self.probe.STAGES:
            reports.append({'model': model, 'batch_size': batch, 'status': 'complete',
                            'hardware': {'device_name': 'gpu'}, 'gpu_uuid': 'same', 'driver_version': 'same', 'probe_sha256': 'same',
                            'common_identity': {'manifest': 'same'},
                            'input_sha256': str(batch), 'protocol': {'steps': 50}})
        self.probe.validate_pair(reports)
        reports[-1]['hardware']['device_name'] = 'other'
        with self.assertRaises(ValueError):
            self.probe.validate_pair(reports)
        with self.assertRaises(ValueError):
            self.probe.validate_pair(reports[:3])


if __name__ == '__main__':
    unittest.main()
