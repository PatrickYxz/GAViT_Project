"""Run only the changed HGTNet and protected DFAGCN regression scope."""
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[4]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
started = time.perf_counter()
loader = unittest.TestLoader()
suite = unittest.TestSuite([loader.discover('tests', pattern='test_dfagcn_*.py'),
                           loader.discover('tests', pattern='test_hgtnet_*.py')])
result = unittest.TextTestRunner(verbosity=2).run(suite)
record = {'scope': ['test_dfagcn_*.py', 'test_hgtnet_*.py'], 'tests_run': result.testsRun,
          'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped),
          'successful': result.wasSuccessful(), 'seconds': time.perf_counter() - started,
          'command': list(sys.argv), 'python': sys.version,
          'not_claimed': 'full-repository legacy dependency suite; remote/GPU/dataset accuracy'}
target = Path(__file__).with_name('related_suite.json')
target.write_text(json.dumps(record, indent=2) + '\n')
sys.exit(0 if result.wasSuccessful() else 1)
