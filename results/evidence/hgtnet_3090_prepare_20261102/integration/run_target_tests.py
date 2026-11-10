"""Hardware/CLI/delivery tests requiring no unavailable local ML runtime."""
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[4]
os.chdir(ROOT);sys.path.insert(0,str(ROOT));started=time.perf_counter()
loader=unittest.TestLoader();patterns=['test_hgtnet_3090_*.py','test_hgtnet_resources.py','test_hgtnet_nwpu_delivery.py']
suite=unittest.TestSuite([loader.discover('tests',pattern=pattern) for pattern in patterns])
result=unittest.TextTestRunner(verbosity=2).run(suite)
record={'scope':patterns,'tests_run':result.testsRun,'successful':result.wasSuccessful(),
        'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'seconds':time.perf_counter()-started,'python':sys.version,
        'limitation':'full numeric suite could not import torch/numpy in the expired temporary runtime; see full_suite_attempt.log. No numeric model/data/engine code changed.'}
Path(__file__).with_name('targeted_suite.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
