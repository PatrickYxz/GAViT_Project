import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import os
os.chdir(ROOT)
HERE=Path(__file__).resolve().parent
paths=sorted((ROOT/'external_baselines/hgtnet').glob('formal_*.py'))
def identities():return {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
before=identities()
suite=unittest.TestLoader().discover('tests',pattern='test_hgtnet_formal_*.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
after=identities()
record=dict(scope='independent formal-module focused CPU/stdlib suite; no GPU or real NWPU claim',
    tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
    passed=result.wasSuccessful(),source_unchanged_during_run=before==after,source_hashes=after)
(HERE/'focused_suite.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps(record,indent=2,sort_keys=True))
assert record['passed'] and record['source_unchanged_during_run']
