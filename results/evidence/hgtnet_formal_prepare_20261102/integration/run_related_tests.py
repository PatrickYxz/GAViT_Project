import json,os,sys,time,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];os.chdir(ROOT);sys.path.insert(0,str(ROOT))
started=time.perf_counter();loader=unittest.TestLoader()
suite=unittest.TestSuite([loader.discover('tests',pattern='test_dfagcn_*.py'),loader.discover('tests',pattern='test_hgtnet_*.py')])
result=unittest.TextTestRunner(verbosity=2).run(suite)
record=dict(scope=['test_dfagcn_*.py','test_hgtnet_*.py'],tests_run=result.testsRun,failures=len(result.failures),
    errors=len(result.errors),skipped=len(result.skipped),successful=result.wasSuccessful(),seconds=time.perf_counter()-started,
    python=sys.version,not_claimed='unrelated legacy dependency suite, targetGPU, realNWPU or formal metrics')
Path(__file__).with_name('related_suite.json').write_text(json.dumps(record,indent=2)+'\n')
sys.exit(0 if result.wasSuccessful() else 1)
