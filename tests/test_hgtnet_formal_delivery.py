import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from tools import package_hgtnet_nwpu_smoke as delivery

class FormalDeliveryTests(unittest.TestCase):
    def test_formal_registered_runner_contains_exact_recipe(self):
        self.assertEqual(delivery.FORMAL_RUNNER,'external_baselines/hgtnet/formal_entry.py')
        with tempfile.TemporaryDirectory() as d:
            bundle=Path(d).resolve()/'formal.pyz'
            result=delivery.build_bundle(bundle,target_gpu='3090',runner_member=delivery.FORMAL_RUNNER)
            manifest,raw,files=delivery._read_bundle(bundle)
            self.assertEqual(manifest['runner'],delivery.FORMAL_RUNNER)
            self.assertEqual(files['configs/hgtnet_nwpu_formal_v1.json'],Path('configs/hgtnet_nwpu_formal_v1.json').read_bytes())
            self.assertEqual(hashlib.sha256(bundle.read_bytes()).hexdigest(),result['sha256'])

    def test_existing_default_runner_does_not_gain_formal_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            result=delivery.build_bundle(Path(d).resolve()/'default.pyz')
            self.assertEqual(result['manifest']['runner'],delivery.RUNNER)
            self.assertNotIn('configs/hgtnet_nwpu_formal_v1.json',result['manifest']['files'])

    def test_formal_runner_wrong_gpu_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                delivery.build_bundle(Path(d).resolve()/'wrong.pyz',target_gpu='3080',runner_member=delivery.FORMAL_RUNNER)

    def test_actual_formal_zipapp_resume_reaches_parser(self):
        with tempfile.TemporaryDirectory() as d:
            bundle=Path(d).resolve()/'formal.pyz'
            delivery.build_bundle(bundle,target_gpu='3090',runner_member=delivery.FORMAL_RUNNER)
            result=subprocess.run([sys.executable,str(bundle),'resume','--help'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--gpu-uuid',result.stdout)

if __name__=='__main__':unittest.main()
