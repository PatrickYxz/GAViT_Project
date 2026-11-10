"""Registered probe delivery uses the real immutable zipapp boundary."""
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import package_hgtnet_nwpu_smoke as delivery


PROBE = 'external_baselines/hgtnet/batch_probe.py'


class BatchProbeDeliveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repository = self.root / 'repository'
        sources = {
            'external_baselines/__init__.py': '',
            'external_baselines/hgtnet/__init__.py': '',
            'external_baselines/hgtnet/model.py': 'raise RuntimeError("ML module imported")\n',
            'configs/hgtnet_independent_v0.json': '{}\n',
            'tools/run_dfagcn_nwpu_smoke.py': 'HELPER = 7\n',
            'tools/run_hgtnet_nwpu_smoke.py': 'HELPER = 42\n',
            PROBE: (
                'import json,os,sys\nfrom pathlib import Path\n'
                'from tools import run_hgtnet_nwpu_smoke as common\n'
                'ROOT=Path(__file__).resolve().parents[2]\n'
                'if sys.argv[1:2]==["run"]: (ROOT/"runs").mkdir()\n'
                'print(json.dumps({"route":"batch_probe","args":sys.argv[1:],'
                '"target":os.environ.get("HGTNET_TARGET_GPU"),"helper":common.HELPER,'
                '"root":str(ROOT),"entry":os.environ.get("HGTNET_DELIVERY_ENTRY"),'
                '"ml_imported":any(name in sys.modules for name in '
                '["torch","torchvision","numpy","PIL"])}))\n'),
        }
        for name, source in sources.items():
            path = self.repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        self.bundle = self.root / 'batch16.pyz'

    def build(self, **kwargs):
        with patch.object(delivery, 'ROOT', self.repository):
            return delivery.build_bundle(self.bundle, **kwargs)

    def rewrite_manifest(self, name, change):
        destination = self.root / name
        with zipfile.ZipFile(self.bundle) as source:
            entries = {item.filename:(item, source.read(item.filename)) for item in source.infolist()}
        manifest = json.loads(entries['delivery_manifest.json'][1])
        removed = change(manifest)
        description = {key:value for key,value in manifest.items() if key != 'bundle_id'}
        manifest['bundle_id'] = hashlib.sha256(json.dumps(
            description, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        entries['delivery_manifest.json'] = ('delivery_manifest.json', json.dumps(manifest).encode())
        if removed:
            del entries[removed]
        with zipfile.ZipFile(destination, 'w') as output:
            for item, raw in entries.values():
                output.writestr(item, raw)
        return destination

    def test_registered_probe_is_selected_once_and_keeps_common_runner(self):
        self.assertIn('runner_member', inspect.signature(delivery.build_bundle).parameters,
                      'builder lacks registered runner selection')
        record = self.build(target_gpu='3090', runner_member=PROBE)
        self.assertEqual(record['manifest']['runner'], PROBE)
        self.assertEqual(record['manifest']['target_gpu'], '3090')
        with zipfile.ZipFile(self.bundle) as archive:
            self.assertEqual(archive.namelist().count(PROBE), 1)
            self.assertIn('tools/run_hgtnet_nwpu_smoke.py', archive.namelist())
        _, installed = delivery.install_bundle(self.bundle)
        self.assertEqual(installed['runner'], PROBE)

    def test_cli_probe_inspect_and_status_dispatch_without_ml_or_runs(self):
        builder = self.repository / 'tools/package_hgtnet_nwpu_smoke.py'
        shutil.copyfile(delivery.__file__, builder)
        built = subprocess.run([sys.executable, str(builder), 'build', '--output', str(self.bundle),
            '--target-gpu', '3090', '--runner-member', PROBE], capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertEqual(json.loads(built.stdout)['manifest']['runner'], PROBE)
        guard = self.root / 'guard'
        guard.mkdir()
        (guard / 'sitecustomize.py').write_text(
            'import sys\nclass Guard:\n'
            ' def find_spec(self,fullname,*args):\n'
            '  if fullname.split(".")[0] in {"torch","torchvision","numpy","PIL"}:\n'
            '   raise RuntimeError("forbidden ML import: "+fullname)\n'
            'sys.meta_path.insert(0,Guard())\n')
        environment = dict(os.environ, PYTHONPATH=str(guard), HGTNET_TARGET_GPU='3080')
        for action in ('inspect', 'status'):
            with self.subTest(action=action):
                result = subprocess.run([sys.executable, str(self.bundle), action, '--output',
                    'path with spaces'], capture_output=True, text=True, env=environment)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout), {
                    'route':'batch_probe', 'args':[action, '--output', 'path with spaces'],
                    'target':'3090', 'helper':42, 'root':str(self.bundle.with_suffix('')),
                    'entry':str(self.bundle), 'ml_imported':False})
                self.assertFalse((self.bundle.with_suffix('') / 'runs').exists())

    def test_unregistered_runners_are_rejected_without_files(self):
        for runner in ('external_baselines/hgtnet/model.py', 'tools/unknown.py', '', None, 42):
            with self.subTest(runner=runner), self.assertRaises(ValueError):
                self.build(runner_member=runner)
            self.assertFalse(self.bundle.exists())
            self.assertFalse(self.bundle.with_suffix('.pyz.manifest.json').exists())

    def test_selected_probe_must_exist_before_bundle_creation(self):
        (self.repository / PROBE).unlink()
        with self.assertRaisesRegex(ValueError, 'runner|missing'):
            self.build(runner_member=PROBE)
        self.assertFalse(self.bundle.exists())
        self.assertFalse(self.bundle.with_suffix('.pyz.manifest.json').exists())

    def test_rehashed_manifest_cannot_select_an_unregistered_source(self):
        self.build(runner_member=PROBE)
        for index, runner in enumerate(('external_baselines/hgtnet/model.py',
                                       'external_baselines/hgtnet/batch_porbe.py')):
            with self.subTest(runner=runner):
                def change(manifest):
                    manifest['runner'] = runner
                invalid = self.rewrite_manifest('unregistered' + str(index) + '.pyz', change)
                with self.assertRaises(ValueError):
                    delivery.install_bundle(invalid)
                self.assertFalse(invalid.with_suffix('').exists())

    def test_rehashed_manifest_cannot_select_a_registered_but_absent_probe(self):
        self.build(runner_member=PROBE)
        def change(manifest):
            del manifest['files'][PROBE]
            return PROBE
        invalid = self.rewrite_manifest('absent.pyz', change)
        with self.assertRaises(ValueError):
            delivery.install_bundle(invalid)
        self.assertFalse(invalid.with_suffix('').exists())

    def test_reuse_refuses_changed_selected_probe(self):
        self.build(runner_member=PROBE)
        workspace, _ = delivery.install_bundle(self.bundle)
        source = workspace / PROBE
        source.write_text('raise RuntimeError("changed")\n')
        with self.assertRaisesRegex(ValueError, 'immutable source changed'):
            delivery.install_bundle(self.bundle)
        self.assertEqual(source.read_text(), 'raise RuntimeError("changed")\n')

    def test_cli_rejects_unregistered_runner_before_creating_bundle(self):
        builder = self.repository / 'tools/package_hgtnet_nwpu_smoke.py'
        shutil.copyfile(delivery.__file__, builder)
        result = subprocess.run([sys.executable, str(builder), 'build', '--output', str(self.bundle),
            '--runner-member', 'external_baselines/hgtnet/model.py'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.bundle.exists())


if __name__ == '__main__':
    unittest.main()
